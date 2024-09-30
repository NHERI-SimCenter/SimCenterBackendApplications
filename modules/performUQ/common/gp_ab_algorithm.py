import importlib
import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import pydantic

# sys.path.append(
#     "/Users/aakash/SimCenter/SimCenterBackendApplications/modules/performUQ/common"
# )
import uq_utilities
from scipy.stats import invgamma, norm

import common_datamodels
import convergence_metrics
from adaptive_doe import AdaptiveDesignOfExperiments
from gp_model import GaussianProcessModel
from principal_component_analysis import PrincipalComponentAnalysis
from space_filling_doe import LatinHypercubeSampling
from tmcmc import TMCMC, calculate_warm_start_stage


def log_likelihood(prediction_error_vector, prediction_error_variance):
    return np.sum(
        norm.logpdf(prediction_error_vector, 0, np.sqrt(prediction_error_variance))
    )


def _response_approximation(current_gp, current_pca, model_parameters):
    latent_predictions, _ = current_gp.predict(model_parameters)
    gp_prediction = current_pca.project_back_to_original_space(latent_predictions)
    return gp_prediction


class GP_AB_Algorithm:
    def __init__(
        self,
        data,
        output_length_list,
        output_names_list,
        input_dimension,
        output_dimension,
        domain,
        model_evaluation_function,
        sample_transformation_function,
        prior_pdf_function,
        log_likelihood_function,
        prior_variances,
        max_simulations=np.inf,
        max_computational_time=np.inf,
        pca_threshold=0.999,
        run_type="runningLocal",
        loocv_threshold=0.2,
        recalibration_ratio=0.1,
        gkl_threshold=0.01,
        gmap_threshold=0.01,
    ):
        self.data = data
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.output_length_list = output_length_list
        self.domain = domain
        self.prior_variances = prior_variances

        self.inputs = None
        self.outputs = None
        self.latent_outputs = None

        self.prior_pdf_function = prior_pdf_function
        self.log_likelihood_function = log_likelihood_function

        self.pca_threshold = pca_threshold
        self.num_pca_components_list = []

        self.gkl_threshold = gkl_threshold
        self.gmap_threshold = gmap_threshold

        self.max_simulations = max_simulations
        self.max_computational_time = max_computational_time
        self.start_time = time.time()

        self.converged = False
        self.budget_exceeded = False
        self.terminate = False

        self.num_experiments = [0]
        self.num_recalibration_experiments = 0
        self.recalibration_ratio = recalibration_ratio

        self.sample_transformation_function = sample_transformation_function
        self.model_evaluation_function = model_evaluation_function
        self.run_type = run_type
        self.parallel_pool = uq_utilities.get_parallel_pool_instance(run_type)
        self.parallel_evaluation_function = self.parallel_pool.pool.starmap

        self.loocv_threshold = loocv_threshold

        self.results = dict()
        self.current_gp_model = GaussianProcessModel(
            self.input_dimension, 1, ARD=True
        )
        self.current_pca = PrincipalComponentAnalysis(self.pca_threshold)

        self.samples_dict = dict()
        self.betas_dict = dict()
        self.log_likelihoods_dict = dict()
        self.log_target_density_values_dict = dict()
        self.log_evidence_dict = dict()

    def _evaluate_in_parallel(self, func, samples):
        transformed_samples = self.sample_transformation_function(samples)
        simulation_numbers = np.arange(len(samples))
        iterable = zip(simulation_numbers, transformed_samples)
        outputs = np.atleast_2d(
            list(self.parallel_evaluation_function(func, iterable))
        )
        return outputs

    def _perform_initial_doe(self, n_samples):
        self.initial_doe = LatinHypercubeSampling(
            n_samples=n_samples, n_dimensions=self.input_dimension
        )
        samples = self.initial_doe.generate(self.domain)
        return samples

    def _get_initial_training_set(self, n_samples):
        inputs = self._perform_initial_doe(n_samples)
        outputs = self._evaluate_in_parallel(self.model_evaluation_function, inputs)
        # initial_training_set = np.hstack((inputs, outputs))
        # return initial_training_set
        return inputs, outputs

    def _log_likelihood_approximation(self, response_approximation, samples):
        u_values = samples[:, : self.input_dimension]
        model_parameters = self.sample_transformation_function(u_values).reshape(
            1, -1
        )
        predictions = response_approximation(model_parameters)
        prediction_errors = self.data - predictions

        q = samples[:, self.input_dimension :]

        log_likes = []
        num_samples = samples.shape[0]
        for i in range(num_samples):
            ll = []
            start = 0
            for j in range(q.shape[1]):
                ll.append(
                    self.log_likelihood_function(
                        prediction_errors[
                            i, start : start + self.output_length_list[j]
                        ],
                        q[i, j],
                    )
                )
                start += self.output_length_list[j]
            log_likes.append(np.sum(ll))
        return np.array(log_likes).reshape((num_samples, 1))

    def _log_prior_pdf(self, samples):
        u_values = samples[:, : self.input_dimension]
        model_parameters = self.sample_transformation_function(u_values)
        log_prior_model_parameters = np.log(
            self.prior_pdf_function(model_parameters)
        ).reshape((-1, 1))
        # TODO: (ABS) Decide what prior to use for q
        q = samples[:, self.input_dimension :]
        log_prior_q = np.sum(invgamma.logpdf(q, 1, scale=0.5), axis=1)
        prior_log_pdf = log_prior_model_parameters + log_prior_q
        return prior_log_pdf

    def _log_posterior_approximation(self, samples, log_likelihoods):
        log_prior = self._log_prior_pdf(samples)
        log_posterior = log_likelihoods + log_prior
        return log_posterior

    def loocv_measure(self, log_likelihood_approximation):
        lls = log_likelihood_approximation(self.inputs)

        # weights = (
        #     2 / 3 * np.ones((self.database.shape[0], 1))
        #     + 1 / 3 * self.adaptive_weights_per_stage
        # )
        loo_predictions = self.current_gp_model.loo_predictions(self.outputs)
        loocv_measure = convergence_metrics.calculate_loo_nrmse_w(
            loo_predictions, lls
        )
        return loocv_measure

    def calibrate_gp(self, model_parameters, model_outputs):
        latent_outputs = self.current_pca.project_to_latent_space(model_outputs)
        self.num_pca_components_list.append(np.shape(latent_outputs)[1])
        self.current_gp_model = GaussianProcessModel(
            self.input_dimension, self.num_pca_components_list[-1], ARD=True
        )
        self.current_gp_model.fit(model_parameters, latent_outputs)
        self.num_recalibration_experiments = self.num_experiments[-1]

    def run_iteration(self, k):
        self.iteration_number = k
        model_parameters = self.inputs
        model_outputs = self.outputs

        # Step 1.1: GP calibration
        if (
            self.num_experiments[-1] - self.num_recalibration_experiments
            >= self.recalibration_ratio * self.num_recalibration_experiments
        ):
            self.previous_gp_model = self.current_gp_model
            self.latent_outputs = self.current_pca.project_to_latent_space(
                model_outputs
            )
            self.num_pca_components_list.append(np.shape(self.latent_outputs)[1])
            self.current_gp_model = GaussianProcessModel(
                self.input_dimension, self.num_pca_components_list[-1], ARD=True
            )
            self.current_gp_model.fit(model_parameters, self.latent_outputs)
            self.num_recalibration_experiments = self.num_experiments[-1]
        else:
            self.current_gp_model = self.previous_gp_model
            self.latent_outputs = self.current_pca.project_to_latent_space(
                model_outputs
            )
            self.num_pca_components_list.append(np.shape(self.latent_outputs)[1])

        # Step 1.2: GP predictive model
        gp_prediction_latent_mean, gp_prediction_latent_variance = (
            self.current_gp_model.predict(model_parameters)
        )
        gp_prediction_mean = self.current_pca.project_back_to_original_space(
            gp_prediction_latent_mean
        )
        loo_predictions_latent_space = self.current_gp_model.loo_predictions(
            self.latent_outputs
        )
        loo_predictions = self.current_pca.project_back_to_original_space(
            loo_predictions_latent_space
        )

        # Step 1.3: Posterior distribution approximation
        response_approximation = partial(
            _response_approximation, self.current_gp_model, self.current_pca
        )
        log_likelihood_approximation = partial(
            self._log_likelihood_approximation, response_approximation
        )
        log_posterior_approximation = self._log_posterior_approximation

        # Step 2.1: Evaluate warm-starting for TMCMC
        tmcmc = TMCMC(log_likelihood_approximation, log_posterior_approximation)

        j_star = 0
        beta = 0

        samples_dict = dict()
        betas_dict = dict()
        log_likelihoods_dict = dict()
        log_target_density_values_dict = dict()
        log_evidence_dict = dict()

        num_samples_per_stage = 50
        if k > 0:
            # loocv_measure = self.loocv_measure()
            # if loocv_measure < self.loocv_threshold:
            #     j_star = self._calculate_warm_start_stage(
            #         log_likelihood_approximation, self.betas_dict
            #     )
            j_star = calculate_warm_start_stage(
                log_likelihood_approximation, current_tmcmc_results
            )
            previous_log_posterior_approximation = log_posterior_approximation

        # Step 2.2: Sequential MC sampling
        if j_star == 0:
            samples = self._perform_initial_doe(num_samples_per_stage)
            log_target_density_values = np.log(
                self.prior_pdf_function(samples)
            ).reshape((-1, 1))
            log_likelihood_values = np.ones_like(log_target_density_values)
            log_evidence = 0
            beta = 0

            samples_dict[j_star] = samples
            betas_dict[j_star] = beta
            log_likelihoods_dict[j_star] = log_likelihood_values
            log_target_density_values_dict[j_star] = log_target_density_values
            log_evidence_dict[j_star] = log_evidence
        else:
            for j in range(j_star):
                samples_dict[j] = self.samples_dict[j]
                betas_dict[j] = self.betas_dict[j]
                log_likelihoods_dict[j] = self.log_likelihoods_dict[j]
                log_target_density_values_dict[j] = (
                    self.log_target_density_values_dict[j]
                )
                log_evidence_dict[j] = self.log_evidence_dict[j]

        current_tmcmc_results = tmcmc.run(
            samples_dict,
            betas_dict,
            log_likelihoods_dict,
            log_target_density_values_dict,
            log_evidence_dict,
            np.random.default_rng(),
            j_star,
            num_burn_in=0,
        )

        # Step 3.1: Assess convergence
        gkl = convergence_metrics.calculate_gkl(
            log_posterior_approximation,
            previous_log_posterior_approximation,
            samples,
        )
        gmap = convergence_metrics.calculate_gmap(
            log_posterior_approximation,
            previous_log_posterior_approximation,
            samples,
            self.prior_variances,
        )
        self.converged = gkl < self.gkl_threshold and gmap < self.gmap_threshold

        # Step 3.2: Computational budget related termination
        self.budget_exceeded = (len(self.inputs) >= self.max_simulations) or (
            time.time() - self.start_time >= self.max_computational_time
        )
        self.terminate = self.converged or self.budget_exceeded
        if self.terminate:
            return self.terminate, current_tmcmc_results

        # Step 4.1: Select variance for DoE
        n_training_points = 2 * self.input_dimension
        self.exploitation_proportion = 0.5
        n_exploit = np.ceil(self.exploitation_proportion * n_training_points)
        current_doe = AdaptiveDesignOfExperiments(
            self.current_gp_model, self.current_pca, self.domain
        )

        # Step 4.2: Exploitation DoE
        exploitation_training_points = current_doe.select_training_points(
            self.inputs, n_exploit, samples, 1000
        )
        self.inputs = np.vstack([self.inputs, exploitation_training_points])

        # Step 4.3: Exploration DoE
        exploration_training_points = current_doe.select_training_points(
            self.inputs, n_training_points - n_exploit, samples, 1000
        )
        self.inputs = np.vstack([self.inputs, exploration_training_points])

        new_training_points = np.vstack(
            [exploitation_training_points, exploration_training_points]
        )
        self.num_experiments.append(len(self.inputs))

        # Step 5: Response estimation
        new_training_outputs = self._evaluate_in_parallel(
            self.model_evaluation_function, new_training_points
        )
        self.outputs = np.vstack([self.outputs, new_training_outputs])

        return self.terminate, current_tmcmc_results

    def run_initial_doe(self, num_initial_doe_per_dim=2):
        num_initial_doe_samples = num_initial_doe_per_dim * self.input_dimension
        inputs, outputs = self._get_initial_training_set(num_initial_doe_samples)
        return inputs, outputs, num_initial_doe_samples

    def write_results(self):
        print(f"{self.iteration_number = }")
        print("Results written to file")


def main(input_arguments):
    gp_ab = GP_AB_Algorithm(*preprocess(input_arguments))
    inputs, outputs, num_initial_doe_samples = gp_ab.run_initial_doe(
        num_initial_doe_per_dim=2
    )
    gp_ab.inputs = inputs
    gp_ab.outputs = outputs
    gp_ab.num_experiments.append(num_initial_doe_samples)
    iteration = 0
    while not gp_ab.terminate:
        gp_ab.terminate, gp_ab.results = gp_ab.run_iteration(iteration)
        gp_ab.write_results()
        iteration += 1


def read_inputs(input_json_file):
    with open(input_json_file, encoding="utf-8") as f:
        inputs = json.load(f)

    # application_inputs = inputs["Applications"]
    # uq_inputs = inputs["UQ"]
    # rv_inputs = inputs["randomVariables"]
    # correlation_matrix_inputs = inputs["correlationMatrix"]
    # edp_inputs = inputs["EDP"]

    input_data = common_datamodels.Model.model_validate(inputs)
    uq_inputs = input_data.UQ
    rv_inputs = input_data.randomVariables
    correlation_matrix_inputs = input_data.correlationMatrix
    edp_inputs = inputs["EDP"]
    application_inputs = input_data.Applications

    # application_inputs = common_datamodels.Applications.model_validate(
    #     inputs["Applications"]
    # )
    # uq_inputs = GP_AB_UQData.model_validate(inputs["UQ"])
    # rv_inputs = inputs["randomVariables"]
    # correlation_matrix_inputs = common_datamodels.CorrelationMatrix.from_list(
    #     inputs["correlationMatrix"]
    # )
    # edp_inputs = common_datamodels.EDP(inputs["EDP"])

    return (
        uq_inputs,
        rv_inputs,
        correlation_matrix_inputs,
        edp_inputs,
        application_inputs,
    )


def preprocess(input_arguments):
    (
        uq_inputs,
        rv_inputs,
        correlation_matrix_inputs,
        edp_inputs,
        application_inputs,
    ) = read_inputs(input_arguments.input_json_file)

    data_file = (
        uq_inputs.calibration_data_path / uq_inputs.calibration_data_file_name
    )
    with open(data_file, "r") as f:
        data = np.genfromtxt(f, delimiter=",")

    log_likelihood_file_name = uq_inputs.log_likelihood_file_name
    log_likelihood_path = uq_inputs.log_likelihood_path
    log_likelihood_function = log_likelihood
    if log_likelihood_file_name:
        sys.path.append(str(log_likelihood_path))
        ll_module = importlib.import_module(log_likelihood_file_name)
        log_likelihood_function = getattr(ll_module, "log_likelihood")

    joint_distribution = uq_utilities.ERANatafJointDistribution(
        rv_inputs,
        correlation_matrix_inputs,
    )
    domain = [(-3, 3) for _ in range(len(rv_inputs))]
    prior_variances = [1 for _ in range(len(rv_inputs))]  # TODO: (ABS) Validate this
    # Transformation function from standard to physical space
    sample_transformation_function = joint_distribution.u_to_x

    # Prior logpdf function
    prior_pdf_function = joint_distribution.pdf

    main_script_path = str(application_inputs.FEM.ApplicationData.MS_Path)

    model = uq_utilities.get_default_model(
        list_of_rv_data=rv_inputs,
        edp_data=edp_inputs,
        list_of_dir_names_to_copy_files_from=[main_script_path],
        run_directory=input_arguments.path_to_working_directory,
        driver_filename=str(input_arguments.driver_file_name),
        workdir_prefix="workdir",
    )
    model_evaluation_function = model.evaluate_model_once

    edp_names_list = [edp["name"] for edp in edp_inputs]
    edp_lengths_list = [edp["length"] for edp in edp_inputs]
    input_dimension = len(rv_inputs)
    output_dimension = sum(
        edp_lengths_list
    )  # TODO: (ABS) Validate this against length of data

    max_simulations = np.inf
    max_computational_time = np.inf
    pca_threshold = 0.999
    run_type = input_arguments.run_type
    loocv_threshold = 0.2
    recalibration_ratio = 0.1

    return (
        data,
        edp_lengths_list,
        edp_names_list,
        input_dimension,
        output_dimension,
        domain,
        model_evaluation_function,
        sample_transformation_function,
        prior_pdf_function,
        log_likelihood_function,
        prior_variances,
        max_simulations,
        max_computational_time,
        pca_threshold,
        run_type,
        loocv_threshold,
        recalibration_ratio,
    )


class InputArguments(pydantic.BaseModel):
    path_to_working_directory: Path
    path_to_template_directory: Path
    run_type: Literal["runningLocal", "runningRemote"]
    driver_file_name: Path
    input_json_file: Path

    model_config = pydantic.ConfigDict(revalidate_instances="always")


if __name__ == "__main__":
    this = Path(__file__).resolve()
    os.chdir("/Users/aakash/Documents/quoFEM/LocalWorkDir/tmp.SimCenter")
    args = {
        "path_to_working_directory": Path(sys.argv[1]).resolve(),
        "path_to_template_directory": Path(sys.argv[2]).resolve(),
        "run_type": sys.argv[3],
        "driver_file_name": Path(sys.argv[4]).resolve(),
        "input_json_file": Path(sys.argv[5]).resolve(),
    }
    input_arguments = InputArguments.model_validate(args)
    main(input_arguments)
    os.chdir(this.parent)
