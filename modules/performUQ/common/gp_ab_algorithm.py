"""
Module implementing the GP-AB Algorithm for Bayesian calibration.

It includes classes and functions for performing Gaussian Process modeling,
Principal Component Analysis, and various convergence metrics.
"""

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Literal

import common_datamodels
import convergence_metrics
import numpy as np
import pydantic

# sys.path.append(
#     "/Users/aakash/SimCenter/SimCenterBackendApplications/modules/performUQ/common"
# )
import uq_utilities
from adaptive_doe import AdaptiveDesignOfExperiments
from gp_model import GaussianProcessModel
from principal_component_analysis import PrincipalComponentAnalysis
from scipy.stats import invgamma, norm
from space_filling_doe import LatinHypercubeSampling
from tmcmc import TMCMC, calculate_warm_start_stage


def log_likelihood(prediction_error_vector, prediction_error_variance):
    """
    Calculate the log-likelihood of the prediction errors given the variance.

    Args:
        prediction_error_vector (np.ndarray): The vector of prediction errors.
        prediction_error_variance (float): The variance of the prediction errors.

    Returns
    -------
        float: The log-likelihood value.
    """
    return np.sum(
        norm.logpdf(prediction_error_vector, 0, np.sqrt(prediction_error_variance))
    )


def _response_approximation(current_gp, current_pca, model_parameters):
    """
    Approximate the response using the current GP model and PCA.

    Args:
        current_gp (GaussianProcessModel): The current Gaussian Process model.
        current_pca (PrincipalComponentAnalysis): The current PCA model.
        model_parameters (np.ndarray): The model parameters.

    Returns
    -------
        np.ndarray: The approximated response.
    """
    latent_predictions, _ = current_gp.predict(model_parameters)
    gp_prediction = current_pca.project_back_to_original_space(latent_predictions)
    return gp_prediction  # noqa: RET504


class GP_AB_Algorithm:
    """
    A class to represent the GP-AB Algorithm for Bayesian calibration.

    Attributes
    ----------
        data (np.ndarray): The observed data.
        input_dimension (int): The input dimension of the model.
        output_dimension (int): The output dimension of the model.
        output_length_list (list[int]): The list of output lengths.
        domain (list[tuple[float, float]]): The domain for each dimension.
        prior_variances (np.ndarray): The prior variances.
        prior_pdf_function (callable): The prior PDF function.
        log_likelihood_function (callable): The log-likelihood function.
        pca_threshold (float): The threshold for PCA.
        gkl_threshold (float): The threshold for GKL.
        gmap_threshold (float): The threshold for GMAP.
        max_simulations (int): The maximum number of simulations.
        max_computational_time (float): The maximum computational time.
        start_time (float): The start time of the algorithm.
        converged (bool): Whether the algorithm has converged.
        budget_exceeded (bool): Whether the budget has been exceeded.
        terminate (bool): Whether to terminate the algorithm.
        num_experiments (list[int]): The number of experiments.
        num_recalibration_experiments (int): The number of recalibration experiments.
        recalibration_ratio (float): The recalibration ratio.
        sample_transformation_function (callable): The sample transformation function.
        model_evaluation_function (callable): The model evaluation function.
        run_type (str): The run type (e.g., "runningLocal").
        parallel_pool (uq_utilities.ParallelPool): The parallel pool instance.
        parallel_evaluation_function (callable): The parallel evaluation function.
        loocv_threshold (float): The threshold for LOOCV.
        results (dict): The results dictionary.
        current_gp_model (GaussianProcessModel): The current GP model.
        current_pca (PrincipalComponentAnalysis): The current PCA model.
        samples_dict (dict): The dictionary of samples.
        betas_dict (dict): The dictionary of betas.
        log_likelihoods_dict (dict): The dictionary of log-likelihoods.
        log_target_density_values_dict (dict): The dictionary of log-target density values.
        log_evidence_dict (dict): The dictionary of log evidence.
    """

    def __init__(  # noqa: PLR0913
        self,
        data,
        output_length_list,
        output_names_list,  # noqa: ARG002
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
        run_type='runningLocal',
        loocv_threshold=0.2,
        recalibration_ratio=0.1,
        gkl_threshold=0.01,
        gmap_threshold=0.01,
    ):
        """
        Initialize the GP_AB_Algorithm class.

        Args:
            data (np.ndarray): The observed data.
            output_length_list (list[int]): The list of output lengths.
            output_names_list (list[str]): The list of output names.
            input_dimension (int): The input dimension of the model.
            output_dimension (int): The output dimension of the model.
            domain (list[tuple[float, float]]): The domain for each dimension.
            model_evaluation_function (callable): The model evaluation function.
            sample_transformation_function (callable): The sample transformation function.
            prior_pdf_function (callable): The prior PDF function.
            log_likelihood_function (callable): The log-likelihood function.
            prior_variances (np.ndarray): The prior variances.
            max_simulations (int, optional): The maximum number of simulations. Defaults to np.inf.
            max_computational_time (float, optional): The maximum computational time. Defaults to np.inf.
            pca_threshold (float, optional): The threshold for PCA. Defaults to 0.999.
            run_type (str, optional): The run type (e.g., "runningLocal"). Defaults to "runningLocal".
            loocv_threshold (float, optional): The threshold for LOOCV. Defaults to 0.2.
            recalibration_ratio (float, optional): The recalibration ratio. Defaults to 0.1.
            gkl_threshold (float, optional): The threshold for GKL. Defaults to 0.01.
            gmap_threshold (float, optional): The threshold for GMAP. Defaults to 0.01.
        """
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

        self.results = {}
        self.current_gp_model = GaussianProcessModel(
            self.input_dimension, 1, ARD=True
        )
        self.current_pca = PrincipalComponentAnalysis(self.pca_threshold)

        self.samples_dict = {}
        self.betas_dict = {}
        self.log_likelihoods_dict = {}
        self.log_target_density_values_dict = {}
        self.log_evidence_dict = {}

    def _evaluate_in_parallel(self, func, samples):
        """
        Evaluate the model in parallel using the provided function and samples.

        Args:
            func (callable): The function to evaluate.
            samples (np.ndarray): The samples to evaluate.

        Returns
        -------
            np.ndarray: The evaluated outputs.
        """
        transformed_samples = self.sample_transformation_function(samples)
        simulation_numbers = np.arange(len(samples))
        iterable = zip(simulation_numbers, transformed_samples)
        outputs = np.atleast_2d(
            list(self.parallel_evaluation_function(func, iterable))
        )
        return outputs  # noqa: RET504

    def _perform_initial_doe(self, n_samples):
        """
        Perform the initial Design of Experiments (DoE) using Latin Hypercube Sampling.

        Args:
            n_samples (int): The number of samples to generate.

        Returns
        -------
            np.ndarray: The generated samples.
        """
        self.initial_doe = LatinHypercubeSampling(
            n_samples=n_samples, n_dimensions=self.input_dimension
        )
        samples = self.initial_doe.generate(self.domain)
        return samples  # noqa: RET504

    def _get_initial_training_set(self, n_samples):
        """
        Get the initial training set by performing DoE and evaluating the model.

        Args:
            n_samples (int): The number of samples to generate.

        Returns
        -------
            tuple: A tuple containing the inputs and outputs of the initial training set.
        """
        inputs = self._perform_initial_doe(n_samples)
        outputs = self._evaluate_in_parallel(self.model_evaluation_function, inputs)
        return inputs, outputs

    def _log_likelihood_approximation(self, response_approximation, samples):
        """
        Approximate the log-likelihood for the given samples.

        Args:
            response_approximation (callable): The response approximation function.
            samples (np.ndarray): The samples to evaluate.

        Returns
        -------
            np.ndarray: The approximated log-likelihood values.
        """
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
        """
        Calculate the log-prior PDF for the given samples.

        Args:
            samples (np.ndarray): The samples to evaluate.

        Returns
        -------
            np.ndarray: The log-prior PDF values.
        """
        u_values = samples[:, : self.input_dimension]
        model_parameters = self.sample_transformation_function(u_values)
        log_prior_model_parameters = np.log(
            self.prior_pdf_function(model_parameters)
        ).reshape((-1, 1))
        q = samples[:, self.input_dimension :]
        log_prior_q = np.sum(invgamma.logpdf(q, 1, scale=0.5), axis=1)
        prior_log_pdf = log_prior_model_parameters + log_prior_q
        return prior_log_pdf  # noqa: RET504

    def _log_posterior_approximation(self, samples, log_likelihoods):
        """
        Approximate the log-posterior for the given samples and log-likelihoods.

        Args:
            samples (np.ndarray): The samples to evaluate.
            log_likelihoods (np.ndarray): The log-likelihood values.

        Returns
        -------
            np.ndarray: The approximated log-posterior values.
        """
        log_prior = self._log_prior_pdf(samples)
        log_posterior = log_likelihoods + log_prior
        return log_posterior  # noqa: RET504

    def loocv_measure(self, log_likelihood_approximation):
        """
        Calculate the Leave-One-Out Cross-Validation (LOOCV) measure.

        Args:
            log_likelihood_approximation (callable): The log-likelihood approximation function.

        Returns
        -------
            float: The LOOCV measure.
        """
        lls = log_likelihood_approximation(self.inputs)
        loo_predictions = self.current_gp_model.loo_predictions(self.outputs)
        loocv_measure = convergence_metrics.calculate_loo_nrmse_w(
            loo_predictions, lls
        )
        return loocv_measure  # noqa: RET504

    def run_iteration(self, k):
        """
        Run a single iteration of the GP-AB Algorithm.

        Args:
            k (int): The iteration number.

        Returns
        -------
            tuple: A tuple containing a boolean indicating whether to terminate and the current TMCMC results.
        """
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
            self.current_gp_model.fit(
                model_parameters, self.latent_outputs, reoptimize=False
            )

        # Step 1.2: GP predictive model
        gp_prediction_latent_mean, gp_prediction_latent_variance = (
            self.current_gp_model.predict(model_parameters)
        )
        gp_prediction_mean = self.current_pca.project_back_to_original_space(  # noqa: F841
            gp_prediction_latent_mean
        )
        loo_predictions_latent_space = self.current_gp_model.loo_predictions(
            self.latent_outputs
        )
        loo_predictions = self.current_pca.project_back_to_original_space(  # noqa: F841
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

        samples_dict = {}
        betas_dict = {}
        log_likelihoods_dict = {}
        log_target_density_values_dict = {}
        log_evidence_dict = {}

        num_samples_per_stage = 50
        if k > 0:
            # loocv_measure = self.loocv_measure()
            # if loocv_measure < self.loocv_threshold:
            #     j_star = self._calculate_warm_start_stage(
            #         log_likelihood_approximation, self.betas_dict
            #     )
            j_star = calculate_warm_start_stage(
                log_likelihood_approximation,
                self.results,
            )
            previous_log_likelihood_approximation = log_likelihood_approximation
            # previous_log_posterior_approximation = log_posterior_approximation

        # Step 2.2: Sequential MC sampling
        if j_star == 0:
            samples = self._perform_initial_doe(num_samples_per_stage)
            samples_transformed = self.sample_transformation_function(samples)
            log_target_density_values = np.log(
                self.prior_pdf_function(samples_transformed)
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
                samples_dict[j] = self.results['samples_dict'][j]
                betas_dict[j] = self.results['betas_dict'][j]
                log_likelihoods_dict[j] = self.results['log_likelihood_values_dict'][
                    j
                ]
                log_target_density_values_dict[j] = self.results[
                    'log_target_density_values_dict'
                ][j]
                log_evidence_dict[j] = self.results['log_evidence_dict'][j]

        self.results = tmcmc.run(
            samples_dict,
            betas_dict,
            log_likelihoods_dict,
            log_target_density_values_dict,
            log_evidence_dict,
            np.random.default_rng(),
            j_star,
            num_burn_in=0,
        )

        self.converged = False
        if k > 0:
            # Step 3.1: Assess convergence
            gkl = convergence_metrics.calculate_gkl(
                log_likelihood_approximation,
                previous_log_likelihood_approximation,
                self.prior_pdf_function,
                samples,
            )
            gmap = convergence_metrics.calculate_gmap(
                log_likelihood_approximation,
                previous_log_likelihood_approximation,
                self.prior_pdf_function,
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
            return self.terminate, self.results

        # Step 4.1: Select variance for DoE
        n_training_points = 2 * self.input_dimension
        self.exploitation_proportion = 0.5
        n_exploit = int(np.ceil(self.exploitation_proportion * n_training_points))
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

        return self.terminate, self.results

    def run_initial_doe(self, num_initial_doe_per_dim=2):
        """
        Run the initial Design of Experiments (DoE).

        Args:
            num_initial_doe_per_dim (int, optional): Number of initial DoE samples per dimension. Defaults to 2.

        Returns
        -------
            tuple: A tuple containing the inputs, outputs, and number of initial DoE samples.
        """
        num_initial_doe_samples = num_initial_doe_per_dim * self.input_dimension
        inputs, outputs = self._get_initial_training_set(num_initial_doe_samples)
        return inputs, outputs, num_initial_doe_samples

    def write_results(self):
        """Write the results of the GP-AB Algorithm to a file."""
        # print(f'{self.iteration_number = }')
        # print('Results written to file')


def read_inputs(input_json_file):
    """
    Read and parse the input JSON file.

    Args:
        input_json_file (str): Path to the input JSON file.

    Returns
    -------
        tuple: A tuple containing UQ inputs, random variables, correlation matrix, EDP inputs, and application inputs.
    """
    with Path(input_json_file).open(encoding='utf-8') as f:
        inputs = json.load(f)

    input_data = common_datamodels.Model.model_validate(inputs)
    uq_inputs = input_data.UQ
    rv_inputs = input_data.randomVariables
    correlation_matrix_inputs = input_data.correlationMatrix
    edp_inputs = inputs['EDP']
    application_inputs = input_data.Applications

    return (
        uq_inputs,
        rv_inputs,
        correlation_matrix_inputs,
        edp_inputs,
        application_inputs,
    )


def preprocess(input_arguments):
    """
    Preprocess the input arguments for the GP-AB Algorithm.

    Args:
        input_arguments (InputArguments): The input arguments for the algorithm.

    Returns
    -------
        tuple: A tuple containing the preprocessed data required for the GP-AB Algorithm.
    """
    input_file_full_path = (
        Path(input_arguments.path_to_template_directory)
        / input_arguments.input_json_file
    )
    (
        uq_inputs,
        rv_inputs,
        correlation_matrix_inputs,
        edp_inputs,
        application_inputs,
    ) = read_inputs(input_file_full_path)

    data_file = uq_inputs.calDataFilePath / uq_inputs.calDataFile
    with data_file.open() as f:
        data = np.genfromtxt(f, delimiter=',')

    log_likelihood_file_name = uq_inputs.logLikelihoodFile
    log_likelihood_path = uq_inputs.logLikelihoodPath
    log_likelihood_function = log_likelihood
    if log_likelihood_file_name:
        sys.path.append(str(log_likelihood_path))
        ll_module = importlib.import_module(log_likelihood_file_name)
        log_likelihood_function = ll_module.log_likelihood

    joint_distribution = uq_utilities.ERANatafJointDistribution(
        rv_inputs,
        correlation_matrix_inputs,
    )
    domain = [(-3, 3) for _ in range(len(rv_inputs))]
    prior_variances = [1 for _ in range(len(rv_inputs))]  # TODO(ABS): Validate this
    # Transformation function from standard to physical space
    sample_transformation_function = joint_distribution.u_to_x

    # Prior logpdf function
    prior_pdf_function = joint_distribution.pdf

    main_script_path = str(input_arguments.path_to_template_directory)

    model = uq_utilities.get_default_model(
        list_of_rv_data=rv_inputs,
        edp_data=edp_inputs,
        list_of_dir_names_to_copy_files_from=[main_script_path],
        run_directory=input_arguments.path_to_working_directory,
        driver_filename=str(input_arguments.driver_file_name),
        workdir_prefix='workdir',
    )
    model_evaluation_function = model.evaluate_model_once

    edp_names_list = [edp['name'] for edp in edp_inputs]
    edp_lengths_list = [edp['length'] for edp in edp_inputs]
    input_dimension = len(rv_inputs)
    output_dimension = sum(
        edp_lengths_list
    )  # TODO(ABS): Validate this against length of data

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
    """
    A class to represent the input arguments for the GP-AB Algorithm.

    Attributes
    ----------
    path_to_working_directory : Path
        The path to the working directory.
    path_to_template_directory : Path
        The path to the template directory.
    run_type : Literal['runningLocal', 'runningRemote']
        The type of run (local or remote).
    driver_file_name : Path
        The name of the driver file.
    input_json_file : Path
        The path to the input JSON file.
    """

    path_to_working_directory: Path
    path_to_template_directory: Path
    run_type: Literal['runningLocal', 'runningRemote']
    driver_file_name: Path
    input_json_file: Path

    model_config = pydantic.ConfigDict(revalidate_instances='always')


def run_gp_ab_algorithm(input_arguments):
    """
    Run the GP-AB Algorithm.

    Args:
        input_arguments (InputArguments): The input arguments for the algorithm.
    """
    gp_ab = GP_AB_Algorithm(*preprocess(input_arguments))
    inputs, outputs, num_initial_doe_samples = gp_ab.run_initial_doe(
        num_initial_doe_per_dim=2
    )
    gp_ab.inputs = inputs
    gp_ab.outputs = outputs
    gp_ab.num_experiments.append(num_initial_doe_samples)
    iteration = 0
    terminate = False
    while not terminate:
        terminate, results = gp_ab.run_iteration(iteration)
        gp_ab.write_results()
        iteration += 1


def parse_arguments(args=None):
    """
    Parse command-line arguments.

    Args:
        args (list, optional): List of arguments to parse (for function calls).
                               If None, arguments are taken from `sys.argv`.

    Returns
    -------
        dict: Parsed arguments in dictionary form.
    """
    parser = argparse.ArgumentParser(
        description='Run the GP-AB Algorithm with the specified arguments.'
    )
    parser.add_argument(
        'path_to_working_directory',
        type=Path,
        help='Absolute path to the working directory.',
    )
    parser.add_argument(
        'path_to_template_directory',
        type=Path,
        help='Absolute path to the template directory.',
    )
    parser.add_argument(
        'run_type',
        choices=['runningLocal', 'runningRemote'],
        help='Type of run (local or remote).',
    )
    parser.add_argument(
        'driver_file_name', type=Path, help='Name of the driver file.'
    )
    parser.add_argument(
        'input_json_file', type=Path, help='Name of the input JSON file.'
    )

    return vars(parser.parse_args(args))  # Returns arguments as a dictionary


def main(command_args=None):
    """
    Run the GP-AB Algorithm.

    Args:
        command_args (list, optional): A list of command-line arguments.
                                       If None, uses `sys.argv[1:]`.
    """
    try:
        # Parse arguments (either from function call or command line)
        args = parse_arguments(command_args)

        # Change to the working directory
        os.chdir(args['path_to_working_directory'])

        # Validate input arguments
        input_arguments = InputArguments.model_validate(args)

        # Run the GP-AB Algorithm
        run_gp_ab_algorithm(input_arguments)

    except Exception as e:
        err_msg = f'ERROR: An exception occurred:\n{traceback.format_exc()}\n'
        sys.stderr.write(err_msg)
        raise RuntimeError(err_msg) from e


if __name__ == '__main__':
    try:
        main(sys.argv[1:])  # Runs with command-line arguments
    except Exception as e:  # noqa: BLE001
        print(f'Error: {e}', file=sys.stderr)  # noqa: T201
        sys.exit(1)
