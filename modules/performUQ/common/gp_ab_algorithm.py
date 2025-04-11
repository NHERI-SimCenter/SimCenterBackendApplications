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
import pandas as pd
import pydantic

# sys.path.append(
#     "/Users/aakash/SimCenter/SimCenterBackendApplications/modules/performUQ/common"
# )
import uq_utilities
from adaptive_doe import AdaptiveDesignOfExperiments
from gp_model import GaussianProcessModel
from principal_component_analysis import PrincipalComponentAnalysis
from scipy.special import logsumexp
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
        gcv_threshold (float): The threshold for LOOCV.
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
        output_names_list,
        rv_names_list,
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
        gcv_threshold=0.2,
        recalibration_ratio=0.1,
        num_samples_per_stage=5000,
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
            gcv_threshold (float, optional): The threshold for LOOCV. Defaults to 0.2.
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

        self.rv_names_list = rv_names_list
        self.output_names_list = output_names_list

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

        self.gcv_threshold = gcv_threshold

        self.results = {}
        self.current_gp_model = GaussianProcessModel(
            self.input_dimension, 1, ARD=True
        )
        self.current_pca = PrincipalComponentAnalysis(
            self.pca_threshold, perform_scaling=True
        )
        self.gp_recalibrated = False

        self.samples_dict = {}
        self.betas_dict = {}
        self.log_likelihoods_dict = {}
        self.log_target_density_values_dict = {}
        self.log_evidence_dict = {}

        self.previous_posterior_samples = None
        self.current_posterior_samples = None
        self.previous_model_parameters = None
        self.current_model_parameters = None

        self.num_samples_per_stage = num_samples_per_stage

    def _evaluate_in_parallel(
        self, func, model_parameters, simulation_number_start=0
    ):
        """
        Evaluate the model in parallel using the provided function and samples.

        Args:
            func (callable): The function to evaluate.
            samples (np.ndarray): The samples to evaluate.

        Returns
        -------
            np.ndarray: The evaluated outputs.
        """
        # transformed_samples = self.sample_transformation_function(samples)
        simulation_numbers = np.arange(
            simulation_number_start, simulation_number_start + len(model_parameters)
        )
        iterable = zip(simulation_numbers, model_parameters)
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
        inputs = self.sample_transformation_function(
            self._perform_initial_doe(n_samples)
        )
        outputs = self._evaluate_in_parallel(self.model_evaluation_function, inputs)
        return inputs, outputs

    def _log_like(self, predictions):
        nd, ny = self.data.shape
        nynd = ny * nd
        epsilon = 1e-12  # to handle cases where the variance is zero

        sum_y2 = np.sum(self.data**2, axis=0) + epsilon

        log_likes = []
        num_samples = predictions.shape[0]
        for i in range(num_samples):
            prediction_errors = self.data - predictions[i, :].reshape((1, ny))
            sse = np.sum(prediction_errors**2, axis=0) + epsilon
            log_ratios = np.log(sse) - np.log(sum_y2)
            log_sum = logsumexp(log_ratios)
            ll = -0.5 * nynd * log_sum
            log_likes.append(ll)
        return np.array(log_likes).reshape((num_samples, 1))

    def _log_likelihood_approximation(
        self, response_approximation_function, model_parameters
    ):
        """
        Approximate the log-likelihood for the given samples.

        Args:
            response_approximation (callable): The response approximation function.
            samples (np.ndarray): The samples to evaluate.

        Returns
        -------
            np.ndarray: The approximated log-likelihood values.
        """
        # u_values = np.atleast_2d(samples[:, : self.input_dimension])
        # model_parameters = self.sample_transformation_function(u_values).reshape(
        #     u_values.shape
        # )
        predictions = response_approximation_function(model_parameters)
        log_likes = self._log_like(predictions)
        return log_likes  # noqa: RET504

    def _log_prior_pdf(self, model_parameters):
        """
        Calculate the log-prior PDF for the given samples.

        Args:
            samples (np.ndarray): The samples to evaluate.

        Returns
        -------
            np.ndarray: The log-prior PDF values.
        """
        # u_values = samples[:, : self.input_dimension]
        # model_parameters = self.sample_transformation_function(u_values)
        log_prior_model_parameters = np.log(
            self.prior_pdf_function(model_parameters)
        ).reshape((-1, 1))
        return log_prior_model_parameters  # noqa: RET504

    def _log_posterior_approximation(self, model_parameters, log_likelihoods):
        """
        Approximate the log-posterior for the given samples and log-likelihoods.

        Args:
            samples (np.ndarray): The samples to evaluate.
            log_likelihoods (np.ndarray): The log-likelihood values.

        Returns
        -------
            np.ndarray: The approximated log-posterior values.
        """
        log_prior = self._log_prior_pdf(model_parameters)
        log_posterior = log_likelihoods + log_prior
        return log_posterior  # noqa: RET504

    def _calculate_gcv(self, weights=None):
        """
        Calculate the Leave-One-Out Cross-Validation (LOOCV) measure.

        Args:
            log_likelihood_approximation (callable): The log-likelihood approximation function.

        Returns
        -------
            float: The LOOCV measure.
        """
        loo_predictions = self.current_gp_model.loo_predictions(self.outputs)
        loocv_measure = convergence_metrics.calculate_gcv(
            loo_predictions, self.outputs, weights=weights
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
        ):  # sufficient number of experiments for recalibration
            if k > 0:
                self.previous_gp_model = self.current_gp_model
                self.previous_pca = self.current_pca
                self.previous_response_approximation = partial(
                    _response_approximation,
                    self.previous_gp_model,
                    self.previous_pca,
                )
                self.previous_log_likelihood_approximation = partial(
                    self._log_likelihood_approximation,
                    self.previous_response_approximation,
                )

            self.current_pca = PrincipalComponentAnalysis(self.pca_threshold)
            self.current_pca.fit(model_outputs)
            self.latent_outputs = self.current_pca.project_to_latent_space(
                model_outputs
            )
            self.num_pca_components_list.append(self.current_pca.n_components)
            self.current_gp_model = GaussianProcessModel(
                self.input_dimension, self.num_pca_components_list[-1], ARD=True
            )
            self.current_gp_model.update(
                model_parameters, self.latent_outputs, reoptimize=True
            )
            self.num_recalibration_experiments = self.num_experiments[-1]
            self.gp_recalibrated = True

        else:  # no recalibration of GP model
            self.current_gp_model = self.previous_gp_model
            self.current_pca = self.previous_pca

            self.latent_outputs = self.current_pca.project_to_latent_space(
                model_outputs
            )
            self.num_pca_components_list.append(np.shape(self.latent_outputs)[1])
            self.current_gp_model.update(
                model_parameters, self.latent_outputs, reoptimize=False
            )
            self.gp_recalibrated = False

        # Step 1.2: GP predictive model
        gp_prediction_latent_mean, gp_prediction_latent_variance = (
            self.current_gp_model.predict(model_parameters)
        )
        self.gp_prediction_mean = self.current_pca.project_back_to_original_space(
            gp_prediction_latent_mean
        )
        loo_predictions_latent_space = self.current_gp_model.loo_predictions(
            self.latent_outputs
        )
        self.loo_predictions = self.current_pca.project_back_to_original_space(
            loo_predictions_latent_space
        )

        # Step 1.3: Posterior distribution approximation
        self.current_response_approximation = partial(
            _response_approximation, self.current_gp_model, self.current_pca
        )
        self.current_log_likelihood_approximation = partial(
            self._log_likelihood_approximation, self.current_response_approximation
        )
        self.log_posterior_approximation = self._log_posterior_approximation

        # Step 2.1: Evaluate warm-starting for TMCMC
        tmcmc = TMCMC(
            self.current_log_likelihood_approximation,
            self.log_posterior_approximation,
            self.sample_transformation_function,
        )

        self.j_star = 0
        beta = 0

        samples_dict = {}
        model_parameters_dict = {}
        betas_dict = {}
        log_likelihoods_dict = {}
        log_target_density_values_dict = {}
        log_evidence_dict = {}

        if k > 0:
            self.gcv = self._calculate_gcv()
            self.warm_start_possible = self.gcv < self.gcv_threshold
            if self.warm_start_possible:
                self.j_star = calculate_warm_start_stage(
                    self.current_log_likelihood_approximation,
                    self.results,
                )

            self.previous_model_parameters = self.results['model_parameters_dict']
            max_stage = len(self.previous_model_parameters) - 1
            self.previous_posterior_samples = self.previous_model_parameters[
                max_stage
            ]

        # Step 2.2: Sequential MC sampling
        if self.j_star == 0:
            initial_samples = self._perform_initial_doe(self.num_samples_per_stage)
            model_parameters_initial = self.sample_transformation_function(
                initial_samples
            )
            log_target_density_values = np.log(
                self.prior_pdf_function(model_parameters_initial)
            ).reshape((-1, 1))
            log_likelihood_values = self.current_log_likelihood_approximation(
                model_parameters_initial
            )
            log_evidence = 0
            beta = 0
            stage_num = 0

            samples_dict[self.j_star] = initial_samples
            model_parameters_dict[self.j_star] = model_parameters_initial
            betas_dict[self.j_star] = beta
            log_likelihoods_dict[self.j_star] = log_likelihood_values
            log_target_density_values_dict[self.j_star] = log_target_density_values
            log_evidence_dict[self.j_star] = log_evidence
        else:
            for j in range(self.j_star):
                samples_dict[j] = self.results['samples_dict'][j]
                model_parameters_dict[j] = self.results['model_parameters_dict'][j]
                betas_dict[j] = self.results['betas_dict'][j]
                log_likelihoods_dict[j] = self.results['log_likelihood_values_dict'][
                    j
                ]
                log_target_density_values_dict[j] = self.results[
                    'log_target_density_values_dict'
                ][j]
                log_evidence_dict[j] = self.results['log_evidence_dict'][j]
            stage_num = self.j_star - 1

        self.results = tmcmc.run(
            samples_dict,
            model_parameters_dict,
            betas_dict,
            log_likelihoods_dict,
            log_target_density_values_dict,
            log_evidence_dict,
            np.random.default_rng(),
            stage_num,
            num_burn_in=0,
        )

        self.current_model_parameters = self.results['model_parameters_dict']
        max_stage = len(self.current_model_parameters) - 1
        self.current_posterior_samples = self.current_model_parameters[max_stage]

        self.converged = False
        if k > 0:
            combined_samples = np.vstack(
                [self.previous_posterior_samples, self.current_posterior_samples]
            )  # type: ignore
            # Step 3.1: Assess convergence
            self.gkl = convergence_metrics.calculate_gkl(
                self.current_log_likelihood_approximation,
                self.previous_log_likelihood_approximation,
                self._log_prior_pdf,
                combined_samples,
            )
            self.gkl_converged = self.gkl < self.gkl_threshold
            self.gmap = convergence_metrics.calculate_gmap(
                self.current_log_likelihood_approximation,
                self.previous_log_likelihood_approximation,
                self.prior_pdf_function,
                combined_samples,
                self.prior_variances,
            )
            self.gmap_converged = self.gmap < self.gmap_threshold
            # self.converged = self.gkl_converged and self.gmap_converged
            self.converged = self.gkl_converged

        # Step 3.2: Computational budget related termination
        num_simulations = 0
        if self.inputs is not None:
            num_simulations = len(self.inputs)
        self.budget_exceeded = (num_simulations >= self.max_simulations) or (
            time.time() - self.start_time >= self.max_computational_time
        )
        self.terminate = self.converged or self.budget_exceeded
        if self.terminate:
            return self.terminate, self.results

        # Step 4.1: Select variance for DoE
        n_training_points = 2 * self.input_dimension
        self.exploitation_proportion = 0.5
        n_exploit = int(np.ceil(self.exploitation_proportion * n_training_points))
        n_explore = n_training_points - n_exploit
        current_doe = AdaptiveDesignOfExperiments(
            self.current_gp_model, self.current_pca, self.domain
        )

        # Step 4.2: Exploitation DoE
        self.exploitation_training_points = current_doe.select_training_points(
            self.inputs, n_exploit, self.current_posterior_samples, 1000
        )
        self.inputs = np.vstack([self.inputs, self.exploitation_training_points])

        # Step 4.3: Exploration DoE
        self.exploration_training_points = current_doe.select_training_points(
            self.inputs, n_explore, self.current_posterior_samples, 1000
        )
        self.inputs = np.vstack([self.inputs, self.exploration_training_points])

        self.new_training_points = np.vstack(
            [self.exploitation_training_points, self.exploration_training_points]
        )
        self.num_experiments.append(len(self.inputs))

        # Step 5: Response estimation
        simulation_number_start = 0
        if self.outputs is not None:
            simulation_number_start = len(self.outputs)
        self.new_training_outputs = self._evaluate_in_parallel(
            self.model_evaluation_function,
            self.new_training_points,
            simulation_number_start=simulation_number_start,
        )
        self.outputs = np.vstack([self.outputs, self.new_training_outputs])

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

    def save_tabular_results(
        self,
        samples,
        predictions,
        rv_names_list,
        output_names_list,
        output_length_list,
        iteration_number,
        output_dir: Path,
    ):
        # Step 1: Construct headers
        pred_headers = []
        for name, length in zip(output_names_list, output_length_list):
            if length == 1:
                pred_headers.append(name)
            else:
                pred_headers.extend([f'{name}_{i+1}' for i in range(length)])

        # Step 2: Create DataFrame
        tabular_data = pd.DataFrame(samples, columns=rv_names_list)

        pred_df = pd.DataFrame(predictions, columns=pred_headers)
        df_combined = pd.concat([tabular_data, pred_df], axis=1)

        # Step 3: Save to TSV
        filename = output_dir / f'tabular_results_{iteration_number}.txt'
        df_combined.to_csv(filename, sep='\t', index=False)
        print(f'Saved: {filename}')

    def _make_json_serializable(self, obj):
        """Recursively convert NumPy and other non-serializable types to JSON-serializable Python types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):  # noqa: RET505
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(item) for item in obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif obj is None:
            return None
        else:
            msg = f'Object of type {type(obj)} is not JSON serializable: {obj}'
            raise TypeError(msg)

    def _save_gp_ab_progress(self):
        """Save the current progress of the GP-AB Algorithm to a file."""
        data = {
            'iteration_number': self.iteration_number,
            'inputs': self.inputs[: len(self.gp_prediction_mean)],
            'outputs': self.outputs[: len(self.gp_prediction_mean)],
            'gp_prediction_mean': self.gp_prediction_mean,
            'loo_predictions': self.loo_predictions,
            'exploitation_training_points': self.exploitation_training_points,
            'exploration_training_points': self.exploration_training_points,
            'num_latent_variables': self.current_pca.n_components,
            'explained_variance_ratio': self.current_pca.pca.explained_variance_ratio_,
            'gp_recalibrated': self.gp_recalibrated,
        }

        if self.iteration_number > 0:
            data.update(
                {
                    'gcv': self.gcv,
                    'warm_start_possible': self.warm_start_possible,
                    'warm_start_stage': self.j_star,
                    'gkl': self.gkl,
                    'gkl_converged': self.gkl_converged,
                    'gmap': self.gmap,
                    'gmap_converged': self.gmap_converged,
                    'converged': self.converged,
                    'budget_exceeded': self.budget_exceeded,
                    'terminate': self.terminate,
                }
            )

        return self._make_json_serializable(data)

    def write_results(self, results_dir='results'):
        """Write the results of the GP-AB Algorithm to a file."""
        res_dir = Path(results_dir)
        res_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        serializable_data = self._make_json_serializable(self.results)
        outfile_path = res_dir / f'tmcmc_results_{self.iteration_number}.json'
        with outfile_path.open('w') as f:
            json.dump(serializable_data, f, indent=4)

        outfile_gp_ab_path = res_dir / f'gp_ab_progress_{self.iteration_number}.json'
        with outfile_gp_ab_path.open('w') as f:
            json.dump(self._save_gp_ab_progress(), f, indent=4)

        samples = self.current_posterior_samples
        predictions = _response_approximation(
            self.current_gp_model, self.current_pca, samples
        )
        self.save_tabular_results(
            samples=samples,
            predictions=predictions,
            rv_names_list=self.rv_names_list,
            output_names_list=self.output_names_list,
            output_length_list=self.output_length_list,
            iteration_number=self.iteration_number,
            output_dir=res_dir,
        )


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
        input_arguments.path_to_template_directory / input_arguments.input_json_file
    )
    (
        uq_inputs,
        rv_inputs,
        correlation_matrix_inputs,
        edp_inputs,
        application_inputs,
    ) = read_inputs(input_file_full_path)

    data_file = input_arguments.path_to_working_directory / uq_inputs.calDataFile
    with data_file.open() as f:
        data = np.atleast_2d(np.genfromtxt(f, delimiter=','))

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
    prior_variances = [
        (marginal.Dist.var()) ** 2
        for marginal in joint_distribution.ERANataf_object.Marginals
    ]
    # Transformation function from standard to physical space
    sample_transformation_function = joint_distribution.u_to_x
    # Prior logpdf function
    prior_pdf_function = joint_distribution.pdf

    domain = [(-3, 3) for _ in range(len(rv_inputs))]

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

    rv_names_list = [rv['name'] for rv in rv_inputs]

    edp_names_list = [edp['name'] for edp in edp_inputs]
    edp_lengths_list = [edp['length'] for edp in edp_inputs]
    input_dimension = len(rv_inputs)
    output_dimension = sum(
        edp_lengths_list
    )  # TODO(ABS): Validate this against length of data

    # TODO(ABS): Make the following parameters configurable by reading from a
    # config.json file
    max_simulations = np.inf
    max_computational_time = np.inf
    pca_threshold = 0.999
    run_type = input_arguments.run_type
    gcv_threshold = 0.2
    recalibration_ratio = 0.1
    num_samples_per_stage = 1000
    gkl_threshold = 0.01
    gmap_threshold = 0.01

    return (
        data,
        edp_lengths_list,
        edp_names_list,
        rv_names_list,
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
        gcv_threshold,
        recalibration_ratio,
        num_samples_per_stage,
        gkl_threshold,
        gmap_threshold,
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
    gp_ab.parallel_pool.close_pool()


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
