"""
Module implementing the GP-AB Algorithm for Bayesian calibration.

It includes classes and functions for performing Gaussian Process modeling,
Bayesian calibration, design of computer experiments, and convergence monitoring.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import shutil
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import common_datamodels
import convergence_metrics
import numpy as np
import pandas as pd
import pydantic
import uq_utilities
from adaptive_doe import AdaptiveDesignOfExperiments
from gp_model import create_gp_model
from kernel_density_estimation import GaussianKDE
from log_likelihood_functions import (
    log_like,
    log_likelihood_approx,
    log_prior,
    response_approximation,
)
from logging_utilities import (
    LoggerAutoFlusher,
    LogStepContext,
    _format_duration,
    log_exception,
    make_log_info,
    make_logger_context,
    setup_logger,
)
from space_filling_doe import LatinHypercubeSampling
from tmcmc import TMCMC, calculate_log_evidence, calculate_warm_start_stage

if TYPE_CHECKING:
    import logging

# warnings.simplefilter('error', RuntimeWarning)


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
        pca_threshold (float): Threshold for variance explained when applying PCA (used internally in the GP model).
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
        use_pca=False,  # noqa: FBT002
        pca_threshold=0.999,
        run_type='runningLocal',
        gcv_threshold=0.2,
        recalibration_ratio=0.1,
        num_samples_per_stage=5000,
        gkl_threshold=0.01,
        gmap_threshold=0.01,
        batch_size_factor=2,
        num_candidate_training_points=4000,
        logger: logging.Logger | None = None,
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
        self.logger = logger or setup_logger()
        self.log_step = make_logger_context(self.logger)
        self.loginfo = make_log_info(self.logger)

        with self.log_step('Initializing GP_AB_Algorithm.'):
            self.data = data
            self.input_dimension = input_dimension
            self.output_dimension = output_dimension
            self.output_length_list = output_length_list
            self.domain = domain
            self.prior_variances = prior_variances

            self.rv_names_list = rv_names_list
            self.output_names_list = output_names_list

            self.inputs = np.empty((0, self.input_dimension), dtype=float)
            self.outputs = np.empty((0, self.output_dimension), dtype=float)

            self.prior_pdf_function = prior_pdf_function
            self.log_likelihood_function = log_likelihood_function

            self.use_pca = use_pca
            # pca_output_dimension_threshold = 10
            # if self.output_dimension > pca_output_dimension_threshold:
            #     self.use_pca = True
            self.pca_threshold = pca_threshold
            self.gp_output_dimension_list = []

            self.gkl_threshold = gkl_threshold
            self.gmap_threshold = gmap_threshold

            self.max_simulations = max_simulations
            self.max_computational_time = max_computational_time
            self.start_time = time.time()

            self.converged = False
            self.budget_exceeded = False
            self.terminate = False

            self.num_experiments = [0]
            self.num_attempted_experiments = [0]
            self.num_recalibration_experiments = 0
            self.recalibration_ratio = recalibration_ratio

            self.sample_transformation_function = uq_utilities.Ensure2DOutputShape(
                sample_transformation_function,
                expected_dim=self.input_dimension,
                label='sample_transformation_function',
            )

            self.model_evaluation_function = uq_utilities.Ensure2DOutputShape(
                model_evaluation_function,
                expected_dim=self.output_dimension,
                label='model_evaluation_function',
            )

            self.run_type = run_type
            self.parallel_pool = uq_utilities.get_parallel_pool_instance(run_type)
            self.parallel_evaluation_function = self.parallel_pool.pool.starmap  # type: ignore

            self.gcv_threshold = gcv_threshold
            self.gcv = None

            self.batch_size_factor = batch_size_factor
            self.num_candidate_training_points = num_candidate_training_points

            self.results = {}

            self.current_gp_model = None
            self.gp_recalibrated = False

            self.kde = None

            self.samples_dict = {}
            self.betas_dict = {}
            self.log_likelihoods_dict = {}
            self.log_target_density_values_dict = {}
            self.log_evidence_dict = {}

            self.previous_posterior_samples = np.empty(
                (0, self.input_dimension), dtype=float
            )
            self.current_posterior_samples = np.empty(
                (0, self.input_dimension), dtype=float
            )
            self.previous_model_parameters = np.empty(
                (0, self.input_dimension), dtype=float
            )
            self.current_model_parameters = np.empty(
                (0, self.input_dimension), dtype=float
            )

            self.num_samples_per_stage = num_samples_per_stage

            self.save_outputs = True

            self.model_evaluation_time = 0.0
            self.gp_training_time = 0.0
            self.posterior_sampling_time = 0.0
            self.doe_time = 0.0
            self.exploitation_doe_time = 0.0
            self.exploration_doe_time = 0.0

        # # Decorate selected methods with timing/logging
        # decorate_methods_with_log_step(
        #     self,
        #     method_names=['run_iteration'],
        #     logger=self.logger,
        #     warn_if_longer_than=300.0,
        # )

    # def _evaluate_in_parallel(
    #     self, func, model_parameters, simulation_number_start=0
    # ):
    #     """
    #     Evaluate the model in parallel using the provided function and samples.

    #     Args:
    #         func (callable): The function to evaluate.
    #         samples (np.ndarray): The samples to evaluate.

    #     Returns
    #     -------
    #         np.ndarray: The evaluated outputs.
    #     """
    #     simulation_numbers = np.arange(
    #         simulation_number_start, simulation_number_start + len(model_parameters)
    #     )
    #     iterable = zip(simulation_numbers, model_parameters)
    #     outputs = np.atleast_2d(
    #         list(self.parallel_evaluation_function(func, iterable))
    #     )
    #     # Fix scalar outputs → (n, 1)
    #     if outputs.ndim == 1:
    #         outputs = outputs.reshape(-1, 1)

    #     # Fix shape (n, 1, d) → (n, d)
    #     elif outputs.ndim == 3 and outputs.shape[1] == 1:
    #         outputs = np.squeeze(outputs, axis=1)
    #     return outputs

    def _evaluate_in_parallel(
        self, func, model_parameters, simulation_number_start=0
    ):
        """
        Safely evaluate the model in parallel over a batch of input samples.

        Each sample is evaluated using the provided function in a separate workdir.
        Failed model evaluations (e.g., due to runtime errors or invalid output)
        are skipped and logged to a JSON file with error messages.

        Output shape handling is preserved for compatibility with downstream GP training:
        - Scalar outputs are reshaped to (n, 1)
        - Extra singleton dimensions (e.g., shape (n, 1, d)) are squeezed to (n, d)

        Parameters
        ----------
        func : callable
            The model evaluation function of the form (sim_number: int, x: np.ndarray) -> np.ndarray.
        model_parameters : np.ndarray
            Array of shape (n_samples, n_inputs) containing the parameter samples to evaluate.
        simulation_number_start : int, optional
            The starting index for naming simulation workdirs, by default 0.

        Returns
        -------
        y_valid : np.ndarray
            Array of shape (n_valid, n_outputs) containing the outputs from successful model runs.
        x_valid : np.ndarray
            Array of shape (n_valid, n_inputs) containing the corresponding input samples.
        """
        simulation_numbers = np.arange(
            simulation_number_start, simulation_number_start + len(model_parameters)
        )
        iterable = zip(simulation_numbers, model_parameters)

        wrapped_func = partial(
            uq_utilities.safe_evaluate_model_for_gp_ab,
            model_callable=func,
            logger=self.logger,
        )
        results = list(self.parallel_evaluation_function(wrapped_func, iterable))

        x_valid, y_valid, failed = [], [], []

        for x, y, msg in results:
            if y is not None:
                x_valid.append(x)
                y_valid.append(y)
            else:
                failed.append((x, msg))

        if failed:
            output_dir = Path('results')
            out_path = (
                output_dir / f'failed_model_inputs_iter_{self.iteration_number}.json'
            )
            self.logger.warning(
                f'Skipping {len(failed)} failed model evaluations. Details in {out_path}.'
            )
            uq_utilities.log_failed_points_to_file(
                failed,
                iteration=self.iteration_number,
                logger=self.logger,
                output_dir=output_dir,
            )

        y_valid = np.array(y_valid)
        if y_valid.ndim == 1:
            y_valid = y_valid.reshape(-1, 1)
        elif y_valid.ndim == 3 and y_valid.shape[1] == 1:  # noqa: PLR2004
            y_valid = np.squeeze(y_valid, axis=1)

        return y_valid, np.array(x_valid)

    def _perform_space_filling_doe(self, n_samples):
        """
        Perform the initial Design of Experiments (DoE) using Latin Hypercube Sampling.

        Args:
            n_samples (int): The number of samples to generate.

        Returns
        -------
            np.ndarray: The generated samples.
        """
        self.space_filling_design = LatinHypercubeSampling(
            n_samples=n_samples, n_dimensions=self.input_dimension
        )
        samples = self.space_filling_design.generate()
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
        self.loginfo('Using a space filling strategy')
        proposed_inputs = self.sample_transformation_function(
            self._perform_space_filling_doe(n_samples)
        )
        self.loginfo(
            f'Generated {proposed_inputs.shape[0]} samples for initial training set. Evaluating the model at these samples.'
        )
        successful_outputs, successful_inputs = self._evaluate_in_parallel(
            self.model_evaluation_function, proposed_inputs
        )
        return successful_inputs, successful_outputs

    # def _log_like(self, predictions):
    #     predictions = np.atleast_2d(predictions)
    #     num_rows, num_cols = self.data.shape
    #     num_samples = predictions.shape[0]

    #     # Precompute weights: w_i = 1 / mean(y_obs_i^2)
    #     weights = []
    #     start = 0
    #     for length in self.output_length_list:
    #         end = start + length
    #         y_obs = self.data[:, start:end]
    #         mse = np.mean(y_obs**2)
    #         weights.append(1.0 / (mse + 1e-12))
    #         start = end

    #     # Stack results for all output groups
    #     weighted_sse_per_sample = np.zeros(num_samples)
    #     start = 0

    #     for j, length in enumerate(self.output_length_list):
    #         end = start + length
    #         y_obs = self.data[:, start:end]  # shape (nd, d_j)
    #         y_obs_exp = y_obs[None, :, :]  # shape (1, nd, d_j)

    #         # predictions[:, start:end]: shape (num_samples, d_j)
    #         # broadcast to shape (num_samples, nd, d_j)
    #         y_pred_exp = predictions[:, start:end][
    #             :, None, :
    #         ]  # shape (num_samples, 1, d_j)
    #         err = y_obs_exp - y_pred_exp  # shape (num_samples, nd, d_j)

    #         sse = np.einsum('ijk,ijk->i', err, err)  # shape (num_samples,)
    #         weighted_sse_per_sample += weights[j] * sse

    #         start = end

    #     # Compute log-likelihood
    #     exponent = -0.5 * num_rows * num_cols
    #     log_likes = exponent * np.log(weighted_sse_per_sample + 1e-12)
    #     return log_likes.reshape((num_samples, 1))

    # def _log_likelihood_approximation(
    #     self, response_approximation_function, model_parameters
    # ):
    #     """
    #     Approximate the log-likelihood for the given samples.

    #     Args:
    #         response_approximation (callable): The response approximation function.
    #         samples (np.ndarray): The samples to evaluate.

    #     Returns
    #     -------
    #         np.ndarray: The approximated log-likelihood values.
    #     """
    #     predictions = response_approximation_function(model_parameters)
    #     log_likes = self._log_like(predictions)
    #     return log_likes

    # def _log_prior_pdf(self, model_parameters):
    #     """
    #     Calculate the log-prior PDF for the given samples.

    #     Args:
    #         samples (np.ndarray): The samples to evaluate.

    #     Returns
    #     -------
    #         np.ndarray: The log-prior PDF values.
    #     """
    #     log_prior_model_parameters = np.log(
    #         self.prior_pdf_function(model_parameters)
    #     ).reshape((-1, 1))
    #     return log_prior_model_parameters

    # def _log_posterior_approximation(self, model_parameters, log_likelihoods):
    #     """
    #     Approximate the log-posterior for the given samples and log-likelihoods.

    #     Args:
    #         samples (np.ndarray): The samples to evaluate.
    #         log_likelihoods (np.ndarray): The log-likelihood values.

    #     Returns
    #     -------
    #         np.ndarray: The approximated log-posterior values.
    #     """
    #     log_prior = self._log_prior_pdf(model_parameters)
    #     log_posterior = log_likelihoods + log_prior
    #     return log_posterior

    def _calculate_gcv(self, weights=None):
        """
        Calculate the Leave-One-Out Cross-Validation (LOOCV) measure.

        Args:
            log_likelihood_approximation (callable): The log-likelihood approximation function.

        Returns
        -------
            float: The LOOCV measure.
        """
        # start_time = time.time()
        self.loginfo('Calculating leave-one-out cross validation error measure gCV.')
        x_train_unique, y_train_unique = (
            self.current_gp_model.x_train,  # type: ignore
            self.current_gp_model.y_train,  # type: ignore
        )
        loo_predictions = self.current_gp_model.loo_predictions(  # type: ignore
            x_train_unique, y_train_unique
        )
        loocv_measure = convergence_metrics.calculate_gcv(
            loo_predictions,
            y_train_unique,
            self.output_length_list,
            weights=weights,
            weight_combination=(2 / 3, 1 / 3),
        )
        # time_taken = time.time() - start_time
        self.loginfo(f'gCV: {loocv_measure:.4f}')
        return loocv_measure

    def _get_exploitation_candidates(self):
        """
        Assemble candidate training points for exploitation.

        This is done by sampling from TMCMC stages after the warm start,
        proportionally to the provided stage weights.

        Returns
        -------
        np.ndarray:
            The exploitation candidate training points of shape (n_candidates, n_parameters).
        dict:
            Dictionary mapping stage number to the number of points sampled from that stage.
        """
        stage_weights = np.asarray(self.stage_weights)
        stage_weights /= np.sum(stage_weights)  # Normalize weights

        n_candidates = self.num_candidate_training_points
        candidate_training_points_exploitation = []
        stage_sample_counts = {}

        for stage_idx, stage_num in enumerate(self.stages_after_warm_start):
            samples_stage = self.results['model_parameters_dict'][stage_num]
            n_available = samples_stage.shape[0]

            n_stage_samples = int(np.round(stage_weights[stage_idx] * n_candidates))
            if n_stage_samples == 0:
                continue

            if n_stage_samples <= n_available:
                selected_indices = np.random.choice(
                    n_available, size=n_stage_samples, replace=False
                )
                selected = samples_stage[selected_indices]
            else:
                # Use all available samples
                selected = samples_stage

            candidate_training_points_exploitation.append(selected)
            stage_sample_counts[stage_num] = selected.shape[0]

        # Combine all stage samples
        candidate_training_points_exploitation = np.vstack(
            candidate_training_points_exploitation
        )

        # Optional: globally truncate if total exceeds n_candidates (e.g., due to rounding)
        if candidate_training_points_exploitation.shape[0] > n_candidates:
            trim_indices = np.random.choice(
                candidate_training_points_exploitation.shape[0],
                size=n_candidates,
                replace=False,
            )
            candidate_training_points_exploitation = (
                candidate_training_points_exploitation[trim_indices]
            )

        return candidate_training_points_exploitation, stage_sample_counts

    def _summarize_iteration(self):
        with self.log_step(
            f'Summary after iteration {self.iteration_number}',
            highlight='success',
        ):
            self.loginfo(f'Number of model evaluations till now: {len(self.inputs)}')
            self.loginfo(
                f'Time for model evaluation till now: {_format_duration(self.model_evaluation_time)}'
            )
            self.loginfo(
                f'Time for design of experiments till now: {_format_duration(self.doe_time)}'
            )
            self.loginfo(
                f'Time for GP calibration till now: {_format_duration(self.gp_training_time)}'
            )
            self.loginfo(
                f'Time for posterior sampling till now: {_format_duration(self.posterior_sampling_time)}'
            )

    def run(self):
        """
        Execute the GP-AB Algorithm.

        This method initializes the algorithm, generates initial training points,
        and iteratively performs Bayesian updating, convergence checks, and adaptive
        training point selection until termination criteria are met.

        Args:
            batch_size_factor (int, optional): Factor to determine the batch size for training points. Defaults to 2.
        """
        with self.log_step('Running GP-AB Algorithm', highlight='major'):
            with self.log_step('Generation of Initial Training Points'):
                self.iteration_number = -1
                inputs, outputs, num_initial = self.run_initial_doe(
                    num_initial_doe_per_dim=self.batch_size_factor
                )
                self.inputs = inputs
                self.outputs = outputs
                self.num_experiments.append(len(inputs))
                self.num_attempted_experiments.append(num_initial)

                # First time
                self.current_gp_model = create_gp_model(
                    input_dimension=self.input_dimension,
                    output_dimension=self.output_dimension,
                    mean_function='linear',
                    # mean_function='none',
                    # fix_nugget=True,
                    fix_nugget=False,
                    nugget_value=1e-6,
                    logger=self.logger,
                    use_pca=self.use_pca,
                    pca_threshold=self.pca_threshold,
                )
                self.current_gp_model.initialize(
                    self.inputs, self.outputs, reoptimize=True
                )

            try:
                iteration = 0
                terminate = False
                while not terminate:
                    with self.log_step(
                        f'Iteration {iteration}',
                        highlight='major',
                    ):
                        terminate, results = self.run_iteration(iteration)
                        self.write_results()
                        self._summarize_iteration()
                    iteration += 1

            finally:
                with self.log_step(
                    'GP-AB Algorithm run completed.',
                    highlight='major',
                ):
                    self.parallel_pool.close_pool()

    def run_iteration(self, k: int) -> tuple[bool, dict]:
        """
        Run a single iteration of the GP-AB (Gaussian Process Adaptive Bayesian) algorithm.

        The iteration performs five core steps:
            1. Posterior approximation via GP model and PCA.
            2. Bayesian updating via TMCMC (with warm-start support).
            3. Convergence and budget checks based on gKL and gMAP metrics.
            4. Adaptive selection of new training points (exploitation + exploration).
            5. Evaluation of the computational model at selected points.

        Parameters
        ----------
        k : int
            The current iteration number.

        Returns
        -------
        tuple of (bool, dict)
            A tuple with:
                - `terminate` : bool, indicating whether the algorithm should stop.
                - `results` : dict, TMCMC results for the current iteration.
        """
        self.loginfo(f'Setting up for iteration {k}.')
        self.iteration_number = k
        model_parameters = self.inputs
        model_outputs = self.outputs

        log_like_fn = partial(
            log_like, data=self.data, output_length_list=self.output_length_list
        )
        log_prior_fn = partial(log_prior, prior_pdf_function=self.prior_pdf_function)

        self._step_1_posterior_approximation(
            model_parameters, model_outputs, log_like_fn
        )
        self._step_2_bayesian_updating(log_prior_fn, log_like_fn)
        self._step_3_assess_convergence(log_prior_fn)

        if self.terminate:
            return self.terminate, self.results

        self._step_4_adaptive_training_point_selection()
        self._step_5_evaluate_responses()

        return self.terminate, self.results

    def _step_1_posterior_approximation(
        self, model_parameters, model_outputs, log_like_fn
    ):
        """
        Step 1: Recalibrate or reuse the GP surrogate model.

        Construct an approximation to the posterior distribution using PCA and GP prediction.
        """
        with LogStepContext(
            'Posterior Approximation',
            logger=self.logger,
            highlight='submajor',
        ):
            if self.iteration_number > 0:
                self.loginfo('Saving previous GP model and posterior approximation.')
                self.previous_gp_model = copy.deepcopy(self.current_gp_model)

                self.previous_response_approximation = partial(
                    response_approximation,
                    self.previous_gp_model,
                )
                self.previous_log_likelihood_approximation = partial(
                    log_likelihood_approx,
                    log_like_fn=log_like_fn,
                    response_approx_fn=self.previous_response_approximation,
                )
            with self.log_step('Evaluating if GP Recalibration Needed.'):
                delta_experiments = (
                    self.num_experiments[-1] - self.num_recalibration_experiments
                )
                recalib_threshold = (
                    self.recalibration_ratio * self.num_recalibration_experiments
                )
                self.loginfo(f'Recalibration threshold: {recalib_threshold:.1f}')
                self.loginfo(f'New experiments added: {delta_experiments}')

                if delta_experiments > recalib_threshold:
                    self.loginfo(
                        'Sufficient number of experiments for recalibration.'
                    )

                    # if self.iteration_number > 0:
                    #     self.loginfo(
                    #         'Saving previous GP model and posterior approximation.'
                    #     )
                    #     self.previous_gp_model = copy.deepcopy(self.current_gp_model)

                    #     self.previous_response_approximation = partial(
                    #         response_approximation,
                    #         self.previous_gp_model,
                    #     )
                    #     self.previous_log_likelihood_approximation = partial(
                    #         log_likelihood_approx,
                    #         log_like_fn=log_like_fn,
                    #         response_approx_fn=self.previous_response_approximation,
                    #     )

                    start_time = time.time()
                    with self.log_step('Calibrating the GP model.'):
                        # Later iterations
                        self.current_gp_model.update_training_dataset(  # type: ignore
                            model_parameters, model_outputs, reoptimize=True
                        )

                        # self.current_gp_model.update(
                        #     model_parameters, model_outputs, reoptimize=True
                        # )
                    time_elapsed = time.time() - start_time
                    self.gp_training_time += time_elapsed

                    gp_output_dimension = self.current_gp_model.pca_info.get(  # type: ignore
                        'n_components', self.output_dimension
                    )
                    self.gp_output_dimension_list.append(gp_output_dimension)
                    self.num_recalibration_experiments = self.num_experiments[-1]
                    self.gp_recalibrated = True
                else:
                    self.loginfo(
                        'Insufficient number of experiments for recalibration.'
                    )
                    # self.current_gp_model = copy.deepcopy(self.previous_gp_model)
                    gp_output_dimension = self.current_gp_model.pca_info.get(  # type: ignore
                        'n_components', self.output_dimension
                    )
                    self.gp_output_dimension_list.append(gp_output_dimension)
                    self.current_gp_model.update_training_dataset(  # type: ignore
                        model_parameters, model_outputs, reoptimize=False
                    )
                    self.gp_recalibrated = False

                self.loginfo(
                    f'Current GP model output dimension: {gp_output_dimension}'
                )

            with self.log_step('Constructing Posterior Distribution Approximation.'):
                self.gp_prediction_mean, _ = self.current_gp_model.predict(  # type: ignore
                    model_parameters
                )
                # self.loo_predictions = self.current_gp_model.loo_predictions(  # type: ignore
                #     model_parameters, model_outputs
                # )

                self.loo_predictions = self.current_gp_model.loo_predictions(  # type: ignore
                    self.current_gp_model.x_train,  # type: ignore
                    self.current_gp_model.y_train,  # type: ignore
                )

                self.current_response_approximation = partial(
                    response_approximation, self.current_gp_model
                )
                self.current_log_likelihood_approximation = partial(
                    log_likelihood_approx,
                    log_like_fn=log_like_fn,
                    response_approx_fn=self.current_response_approximation,
                )

            res_dir = Path('results')
            res_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            outfile_path = (
                res_dir / f'gp_model_parameters_{self.iteration_number}.json'
            )
            self.current_gp_model.write_model_parameters_to_json(  # type: ignore
                outfile_path,
                include_training_data=True,
            )

    def _step_2_bayesian_updating(self, log_prior_fn, log_like_fn):
        """
        Step 2: Perform Bayesian updating using TMCMC.

        Supports warm-starting from intermediate stages when gCV is low.
        """
        current_response_approximation = partial(
            response_approximation, self.current_gp_model
        )
        current_log_likelihood_approximation = partial(
            log_likelihood_approx,
            log_like_fn=log_like_fn,
            response_approx_fn=current_response_approximation,
        )
        with LogStepContext(
            'Bayesian Updating',
            logger=self.logger,
            highlight='submajor',
        ):
            self.j_star = 0
            tmcmc = TMCMC(
                current_log_likelihood_approximation,
                log_prior_fn,
                self.sample_transformation_function,
                run_parallel=True,
                logger=self.logger,
                run_type=self.run_type,
            )
            self._initialize_tmcmc_result_dicts()

            with self.log_step('Evaluating warm-start of TMCMC.'):
                if self.iteration_number > 0:
                    weights = self.kde.evaluate(  # type: ignore
                        self.current_gp_model.x_train  # type: ignore
                    )  # inputs deduplicated in the gp model
                    self.gcv = self._calculate_gcv(weights)
                    self.warm_start_possible = self.gcv <= self.gcv_threshold

                    if self.warm_start_possible:
                        self.loginfo(
                            f'Warm start possible since gCV: {self.gcv:.4f} <= {self.gcv_threshold:.4f}'
                        )
                        self.j_star = calculate_warm_start_stage(
                            self.current_log_likelihood_approximation, self.results
                        )
                        self.loginfo(f'Warm start stage: {self.j_star}')
                    else:
                        self.loginfo(
                            f'Warm start not possible since gCV: {self.gcv:.4f} > {self.gcv_threshold:.4f}'
                        )
                    self.previous_model_parameters = self.results[
                        'model_parameters_dict'
                    ]
                    self.num_tmcmc_stages = len(self.previous_model_parameters)
                    self.previous_posterior_samples = self.previous_model_parameters[
                        self.num_tmcmc_stages - 1
                    ]
                else:
                    self.loginfo('No previous samples available for warm start.')

            with self.log_step('TMCMC Sampling'):
                if self.j_star == 0:
                    # if self.iteration_number == 0:
                    self._initialize_tmcmc_from_prior(
                        log_prior_fn, self.current_log_likelihood_approximation
                    )
                else:
                    self._load_tmcmc_from_previous_results()

                start_time = time.time()
                self.results = tmcmc.run(
                    self.samples_dict,
                    self.model_parameters_dict,
                    self.betas_dict,
                    self.log_likelihoods_dict,
                    self.log_target_density_values_dict,
                    self.log_evidence_dict,
                    self.num_model_evals_dict,
                    self.scale_factor_dict,
                    self.j_star,
                    num_burn_in=5,
                )
                time_elapsed = time.time() - start_time
                self.posterior_sampling_time += time_elapsed

                self.current_model_parameters = self.results['model_parameters_dict']
                self.num_tmcmc_stages = len(self.current_model_parameters)
                self.current_posterior_samples = self.current_model_parameters[
                    self.num_tmcmc_stages - 1
                ]

    def _step_3_assess_convergence(self, log_prior_fn):
        """
        Step 3: Assess convergence using gKL and gMAP divergence metrics.

        Check against computational budget constraints.
        """
        with LogStepContext(
            'Assessing Convergence',
            logger=self.logger,
            highlight='submajor',
        ):
            self.converged = False
            self.terminate = False

            if self.iteration_number > 0:
                with self.log_step(
                    'Calculating KL divergence based convergence metric gKL.'
                ):
                    combined_samples = np.vstack(
                        [
                            self.previous_posterior_samples,
                            self.current_posterior_samples,
                        ]
                    )
                    self.gkl = convergence_metrics.calculate_gkl(
                        self.current_log_likelihood_approximation,
                        self.previous_log_likelihood_approximation,
                        log_prior_fn,
                        combined_samples,
                    )
                    self.gkl_converged = self.gkl < self.gkl_threshold
                    self.loginfo(
                        f'gKL: {self.gkl:.4g}, threshold: {self.gkl_threshold:.4f}'
                    )
                    if not self.gkl_converged:
                        self.loginfo('Convergence based on gKL not achieved.')

                self.gmap = convergence_metrics.calculate_gmap(
                    self.current_log_likelihood_approximation,
                    self.previous_log_likelihood_approximation,
                    log_prior_fn,
                    combined_samples,
                    self.prior_variances,
                )
                self.gmap_converged = self.gmap < self.gmap_threshold
                self.converged = self.gkl_converged

                num_simulations = self.num_attempted_experiments[-1]
                elapsed_time = time.time() - self.start_time
                self.budget_exceeded = (
                    num_simulations >= self.max_simulations
                    or elapsed_time >= self.max_computational_time
                )
                self.terminate = self.converged or self.budget_exceeded

            else:
                self.loginfo('No previous samples available to assess convergence.')

            if self.terminate:
                if self.gkl_converged:
                    self.loginfo('Terminating: convergence based on gKL')
                if self.budget_exceeded:
                    self.loginfo(
                        f'Terminating: computational budget exceeded '
                        f'(simulations: {num_simulations}/{self.max_simulations}, '
                        f'time: {elapsed_time:.2f}/{self.max_computational_time} sec)'
                    )

    def _step_4_adaptive_training_point_selection(self):
        """
        Step 4: Select new training points using an adaptive Design of Experiments (DoE).

        Balancing exploitation and exploration strategies.
        """
        with LogStepContext(
            'Selection of New Training Points',
            logger=self.logger,
            highlight='submajor',
        ):
            self.n_training_points = max(
                2, self.batch_size_factor * self.input_dimension
            )
            self.loginfo(
                f'Selecting {self.n_training_points} new training points, based on batch size factor {self.batch_size_factor}.'
            )
            self.loginfo('Using an adaptive strategy to select new training points.')
            self.loginfo(
                'Adjusting the ratio of exploitation to exploration training points.'
            )
            self.exploitation_proportion = calculate_exploitation_proportion(
                self.gcv, self.n_training_points
            )
            self.loginfo(
                f'Proportion of exploitation points: {self.exploitation_proportion:.2f}'
            )
            self.n_exploit = max(
                1,
                int(np.floor(self.exploitation_proportion * self.n_training_points)),
            )
            self.n_explore = self.n_training_points - self.n_exploit
            self.loginfo(f'Number of exploitation points: {self.n_exploit}')
            self.loginfo(f'Number of exploration points: {self.n_explore}')

            self.loginfo('Setting up GP for adaptive design of experiments.')
            current_doe = AdaptiveDesignOfExperiments(self.current_gp_model)

            res_dir = Path('results')
            res_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            outfile_path = (
                res_dir / f'doe_kernel_summary_{self.iteration_number}.json'
            )
            current_doe.write_gp_for_doe_to_json(outfile_path)

            with LogStepContext(
                'Exploitation Design of Experiments',
                logger=self.logger,
                highlight='minor',
            ):
                if self.n_exploit > 0:
                    self.loginfo('Setting up exploitation weight distribution.')
                start_time = time.time()
                start_stage = max(1, self.j_star)
                self.stages_after_warm_start = list(
                    range(start_stage, self.num_tmcmc_stages)
                )
                self.stage_weights = np.ones(len(self.stages_after_warm_start))

                candidates_exploit, self.stage_sample_counts = (
                    self._get_exploitation_candidates()
                )
                save_exploitation_candidates_by_stage_json(
                    Path('results')
                    / f'exploitation_candidates_{self.iteration_number}.json',
                    candidates_exploit,
                    self.stage_sample_counts,
                )

                self.kde = GaussianKDE(candidates_exploit)
                weights = self.kde.evaluate(candidates_exploit)
                weights_normalized = weights / np.sum(weights)

                self.exploitation_training_points = np.empty(
                    (0, self.input_dimension)
                )

                current_inputs = self.current_gp_model.x_train.copy()  # type: ignore
                if self.n_exploit > 0:
                    self.exploitation_training_points = (
                        current_doe.select_training_points(
                            current_inputs,
                            self.n_exploit,
                            candidates_exploit,
                            use_mse_w=True,
                            weights=weights_normalized,
                        )
                    )
                    current_inputs = np.vstack(
                        [current_inputs, self.exploitation_training_points]
                    )
                    self.loginfo(f'{self.n_exploit} exploitation points selected.')
                else:
                    self.loginfo('No exploitation points selected.')
            time_elapsed = time.time() - start_time
            self.exploitation_doe_time += time_elapsed
            with LogStepContext(
                'Exploration Design of Experiments',
                logger=self.logger,
                highlight='minor',
            ):
                start_time_explore = time.time()
                candidates_explore = self.sample_transformation_function(
                    self._perform_space_filling_doe(
                        self.num_candidate_training_points
                    )
                )
                self.exploration_training_points = np.empty(
                    (0, self.input_dimension)
                )
                if self.n_explore > 0:
                    self.exploration_training_points = (
                        current_doe.select_training_points(
                            current_inputs,
                            self.n_explore,
                            candidates_explore,
                            use_mse_w=False,
                            weights=None,
                        )
                    )
                    current_inputs = np.vstack(
                        [current_inputs, self.exploration_training_points]
                    )
                    self.loginfo(f'{self.n_explore} exploration points selected.')
                else:
                    self.loginfo('No exploration points selected.')
                time_elapsed_explore = time.time() - start_time_explore
                self.exploration_doe_time += time_elapsed_explore
            time_elapsed = time.time() - start_time
            self.doe_time += time_elapsed

            self.new_training_points = np.vstack(
                [self.exploitation_training_points, self.exploration_training_points]
            )

    def _step_5_evaluate_responses(self):
        """
        Step 5: Evaluate the computational model at newly selected training points.

        Append the outputs to the training dataset.
        """
        with LogStepContext(
            'Response Evaluation at New Training Points',
            logger=self.logger,
            highlight='submajor',
        ):
            simulation_number_start = (
                self.num_attempted_experiments[-1]
                if self.num_attempted_experiments
                else 0
            )
            self.loginfo(
                f'Evaluating {len(self.new_training_points)} new training points.'
            )
            self.loginfo(
                f'Running {min(len(self.new_training_points), self.parallel_pool.num_processors)} model evaluations in parallel'
            )
            start_time = time.time()
            (
                self.successful_new_training_outputs,
                self.successful_new_training_inputs,
            ) = self._evaluate_in_parallel(
                self.model_evaluation_function,
                self.new_training_points,
                simulation_number_start=simulation_number_start,
            )
            self.inputs = np.vstack(
                [self.inputs, self.successful_new_training_inputs]
            )
            self.outputs = np.vstack(
                [self.outputs, self.successful_new_training_outputs]
            )
            self.num_experiments.append(len(self.inputs))
            self.num_attempted_experiments.append(
                (
                    self.num_attempted_experiments[-1]
                    if self.num_attempted_experiments
                    else 0
                )
                + len(self.new_training_points)
            )
            time_elapsed = time.time() - start_time
            self.model_evaluation_time += time_elapsed

    def _initialize_tmcmc_result_dicts(self):
        self.samples_dict = {}
        self.model_parameters_dict = {}
        self.betas_dict = {}
        self.log_likelihoods_dict = {}
        self.log_target_density_values_dict = {}
        self.log_evidence_dict = {}
        self.num_model_evals_dict = {}
        self.scale_factor_dict = {}

    def _initialize_tmcmc_from_prior(self, log_prior_fn, log_likelihood_fn):
        self.loginfo('Preparing for sequential sampling starting from prior.')
        initial_samples = self._perform_space_filling_doe(self.num_samples_per_stage)
        model_parameters = self.sample_transformation_function(initial_samples)
        log_target = log_prior_fn(model_parameters)
        log_likelihood = log_likelihood_fn(model_parameters)

        self.samples_dict[0] = initial_samples
        self.model_parameters_dict[0] = model_parameters
        self.betas_dict[0] = 0
        self.log_likelihoods_dict[0] = log_likelihood
        self.log_target_density_values_dict[0] = log_target
        self.log_evidence_dict[0] = 0
        self.num_model_evals_dict[0] = self.num_samples_per_stage
        if (
            self.results is not None
            and 'scale_factor_dict' in self.results
            and 0 in self.results['scale_factor_dict']
        ):
            self.scale_factor_dict[0] = self.results['scale_factor_dict'][0]
        else:
            self.scale_factor_dict[0] = 2.4 / np.sqrt(self.input_dimension)

    def _load_tmcmc_from_previous_results(self):
        if self.j_star == 0:
            self.loginfo('Preparing for sequential sampling starting from prior.')
        else:
            self.loginfo(
                'Preparing for sequential sampling warm starting from intermediate stage.'
            )
        for j in range(self.j_star + 1):
            self.samples_dict[j] = self.results['samples_dict'][j]
            self.model_parameters_dict[j] = self.results['model_parameters_dict'][j]
            self.betas_dict[j] = self.results['betas_dict'][j]
            self.num_model_evals_dict[j] = self.results['num_model_evals_dict'][j]
            self.scale_factor_dict[j] = self.results['scale_factor_dict'][j]
            # self.log_likelihoods_dict[j] = self.results[
            #     'log_likelihood_values_dict'
            # ][j]
            # self.log_target_density_values_dict[j] = self.results[
            #     'log_target_density_values_dict'
            # ][j]
            # self.log_evidence_dict[j] = self.results['log_evidence_dict'][j]
            # self.num_model_evals_dict[j] = self.results['num_model_evals_dict'][j]
            ll_shape = self.results['log_likelihood_values_dict'][j].shape
            lt_shape = self.results['log_target_density_values_dict'][j].shape
            log_likes = []
            log_targets = []
            log_prior_fn = partial(
                log_prior, prior_pdf_function=self.prior_pdf_function
            )
            for i, params in enumerate(self.model_parameters_dict[j]):
                model_parameters = np.reshape(params, (1, -1))
                loglike = self.current_log_likelihood_approximation(
                    model_parameters, simulation_number=i
                )
                logprior = log_prior_fn(model_parameters)
                logtarget = self.results['betas_dict'][j] * loglike + logprior
                log_likes.append(loglike)
                log_targets.append(logtarget)

            log_likelihood_values = np.array(log_likes).reshape(ll_shape)
            log_target_density_values = np.array(log_targets).reshape(lt_shape)
            if j > 0:
                beta_increment = (
                    self.results['betas_dict'][j] - self.results['betas_dict'][j - 1]
                )
                log_evidence = calculate_log_evidence(
                    beta_increment, log_likelihood_values
                )
            else:
                log_evidence = 0.0
            self.log_likelihoods_dict[j] = log_likelihood_values
            self.log_target_density_values_dict[j] = log_target_density_values
            self.log_evidence_dict[j] = log_evidence

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
        self.loginfo(f'Number of model evaluations till now: {len(outputs)}')
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
        terminate: bool = False,  # noqa: FBT001, FBT002
    ):
        """
        Save tabular results of the GP-AB Algorithm.

        Args:
            samples (np.ndarray): The samples used in the algorithm.
            predictions (np.ndarray): The predictions corresponding to the samples.
            rv_names_list (list[str]): List of random variable names.
            output_names_list (list[str]): List of output variable names.
            output_length_list (list[int]): List of output variable lengths.
            iteration_number (int): The current iteration number.
            output_dir (Path): The directory to save the results.
            terminate (bool, optional): Whether this is the final iteration. Defaults to False.
        """
        # Step 1: Construct prediction headers
        pred_headers = []
        for name, length in zip(output_names_list, output_length_list):
            if length == 1:
                pred_headers.append(name)
            else:
                pred_headers.extend([f'{name}_{i+1}' for i in range(length)])

        # Step 2: Create base dataframes
        df_samples = pd.DataFrame(samples, columns=rv_names_list)
        df_preds = pd.DataFrame(predictions, columns=pred_headers)
        df_combined = pd.concat([df_samples, df_preds], axis=1)

        # Step 3: Save regular tabular result file
        tsv_path = output_dir / f'tabular_results_{iteration_number}.txt'
        df_combined.to_csv(tsv_path, sep='\t', index=False)
        print(f'Saved: {tsv_path}')

        # Step 4: If terminate, create dakotaTab and dakotaTabPrior
        if terminate:
            # --- COMMON HEADER PREP ---
            final_headers = ['eval_id', 'interface', *rv_names_list, *pred_headers]

            # --- dakotaTab.out ---
            n_samples = len(df_combined)
            df_combined_with_meta = df_combined.copy()
            df_combined_with_meta.insert(0, 'interface', 1)
            df_combined_with_meta.insert(0, 'eval_id', range(1, n_samples + 1))
            dakota_tab_path = 'dakotaTab.out'
            df_combined_with_meta.to_csv(
                dakota_tab_path, sep='\t', index=False, header=final_headers
            )
            print(f'Saved: {dakota_tab_path}')

            # --- dakotaTabPrior.out ---
            prior_samples = self.results['model_parameters_dict'][0]
            prior_predictions = response_approximation(
                self.current_gp_model, prior_samples
            )

            df_prior_samples = pd.DataFrame(prior_samples, columns=rv_names_list)
            df_prior_preds = pd.DataFrame(prior_predictions, columns=pred_headers)
            df_prior_combined = pd.concat([df_prior_samples, df_prior_preds], axis=1)
            df_prior_combined.insert(0, 'interface', 1)
            df_prior_combined.insert(
                0, 'eval_id', range(1, len(df_prior_combined) + 1)
            )

            dakota_tab_prior_path = 'dakotaTabPrior.out'
            df_prior_combined.to_csv(
                dakota_tab_prior_path, sep='\t', index=False, header=final_headers
            )
            print(f'Saved: {dakota_tab_prior_path}')

    def _save_gp_ab_progress(self):
        """Save the current progress of the GP-AB Algorithm to a file."""
        data = {
            'iteration_number': self.iteration_number,
            'inputs': self.inputs[: len(self.gp_prediction_mean)],
            'exploitation_proportion': self.exploitation_proportion,
            'n_training_points': self.n_training_points,
            'n_exploit': self.n_exploit,
            'n_explore': self.n_explore,
            'exploitation_candidates_stage_sample_counts': self.stage_sample_counts,
            'exploitation_training_points': self.exploitation_training_points,
            'exploration_training_points': self.exploration_training_points,
            'num_latent_variables': self.current_gp_model.pca_info.get(  # type: ignore
                'n_components', None
            ),
            'explained_variance_ratio': self.current_gp_model.pca_info.get(  # type: ignore
                'explained_variance_ratio', None
            ),
            'gp_recalibrated': self.gp_recalibrated,
        }

        if self.save_outputs:
            data.update(
                {
                    'outputs': self.outputs[: len(self.gp_prediction_mean)],
                    'gp_prediction_mean': self.gp_prediction_mean,
                    'loo_predictions': self.loo_predictions,
                }
            )

        if self.iteration_number > 0:
            data.update(
                {
                    'gcv': self.gcv,
                    'warm_start_possible': self.warm_start_possible,
                    'warm_start_stage': self.j_star,
                    'num_tmcmc_stages': self.num_tmcmc_stages,
                    'gkl': self.gkl,
                    'gkl_converged': self.gkl_converged,
                    'gmap': self.gmap,
                    'gmap_converged': self.gmap_converged,
                    'converged': self.converged,
                    'budget_exceeded': self.budget_exceeded,
                    'terminate': self.terminate,
                }
            )

        return uq_utilities.make_json_serializable(data)

    def write_results(self, results_dir='results'):
        """Write the results of the GP-AB Algorithm to a file."""
        res_dir = Path(results_dir)
        res_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        serializable_data = uq_utilities.make_json_serializable(self.results)
        outfile_path = res_dir / f'tmcmc_results_{self.iteration_number}.json'
        with outfile_path.open('w') as f:
            json.dump(serializable_data, f, indent=4)

        outfile_gp_ab_path = res_dir / f'gp_ab_progress_{self.iteration_number}.json'
        with outfile_gp_ab_path.open('w') as f:
            json.dump(self._save_gp_ab_progress(), f, indent=4)

        samples = self.current_posterior_samples
        predictions = response_approximation(self.current_gp_model, samples)
        self.save_tabular_results(
            samples=samples,
            predictions=predictions,
            rv_names_list=self.rv_names_list,
            output_names_list=self.output_names_list,
            output_length_list=self.output_length_list,
            iteration_number=self.iteration_number,
            output_dir=res_dir,
            terminate=self.terminate,
        )


def save_exploitation_candidates_by_stage_json(
    out_file: Path, samples: np.ndarray, stage_sample_counts: dict[int, int]
):
    """
    Save exploitation candidate samples grouped by TMCMC stage to a JSON file.

    Parameters
    ----------
    out_file : Path
        Path to the output JSON file.
    samples : np.ndarray
        Array of shape (n_candidates, n_parameters), assumed to be ordered stage-by-stage.
    stage_sample_counts : dict[int, int]
        Dictionary mapping stage number to number of samples drawn from that stage.

    The output JSON file will have:
    - 'stage_sample_counts': summary of counts per stage
    - 'samples_by_stage': mapping of stage number to list of samples
    """
    samples_by_stage = {}
    start_idx = 0
    for stage, count in stage_sample_counts.items():
        end_idx = start_idx + count
        stage_samples = samples[start_idx:end_idx].tolist()
        samples_by_stage[str(stage)] = stage_samples
        start_idx = end_idx

    output = {
        'stage_sample_counts': {str(k): v for k, v in stage_sample_counts.items()},
        'samples_by_stage': samples_by_stage,
    }
    output_json_serializable = uq_utilities.make_json_serializable(output)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open('w') as f:
        json.dump(output_json_serializable, f, indent=2)


def save_exploitation_candidates_json(
    out_file: Path, samples: np.ndarray, stage_sample_counts: dict[int, int]
):
    """
    Save exploitation candidate samples and their originating TMCMC stage to a JSON file.

    Each sample is tagged with the stage it was drawn from, allowing later analysis of
    stage-wise contribution to the candidate pool.

    Parameters
    ----------
    out_file : Path
        Path to the output JSON file.
    samples : np.ndarray
        Array of candidate samples of shape (n_candidates, n_parameters),
        assumed to be ordered stage-by-stage.
    stage_sample_counts : dict[int, int]
        Dictionary mapping stage number to the number of samples drawn from that stage.

    Notes
    -----
    The JSON file will contain:
    - 'stage_sample_counts': summary of number of samples per stage.
    - 'samples': a list of dicts with 'stage' and 'values' keys.
    """
    # Create list of samples with their corresponding stage
    sample_dicts = []
    start_idx = 0
    for stage, count in stage_sample_counts.items():
        for i in range(count):
            sample_values = samples[start_idx + i].tolist()
            sample_dicts.append({'stage': stage, 'values': sample_values})
        start_idx += count

    # Structure to write
    output = {
        'stage_sample_counts': stage_sample_counts,
        'samples': sample_dicts,
    }

    output_json_serializable = uq_utilities.make_json_serializable(output)

    # Ensure parent directory exists
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with out_file.open('w') as f:
        json.dump(output_json_serializable, f, indent=2)


def calculate_exploitation_proportion(gcv, num_training_points):
    """
    Compute the exploitation proportion r_ex based on gcv using log-scale interpolation.

    This function maps the cross-validation score `gcv` to an exploitation proportion `r_ex`
    in the range [r_min, r_max], using interpolation in log-space. Higher `gcv` implies
    greater uncertainty and favors exploration (lower r_ex), while lower `gcv` implies
    model confidence and favors exploitation (higher r_ex).

    Parameters
    ----------
    gcv : float
        Cross-validation score g_cv^(k). Must be positive.

    Returns
    -------
    float
        Exploitation proportion r_ex in [r_min, r_max].
    """
    g_upper = 0.2
    g_lower = 0.005
    # r_max = 0.9
    # r_min = 0.0
    r_min = 1 / (num_training_points)
    r_max = 1 - r_min

    if gcv is None:
        return r_min
    if gcv > g_upper:
        return r_min
    if gcv < g_lower:
        return r_max

    # Ensure gcv is valid for log computation
    assert gcv > 0, 'gcv must be positive for log-scale interpolation.'

    # Linear interpolation in log scale
    log_g = np.log(gcv)
    log_g_upper = np.log(g_upper)
    log_g_lower = np.log(g_lower)

    # Interpolation factor: 0 when gcv = g_upper, 1 when gcv = g_lower
    alpha = (log_g_upper - log_g) / (log_g_upper - log_g_lower)
    return r_min + alpha * (r_max - r_min)


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

    joint_distribution = uq_utilities.ERANatafJointDistribution(
        rv_inputs,
        correlation_matrix_inputs,  # type: ignore
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
    output_dimension = sum(edp_lengths_list)

    # Move file from template to main working directory
    src = input_arguments.path_to_template_directory / uq_inputs.calDataFile
    dst = input_arguments.path_to_working_directory / uq_inputs.calDataFile
    shutil.move(src, dst)

    cal_data_file = dst
    tmp_file = (
        input_arguments.path_to_working_directory
        / 'quoFEMTempCalibrationDataFile.cal'
    )
    num_experiments = 0

    with cal_data_file.open('r') as f_in, tmp_file.open('w') as f_out:
        headings = 'Exp_num interface '
        for name, count in zip(edp_names_list, edp_lengths_list):
            if count == 1:
                headings += f'{name} '
            else:
                headings += ' '.join(f'{name}_{i+1}' for i in range(count)) + ' '
        f_out.write(headings.strip() + '\n')

        linenum = 0
        for line in f_in:
            linenum += 1
            if not line.strip():
                continue

            cleaned_line = line.replace(',', ' ')
            words = cleaned_line.split()

            if len(words) != output_dimension:
                msg = f"Line {linenum} in '{cal_data_file}' has {len(words)} entries, expected {output_dimension}."
                raise RuntimeError(msg)

            num_experiments += 1
            new_line = f'{num_experiments} 1 ' + ' '.join(words)
            f_out.write(new_line + '\n')

    data = np.atleast_2d(
        np.genfromtxt(
            tmp_file,
            skip_header=1,
            usecols=np.arange(2, 2 + output_dimension),  # type: ignore
        )
    )

    log_likelihood_file_name = uq_inputs.logLikelihoodFile
    log_likelihood_path = uq_inputs.logLikelihoodPath
    log_likelihood_function = partial(
        log_like,
        data=data,
        output_length_list=edp_lengths_list,
    )
    if log_likelihood_file_name:
        sys.path.append(str(log_likelihood_path))
        ll_module = importlib.import_module(log_likelihood_file_name)
        log_likelihood_function = ll_module.log_likelihood

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

    use_pca = False
    pca_output_dimension_threshold = 10
    if output_dimension > pca_output_dimension_threshold:
        use_pca = True

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
        use_pca,
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


def run_gp_ab_algorithm(input_arguments, logger: logging.Logger | None = None):
    """
    Run the GP-AB Algorithm.

    Args:
        input_arguments (InputArguments): The input arguments for the algorithm.
    """
    logger = logger or setup_logger()
    inputs = preprocess(input_arguments)
    gp_ab = GP_AB_Algorithm(*inputs, logger=logger)
    gp_ab.run()


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
    # Parse arguments first
    args = parse_arguments(command_args)

    # Change to working directory
    os.chdir(args['path_to_working_directory'])

    # Now create the logger
    logger = setup_logger(
        log_filename='logFileTMCMC.txt',
        prefix='',
        style='compact',
    )

    # Start the auto-flusher
    flusher = LoggerAutoFlusher(logger, interval=10)
    flusher.start()

    try:
        # Validate input arguments
        input_arguments = InputArguments.model_validate(args)

        # Run the GP-AB Algorithm
        run_gp_ab_algorithm(input_arguments, logger=logger)

    except Exception as e:
        err_msg = f'ERROR: An exception occurred:\n{traceback.format_exc()}\n'
        sys.stderr.write(err_msg)
        err_file = input_arguments.path_to_working_directory / 'UCSD_UQ.err'
        with err_file.open('a') as f:
            f.write(err_msg)
        log_exception(logger, e, message='Error when running GP_AB_Algorithm')
        raise RuntimeError(err_msg) from e

    finally:
        # Always stop flusher
        flusher.stop()

        if input_arguments.run_type == 'runningRemote':
            from mpi4py import MPI  # type: ignore

            MPI.COMM_WORLD.Abort(0)


if __name__ == '__main__':
    try:
        main(sys.argv[1:])  # Runs with command-line arguments
    except Exception as e:  # noqa: BLE001
        print(f'Error: {e}', file=sys.stderr)  # noqa: T201
        sys.exit(1)
