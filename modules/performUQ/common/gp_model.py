"""Module for creating and using Gaussian Process models with the GaussianProcessModel class."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import GPy
import numpy as np
from config_utilities import load_settings_from_config, save_used_settings_as_config
from logging_utilities import (  # decorate_methods_with_log_step,
    flush_logger,
    make_log_info,
    make_logger_context,
    setup_logger,
)
from principal_component_analysis import (
    PrincipalComponentAnalysis,
    SafeStandardScaler,
)
from pydantic import BaseModel, Field, model_validator
from sklearn.linear_model import LinearRegression
from uq_utilities import make_json_serializable

# =========================================================
# Top-level Classes
# =========================================================


def remove_duplicate_inputs(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove duplicate rows in X and corresponding rows in Y.

    Keeps only the first occurrence.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (n_samples, n_features).
    y : np.ndarray
        Output array of shape (n_samples, n_outputs).

    Returns
    -------
    x_unique, y_unique : tuple[np.ndarray, np.ndarray]
        Deduplicated input-output arrays.
    """
    _, unique_indices = np.unique(x, axis=0, return_index=True)
    sorted_indices = np.sort(unique_indices)
    return x[sorted_indices], y[sorted_indices]


class GaussianProcessModelSettings(BaseModel):
    """Settings for creating a Gaussian Process Model."""

    input_dimension: int = Field(..., gt=0)  # Required
    output_dimension: int = Field(..., gt=0)  # Required
    ARD: bool = True
    kernel_type: str = Field(
        'matern52', pattern='^(exponential|matern32|matern52|rbf)$'
    )
    mean_function: str = Field('none', pattern='^(none|linear)$')
    nugget_value: float = 1e-6
    fix_nugget: bool = True
    use_pca: bool = False
    pca_threshold: float = 0.999
    scale_inputs: bool = True
    scale_outputs: bool = True

    @model_validator(mode='after')
    def check_nugget_value(self):  # noqa: D102
        if self.fix_nugget and self.nugget_value <= 0:
            msg = 'If nugget is fixed, its value must be positive.'
            raise ValueError(msg)
        return self


class GaussianProcessModel:
    """A class to represent a Gaussian Process Model."""

    def __init__(
        self,
        settings: GaussianProcessModelSettings,
        logger: logging.Logger | None = None,
    ):
        self.settings = settings
        self.logger = logger or setup_logger()
        self.loginfo = make_log_info(self.logger)
        self.log_step = make_logger_context(self.logger)

        self.input_dimension = settings.input_dimension
        self.model_output_dimension = settings.output_dimension
        self.ARD = settings.ARD
        self.kernel_type = settings.kernel_type.lower()
        self.mean_function_type = settings.mean_function.lower()
        self.nugget_value = settings.nugget_value
        self.fix_nugget = settings.fix_nugget
        self.use_pca = settings.use_pca
        self.pca_threshold = settings.pca_threshold
        self.scale_inputs = settings.scale_inputs

        self.pca = (
            PrincipalComponentAnalysis(self.pca_threshold, perform_scaling=False)
            if self.use_pca
            else None
        )
        self.latent_dimension = None

        self.input_scaler = SafeStandardScaler() if self.scale_inputs else None
        self.output_scaler = SafeStandardScaler() if settings.scale_outputs else None

        self.kernel = self._create_kernel()

        self.linear_models: list[LinearRegression | None] = []
        self.model: list[GPy.models.GPRegression] = []

        self.x_train = None
        self.y_train = None
        self.x_train_scaled = None

    def apply_input_scaling(self, x, *, fit=False):
        """Scale inputs using the input scaler."""
        if self.input_scaler is None:
            return x

        if fit:
            return self.input_scaler.fit_transform(x)
        return self.input_scaler.transform(x)

    def _preprocess_outputs(self, y, *, fit: bool) -> np.ndarray:
        """Scale outputs and apply PCA. If fit=False, use existing transformers."""
        y_processed = y
        if self.output_scaler is not None:
            y_processed = (
                self.output_scaler.fit_transform(y_processed)
                if fit
                else self.output_scaler.transform(y_processed)
            )
        if self.pca is not None:
            if fit:
                self.pca.fit(y_processed)
            y_processed = self.pca.project_to_latent_space(y_processed)
        return y_processed

    def _postprocess_outputs(self, y_latent):
        """Single point for all output postprocessing."""
        y_processed = y_latent

        # Step 1: Inverse PCA if used
        if self.pca is not None:
            y_processed = self.pca.project_back_to_original_space(y_processed)

        # Step 2: Inverse scaling if used
        if self.output_scaler is not None:
            y_processed = self.output_scaler.inverse_transform(y_processed)

        return y_processed

    # def _create_surrogate(self):
    #     use_gpy_normalizer = self.output_scaler is None
    #     model = GPy.models.GPRegression(
    #         X=np.zeros((1, self.input_dimension), dtype=float),
    #         Y=np.zeros((1, 1), dtype=float),
    #         kernel=self.kernel.copy(),
    #         mean_function=None,
    #         normalizer=use_gpy_normalizer,
    #     )
    #     self._configure_nugget(model)
    #     return model

    def _create_kernel(self):
        if self.kernel_type == 'exponential':
            return GPy.kern.Exponential(input_dim=self.input_dimension, ARD=self.ARD)
        if self.kernel_type == 'matern32':
            return GPy.kern.Matern32(input_dim=self.input_dimension, ARD=self.ARD)
        if self.kernel_type == 'matern52':
            return GPy.kern.Matern52(input_dim=self.input_dimension, ARD=self.ARD)
        if self.kernel_type == 'rbf':
            return GPy.kern.RBF(input_dim=self.input_dimension, ARD=self.ARD)
        msg = f"Unknown kernel_type '{self.kernel_type}'"
        raise ValueError(msg)

    def _configure_nugget(self, model: GPy.models.GPRegression):
        """Configure the nugget (Gaussian noise variance) for each GP model."""
        if self.settings.fix_nugget:
            model.likelihood.variance = self.settings.nugget_value
            model.likelihood.variance.fix()
        else:
            model.Gaussian_noise.variance = self.settings.nugget_value
            model.likelihood.variance.constrain_bounded(1e-8, 1e-3)

    # def _set_kernel_hyperparameter_bounds(self, kernel):
    #     """Set reasonable hyperparameter bounds for scaled inputs."""
    #     if self.scale_inputs:
    #         # For standardized inputs, set reasonable bounds
    #         if hasattr(kernel, 'lengthscale'):
    #             kernel.lengthscale.constrain_bounded(0.01, 10.0)
    #         # if hasattr(kernel, 'variance'):
    #         #     kernel.variance.constrain_bounded(0.01, 100.0)
    #     # TODO (ABS): For unscaled inputs, use default bounds or wider ranges

    def _set_kernel_hyperparameter_bounds(self, kernel):
        """Set reasonable hyperparameter bounds for inputs."""
        if hasattr(kernel, 'lengthscale'):
            if self.scale_inputs:
                # Standard approach for scaled inputs
                kernel.lengthscale.constrain_bounded(0.01, 10.0)
            # Engineering approach: bounds based on physical scales
            elif hasattr(self, 'x_train') and self.x_train is not None:
                input_ranges = np.ptp(self.x_train, axis=0)

                # Handle zero-variance dimensions
                input_ranges = np.where(input_ranges == 0, 1.0, input_ranges)

                # Apply bounds to lengthscale parameter
                if self.ARD and len(input_ranges) > 1:
                    # For ARD kernels, set bounds per dimension using GPy's indexing
                    for nx in range(len(input_ranges)):
                        lb = input_ranges[nx] * 0.01  # 1% of range
                        ub = input_ranges[nx] * 10.0  # 10x range

                        # Ensure valid bounds
                        lb = max(lb, 1e-6)
                        if lb >= ub:
                            lb = ub * 0.01

                        # Use GPy's double bracket indexing for individual lengthscale parameters
                        kernel.lengthscale[[nx]].constrain_bounded(
                            lb, ub, warning=False
                        )
                else:
                    # Single lengthscale case
                    lb = float(np.min(input_ranges) * 0.01)
                    ub = float(np.max(input_ranges) * 10.0)
                    lb = max(lb, 1e-6)
                    kernel.lengthscale.constrain_bounded(lb, ub)

    def _initialize_kernel_lengthscales(self, kernel, *, force_initialization=False):
        """Initialize kernel lengthscales to reasonable values based on input data.

        Parameters
        ----------
        kernel : GPy kernel
            Kernel to initialize
        force_initialization : bool
            If True, always initialize. If False, only initialize if lengthscales
            are at default values (close to 1.0).
        """
        if not hasattr(kernel, 'lengthscale'):
            return

        # Check if training data is available
        if self.x_train is None:
            return  # Can't initialize without training data

        # Check if hyperparameters appear to be at default values
        if not force_initialization:
            current_lengthscales = kernel.lengthscale.values.flatten()  # noqa: PD011
            if not np.allclose(current_lengthscales, 1.0, rtol=0.01):
                return  # Skip initialization - hyperparameters appear optimized

        # Simple consistent initialization: use input data ranges
        input_ranges = np.ptp(self.x_train, axis=0)
        # Handle zero-variance dimensions
        input_ranges = np.where(input_ranges == 0, 1.0, input_ranges)

        # Use 20% of range as initial lengthscale
        initial_lengthscales = input_ranges * 0.2

        # Apply to kernel
        if self.ARD and len(initial_lengthscales) > 1:
            kernel.lengthscale[:] = initial_lengthscales
        else:
            kernel.lengthscale = float(np.median(initial_lengthscales))

    def _fit_pca_and_create_models(self, *, reoptimize=True, num_random_restarts=10):
        self.x_train_scaled = self.apply_input_scaling(self.x_train, fit=True)
        if self.scale_inputs:
            self.loginfo(
                f'Input scaling enabled. Scaled training inputs with shape {self.x_train_scaled.shape}'
            )
        else:
            self.loginfo('Input scaling disabled. Using original input scale.')

        y_latent = self._preprocess_outputs(self.y_train, fit=True)
        if self.output_scaler is not None:
            self.loginfo('Output scaling enabled. Scaled training outputs.')
        else:
            self.loginfo('Output scaling disabled. Using original output scale.')

        if self.pca is not None:
            # self.pca.fit(self.y_train)
            # y_latent = self.pca.project_to_latent_space(self.y_train)
            self.latent_dimension = int(self.pca.n_components)  # type: ignore
            self.output_dimension = self.latent_dimension
            self.loginfo(
                f'PCA reduced output dimension from {self.model_output_dimension} '
                f'to {self.output_dimension}, capturing at least '
                f'{self.pca_threshold:.2%} variance.'
            )
        else:
            # y_latent = self.y_train
            self.output_dimension = self.model_output_dimension
            self.loginfo(
                f'Using model outputs of dimension {self.model_output_dimension} '
                'for training GP. Not using PCA.'
            )

        self.model = []
        self.linear_models = []

        for i in range(self.output_dimension):
            x = self.x_train_scaled
            y = np.reshape(y_latent[:, i], (-1, 1))  # type: ignore

            linear_model = None
            if self.mean_function_type == 'linear':
                linear_model = LinearRegression()
                linear_model.fit(x, y)  # type: ignore
                y_detrended = y - linear_model.predict(x)  # type: ignore
            else:
                y_detrended = y

            kernel_copy = self.kernel.copy()
            # Data-adaptive lengthscale initialization (always for new models)
            self._initialize_kernel_lengthscales(
                kernel_copy, force_initialization=True
            )
            self._set_kernel_hyperparameter_bounds(kernel_copy)

            use_gpy_normalizer = self.output_scaler is None

            gp = GPy.models.GPRegression(
                X=x,
                Y=y_detrended,
                kernel=kernel_copy,
                mean_function=None,
                normalizer=use_gpy_normalizer,
            )
            self._configure_nugget(gp)
            if reoptimize:
                gp.optimize_restarts(num_random_restarts)
            else:
                _ = gp.posterior

            self.model.append(gp)
            self.linear_models.append(linear_model)

    def initialize(
        self, x_train, y_train, *, reoptimize=True, num_random_restarts=10
    ):
        """
        Initialize the Gaussian Process Model with training data.

        Parameters
        ----------
        x_train : ndarray
            Training input data of shape (n_samples, input_dimension).
        y_train : ndarray
            Training output data of shape (n_samples, output_dimension).
        reoptimize : bool, optional
            Whether to reoptimize the model hyperparameters (default is True).
        num_random_restarts : int, optional
            Number of random restarts for optimization (default is 10).
        """
        orig_n = x_train.shape[0]
        inputs, outputs = remove_duplicate_inputs(x_train, y_train)
        deduped_n = inputs.shape[0]
        if self.logger and deduped_n < orig_n:
            self.logger.info(f'Removed {orig_n - deduped_n} duplicate input points.')

        self.x_train = inputs
        self.y_train = outputs
        self._fit_pca_and_create_models(
            reoptimize=reoptimize, num_random_restarts=num_random_restarts
        )

    def update_training_dataset(
        self, x_train, y_train, *, reoptimize=False, num_random_restarts=10
    ):
        """
        Add new training data to the Gaussian Process Model.

        Parameters
        ----------
        x_train : ndarray
            New training input data of shape (n_samples, input_dimension).
        y_train : ndarray
            New training output data of shape (n_samples, output_dimension).
        reoptimize : bool, optional
            Whether to reoptimize the model hyperparameters (default is False).
        num_random_restarts : int, optional
            Number of random restarts for optimization (default is 10).
        """
        if not self.model:
            msg = 'GP model not initialized. Call `initialize` first.'
            raise ValueError(msg)

        orig_n = x_train.shape[0]
        inputs, outputs = remove_duplicate_inputs(x_train, y_train)
        deduped_n = inputs.shape[0]
        if self.logger and deduped_n < orig_n:
            self.logger.info(f'Removed {orig_n - deduped_n} duplicate input points.')

        self.x_train = inputs
        self.y_train = outputs

        self.x_train_scaled = self.apply_input_scaling(self.x_train, fit=True)
        y_latent = self._preprocess_outputs(self.y_train, fit=True)

        if self.pca is not None:
            # self.pca.fit(self.y_train)
            # y_latent = self.pca.project_to_latent_space(self.y_train)
            new_latent_dim = int(self.pca.n_components)  # type: ignore
        else:
            # y_latent = self.y_train
            new_latent_dim = self.model_output_dimension

        if (
            new_latent_dim != self.output_dimension
        ):  # PCA dimension has changed — retrain all GP models from scratch
            self.loginfo('Latent dimension changed. Reinitializing GP models.')
            self.output_dimension = new_latent_dim
            self._fit_pca_and_create_models(
                reoptimize=True, num_random_restarts=num_random_restarts
            )
        else:
            for i in range(self.output_dimension):
                y = np.reshape(y_latent[:, i], (-1, 1))
                if self.linear_models[i] is not None:
                    y_detrended = y - self.linear_models[i].predict(  # type: ignore
                        self.x_train_scaled
                    )
                else:
                    y_detrended = y

                # Initialize only if hyperparameters appear unoptimized
                self._initialize_kernel_lengthscales(
                    self.model[i].kern, force_initialization=False
                )
                self.model[i].set_XY(self.x_train_scaled, y_detrended)
                if reoptimize:
                    self.model[i].optimize_restarts(num_random_restarts)
                else:
                    _ = self.model[i].posterior

    def predict(self, x_predict):
        """
        Predict the mean and variance for the given input data.

        Parameters
        ----------
        x_predict : ndarray
            Input data of shape (n_samples, input_dimension) for which predictions are required.

        Returns
        -------
        tuple
            A tuple containing:
            - y_mean (ndarray): Predicted mean values of shape (n_samples, output_dimension).
            - y_var (ndarray): Predicted variance values of shape (n_samples, output_dimension).
        """
        if not self.model:
            msg = 'GP model not initialized. Call `initialize` first.'
            raise ValueError(msg)

        x_predict_scaled = self.apply_input_scaling(x_predict, fit=False)

        y_mean = np.zeros((x_predict.shape[0], self.output_dimension))
        y_var = np.zeros((x_predict.shape[0], self.output_dimension))

        for i in range(self.output_dimension):
            mean, var = self.model[i].predict(x_predict_scaled)
            if self.linear_models[i] is not None:
                trend = self.linear_models[i].predict(x_predict_scaled)  # type: ignore
                mean += trend

            y_mean[:, i] = mean.flatten()
            y_var[:, i] = var.flatten()

        # if self.pca is not None:
        #     y_mean = self.pca.project_back_to_original_space(y_mean)
        #     # y_var = self.pca.inverse_transform_variance(y_var) # Not transforming variance for now

        y_mean_final = self._postprocess_outputs(y_mean)

        return y_mean_final, y_var  # variance is in latent space, if pca is used

    def loo_predictions(self, x_train, y_train, epsilon=1e-8):
        """
        Compute Leave-One-Out (LOO) predictions, accounting for linear trend if present.

        Parameters
        ----------
        x_train : ndarray
            Training inputs of shape (n_samples, input_dimension).
        y_train : ndarray
            Training outputs of shape (n_samples, output_dimension).
        epsilon : float
            Small value to avoid divide-by-zero in numerical instability cases.

        Returns
        -------
        loo_pred : ndarray
            LOO predictions with trend restored, shape (n_samples, output_dimension).
        """
        if not self.model:
            msg = 'GP model not initialized. Call `initialize` first.'
            raise ValueError(msg)

        if not hasattr(self.model[0], 'posterior'):
            msg = 'Model must be trained before computing LOO predictions.'
            raise ValueError(msg)

        x_train_scaled = self.apply_input_scaling(x_train, fit=False)
        y_train_processed = self._preprocess_outputs(y_train, fit=False)

        # if self.pca is not None:
        #     y_train = self.pca.project_to_latent_space(y_train)

        loo_pred_latent = np.zeros_like(y_train_processed, dtype=np.float64)

        for i in range(self.output_dimension):
            x = x_train_scaled
            y = y_train_processed[:, i].reshape(-1, 1)

            # Step 1: Detrend
            if self.linear_models[i] is not None:
                trend = self.linear_models[i].predict(x)  # type: ignore
                y_detrended = y - trend
            else:
                trend = np.zeros_like(y)
                y_detrended = y

            # Step 2: LOO for GP residual
            posterior = self.model[i].posterior  # type: ignore
            alpha = posterior.woodbury_vector  # type: ignore # shape (N, 1)
            k_inv_diag = np.diag(posterior.woodbury_inv)  # type: ignore

            # Step 3: Guard against numerical issues
            if np.any(k_inv_diag == 0):
                self.logger.warning(
                    f'Zero detected on diagonal for output {i}. Adding epsilon={epsilon}.'
                )
                k_inv_diag = np.where(k_inv_diag == 0, epsilon, k_inv_diag)

            loo_gp = y_detrended.flatten() - (alpha.flatten() / k_inv_diag)

            # Step 4: Add back trend to get final prediction
            loo_pred_latent[:, i] = loo_gp + trend.flatten()

        # Step 5: Postprocess outputs
        # if self.pca is not None:
        #     loo_pred_latent = self.pca.project_back_to_original_space(
        #         loo_pred_latent
        #     )
        loo_pred_latent = self._postprocess_outputs(loo_pred_latent)
        return loo_pred_latent  # noqa: RET504

    @property
    def pca_info(self):
        """
        Retrieve information about the PCA transformation.

        Returns
        -------
        dict
            A dictionary containing PCA information such as the number of components
            and the explained variance ratio, if PCA is used. Otherwise, an empty dictionary.
        """
        if self.use_pca:
            return {
                'n_components': self.pca.n_components,  # type: ignore
                'explained_variance_ratio': self.pca.pca.explained_variance_ratio_,  # type: ignore
            }
        return {}

    def flush_logs(self):
        """Flush the logs for the Gaussian Process Model."""
        flush_logger(self.logger)

    def write_model_parameters_to_json(
        self,
        filepath: str | Path,
        *,
        include_training_data: bool = True,
    ):
        """
        Write GP model settings, kernel, linear regression, and optionally training data to a JSON file.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        include_training_data : bool
            Whether to include training data in the JSON file (default is True).
        """
        if not self.model:
            msg = 'Model has not been initialized or trained.'
            raise RuntimeError(msg)

        model_params = {
            'input_dimension': self.input_dimension,
            'output_dimension': self.output_dimension,
            'ARD': self.ARD,
            'kernel_type': self.kernel_type,
            'mean_function': self.mean_function_type,
            'nugget_value': self.nugget_value,
            'fix_nugget': self.fix_nugget,
            'use_pca': self.use_pca,
            'pca_threshold': self.pca_threshold,
            'scale_inputs': self.scale_inputs,
            'scale_outputs': bool(self.output_scaler is not None),
            'pca_info': self.pca_info,
            'models': [],
        }

        if self.input_scaler is not None:
            model_params['input_scaler'] = {
                'mean_': self.input_scaler.mean_.tolist(),  # type: ignore
                'scale_': self.input_scaler.scale_.tolist(),  # type: ignore
                'n_features_in_': int(self.input_scaler.n_features_in_),  # type: ignore
            }

        if self.output_scaler is not None:
            model_params['output_scaler'] = {
                'mean_': self.output_scaler.mean_.tolist(),  # type: ignore
                'scale_': self.output_scaler.scale_.tolist(),  # type: ignore
                'n_features_in_': int(self.output_scaler.n_features_in_),
            }

        if include_training_data:
            if self.x_train is None or self.y_train is None:
                self.logger.warning(
                    'Training data is None. Skipping training data export.'
                )
            else:
                model_params['x_train'] = self.x_train.tolist()
                model_params['y_train'] = self.y_train.tolist()

        for i, gp in enumerate(self.model):
            params = {
                'output_index': i,
                'kernel_parameters': {
                    param.name: {
                        'value': param.values.tolist()  # noqa: PD011
                        if param.size > 1
                        else float(param.values),
                        'shape': param.shape,
                    }
                    for param in gp.kern.parameters
                },
                'nugget': float(gp.likelihood.variance[0]),
            }

            if self.linear_models[i] is not None:
                linear_model = self.linear_models[i]
                params['linear_regression'] = {
                    'coefficients': linear_model.coef_.tolist(),  # type: ignore
                    'intercept': float(linear_model.intercept_),  # type: ignore
                }
            else:
                params['linear_regression'] = None

            model_params['models'].append(params)

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with filepath.open('w') as f:
            json.dump(make_json_serializable(model_params), f, indent=2)

    def load_model_parameters_from_json(  # noqa: C901
        self,
        filepath: str | Path,
        *,
        has_training_data: bool = True,
    ):
        """Load GP model settings, kernel, linear regression, and optionally training data from a JSON file."""
        with Path(filepath).open() as f:
            model_params = json.load(f)

        # --- Settings / flags ---
        self.input_dimension = model_params['input_dimension']
        self.output_dimension = model_params['output_dimension']
        self.ARD = model_params['ARD']
        self.kernel_type = model_params['kernel_type']
        self.mean_function_type = model_params['mean_function']
        self.nugget_value = model_params['nugget_value']
        self.fix_nugget = model_params['fix_nugget']
        self.use_pca = model_params['use_pca']
        self.pca_threshold = model_params['pca_threshold']
        self.scale_inputs = model_params.get('scale_inputs', True)
        scale_outputs_flag = model_params.get('scale_outputs', True)

        # --- Input scaler ---
        self.input_scaler = None
        if self.scale_inputs and 'input_scaler' in model_params:
            self.input_scaler = SafeStandardScaler()
            s = model_params['input_scaler']
            self.input_scaler.mean_ = np.array(s['mean_'])
            self.input_scaler.scale_ = np.array(s['scale_'])
            self.input_scaler.n_features_in_ = s['n_features_in_']

        # --- Output scaler ---
        self.output_scaler = None
        if scale_outputs_flag:
            self.output_scaler = SafeStandardScaler()
            if 'output_scaler' in model_params:
                s = model_params['output_scaler']
                self.output_scaler.mean_ = np.array(s['mean_'])
                self.output_scaler.scale_ = np.array(s['scale_'])
                self.output_scaler.n_features_in_ = s['n_features_in_']
            # If missing, it's fine: we'll fit during preprocessing when fit=True.

        # Keep as local (don't assign to property)
        _pca_info_from_file = model_params.get('pca_info')

        # Reset PCA/latent bookkeeping; let training path fit as needed
        self.pca = None
        self.latent_dimension = None
        if self.use_pca:
            # Keep perform_scaling=False for consistency with external scalers
            self.pca = PrincipalComponentAnalysis(
                self.pca_threshold, perform_scaling=False
            )

        if has_training_data:
            # --- Training data path: let _fit_pca_and_create_models do all fitting ---
            self.x_train = np.array(model_params['x_train'])
            self.y_train = np.array(model_params['y_train'])

            # Transform inputs using restored scaler (no fit)
            self.x_train_scaled = self.apply_input_scaling(self.x_train, fit=False)

            # Build models without re-optimizing (since we'll overwrite params below)
            self._fit_pca_and_create_models(reoptimize=False)

        else:
            # --- No training data: rebuild empty models and then overwrite params ---
            models_data = model_params['models']
            # Trust the JSON if it differs from the header output_dimension
            if len(models_data) != self.output_dimension:
                self.output_dimension = len(models_data)

            self.model = []
            self.linear_models = []

            base_kernel = self._create_kernel()
            for _ in models_data:
                x_dummy = np.zeros((1, self.input_dimension))
                y_dummy = np.zeros((1, 1))
                kernel_copy = base_kernel.copy()
                self._set_kernel_hyperparameter_bounds(kernel_copy)
                use_gpy_normalizer = self.output_scaler is None
                gp = GPy.models.GPRegression(
                    X=x_dummy,
                    Y=y_dummy,
                    kernel=kernel_copy,
                    mean_function=None,
                    normalizer=use_gpy_normalizer,
                )
                self._configure_nugget(gp)
                self.model.append(gp)
                self.linear_models.append(None)

        # --- Overwrite kernel / nugget / linear trend params from file ---
        models_data = model_params['models']
        for i, m in enumerate(models_data):
            gp = self.model[i]

            for name, value in m['kernel_parameters'].items():
                if name in gp.kern.parameter_names():
                    gp.kern[name] = np.array(value['value']).reshape(value['shape'])

            gp.likelihood.variance = m['nugget']
            if self.fix_nugget:
                gp.likelihood.variance.fix()

            lr_params = m.get('linear_regression')
            if lr_params is not None:
                lr = LinearRegression()
                # shape: (1, n_features) to match sklearn internals
                lr.coef_ = np.array(lr_params['coefficients']).reshape(1, -1)
                lr.intercept_ = np.array([lr_params['intercept']])
                self.linear_models[i] = lr
            else:
                self.linear_models[i] = None


# ===============
# Public function
# ===============


def create_gp_model(
    command_args=None,
    used_config_dir: str | Path | None = None,
    logger: logging.Logger | None = None,
    **kwargs,
) -> GaussianProcessModel:
    """
    Create a Gaussian Process Model instance.

    Priority:
    1. Keyword arguments (`kwargs`)
    2. Command-line arguments (`command_args`)
    3. Default config file
    4. Raise error if none found
    """
    settings = (
        _create_gp_settings_from_kwargs(**kwargs)
        if kwargs
        else _create_gp_settings_from_command_args(command_args)
        if command_args is not None
        else _create_gp_settings_from_default_config()
    )

    logger = logger or setup_logger()
    loginfo = make_log_info(logger)
    loginfo('Creating GP Regression Model.')

    config_dict = settings.model_dump()

    if used_config_dir is not None:
        used_config_dir = Path(used_config_dir)
        used_config_dir.mkdir(parents=True, exist_ok=True)
        output_path = used_config_dir / 'gp_config_used.json'
    else:
        output_path = 'gp_config_used.json'

    save_used_settings_as_config(config=config_dict, output_path=output_path)

    model = GaussianProcessModel(settings, logger=logger)
    return model  # noqa: RET504


# ========================
# Private helper functions
# ========================


def _create_gp_settings_from_kwargs(**kwargs) -> GaussianProcessModelSettings:
    return GaussianProcessModelSettings(**kwargs)


def _create_gp_settings_from_command_args(
    command_args: list[str],
) -> GaussianProcessModelSettings:
    args = parse_gp_arguments(command_args)
    config_file_path = args.pop('config_file', None)
    cli_args = {k: v for k, v in args.items() if v is not None}

    if config_file_path:
        config_data = load_settings_from_config(config_file_path)
        merged_settings = {**config_data, **cli_args}
        return GaussianProcessModelSettings(**merged_settings)

    return GaussianProcessModelSettings(**cli_args)


def _create_gp_settings_from_default_config(
    logger: logging.Logger | None = None,
) -> GaussianProcessModelSettings:
    config_filename = os.getenv('GP_CONFIG_FILE', 'gp_config.json')
    config_path = Path(config_filename)

    if config_path.exists():
        if logger:
            loginfo = make_log_info(logger)
            loginfo(f"No arguments provided. Loading config from '{config_path}'.")
        config_data = load_settings_from_config(config_path)
        return GaussianProcessModelSettings(**config_data)

    msg = (
        'No keyword arguments, no command-line arguments, and no config file found. '
        'Cannot create GP model because input_dimension and output_dimension are required.'
    )
    raise ValueError(msg)


def _get_mean_function_name(mean_function):
    if mean_function is None:
        return 'None'
    if type(mean_function).__name__ == 'Additive':
        name1 = _get_mean_function_name(mean_function.mapping1)
        name2 = _get_mean_function_name(mean_function.mapping2)
        return f'({name1} + {name2})'
    return type(mean_function).__name__


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Custom formatter for nicer help messages."""


def parse_gp_arguments(args=None) -> dict:
    """
    Parse command-line arguments for creating a Gaussian Process Model.

    Parameters
    ----------
    args : list[str], optional
        List of command-line arguments. Defaults to None.

    Returns
    -------
    dict
        Parsed arguments as a dictionary.
    """
    parser = argparse.ArgumentParser(
        description='Create a Gaussian Process Model.',
        formatter_class=CustomFormatter,
    )
    # Group: Required Arguments
    required = parser.add_argument_group('Required arguments')
    required.add_argument(
        '--input_dimension',
        type=int,
        required=True,
        help='Input dimension.',
    )
    required.add_argument(
        '--output_dimension',
        type=int,
        required=True,
        help='Output dimension.',
    )

    # Group: Optional Arguments
    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument(
        '--ARD',
        type=lambda x: x.lower() in ('true', '1'),
        default=True,
        help='Use ARD (Automatic Relevance Determination).',
    )
    optional.add_argument(
        '--kernel_type',
        type=str.lower,
        choices=['exponential', 'matern32', 'matern52', 'rbf'],
        default='matern52',
        help='Kernel type to use.',
    )
    optional.add_argument(
        '--mean_function',
        type=str.lower,
        choices=['none', 'linear'],
        default='none',
        help='Mean function for the GP.',
    )
    optional.add_argument(
        '--config_file',
        type=str,
        help='Path to a JSON config file.',
    )
    optional.add_argument(
        '--nugget_value',
        type=float,
        default=1e-6,
        help='Nugget value for Gaussian noise (default: 1e-6).',
    )
    optional.add_argument(
        '--fix_nugget',
        action='store_true',
        help='Fix the nugget value during optimization.',
    )
    optional.add_argument(
        '--use_pca',
        action='store_true',
        help='Use PCA to reduce output dimension before training GP.',
    )
    optional.add_argument(
        '--pca_threshold',
        type=float,
        default=0.999,
        help='Variance threshold to retain when using PCA (default: 0.999).',
    )
    optional.add_argument(
        '--scale_inputs',
        type=lambda x: x.lower() in ('true', '1'),
        default=True,
        help='Whether to scale inputs before training the GP.',
    )
    optional.add_argument(
        '--scale_outputs',
        type=lambda x: x.lower() in ('true', '1'),
        default=True,
        help='Whether to scale outputs before training the GP.',
    )

    parsed_args = parser.parse_args(args)
    return vars(parsed_args)


# =========================================================
# CLI interface
# =========================================================

if __name__ == '__main__':
    from logging_utilities import LoggerAutoFlusher, log_exception, make_log_info

    # Setup logger properly
    logger = setup_logger(
        log_filename='logFileGPMODEL.txt',
        prefix='GPMODEL',
        console_level=logging.INFO,
        file_level=logging.DEBUG,
    )
    loginfo = make_log_info(logger)

    flusher = LoggerAutoFlusher(logger, interval=10)
    flusher.start()

    try:
        loginfo('Starting Gaussian Process Model creation...')
        gp_model = create_gp_model(command_args=sys.argv[1:], logger=logger)
        loginfo('GP model created successfully.')
    except Exception as ex:  # noqa: BLE001
        log_exception(logger, ex, message='Fatal error during GP model creation')
        sys.exit(1)
    finally:
        flusher.stop()
        # flush_logger(logger)
        loginfo('Program finished and logs flushed.')
