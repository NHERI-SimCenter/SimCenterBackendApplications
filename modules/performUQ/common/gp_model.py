"""Module for creating and using Gaussian Process models with the GaussianProcessModel class."""

from __future__ import annotations

import argparse
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
from principal_component_analysis import PrincipalComponentAnalysis
from pydantic import BaseModel, Field, model_validator
from sklearn.linear_model import LinearRegression

# =========================================================
# Top-level Classes
# =========================================================


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

        self.pca = (
            PrincipalComponentAnalysis(self.pca_threshold, perform_scaling=True)
            if self.use_pca
            else None
        )
        self.latent_dimension = None

        self.kernel = self._create_kernel()

        self.linear_models: list[LinearRegression | None] = []
        self.model: list[GPy.models.GPRegression] = []

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

    def _create_surrogate(self):
        model = GPy.models.GPRegression(
            X=np.zeros((1, self.input_dimension), dtype=float),
            Y=np.zeros((1, 1), dtype=float),
            kernel=self.kernel.copy(),
            mean_function=None,
            normalizer=True,
        )
        self._configure_nugget(model)
        return model

    def _configure_nugget(self, model: GPy.models.GPRegression):
        """Configure the nugget (Gaussian noise variance) for each GP model."""
        if self.settings.fix_nugget:
            model.likelihood.variance = self.settings.nugget_value
            model.likelihood.variance.fix()
        else:
            model.Gaussian_noise.variance = self.settings.nugget_value

    def update(self, x_train, y_train, *, reoptimize=True, num_random_restarts=10):
        """
        Update the Gaussian Process model with new training data.

        Parameters
        ----------
        x_train : ndarray
            Input training data of shape (n_samples, input_dimension).
        y_train : ndarray
            Output training data of shape (n_samples, output_dimension).
        reoptimize : bool, optional
            Whether to reoptimize the model hyperparameters (default is True).
        num_random_restarts : int, optional
            Number of random restarts for optimization (default is 10).
        """
        if self.pca is not None:
            self.loginfo('Using PCA for output dimensionality reduction.')
            self.pca.fit(y_train)
            y_latent = self.pca.project_to_latent_space(y_train)
            self.latent_dimension = int(self.pca.n_components)  # type: ignore
            self.output_dimension = self.latent_dimension
            self.loginfo(
                f'PCA reduced output dimension from {self.model_output_dimension} '
                f'to {self.output_dimension}, capturing at least '
                f'{self.pca_threshold:.2%} variance.'
            )
        else:
            self.loginfo('Using model outputs for training GP. No PCA applied.')
            y_latent = y_train
            self.output_dimension = self.model_output_dimension

        # Re-create GP models now that output dimension is known
        self.model = []
        self.linear_models = []

        for i in range(self.output_dimension):
            x = x_train
            y = np.reshape(y_latent[:, i], (-1, 1))

            linear_model = None
            if self.mean_function_type == 'linear':
                linear_model = LinearRegression()
                linear_model.fit(x, y)
                y_detrended = y - linear_model.predict(x)
            else:
                y_detrended = y

            gp = GPy.models.GPRegression(
                X=x,
                Y=y_detrended,
                kernel=self.kernel.copy(),
                mean_function=None,
                normalizer=True,
            )
            self._configure_nugget(gp)
            if reoptimize:
                gp.optimize_restarts(num_random_restarts)
            else:
                _ = gp.posterior

            self.model.append(gp)
            self.linear_models.append(linear_model)

            # self.model[i].set_XY(x, y_detrended)
            # if reoptimize:
            #     self.model[i].optimize_restarts(num_random_restarts)
            # else:
            #     _ = self.model[i].posterior

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
        y_mean = np.zeros((x_predict.shape[0], self.output_dimension))
        y_var = np.zeros((x_predict.shape[0], self.output_dimension))

        for i in range(self.output_dimension):
            mean, var = self.model[i].predict(x_predict)  # type: ignore
            if self.linear_models[i] is not None:
                trend = self.linear_models[i].predict(x_predict)  # type: ignore
                mean += trend

            y_mean[:, i] = mean.flatten()
            y_var[:, i] = var.flatten()

        if self.pca is not None:
            y_mean = self.pca.project_back_to_original_space(y_mean)
            # y_var = self.pca.inverse_transform_variance(y_var) # Not transforming variance for now

        return y_mean, y_var  # variance is in latent space, if pca is used

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
        if not hasattr(self.model[0], 'posterior'):
            msg = 'Model must be trained before computing LOO predictions.'
            raise ValueError(msg)

        if self.pca is not None:
            y_train = self.pca.project_to_latent_space(y_train)

        loo_pred_latent = np.zeros_like(y_train, dtype=np.float64)

        for i in range(self.output_dimension):
            x = x_train
            y = y_train[:, i].reshape(-1, 1)

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

        # Step 5: Inverse PCA if used
        if self.pca is not None:
            loo_pred_latent = self.pca.project_back_to_original_space(
                loo_pred_latent
            )
        return loo_pred_latent

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

    # def __repr__(self):
    #     """Provide a string representation of the GaussianProcessModel instance."""
    #     trend_desc = (
    #         'LinearTrend (sklearn)'
    #         if self.mean_function_type == 'linear'
    #         else 'None'
    #     )
    #     variance = float(self.model[0].likelihood.variance)  # Extract value
    #     is_fixed = self.model[0].likelihood.variance.is_fixed  # Check if fixed
    #     return (
    #         f'GaussianProcessModel('
    #         f'input_dimension={self.input_dimension}, '
    #         f'output_dimension={self.output_dimension}, '
    #         f'ARD={self.ARD}, '
    #         f'kernel_type={self.model[0].kern.name}, '
    #         f'mean_function={trend_desc}, '
    #         f'nugget_value={variance}, '
    #         f'fix_nugget={is_fixed})'
    #     )

    def flush_logs(self):
        """Flush the logs for the Gaussian Process Model."""
        flush_logger(self.logger)


# =========================================================
# Public function
# =========================================================


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
    # loginfo(f'Created: {model}')
    return model  # noqa: RET504


# =========================================================
# Private helper functions
# =========================================================


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
        help='Fix the nugget to the specified value instead of optimizing it. '
        'Default behavior is to optimize.',
    )
    optional.add_argument(
        '--use_pca',
        action='store_true',
        help='Enable dimensionality reduction using PCA before GP modeling.',
    )
    optional.add_argument(
        '--pca_threshold',
        type=float,
        default=0.999,
        help='Cumulative explained variance threshold for PCA (default: 0.999).',
    )

    parsed = parser.parse_args(args)
    return vars(parsed)


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
