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
from GPy.mappings import Additive, Constant, Linear
from logging_utilities import ensure_logger, flush_logger, get_default_logger
from pydantic import BaseModel, Field

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
    mean_function: str = Field('none', pattern='^(none|constant|linear)$')


class GaussianProcessModel:
    """A class to represent a Gaussian Process Model."""

    def __init__(
        self,
        settings: GaussianProcessModelSettings,
        logger: logging.Logger | None = None,
    ):
        self.settings = settings
        self._logger = logger or get_default_logger()

        self.input_dimension = settings.input_dimension
        self.output_dimension = settings.output_dimension
        self.ARD = settings.ARD
        self.kernel_type = settings.kernel_type.lower()
        self.mean_function_type = settings.mean_function.lower()

        self.kernel = self._create_kernel()
        self.model = self._create_surrogate()
        self._add_nugget_parameter()

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

    def _create_mean_function(self):
        if self.mean_function_type == 'none':
            return None
        if self.mean_function_type == 'constant':
            return Constant(input_dim=self.input_dimension, output_dim=1)
        if self.mean_function_type == 'linear':
            return Additive(
                Constant(input_dim=self.input_dimension, output_dim=1),
                Linear(input_dim=self.input_dimension, output_dim=1),
            )
        msg = f"Unknown mean_function '{self.mean_function_type}'"
        raise ValueError(msg)

    def _create_surrogate(self):
        mean_function = self._create_mean_function()
        models = []
        for _ in range(self.output_dimension):
            model = GPy.models.GPRegression(
                X=np.zeros((1, self.input_dimension)),
                Y=np.zeros((1, 1)),
                kernel=self.kernel.copy(),
                mean_function=mean_function,
                normalizer=True,
            )
            models.append(model)
        return models

    def _add_nugget_parameter(self, nugget_value=1e-6):
        """Add a nugget parameter to improve numerical stability."""
        for i in range(self.output_dimension):
            self.model[i].Gaussian_noise.variance = nugget_value

    def _fix_nugget_parameter(self, value=1e-8):
        """Fix the nugget parameter to a specific value."""
        for i in range(self.output_dimension):
            self.model[i].likelihood.variance = value
            self.model[i].likelihood.variance.fix()

    def update(self, x_train, y_train, *, reoptimize=True, num_random_restarts=10):
        """Update the GP models with new training data."""
        out_dim = y_train.shape[1]
        for i in range(out_dim):
            self.model[i].set_XY(x_train, np.reshape(y_train[:, i], (-1, 1)))
            if reoptimize:
                self.model[i].optimize_restarts(num_random_restarts)
            else:
                _ = self.model[i].posterior

    def predict(self, x_predict):
        """Predict the output for the given input data."""
        y_mean = np.zeros((x_predict.shape[0], self.output_dimension))
        y_var = np.zeros((x_predict.shape[0], self.output_dimension))
        for i in range(self.output_dimension):
            mean, var = self.model[i].predict(x_predict)
            y_mean[:, i] = mean.flatten()
            y_var[:, i] = var.flatten()
        return y_mean, y_var

    def loo_predictions(self, y_train, epsilon=1e-8):
        """Calculate Leave-One-Out (LOO) predictions."""
        if not hasattr(self.model[0], 'posterior'):
            msg = 'Model must be trained before computing LOO predictions.'
            raise ValueError(msg)

        loo_pred = np.zeros_like(y_train, dtype=np.float64)
        for i in range(self.output_dimension):
            posterior = self.model[i].posterior
            alpha = posterior.woodbury_vector  # (N, 1)
            k_inverse = posterior.woodbury_inv  # (N, N)
            diagonal_k_inverse = np.diag(k_inverse)

            if np.any(diagonal_k_inverse == 0):
                self._logger.warning(
                    f'Zero detected on diagonal for output {i}. Adding epsilon={epsilon}.'
                )
                diagonal_k_inverse = np.where(
                    diagonal_k_inverse == 0, epsilon, diagonal_k_inverse
                )

            loo_pred[:, i] = y_train[:, i].flatten() - (
                alpha.flatten() / diagonal_k_inverse
            )

        return loo_pred

    def __repr__(self):
        """Provide a string representation of the GaussianProcessModel instance."""
        mean_function = self.model[0].mean_function
        names = _get_mean_function_name(mean_function)
        return (
            f'GaussianProcessModel(input_dimension={self.input_dimension}, '
            f'output_dimension={self.output_dimension}, ARD={self.ARD}, '
            f'kernel_type={self.model[0].kern.name}, '
            f'mean_function={names})'
        )

    def flush_logs(self):
        """Flush the logs for the Gaussian Process Model."""
        flush_logger(self._logger)


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

    logger = logger or get_default_logger()
    logger.info('Creating GP Regression Model.')

    config_dict = settings.model_dump()

    if used_config_dir is not None:
        used_config_dir = Path(used_config_dir)
        used_config_dir.mkdir(parents=True, exist_ok=True)
        output_path = used_config_dir / 'gp_config_used.json'
    else:
        output_path = 'gp_config_used.json'

    save_used_settings_as_config(config=config_dict, output_path=output_path)

    model = GaussianProcessModel(settings, logger=logger)
    logger.info(f'Created: {model}')
    return model


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
            logger.info(
                f"No arguments provided. Loading config from '{config_path}'."
            )
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
        choices=['none', 'constant', 'linear'],
        default='none',
        help='Mean function for the GP.',
    )
    optional.add_argument(
        '--config_file',
        type=str,
        help='Path to a JSON config file.',
    )

    parsed = parser.parse_args(args)
    return vars(parsed)


# =========================================================
# CLI interface
# =========================================================

if __name__ == '__main__':
    from logging_utilities import (
        LoggerAutoFlusher,
        ensure_logger,
        log_exception,
        set_default_logger,
    )

    # Setup logger properly
    logger = ensure_logger(
        log_filename='logFileGPMODEL.txt',
        prefix='GPMODEL',
        console_level=logging.INFO,
        file_level=logging.DEBUG,
    )
    set_default_logger(logger)

    flusher = LoggerAutoFlusher(logger, interval=10)
    flusher.start()

    try:
        logger.info('Starting Gaussian Process Model creation...')
        gp_model = create_gp_model(command_args=sys.argv[1:], logger=logger)
        logger.info('GP model created successfully.')
    except Exception as ex:  # noqa: BLE001
        log_exception(logger, ex, message='Fatal error during GP model creation')
        sys.exit(1)
    finally:
        flusher.stop()
        flush_logger(logger)
        logger.info('Program finished and logs flushed.')
