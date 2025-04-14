"""Module for creating and using Gaussian Process models with the GaussianProcessModel class."""

import logging

import GPy
import numpy as np
from GPy.mappings import Additive, Constant, Linear
from scipy.linalg import cho_solve

logger = logging.getLogger(__name__)


class GaussianProcessModel:
    """
    A class to represent a Gaussian Process Model.

    Attributes
    ----------
        input_dimension (int): The input dimension of the model.
        output_dimension (int): The output dimension of the model.
        ARD (bool): Automatic Relevance Determination flag.
        kernel (GPy.kern.Matern52): The kernel used in the model.
        model (list[GPy.models.GPRegression]): The list of GP regression models.
    """

    def __init__(self, input_dimension, output_dimension, ARD):  # noqa: N803
        """
        Initialize the GaussianProcessModel.

        Args:
            input_dimension (int): The input dimension of the model.
            output_dimension (int): The output dimension of the model.
            ARD (bool): Automatic Relevance Determination flag.
        """
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.ARD = ARD

        self.kernel = self._create_kernel()
        self.model = self._create_surrogate()
        # self._add_nugget_parameter()
        self._fix_nugget_parameter()

    def _create_kernel(self):
        """
        Create the kernel for the Gaussian Process.

        Returns
        -------
            GPy.kern.Matern52: The Matern52 kernel.
        """
        return GPy.kern.Matern52(input_dim=self.input_dimension, ARD=self.ARD)

    def _linear_const_mean(self):
        return Additive(
            Linear(input_dim=self.input_dimension, output_dim=1),
            Constant(input_dim=self.input_dimension, output_dim=1),
        )

    def _create_surrogate(self):
        """
        Create the surrogate GP regression models.

        Returns
        -------
            list[GPy.models.GPRegression]: The list of GP regression models.
        """
        m_list = []
        for _ in range(self.output_dimension):
            m_list += [
                GPy.models.GPRegression(
                    np.zeros((1, self.input_dimension)),
                    np.zeros((1, 1)),
                    kernel=self.kernel.copy(),
                    mean_function=self._linear_const_mean(),
                )
            ]
        return m_list

    def _add_nugget_parameter(self):
        """Add a nugget parameter to the GP models to improve numerical stability."""
        for i in range(self.output_dimension):
            self.model[i].Gaussian_noise.variance = 1e-6

    def _fix_nugget_parameter(self, value=1e-8):
        """
        Fix the nugget parameter to a specific value.

        Args:
            value (float): The value to fix the nugget parameter to.
        """
        for i in range(self.output_dimension):
            self.model[i].likelihood.variance = value
            self.model[i].likelihood.variance.fix()

    def update(self, x_train, y_train, reoptimize=True):  # noqa: FBT002
        """
        Update the GP models with new training data.

        This method sets the training inputs and outputs for each GP model.
        Optionally, it re-optimizes the hyperparameters based on the new data.

        Args:
            x_train (np.ndarray): The input training data of shape (n_samples, n_features).
            y_train (np.ndarray): The output training data of shape (n_samples, n_outputs).
            reoptimize (bool, optional): Whether to re-optimize hyperparameters. Defaults to True.
        """
        for i in range(self.output_dimension):
            self.model[i].set_XY(x_train, np.reshape(y_train[:, i], (-1, 1)))
            if reoptimize:
                self.model[i].optimize()
            else:
                # Force recomputation of the posterior
                _ = self.model[i].posterior  # Triggers re-computation if dirty

    def predict(self, x_predict):
        """
        Predict the output for the given input data.

        Args:
            x_predict (np.ndarray): The input data for prediction.

        Returns
        -------
            tuple: A tuple containing the mean and variance of the predictions.
        """
        y_mean = np.zeros((x_predict.shape[0], self.output_dimension))
        y_var = np.zeros((x_predict.shape[0], self.output_dimension))
        for i in range(self.output_dimension):
            mean, var = self.model[i].predict(x_predict)
            y_mean[:, i] = mean.flatten()
            y_var[:, i] = var.flatten()

        return y_mean, y_var

    # def loo_predictions(self, y_train):
    #     """
    #     Calculate the Leave-One-Out (LOO) predictions.

    #     Args:
    #         y_train (np.ndarray): The output training data.

    #     Returns
    #     -------
    #         np.ndarray: The LOO predictions.
    #     """
    #     loo_pred = np.zeros_like(y_train)
    #     for i in range(self.output_dimension):
    #         cholesky_factor = self.model[i].posterior.K_chol
    #         alpha = cho_solve(
    #             (cholesky_factor, True), np.reshape(y_train[:, i], (-1, 1))
    #         )
    #         cholesky_factor_inverse = cho_solve(
    #             (cholesky_factor, True), np.eye(cholesky_factor.shape[0])
    #         )
    #         k_inv_diag = np.reshape(
    #             np.sum(cholesky_factor_inverse**2, axis=1), (-1, 1)
    #         )
    #         loo_pred[:, i] = (
    #             np.reshape(y_train[:, i], (-1, 1)) - alpha / k_inv_diag
    #         ).flatten()
    #     return loo_pred

    def loo_predictions(self, y_train, epsilon=1e-8):
        """
        Calculate the Leave-One-Out (LOO) predictions.

        Args:
            y_train (np.ndarray): The output training data. Shape: (N, output_dimension)
            epsilon (float): Small value added to diag_Kinv only where it's zero.

        Returns
        -------
            np.ndarray: The LOO predictions. Shape: (N, output_dimension)
        """
        if not hasattr(self.model[0], 'posterior'):
            msg = (
                'Model must be trained (optimized) before computing LOO predictions.'
            )
            raise ValueError(msg)

        loo_pred = np.zeros_like(y_train, dtype=np.float64)

        for i in range(self.output_dimension):
            posterior = self.model[i].posterior
            alpha = posterior.woodbury_vector  # (N, 1)
            k_inverse = posterior.woodbury_inv  # (N, N)
            diagonal_k_inverse = np.diag(k_inverse)

            if np.any(diagonal_k_inverse == 0):
                logger.warning(
                    f'Zero detected on diagonal of Woodbury inverse for output {i}. '
                    f'Adding epsilon={epsilon} to avoid division by zero.'
                )
                diagonal_k_inverse = np.where(
                    diagonal_k_inverse == 0, epsilon, diagonal_k_inverse
                )
            loo_pred[:, i] = y_train[:, i].flatten() - (
                alpha.flatten() / diagonal_k_inverse
            )

        return loo_pred
