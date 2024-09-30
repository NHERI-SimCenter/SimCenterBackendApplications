"""Module for creating and using Gaussian Process models with the GaussianProcessModel class."""

import GPy
import numpy as np
from scipy.linalg import cho_solve


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
                    self.kernel.copy(),
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

    def fit(self, x_train, y_train):
        """
        Fit the GP models to the training data.

        Args:
            x_train (np.ndarray): The input training data.
            y_train (np.ndarray): The output training data.
        """
        for i in range(self.output_dimension):
            self.model[i].set_XY(x_train, np.reshape(y_train[:, i], (-1, 1)))
            self.model[i].optimize()

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

    def loo_predictions(self, y_train):
        """
        Calculate the Leave-One-Out (LOO) predictions.

        Args:
            y_train (np.ndarray): The output training data.

        Returns
        -------
            np.ndarray: The LOO predictions.
        """
        loo_pred = np.zeros_like(y_train)
        for i in range(self.output_dimension):
            cholesky_factor = self.model[i].posterior.K_chol
            alpha = cho_solve(
                (cholesky_factor, True), np.reshape(y_train[:, i], (-1, 1))
            )
            cholesky_factor_inverse = cho_solve(
                (cholesky_factor, True), np.eye(cholesky_factor.shape[0])
            )
            k_inv_diag = np.reshape(
                np.sum(cholesky_factor_inverse**2, axis=1), (-1, 1)
            )
            loo_pred[:, i] = (
                np.reshape(y_train[:, i], (-1, 1)) - alpha / k_inv_diag
            ).flatten()
        return loo_pred
