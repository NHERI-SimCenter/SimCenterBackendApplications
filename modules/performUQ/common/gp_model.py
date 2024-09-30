import GPy
import numpy as np
from scipy.linalg import cho_solve


class GaussianProcessModel:
    def __init__(self, input_dimension, output_dimension, ARD):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.ARD = ARD

        self.kernel = self._create_kernel()
        self.model = self._create_surrogate()
        # self._add_nugget_parameter()
        self._fix_nugget_parameter()

    def _create_kernel(self):
        return GPy.kern.Matern52(input_dim=self.input_dimension, ARD=self.ARD)

    def _create_surrogate(self):
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
        for i in range(self.output_dimension):
            self.model[i].Gaussian_noise.variance = 1e-6

    def _fix_nugget_parameter(self, value=1e-8):
        for i in range(self.output_dimension):
            self.model[i].likelihood.variance = value
            self.model[i].likelihood.variance.fix()

    def fit(self, X_train, Y_train):
        for i in range(self.output_dimension):
            self.model[i].set_XY(X_train, np.reshape(Y_train[:, i], (-1, 1)))
            self.model[i].optimize()

    def predict(self, X_predict):
        Y_mean = np.zeros((X_predict.shape[0], self.output_dimension))
        Y_var = np.zeros((X_predict.shape[0], self.output_dimension))
        for i in range(self.output_dimension):
            mean, var = self.model[i].predict(X_predict)
            Y_mean[:, i] = mean.flatten()
            Y_var[:, i] = var.flatten()

        return Y_mean, Y_var

    def loo_predictions(self, Y_train):
        loo_pred = np.zeros_like(Y_train)
        for i in range(self.output_dimension):
            cholesky_factor = self.model[i].posterior.K_chol
            alpha = cho_solve(
                (cholesky_factor, True), np.reshape(Y_train[:, i], (-1, 1))
            )
            cholesky_factor_inverse = cho_solve(
                (cholesky_factor, True), np.eye(cholesky_factor.shape[0])
            )
            K_inv_diag = np.reshape(
                np.sum(cholesky_factor_inverse**2, axis=1), (-1, 1)
            )
            loo_pred[:, i] = (
                np.reshape(Y_train[:, i], (-1, 1)) - alpha / K_inv_diag
            ).flatten()
        return loo_pred
