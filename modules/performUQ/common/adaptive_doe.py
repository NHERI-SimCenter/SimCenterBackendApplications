"""Implements an adaptive design of experiments strategy using Gaussian Process (GP) and Principal Component Analysis (PCA)."""

import numpy as np


class AdaptiveDesignOfExperiments:
    """
    Adaptive Design of Experiments (DoE) using a Gaussian Process (GP) surrogate model.

    This class implements an adaptive design of experiments strategy to select new training points
    based on the Integrated Mean Squared Error (IMSE) criterion.

    Attributes
    ----------
        gp_model (GaussianProcessModel): The GP surrogate model that internally handles PCA (if used).
    """

    def __init__(self, gp_model):
        """
        Initialize the AdaptiveDesignOfExperiments class.

        Parameters
        ----------
            gp_model (GaussianProcessModel): The GP surrogate model.
        """
        self.gp_model = gp_model
        self._hyperparameters_for_doe()
        self._kernel_for_doe()
        self._gp_for_doe()

    def _hyperparameters_for_doe(self):
        """
        Compute the weighted average of kernel hyperparameters for DoE.

        If PCA is used, weights are based on explained variance.
        Otherwise, uniform averaging is used.
        """
        if self.gp_model.use_pca:
            pca_info = self.gp_model.pca_info
            n_components = pca_info['n_components']
            explained_variance_ratio = np.asarray(
                pca_info['explained_variance_ratio']
            )[:n_components]
            if np.sum(explained_variance_ratio) == 0:
                w = np.full(n_components, 1.0 / n_components)
            else:
                w = explained_variance_ratio / np.sum(explained_variance_ratio)
        else:
            n_models = len(self.gp_model.model)
            w = np.full(n_models, 1.0 / n_models)

        hyperparameters_matrix = [
            np.atleast_2d(model.kern.param_array) for model in self.gp_model.model
        ]
        hyperparameters_matrix = np.vstack(hyperparameters_matrix)

        self.weighted_hyperparameters = np.dot(w, hyperparameters_matrix)
        return self.weighted_hyperparameters

    def _kernel_for_doe(self):
        """
        Create a kernel for the design of experiments.

        The kernel is a copy of the kernel of the first GP model, with the lengthscale set to the computed lengthscale.

        Returns
        -------
            Kernel: The created kernel.
        """
        self.kernel = self.gp_model.kernel.copy()
        self.kernel.param_array[:] = self.weighted_hyperparameters
        return self.kernel

    def _gp_for_doe(self):
        """
        Create a Gaussian Process model for the design of experiments.

        The GP model is a copy of the first GP model, with the kernel set to the created kernel.

        Returns
        -------
            GaussianProcessRegressor: The created GP model.
        """
        self.gp_model_for_doe = self.gp_model.model[0].copy()
        self.gp_model_for_doe.kern = self.kernel
        return self.gp_model_for_doe

    def _imse_w_approximation(self, x_train, mci_samples, weights=None):
        """
        Compute the IMSE approximation for candidate training points.

        Parameters
        ----------
            X_train (array-like): The current training data.
            mci_samples (array-like): Monte Carlo integration samples.

        Returns
        -------
            array: The IMSE values for the candidate training points.
        """
        candidate_training_points = mci_samples
        self.gp_model_for_doe.set_XY(
            x_train,
            np.zeros((x_train.shape[0], 1)),
        )
        _, pred_var = self.gp_model_for_doe.predict(mci_samples)
        n_theta = x_train.shape[1]
        beta = 2.0 * n_theta
        imse = np.zeros((candidate_training_points.shape[0], 1))
        for i, candidate in enumerate(candidate_training_points):
            correlation_vector = self.gp_model_for_doe.kern.K(
                mci_samples, np.atleast_2d(candidate)
            )
            imse[i] = (1 / mci_samples.shape[0]) * np.sum(
                (correlation_vector**beta) * pred_var
            )
        return imse

    def _mse_approximation(self, x_train, mci_samples):
        self.gp_model_for_doe.set_XY(
            x_train,
            np.zeros((x_train.shape[0], 1)),
        )
        _, pred_var = self.gp_model_for_doe.predict(mci_samples)
        return np.reshape(pred_var, (-1, 1))

    def _mse_w_approximation(self, x_train, mci_samples, weights):
        self.gp_model_for_doe.set_XY(
            x_train,
            np.zeros((x_train.shape[0], 1)),
        )
        _, pred_var = self.gp_model_for_doe.predict(mci_samples)
        if weights is None:
            weights = np.ones_like(pred_var)
        mse_w = pred_var.flatten() * weights.flatten()
        return np.reshape(mse_w, (-1, 1))

    def select_training_points(
        self,
        x_train,
        n_points,
        mci_samples,
        *,
        use_mse_w=True,
        weights=None,
    ):
        """
        Select new training points based on the IMSE criterion.

        Parameters
        ----------
            X_train (array-like): The current training data.
            n_points (int): The number of new training points to select.
            mci_samples (array-like): Monte Carlo integration samples.

        Returns
        -------
            array: The selected new training points.
        """
        if use_mse_w:
            acquisition_function = self._mse_w_approximation
        else:
            acquisition_function = self._imse_w_approximation
        for _ in range(n_points):
            acquisition_function_values = acquisition_function(
                x_train, mci_samples, weights
            )
            next_training_point = mci_samples[np.argmax(acquisition_function_values)]
            x_train = np.vstack((x_train, next_training_point))
        return x_train[-n_points:, :]
