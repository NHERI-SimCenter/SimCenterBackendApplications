"""Implements an adaptive design of experiments strategy using Gaussian Process (GP) and Principal Component Analysis (PCA)."""

import numpy as np


class AdaptiveDesignOfExperiments:
    """
    Adaptive Design of Experiments (DoE) using Gaussian Process (GP) and Principal Component Analysis (PCA).

    This class implements an adaptive design of experiments strategy to select new training points
    based on the Integrated Mean Squared Error (IMSE) criterion.

    Attributes
    ----------
        gp_model (list): List of Gaussian Process models.
        pca (PCA): Principal Component Analysis object.

    Methods
    -------
        _lengthscale_for_doe():
            Compute the lengthscale for the design of experiments.

        _kernel_for_doe():
            Create a kernel for the design of experiments.

        _gp_for_doe():
            Create a Gaussian Process model for the design of experiments.

        _imse_w_approximation(X_train, mci_samples, candidate_training_points):
            Compute the IMSE approximation for candidate training points.

        select_training_points(X_train, n_points, mci_samples, n_candidates):
            Select new training points based on the IMSE criterion.
    """

    def __init__(self, gp_model, pca):
        """
        Initialize the AdaptiveDesignOfExperiments class.

        Parameters
        ----------
            gp_model (list): List of Gaussian Process models.
            pca (PCA): Principal Component Analysis object.
        """
        self.gp_model = gp_model
        self.pca = pca

        self._hyperparameters_for_doe()
        self._kernel_for_doe()
        self._gp_for_doe()

    def _hyperparameters_for_doe(self):
        """
        Compute the lengthscale for the design of experiments.

        The lengthscale is computed as a weighted sum of the lengthscales of the individual GP models,
        where the weights are the explained variances of the PCA components.

        Returns
        -------
            float: The computed lengthscale.
        """
        n_components = self.pca.n_components
        eigenvalues = self.pca.pca.explained_variance_[:n_components]
        w = eigenvalues / np.sum(eigenvalues)

        hyperparameters_matrix = [
            np.atleast_2d(model.kern.param_array) for model in self.gp_model.model
        ]
        hyperparameters_matrix = np.vstack(hyperparameters_matrix)
        self.pca_weighted_hyperparamters = np.dot(w, hyperparameters_matrix)
        return self.pca_weighted_hyperparamters

    def _kernel_for_doe(self):
        """
        Create a kernel for the design of experiments.

        The kernel is a copy of the kernel of the first GP model, with the lengthscale set to the computed lengthscale.

        Returns
        -------
            Kernel: The created kernel.
        """
        self.kernel = self.gp_model.kernel.copy()
        self.kernel.param_array[:] = self.pca_weighted_hyperparamters
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

    def _imse_w_approximation(self, x_train, mci_samples, weights=None):  # noqa: ARG002
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
        use_mse_w=True,  # noqa: FBT002
        weights=None,
    ):
        """
        Select new training points based on the IMSE criterion.

        Parameters
        ----------
            X_train (array-like): The current training data.
            n_points (int): The number of new training points to select.
            mci_samples (array-like): Monte Carlo integration samples.
            n_candidates (int): The number of candidate training points to generate.

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
