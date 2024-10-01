"""Implements an adaptive design of experiments strategy using Gaussian Process (GP) and Principal Component Analysis (PCA)."""

import numpy as np
from space_filling_doe import LatinHypercubeSampling


class AdaptiveDesignOfExperiments:
    """
    Adaptive Design of Experiments (DoE) using Gaussian Process (GP) and Principal Component Analysis (PCA).

    This class implements an adaptive design of experiments strategy to select new training points
    based on the Integrated Mean Squared Error (IMSE) criterion.

    Attributes
    ----------
        gp_model (list): List of Gaussian Process models.
        pca (PCA): Principal Component Analysis object.
        domain (array-like): The domain of the input space.

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

    def __init__(self, gp_model, pca, domain):
        """
        Initialize the AdaptiveDesignOfExperiments class.

        Parameters
        ----------
            gp_model (list): List of Gaussian Process models.
            pca (PCA): Principal Component Analysis object.
            domain (array-like): The domain of the input space.
        """
        self.gp_model = gp_model
        self.pca = pca
        self.domain = domain

        self._lengthscale_for_doe()
        self._kernel_for_doe()
        self._gp_for_doe()

    def _lengthscale_for_doe(self):
        """
        Compute the lengthscale for the design of experiments.

        The lengthscale is computed as a weighted sum of the lengthscales of the individual GP models,
        where the weights are the explained variances of the PCA components.

        Returns
        -------
            float: The computed lengthscale.
        """
        eigenvalues = self.pca.explained_variance_
        w = eigenvalues / np.sum(eigenvalues)
        lengthscales = np.atleast_2d([m.kernel.lengthscale for m in self.gp_model])
        self.lengthscale_star = np.sum(w * lengthscales)
        return self.lengthscale_star

    def _kernel_for_doe(self):
        """
        Create a kernel for the design of experiments.

        The kernel is a copy of the kernel of the first GP model, with the lengthscale set to the computed lengthscale.

        Returns
        -------
            Kernel: The created kernel.
        """
        self.kernel = self.gp_model.kernel.copy()
        self.kernel.lengthscale = self.lengthscale_star
        return self.kernel

    def _gp_for_doe(self):
        """
        Create a Gaussian Process model for the design of experiments.

        The GP model is a copy of the first GP model, with the kernel set to the created kernel.

        Returns
        -------
            GaussianProcessRegressor: The created GP model.
        """
        self.gp_model_for_doe = self.gp_model[0].copy()
        self.gp_model_for_doe.kernel = self.kernel
        return self.gp_model_for_doe

    def _imse_w_approximation(self, x_train, mci_samples, candidate_training_points):
        """
        Compute the IMSE approximation for candidate training points.

        Parameters
        ----------
            X_train (array-like): The current training data.
            mci_samples (array-like): Monte Carlo integration samples.
            candidate_training_points (array-like): Candidate training points.

        Returns
        -------
            array: The IMSE values for the candidate training points.
        """
        self.gp_model_for_doe.set_XY(
            x_train,
            np.zeros((x_train.shape[0], 1)),
        )
        _, pred_var = self.gp_model_for_doe.predict(mci_samples)
        n_theta = x_train.shape[1]
        beta = 2.0 * n_theta
        imse = np.zeros((candidate_training_points.shape[0], 1))
        for i, candidate in enumerate(candidate_training_points):
            correlation_vector = self.gp_model_for_doe.kern.K(mci_samples, candidate)
            imse[i] = (
                1
                / (mci_samples.shape[0])
                * np.dot(correlation_vector**beta, pred_var)
            )
        return imse

    def select_training_points(self, x_train, n_points, mci_samples, n_candidates):
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
        dimension = x_train.shape[1]
        for _ in range(n_points):
            lhs = LatinHypercubeSampling(
                n_samples=n_candidates, n_dimensions=dimension
            )
            candidate_training_points = lhs.generate(self.domain)
            imse = self._imse_w_approximation(
                x_train, mci_samples, candidate_training_points
            )
            next_training_point = candidate_training_points[np.argmax(imse)]
            x_train = np.vstack((x_train, next_training_point))
        return x_train[-n_points:, :]
