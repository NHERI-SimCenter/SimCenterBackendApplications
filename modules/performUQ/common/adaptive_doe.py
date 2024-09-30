import numpy as np

from space_filling_doe import LatinHypercubeSampling


class AdaptiveDesignOfExperiments:
    def __init__(self, gp_model, pca, domain):
        self.gp_model = gp_model
        self.pca = pca
        self.domain = domain

        self._lengthscale_for_doe()
        self._kernel_for_doe()
        self._gp_for_doe()

    def _lengthscale_for_doe(self):
        eigenvalues = self.pca.explained_variance_
        w = eigenvalues / np.sum(eigenvalues)
        lengthscales = np.atleast_2d([m.kernel.lengthscale for m in self.gp_model])
        self.lengthscale_star = np.sum(w * lengthscales)
        return self.lengthscale_star

    def _kernel_for_doe(self):
        self.kernel = self.gp_model.kernel.copy()
        self.kernel.lengthscale = self.lengthscale_star
        return self.kernel

    def _gp_for_doe(self):
        self.gp_model_for_doe = self.gp_model[0].copy()
        self.gp_model_for_doe.kernel = self.kernel
        return self.gp_model_for_doe

    def _imse_w_approximation(self, X_train, mci_samples, candidate_training_points):
        self.gp_model_for_doe.set_XY(
            X_train,
            np.zeros((X_train.shape[0], 1)),
        )
        _, pred_var = self.gp_model_for_doe.predict(mci_samples)
        n_theta = X_train.shape[1]
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

    def select_training_points(self, X_train, n_points, mci_samples, n_candidates):
        dimension = X_train.shape[1]
        for _ in range(n_points):
            lhs = LatinHypercubeSampling(
                n_samples=n_candidates, n_dimensions=dimension
            )
            candidate_training_points = lhs.generate(self.domain)
            imse = self._imse_w_approximation(
                X_train, mci_samples, candidate_training_points
            )
            next_training_point = candidate_training_points[np.argmax(imse)]
            X_train = np.vstack((X_train, next_training_point))
        new_training_points = X_train[-n_points:, :]
        return new_training_points
