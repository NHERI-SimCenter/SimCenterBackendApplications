"""
kernel_density_estimation.py.

A simple wrapper around scipy.stats.gaussian_kde for multivariate kernel density estimation (KDE),
with automatic bandwidth selection and evaluation support for arbitrary points.

Features:
- Supports multivariate datasets
- Automatically selects bandwidth (Scott's rule by default)
- Provides density and log-density evaluation
"""

import numpy as np
from scipy.stats import gaussian_kde


class GaussianKDE:
    """
    Multivariate Gaussian Kernel Density Estimation (KDE) using scipy.

    This class provides a clean interface to fit a KDE model and evaluate density
    at arbitrary points.

    Parameters
    ----------
    samples : np.ndarray
        Array of shape (n_samples, n_dimensions) containing the training samples.
    bw_method : str or float, optional
        Bandwidth selection method: 'scott', 'silverman', or a float scale factor (default: 'scott').
    """

    def __init__(self, samples, bw_method='scott'):
        self.samples = samples
        self.kde = gaussian_kde(samples.T, bw_method=bw_method)
        self.bandwidth = self.kde.factor

    def evaluate(self, points):
        """
        Evaluate the KDE at the given query points.

        Parameters
        ----------
        points : np.ndarray
            Array of shape (n_eval, n_dimensions) containing the query points.

        Returns
        -------
        np.ndarray
            Array of shape (n_eval,) with estimated density values.
        """
        return self.kde(points.T)

    def log_evaluate(self, points):
        """
        Evaluate the log-density of the KDE at the given query points.

        Parameters
        ----------
        points : np.ndarray
            Array of shape (n_eval, n_dimensions) containing the query points.

        Returns
        -------
        np.ndarray
            Array of shape (n_eval,) with log-density values.
        """
        return np.log(self.evaluate(points))

    def get_bandwidth(self):
        """
        Get the bandwidth factor used in the KDE.

        Returns
        -------
        float
            Bandwidth scaling factor.
        """
        return self.bandwidth


if __name__ == '__main__':
    # Example usage
    np.random.seed(0)

    # Generate 3D Gaussian samples
    samples = np.random.multivariate_normal(mean=[0, 0, 0], cov=np.eye(3), size=1000)

    # Fit KDE
    kde = GaussianKDE(samples)

    # Evaluate on new points
    query_points = np.array([[0, 0, 0], [1, 1, 1], [-1, -1, -1]])
    densities = kde.evaluate(query_points)
    log_densities = kde.log_evaluate(query_points)

    # Print results
    print('Bandwidth factor:', kde.get_bandwidth())
    for pt, d, ld in zip(query_points, densities, log_densities):
        print(f'Point {pt} → Density: {d:.5f}, Log-Density: {ld:.5f}')
