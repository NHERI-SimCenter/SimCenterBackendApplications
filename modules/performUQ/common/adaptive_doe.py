"""Implements an adaptive design of experiments strategy using Gaussian Process (GP) and Principal Component Analysis (PCA)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import uq_utilities
from scipy.stats import qmc


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

    def _scale(self, x):
        return self.gp_model.apply_input_scaling(x, fit=False)

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

    def select_training_points(
        self,
        x_train,
        n_points,
        mci_samples,
        *,
        use_mse_w=True,
        weights=None,
        n_samples=4000,
        seed=None,
    ):
        """
        Fully sequential DoE selection using MSEw or IMSEw acquisition.

        Parameters
        ----------
        x_train : array-like
            Current training data (original space).
        n_points : int
            Number of new training points to select.
        mci_samples : array-like
            Monte Carlo integration samples (original space).
        use_mse_w : bool
            Whether to use MSEw (True) or IMSEw (False).
        weights : array-like or None
            Optional weights applied to predictive variance.
        n_samples : int
            Number of LHS candidate points to generate (IMSEw only).
        seed : int or None
            Random seed for LHS sampling.

        Returns
        -------
        selected_points : np.ndarray of shape (n_points, d)
            Selected new training points.
        """
        selected_points = []

        # Determine candidate points
        if use_mse_w:
            candidate_pool = mci_samples.copy()
        else:
            bounds = compute_lhs_bounds(x_train, mci_samples, padding=0)
            candidate_pool = generate_lhs_candidates(n_samples, bounds, seed=seed)

        for _ in range(n_points):
            # Scale inputs
            scaled_x_train = self._scale(x_train)
            scaled_candidates = self._scale(candidate_pool)
            scaled_mci_samples = self._scale(mci_samples)

            # Update GP with current training set
            self.gp_model_for_doe.set_XY(
                scaled_x_train, np.zeros((scaled_x_train.shape[0], 1))
            )

            # Predict variance at integration points
            _, pred_var = self.gp_model_for_doe.predict(scaled_mci_samples)
            if weights is not None:
                pred_var *= weights.reshape(-1, 1)

            # Compute acquisition value
            if use_mse_w:
                acquisition_values = np.zeros((scaled_candidates.shape[0], 1))
                for i, cand in enumerate(scaled_candidates):
                    acquisition_values[i] = np.mean(pred_var)
            else:
                n_theta = scaled_x_train.shape[1]
                beta = 2.0 * n_theta
                acquisition_values = np.zeros((scaled_candidates.shape[0], 1))
                for i, cand in enumerate(scaled_candidates):
                    corr = self.gp_model_for_doe.kern.K(
                        scaled_mci_samples, np.atleast_2d(cand)
                    )
                    acquisition_values[i] = np.mean((corr**beta) * pred_var)

            # Select best candidate
            idx = np.argmax(acquisition_values)
            next_point = candidate_pool[idx]
            selected_points.append(next_point)

            # Update training data and candidate pool
            x_train = np.vstack([x_train, next_point])
            candidate_pool = np.delete(candidate_pool, idx, axis=0)

        return np.array(selected_points)

    def write_gp_for_doe_to_json(self, filepath: str | Path):
        """
        Write DoE GP kernel hyperparameters and contributing model param_arrays to JSON.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        """
        if not hasattr(self, 'gp_model_for_doe'):
            msg = 'gp_model_for_doe has not been initialized.'
            raise RuntimeError(msg)

        kernel = self.gp_model_for_doe.kern

        # Detailed hyperparameters for the DoE model
        doe_hyperparams = {
            param.name: {
                'value': param.values.tolist()  # noqa: PD011
                if param.size > 1
                else float(param.values),
                'shape': param.shape,
            }
            for param in kernel.parameters
        }

        # Aggregation weights
        if self.gp_model.use_pca:
            weights = np.asarray(self.gp_model.pca_info['explained_variance_ratio'])[
                : self.gp_model.pca_info['n_components']
            ]
            weights = (weights / np.sum(weights)).tolist()
        else:
            weights = (
                np.ones(len(self.gp_model.model)) / len(self.gp_model.model)
            ).tolist()

        # Contributing models' param_arrays only
        contributing_param_arrays = [
            gp.kern.param_array.tolist() for gp in self.gp_model.model
        ]

        # Output structure
        output = {
            'doe_kernel_type': kernel.name,
            'doe_ARD': kernel.ARD,
            'doe_hyperparameters': doe_hyperparams,
            'weighted_param_array': self.weighted_hyperparameters.tolist(),
            'aggregation_weights': weights,
            'contributing_param_arrays': contributing_param_arrays,
        }

        # Write JSON file
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open('w') as f:
            json.dump(uq_utilities.make_json_serializable(output), f, indent=4)


def generate_lhs_candidates(n_samples, input_bounds, seed=None):
    """
    Generate LHS candidate points using scipy's QMC module with an optional random seed.

    Parameters
    ----------
    n_samples : int
        Number of candidate points to generate.
    input_bounds : array-like of shape (d, 2)
        Lower and upper bounds for each input dimension.
    seed : int or None
        Random seed for reproducibility (default: None).

    Returns
    -------
    candidates : np.ndarray of shape (n_samples, d)
        Generated candidate points in the original input space.
    """
    input_bounds = np.asarray(input_bounds)
    d = input_bounds.shape[0]

    sampler = qmc.LatinHypercube(d, seed=seed)
    lhs_unit = sampler.random(n=n_samples)
    candidates = qmc.scale(lhs_unit, input_bounds[:, 0], input_bounds[:, 1])
    return candidates  # noqa: RET504


def compute_lhs_bounds(x_train, mci_samples, padding=0):
    """
    Compute input bounds for LHS based on x_train and mci_samples.

    Parameters
    ----------
    x_train : array-like, shape (n_train, d)
    mci_samples : array-like, shape (n_mci, d)
    padding : float
        Relative padding (e.g., 0.05 adds ±5% range to each side).

    Returns
    -------
    bounds : np.ndarray of shape (d, 2)
        Array of (min, max) bounds for each dimension.
    """
    x_all = np.vstack([x_train, mci_samples])
    min_vals = np.min(x_all, axis=0)
    max_vals = np.max(x_all, axis=0)

    ranges = max_vals - min_vals
    min_vals_padded = min_vals - padding * ranges
    max_vals_padded = max_vals + padding * ranges

    return np.vstack([min_vals_padded, max_vals_padded]).T
