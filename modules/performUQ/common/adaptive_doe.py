"""Implements an adaptive design of experiments strategy using Gaussian Process (GP) and Principal Component Analysis (PCA)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import uq_utilities
from gp_model import remove_duplicate_inputs
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

    def _safe_normalize(self, v: np.ndarray) -> np.ndarray:
        s = float(np.nansum(v))
        if not np.isfinite(s) or s <= 0:
            return np.full_like(v, 1.0 / max(len(v), 1))
        return v / s

    def _compute_doe_weights(self) -> tuple[np.ndarray, dict]:
        """
        Compute weights for aggregating across GP output components.

        Return (weights, meta) for aggregating across GP output components:
        - PCA on  -> eigenvalue weights (explained_variance_ratio)
        - PCA off -> per-output variance in processed training space.
        """
        n_models = len(self.gp_model.model)

        if self.gp_model.use_pca:
            pinfo = self.gp_model.pca_info
            ncomp = int(pinfo['n_components'])
            evr = np.asarray(pinfo['explained_variance_ratio'])[:ncomp].astype(float)
            w = self._safe_normalize(evr)
            if len(w) != n_models:  # defensive
                w = np.full(n_models, 1.0 / max(n_models, 1))
                basis = 'uniform_fallback'
                basis_vals = np.ones(n_models, dtype=float)
            else:
                basis = 'pca_explained_variance_ratio'
                basis_vals = evr
            return w, {
                'mode': 'pca',
                'basis': basis,
                'basis_values': basis_vals.tolist(),
                'sum': float(np.sum(w)),
                'count': len(w),
            }

        # no PCA: weight by per-output variance in the space the GP trained on
        Y = getattr(self.gp_model, 'y_train', None)  # noqa: N806
        if Y is None or Y.size == 0:
            w = np.full(n_models, 1.0 / max(n_models, 1))
            return w, {
                'mode': 'no_pca',
                'basis': 'uniform_no_training_data',
                'basis_values': [1.0] * n_models,
                'sum': float(np.sum(w)),
                'count': len(w),
            }

        if self.gp_model.output_scaler is not None:
            Y_proc = self.gp_model.output_scaler.transform(Y)  # noqa: N806
        else:
            Y_proc = Y  # noqa: N806

        raw = np.nanvar(Y_proc, axis=0, ddof=1).astype(float)
        raw = np.where(np.isfinite(raw) & (raw > 0), raw, 1.0)
        w = self._safe_normalize(raw)

        if len(w) != n_models:  # defensive
            if len(w) > n_models:
                basis_vals = raw[:n_models]
                w = self._safe_normalize(basis_vals)
                basis = 'output_variance_truncated'
            else:
                pad = np.full(
                    n_models - len(w), float(np.mean(raw)) if raw.size else 1.0
                )
                basis_vals = np.concatenate([raw, pad])
                w = self._safe_normalize(basis_vals)
                basis = 'output_variance_padded'
        else:
            basis = 'output_variance'
            basis_vals = raw

        return w, {
            'mode': 'no_pca',
            'basis': basis,
            'basis_values': basis_vals.tolist(),
            'sum': float(np.sum(w)),
            'count': len(w),
        }

    def _hyperparameters_for_doe(self):
        """
        Compute the weighted average of kernel hyperparameters for DoE.

        If PCA is used, weights are based on explained variance.
        Otherwise, uniform averaging is used.
        """
        w, _ = self._compute_doe_weights()

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
        Efficient sequential DoE using MSEw (equation 10) or IMSEw (equation 9) from Taflanidis et al. (https://doi.org/10.1016/j.ymssp.2025.113014).

        Parameters
        ----------
        x_train : array-like
            Current training data (original space).
        n_points : int
            Number of new training points to select.
        mci_samples : array-like
            Monte Carlo integration samples (original space).
        use_mse_w : bool
            Whether to use MSE equation (10) (True) or IMSEw equation (9) (False).
        weights : array-like or None
            Optional importance weights for integration points.
        n_samples : int
            Number of candidate points to generate (IMSEw only).
        seed : int or None
            Random seed for LHS sampling.

        Returns
        -------
        selected_points : np.ndarray of shape (n_points, d)
            Selected new training points.
        """
        x_train = np.atleast_2d(x_train)
        x_train, _ = remove_duplicate_inputs(
            x_train, np.zeros((x_train.shape[0], 1))
        )

        selected_points = []

        # 1. Setup candidate pool based on acquisition function
        if use_mse_w:
            # MSE (equation 10): candidates are the integration points themselves
            mci_samples, weights = remove_duplicate_inputs(
                mci_samples,
                weights
                if weights is not None
                else np.ones((mci_samples.shape[0], 1)),
            )
            candidate_pool = mci_samples.copy()
        else:
            # IMSEw (equation 9): generate separate candidate pool via LHS
            bounds = compute_lhs_bounds(x_train, mci_samples, padding=0)
            candidate_pool = generate_lhs_candidates(n_samples, bounds, seed=seed)

        # 2. Sequential selection
        for _ in range(n_points):
            # Scale all arrays
            scaled_x_train = self._scale(x_train)
            scaled_candidates = self._scale(candidate_pool)
            scaled_mci_samples = self._scale(mci_samples)

            # Set GP with current training set
            self.gp_model_for_doe.set_XY(
                scaled_x_train, np.zeros((scaled_x_train.shape[0], 1))
            )

            # Compute acquisition values based on selected method
            if use_mse_w:
                # MSE (equation 10): σ²(θ_new|Θ^(k))
                _, pred_var = self.gp_model_for_doe.predict(scaled_candidates)

                # Apply importance weights if provided
                if weights is not None:
                    pred_var *= weights.reshape(-1, 1)

                acquisition_values = pred_var

            else:
                # IMSEw (equation 9): (1/N_s) * Σ[R(θ^(l), θ_new)^β * σ²(θ^(l)|Θ^(k))]

                # Predict variance at integration points
                _, pred_var = self.gp_model_for_doe.predict(scaled_mci_samples)

                # Apply importance weights to integration points
                if weights is not None:
                    pred_var *= weights.reshape(-1, 1)

                # Compute covariance K(θ^(l), θ_new)
                k_matrix = self.gp_model_for_doe.kern.K(
                    scaled_mci_samples, scaled_candidates
                )

                # Get diagonal kernel values for covariance normalization
                k_diag_mci = np.diag(
                    self.gp_model_for_doe.kern.K(scaled_mci_samples)
                )
                k_diag_candidates = np.diag(
                    self.gp_model_for_doe.kern.K(scaled_candidates)
                )

                # Correlations : R(θ^(l), θ_new)
                corr_matrix = k_matrix / np.sqrt(
                    np.outer(k_diag_mci, k_diag_candidates)
                )

                # Apply beta exponent
                beta = 2.0 * scaled_x_train.shape[1]

                # IMSEw computation: (corr^beta).T @ pred_var / N_s
                weighted_var = (corr_matrix**beta).T @ pred_var
                acquisition_values = weighted_var / scaled_mci_samples.shape[0]

            # Select candidate with maximum acquisition value
            idx = np.argmax(acquisition_values)
            next_point = candidate_pool[idx]
            selected_points.append(next_point)

            # Update training set and candidate pool
            x_train = np.vstack([x_train, next_point])
            candidate_pool = np.delete(candidate_pool, idx, axis=0)

            if use_mse_w:
                # For MSE: maintain alignment by deleting from mci_samples too
                mci_samples = np.delete(mci_samples, idx, axis=0)
                if weights is not None:
                    weights = np.delete(weights, idx, axis=0)
            # For IMSEw: keep mci_samples constant (integration domain unchanged)

        return np.array(selected_points)

    def write_gp_for_doe_to_json(self, filepath: str | Path):
        """Write DoE GP kernel hyperparameters and contributing model param_arrays to JSON."""
        if not hasattr(self, 'gp_model_for_doe'):
            msg = 'gp_model_for_doe has not been initialized.'
            raise RuntimeError(msg)

        # Ensure we have the current weighted vector available
        if not hasattr(self, 'weighted_hyperparameters'):
            _ = self._hyperparameters_for_doe()

        # Reuse the exact same weights/provenance used for aggregation
        weights, weights_meta = self._compute_doe_weights()

        kernel = self.gp_model_for_doe.kern
        doe_hyperparams = {
            p.name: {
                'value': p.values.tolist() if p.size > 1 else float(p.values),
                'shape': p.shape,
            }
            for p in kernel.parameters
        }

        contributing_param_arrays = [
            gp.kern.param_array.tolist() for gp in self.gp_model.model
        ]

        output = {
            'doe_kernel_type': kernel.name,
            'doe_ARD': getattr(kernel, 'ARD', None),
            'doe_hyperparameters': doe_hyperparams,
            'weighted_param_array': self.weighted_hyperparameters.tolist(),
            'aggregation_weights': weights.tolist(),
            'aggregation_weights_meta': weights_meta,  # provenance for reproducibility
            'contributing_param_arrays': contributing_param_arrays,
        }

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as f:
            json.dump(uq_utilities.make_json_serializable(output), f, indent=4)

    def visualize_acquisition_analysis(  # noqa: C901
        self,
        x_train,
        candidate_pool,
        mci_samples,
        *,
        use_mse_w=True,
        weights=None,
        plot_dims=(0, 1),
        n_grid=100,
        show_before_after=True,
        save_path=None,
    ):
        """
        Visualize acquisition function and other relevant fields.

        Visualize GP variance field, weight function, and acquisition function
        for 2D projections of the design space using Plotly.

        Parameters
        ----------
        x_train : array-like
            Current training data (original space)
        candidate_pool : array-like
            Candidate points for selection
        mci_samples : array-like
            Monte Carlo integration samples
        use_mse_w : bool
            Whether to use MSEw (True) or IMSEw (False)
        weights : array-like or None
            Optional importance weights
        plot_dims : tuple
            Dimensions to plot (default: (0,1))
        n_grid : int
            Grid resolution for plotting
        show_before_after : bool
            Whether to show before/after adding one point
        save_path : str or None
            Path to save figure (HTML format)
        """
        # Ensure we have the DoE GP model
        if not hasattr(self, 'gp_model_for_doe'):
            msg = 'DoE GP model not initialized. Call select_training_points first.'
            raise RuntimeError(msg)

        x_train = np.atleast_2d(x_train)
        x_train, _ = remove_duplicate_inputs(
            x_train, np.zeros((x_train.shape[0], 1))
        )

        dim1, dim2 = plot_dims

        # Create plotting grid
        x1_min = min(np.min(x_train[:, dim1]), np.min(candidate_pool[:, dim1]))
        x1_max = max(np.max(x_train[:, dim1]), np.max(candidate_pool[:, dim1]))
        x2_min = min(np.min(x_train[:, dim2]), np.min(candidate_pool[:, dim2]))
        x2_max = max(np.max(x_train[:, dim2]), np.max(candidate_pool[:, dim2]))

        # Add padding
        x1_range = x1_max - x1_min
        x2_range = x2_max - x2_min
        padding = 0.1
        x1_min -= padding * x1_range
        x1_max += padding * x1_range
        x2_min -= padding * x2_range
        x2_max += padding * x2_range

        x1_grid = np.linspace(x1_min, x1_max, n_grid)
        x2_grid = np.linspace(x2_min, x2_max, n_grid)
        X1, X2 = np.meshgrid(x1_grid, x2_grid)

        # Create grid points (assuming other dimensions at mean of training data)
        n_dims = x_train.shape[1]
        grid_points = np.zeros((n_grid * n_grid, n_dims))
        grid_points[:, dim1] = X1.flatten()
        grid_points[:, dim2] = X2.flatten()

        # Set other dimensions to mean of training data
        for i in range(n_dims):
            if i not in [dim1, dim2]:
                grid_points[:, i] = np.mean(x_train[:, i])

        def compute_fields(current_x_train):
            """Compute variance and acquisition fields for given training set."""
            # Scale training data and grid
            scaled_x_train = self._scale(current_x_train)
            scaled_grid = self._scale(grid_points)
            scaled_candidates = self._scale(candidate_pool)
            scaled_mci = self._scale(mci_samples)

            # Set GP with current training set
            self.gp_model_for_doe.set_XY(
                scaled_x_train, np.zeros((scaled_x_train.shape[0], 1))
            )

            # Compute variance field over grid
            _, var_grid = self.gp_model_for_doe.predict(scaled_grid)
            variance_field = var_grid.reshape(n_grid, n_grid)

            # Compute acquisition values for candidates
            if use_mse_w:
                # MSEw: direct variance at candidates
                _, pred_var = self.gp_model_for_doe.predict(scaled_candidates)
                if weights is not None:
                    pred_var *= weights.reshape(-1, 1)
                acquisition_values = pred_var.flatten()

                # For plotting: acquisition field is just variance field
                acquisition_field = variance_field.copy()

            else:
                # IMSEw: integrated variance weighted by correlations
                _, pred_var_mci = self.gp_model_for_doe.predict(scaled_mci)
                if weights is not None:
                    pred_var_mci *= weights.reshape(-1, 1)

                # Compute acquisition for candidates
                k_matrix = self.gp_model_for_doe.kern.K(
                    scaled_mci, scaled_candidates
                )
                k_diag_mci = np.diag(self.gp_model_for_doe.kern.K(scaled_mci))
                k_diag_candidates = np.diag(
                    self.gp_model_for_doe.kern.K(scaled_candidates)
                )

                corr_matrix = k_matrix / np.sqrt(
                    np.outer(k_diag_mci, k_diag_candidates)
                )
                beta = 2.0 * scaled_x_train.shape[1]
                weighted_var = (corr_matrix**beta).T @ pred_var_mci
                acquisition_values = (weighted_var / scaled_mci.shape[0]).flatten()

                # Compute acquisition field over grid
                k_matrix_grid = self.gp_model_for_doe.kern.K(scaled_mci, scaled_grid)
                k_diag_grid = np.diag(self.gp_model_for_doe.kern.K(scaled_grid))
                corr_matrix_grid = k_matrix_grid / np.sqrt(
                    np.outer(k_diag_mci, k_diag_grid)
                )

                weighted_var_grid = (corr_matrix_grid**beta).T @ pred_var_mci
                acquisition_field = (
                    weighted_var_grid / scaled_mci.shape[0]
                ).reshape(n_grid, n_grid)

            return variance_field, acquisition_field, acquisition_values

        # Compute initial fields
        var_field_before, acq_field_before, acq_values = compute_fields(x_train)

        # Select next point
        best_idx = np.argmax(acq_values)
        next_point = candidate_pool[best_idx]

        # Setup figure
        if show_before_after:
            fig = sp.make_subplots(
                rows=2,
                cols=3,
                subplot_titles=[
                    'Variance Field (Before)',
                    'Acquisition Field (Before)',
                    'Combined (Before)',
                    'Variance Field (After)',
                    'Acquisition Field (After)',
                    'Combined (After)',
                ],
                specs=[
                    [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
                    [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
                ],
            )
        else:
            fig = sp.make_subplots(
                rows=1,
                cols=3,
                subplot_titles=['Variance Field', 'Acquisition Field', 'Combined'],
                specs=[[{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}]],
            )

        def add_field_plots(row, var_field, acq_field, title_suffix=''):
            # Variance field
            fig.add_trace(
                go.Contour(
                    x=x1_grid,
                    y=x2_grid,
                    z=var_field,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Variance', x=0.32, len=0.4),  # noqa: C408
                ),
                row=row,
                col=1,
            )

            # Training points
            fig.add_trace(
                go.Scatter(
                    x=x_train[:, dim1],
                    y=x_train[:, dim2],
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='square'),  # noqa: C408
                    name='Training Points',
                    showlegend=(row == 1),
                ),
                row=row,
                col=1,
            )

            # Candidates
            fig.add_trace(
                go.Scatter(
                    x=candidate_pool[:, dim1],
                    y=candidate_pool[:, dim2],
                    mode='markers',
                    marker=dict(size=4, color='orange', opacity=0.3),  # noqa: C408
                    name='Candidates',
                    showlegend=(row == 1),
                ),
                row=row,
                col=1,
            )

            # Selected point (only for "before" plots)
            if title_suffix == '':
                fig.add_trace(
                    go.Scatter(
                        x=[next_point[dim1]],
                        y=[next_point[dim2]],
                        mode='markers',
                        marker=dict(  # noqa: C408
                            size=15,
                            color='white',
                            symbol='star',
                            line=dict(color='black', width=2),  # noqa: C408
                        ),
                        name='Selected Point',
                        showlegend=(row == 1),
                    ),
                    row=row,
                    col=1,
                )

            # Acquisition field
            acq_name = 'MSEw' if use_mse_w else 'IMSEw'
            fig.add_trace(
                go.Contour(
                    x=x1_grid,
                    y=x2_grid,
                    z=acq_field,
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title=f'{acq_name} Value', x=0.66, len=0.4),  # noqa: C408
                ),
                row=row,
                col=2,
            )

            # Training points
            fig.add_trace(
                go.Scatter(
                    x=x_train[:, dim1],
                    y=x_train[:, dim2],
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='square'),  # noqa: C408
                    showlegend=False,
                ),
                row=row,
                col=2,
            )

            # Candidates
            fig.add_trace(
                go.Scatter(
                    x=candidate_pool[:, dim1],
                    y=candidate_pool[:, dim2],
                    mode='markers',
                    marker=dict(size=4, color='orange', opacity=0.3),  # noqa: C408
                    showlegend=False,
                ),
                row=row,
                col=2,
            )

            # Selected point
            if title_suffix == '':
                fig.add_trace(
                    go.Scatter(
                        x=[next_point[dim1]],
                        y=[next_point[dim2]],
                        mode='markers',
                        marker=dict(  # noqa: C408
                            size=15,
                            color='white',
                            symbol='star',
                            line=dict(color='black', width=2),  # noqa: C408
                        ),
                        showlegend=False,
                    ),
                    row=row,
                    col=2,
                )

            # Combined view - acquisition field with variance contours
            fig.add_trace(
                go.Contour(
                    x=x1_grid,
                    y=x2_grid,
                    z=acq_field,
                    colorscale='Plasma',
                    opacity=0.8,
                    showscale=True,
                    colorbar=dict(title=f'{acq_name} Value', x=1.0, len=0.4),  # noqa: C408
                ),
                row=row,
                col=3,
            )

            # Variance contours
            fig.add_trace(
                go.Contour(
                    x=x1_grid,
                    y=x2_grid,
                    z=var_field,
                    showscale=False,
                    line=dict(color='white', width=1),  # noqa: C408
                    opacity=0.5,
                    contours=dict(  # noqa: C408
                        showlabels=True,
                        labelfont=dict(size=8, color='white'),  # noqa: C408
                    ),
                ),
                row=row,
                col=3,
            )

            # Training points
            fig.add_trace(
                go.Scatter(
                    x=x_train[:, dim1],
                    y=x_train[:, dim2],
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='square'),  # noqa: C408
                    showlegend=False,
                ),
                row=row,
                col=3,
            )

            # Selected point
            if title_suffix == '':
                fig.add_trace(
                    go.Scatter(
                        x=[next_point[dim1]],
                        y=[next_point[dim2]],
                        mode='markers',
                        marker=dict(  # noqa: C408
                            size=15,
                            color='white',
                            symbol='star',
                            line=dict(color='black', width=2),  # noqa: C408
                        ),
                        showlegend=False,
                    ),
                    row=row,
                    col=3,
                )

        # Plot before selection
        add_field_plots(1, var_field_before, acq_field_before, '')

        if show_before_after:
            # Compute fields after adding the selected point
            x_train_after = np.vstack([x_train, next_point])
            var_field_after, acq_field_after, _ = compute_fields(x_train_after)

            # Plot after selection
            add_field_plots(2, var_field_after, acq_field_after, ' (After)')

            # Update training points for after plots
            for col in range(1, 4):
                fig.add_trace(
                    go.Scatter(
                        x=x_train_after[:, dim1],
                        y=x_train_after[:, dim2],
                        mode='markers',
                        marker=dict(size=10, color='red', symbol='square'),  # noqa: C408
                        showlegend=False,
                    ),
                    row=2,
                    col=col,
                )

        # Update layout
        fig.update_layout(
            height=800 if show_before_after else 400,
            width=1200,
            title_text=f'DoE Acquisition Analysis - {("MSEw" if use_mse_w else "IMSEw")}',
            showlegend=True,
        )

        # Update axes labels
        for row in range(1, (3 if show_before_after else 2)):
            for col in range(1, 4):
                fig.update_xaxes(
                    title_text=f'Dimension {dim1 + 1}', row=row, col=col
                )
                fig.update_yaxes(
                    title_text=f'Dimension {dim2 + 1}', row=row, col=col
                )

        # Add kernel information as annotation
        kernel_info = f'Kernel: {self.gp_model_for_doe.kern.name}<br>'
        if hasattr(self.gp_model_for_doe.kern, 'lengthscale'):
            ls = self.gp_model_for_doe.kern.lengthscale.values
            if self.ARD and len(ls) > 1:
                kernel_info += f'Lengthscales: {[f"{l:.3f}" for l in ls]}<br>'
            else:
                kernel_info += f'Lengthscale: {ls[0]:.3f}<br>'

        if hasattr(self.gp_model_for_doe.kern, 'variance'):
            kernel_info += (
                f'Variance: {self.gp_model_for_doe.kern.variance.values[0]:.3f}<br>'
            )

        kernel_info += (
            f'Nugget: {self.gp_model_for_doe.likelihood.variance.values[0]:.3e}<br>'
        )
        kernel_info += f'Training points: {len(x_train)}<br>'
        kernel_info += (
            f'Next point: [{next_point[dim1]:.3f}, {next_point[dim2]:.3f}]'
        )

        fig.add_annotation(
            text=kernel_info,
            xref='paper',
            yref='paper',
            x=0.02,
            y=0.98,
            showarrow=False,
            align='left',
            bgcolor='wheat',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=10),  # noqa: C408
        )

        if save_path:
            fig.write_html(save_path)
            print(f'Figure saved to {save_path}')

        fig.show()

        # Print diagnostic information
        print('\nDIAGNOSTIC INFORMATION:')  # noqa: T201
        print(f'Acquisition method: {"MSEw" if use_mse_w else "IMSEw"}')  # noqa: T201
        print(f'GP Kernel: {self.gp_model_for_doe.kern.name}')  # noqa: T201

        if hasattr(self.gp_model_for_doe.kern, 'lengthscale'):
            ls = self.gp_model_for_doe.kern.lengthscale.values
            print(f'Lengthscales: {ls}')  # noqa: T201
            print(f'Min lengthscale: {np.min(ls):.4f}')  # noqa: T201
            print(f'Max lengthscale: {np.max(ls):.4f}')  # noqa: T201

            # Check if lengthscales are very small (potential clustering cause)
            ls_thresh = 0.1
            if np.any(ls < ls_thresh):
                print(  # noqa: T201
                    'WARNING: Very small lengthscales detected - may cause clustering!'
                )

        print(  # noqa: T201
            f'Variance range in field: [{np.min(var_field_before):.4f}, {np.max(var_field_before):.4f}]'
        )
        print(  # noqa: T201
            f'Acquisition range: [{np.min(acq_values):.4f}, {np.max(acq_values):.4f}]'
        )
        print(f'Selected point acquisition value: {acq_values[best_idx]:.4f}')  # noqa: T201
        print(f'Number of candidates: {len(candidate_pool)}')  # noqa: T201

        return {
            'variance_field_before': var_field_before,
            'acquisition_field_before': acq_field_before,
            'selected_point': next_point,
            'selected_point_value': acq_values[best_idx],
            'kernel_lengthscales': self.gp_model_for_doe.kern.lengthscale.values
            if hasattr(self.gp_model_for_doe.kern, 'lengthscale')
            else None,
        }


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
