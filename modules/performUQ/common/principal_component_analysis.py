"""Provides a class for performing Principal Component Analysis (PCA) for dimensionality reduction."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PrincipalComponentAnalysis:
    """
    A class to perform Principal Component Analysis (PCA) for dimensionality reduction.

    Attributes
    ----------
        pca_threshold (float): The threshold for the cumulative explained variance ratio to determine the number of components.
        scale (bool): Whether to standardize data before PCA.
        pca (PCA): The PCA model.
        scaler (StandardScaler | None): The scaler used to standardize the data, if enabled.
        n_components (int): The number of components to retain.
    """

    def __init__(self, pca_threshold, perform_scaling=True) -> None:  # noqa: FBT002
        """
        Initialize the PrincipalComponentAnalysis class.

        Args:
            pca_threshold (float): The threshold for the cumulative explained variance ratio to determine the number of components.
            scale (bool): Whether to standardize data before PCA.
        """
        self.pca_threshold = pca_threshold
        self.perform_scaling = perform_scaling

        self.scaler = StandardScaler() if self.perform_scaling else None
        self.pca = None
        self.n_components = None

    def fit(self, outputs):
        """
        Fit the PCA model to the output data.

        Args:
            outputs (np.ndarray): The output data to be fitted.

        Returns
        -------
            None
        """
        self.pca = PCA()
        if self.perform_scaling:
            scaled_outputs = self.scaler.fit_transform(outputs)
        else:
            scaled_outputs = outputs

        self.pca.fit(scaled_outputs)

        explained_variance_ratio = self.pca.explained_variance_ratio_
        self.n_components = (
            np.argmax(np.cumsum(explained_variance_ratio) >= self.pca_threshold) + 1
        )

    def project_to_latent_space(self, outputs):
        """
        Project the output data to the latent (reduced) space.

        Args:
            outputs (np.ndarray): The output data to be projected.

        Returns
        -------
            np.ndarray: The latent outputs after PCA transformation.
        """
        if self.pca is None:
            msg = 'No PCA model has been fitted yet.'
            raise ValueError(msg)
        if self.perform_scaling and self.scaler is None:
            msg = 'No scaler model has been fitted yet.'
            raise ValueError(msg)

        outputs_scaled = (
            self.scaler.transform(outputs) if self.perform_scaling else outputs
        )
        outputs_centered = outputs_scaled - self.pca.mean_
        components_selected = self.pca.components_[: self.n_components]
        latent_outputs = outputs_centered @ components_selected.T
        return latent_outputs  # noqa: RET504

    def project_back_to_original_space(self, latent_outputs):
        """
        Inverse transform the latent outputs back to the original space.

        Args:
            latent_outputs (np.ndarray): The latent outputs to be projected back.

        Returns
        -------
            np.ndarray: The outputs in the original space.
        """
        if self.pca is None:
            msg = 'No PCA model has been fitted yet.'
            raise ValueError(msg)
        if self.perform_scaling and self.scaler is None:
            msg = 'No scaler model has been fitted yet.'
            raise ValueError(msg)

        components_selected = self.pca.components_[: self.n_components]
        outputs_scaled = latent_outputs @ components_selected + self.pca.mean_
        outputs = (
            self.scaler.inverse_transform(outputs_scaled)
            if self.perform_scaling
            else outputs_scaled
        )
        return outputs  # noqa: RET504
