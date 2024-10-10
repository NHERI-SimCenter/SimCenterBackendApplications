"""Provides a class for performing Principal Component Analysis (PCA) for dimensionality reduction."""

import numpy as np
from sklearn.decomposition import PCA


class PrincipalComponentAnalysis:
    """
    A class to perform Principal Component Analysis (PCA) for dimensionality reduction.

    Attributes
    ----------
        pca_threshold (float): The threshold for the cumulative explained variance ratio to determine the number of components.
        pca (PCA): The PCA model.
        mean_vec (np.ndarray): The mean vector of the original data.
        proj_matrix (np.ndarray): The projection matrix (PCA components).
        eigenvalues (np.ndarray): The eigenvalues of the PCA components.
        explained_variance_ratio (np.ndarray): The explained variance ratio of the PCA components.
        n_components (int): The number of components to retain.
    """

    def __init__(self, pca_threshold) -> None:
        """
        Initialize the PrincipalComponentAnalysis class.

        Args:
            pca_threshold (float): The threshold for the cumulative explained variance ratio to determine the number of components.
        """
        self.pca_threshold = pca_threshold

        self.pca = None
        self.mean_vec = None
        self.proj_matrix = None
        self.eigenvalues = None
        self.explained_variance_ratio = None
        self.n_components = None

    def project_to_latent_space(self, outputs):
        """
        Perform PCA on the output data to reduce the dimensionality.

        Args:
            outputs (np.ndarray): The output data to be projected to the latent space.

        Returns
        -------
            np.ndarray: The latent outputs after PCA transformation.
        """
        # Perform PCA on the output data to reduce the dimensionality
        pca = PCA()
        pca.fit(outputs)

        # Determine the number of components to retain based on the threshold value
        explained_variance_ratio = pca.explained_variance_ratio_
        self.n_components = (
            np.argmax(np.cumsum(explained_variance_ratio) >= self.pca_threshold) + 1
        )

        # Perform PCA with the specified number of components and transform the output data
        # TODO (ABS): Reimplement without the second PCA fit
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(outputs)
        latent_outputs = self.pca.transform(outputs)

        # Store the PCA parameters for later use in inverse transformation
        self.mean_vec = self.pca.mean_
        self.proj_matrix = self.pca.components_
        self.eigenvalues = self.pca.explained_variance_
        self.explained_variance_ratio = self.pca.explained_variance_ratio_

        return latent_outputs

    def project_back_to_original_space(self, latent_outputs):
        """
        Inverse transform the latent outputs back to the original space using the stored PCA parameters.

        Args:
            latent_outputs (np.ndarray): The latent outputs to be projected back to the original space.

        Returns
        -------
            np.ndarray: The outputs in the original space.

        Raises
        ------
            ValueError: If the PCA model has not been fitted yet.
        """
        # Check if the PCA model has been fitted
        if self.pca is None:
            error_message = 'No PCA model has been fitted yet.'
            raise ValueError(error_message)
        # Inverse transform the latent outputs using the stored PCA parameters
        outputs = self.pca.inverse_transform(latent_outputs)
        return outputs  # noqa: RET504
