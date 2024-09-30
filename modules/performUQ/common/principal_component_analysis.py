import numpy as np
from sklearn.decomposition import PCA


class PrincipalComponentAnalysis:
    def __init__(self, pca_threshold) -> None:
        self.pca_threshold = pca_threshold

        self.pca = None
        self.mean_vec = None
        self.proj_matrix = None
        self.eigenvalues = None
        self.explained_variance_ratio = None
        self.n_components = None

    def project_to_latent_space(self, outputs):
        # Perform PCA on the output data to reduce the dimensionality
        pca = PCA()
        pca.fit(outputs)

        # Determine the number of components to retain based on the threshold value
        explained_variance_ratio = pca.explained_variance_ratio_
        self.n_components = (
            np.argmax(np.cumsum(explained_variance_ratio) >= self.pca_threshold) + 1
        )

        # Perform PCA with the specified number of components and transform the output data
        # TODO(ABS): Reimplement without the second PCA fit
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
        # Check if the PCA model has been fitted
        if self.pca is None:
            raise ValueError("No PCA model has been fitted yet.")
        else:
            # Inverse transform the latent outputs using the stored PCA parameters
            outputs = self.pca.inverse_transform(latent_outputs)
            return outputs
