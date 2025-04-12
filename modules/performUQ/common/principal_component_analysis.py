"""Provides a class for performing Principal Component Analysis (PCA) for dimensionality reduction."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class SafeStandardScaler:
    """
    A wrapper around sklearn's StandardScaler that safely handles cases
    where the input data has only one sample.

    For a single sample:
        - Centers the data by subtracting the mean.
        - Skips variance scaling to avoid division by zero.
        - Allows inverse transformation.

    For multiple samples:
        - Behaves exactly like sklearn's StandardScaler.
    """  # noqa: D205

    def __init__(self):
        """Initialize the SafeStandardScaler."""
        self.scaler = StandardScaler()
        self.is_trivial_case = False
        self.mean_ = None
        self.scale_ = None

    def fit(self, data):
        """
        Fit the scaler to the input data.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.

        Returns
        -------
        self : object
            Fitted scaler instance.
        """
        data = np.asarray(data)
        if data.shape[0] == 1:
            self.is_trivial_case = True
            self.mean_ = data[0]
            self.scale_ = np.ones_like(data[0])  # Avoid divide-by-zero
        else:
            self.scaler.fit(data)
            self.mean_ = self.scaler.mean_
            self.scale_ = self.scaler.scale_
        return self

    def transform(self, data):
        """
        Standardize the input data.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        data_scaled : ndarray
            The standardized version of the input data.
        """
        data = np.asarray(data)
        if self.is_trivial_case:
            return data - self.mean_
        return self.scaler.transform(data)

    def inverse_transform(self, scaled_data):
        """
        Undo the standardization of previously transformed data.

        Parameters
        ----------
        scaled_data : array-like of shape (n_samples, n_features)
            The standardized data to be inverse transformed.

        Returns
        -------
        data_original : ndarray
            The data in the original feature space.
        """
        scaled_data = np.asarray(scaled_data)
        if self.is_trivial_case:
            return scaled_data + self.mean_
        return self.scaler.inverse_transform(scaled_data)

    def fit_transform(self, data):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data to fit and transform.

        Returns
        -------
        data_scaled : ndarray
            The standardized version of the input data.
        """
        return self.fit(data).transform(data)


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

        self.scaler = SafeStandardScaler() if self.perform_scaling else None
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
