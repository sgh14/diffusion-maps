import numpy as np
from numba import njit
from sklearn.base import BaseEstimator, TransformerMixin
from kernels import rbf_kernel, laplacian_kernel


class DiffusionMaps(TransformerMixin, BaseEstimator):
    """
    Diffusion Maps.
    """
    def __init__(self, sigma, n_components, step=1, alpha=1, kernel='rbf'):
        """
        Initialize the Diffusion Maps instance.
        
        Parameters
        ----------
        sigma : float
            Scale parameter for the kernel.
        n_components : int
            Number of diffusion map components to keep.
        step : int, optional (default=1)
            Power to which eigenvalues are raised in the diffusion map.
        alpha : float, optional (default=1)
            Normalization factor.
        kernel : str, optional (default='rbf')
            Type of kernel to use ('rbf' or 'laplacian').
        """
        self.sigma = sigma
        self.n_components = n_components
        self.step = step
        self.alpha = alpha
        self.kernel = kernel


    @staticmethod
    @njit
    def get_kernel(X, Y, sigma, kernel):
        """
        Compute the kernel matrix.

        Parameters
        ----------
        X : array-like, shape (n_samples_X, n_features)
            Input data.
        Y : array-like, shape (n_samples_Y, n_features)
            Input data. If None, compute the kernel with respect to X.
        sigma : float
            Scale parameter for the kernel.
        kernel : str
            Type of kernel to use ('rbf' or 'laplacian').

        Returns
        -------
        K : array-like, shape (n_samples_X, n_samples_Y)
            Kernel matrix.
        """
        gamma = 1 / (2 * sigma ** 2)
        if kernel == 'laplacian':
            K = laplacian_kernel(X, Y, gamma=gamma)
        elif kernel == 'rbf':
            K = rbf_kernel(X, Y, gamma=gamma)
        else:
            raise ValueError("Unsupported kernel")

        return K


    @staticmethod
    @njit
    def _get_P(K, alpha):
        """
        Compute the diffusion matrix P.

        Parameters
        ----------
        K : array-like, shape (n_samples_X, n_samples_Y)
            Kernel matrix.
        alpha : float
            Normalization factor.

        Returns
        -------
        P : array-like, shape (n_samples_X, n_samples_Y)
            Diffusion matrix.
        """
        # Compute 1/d_i^alpha as a diagonal matrix
        D_i_inv = np.diag(np.sum(K, axis=1) ** (-alpha))
        # Compute 1/d_i^alpha as a diagonal matrix
        D_j_inv = np.diag(np.sum(K, axis=0) ** (-alpha))
        # Compute k_ij/(d_i^alpha * d_j^alpha)
        K_alpha = D_i_inv @ K @ D_j_inv
        # Compute 1/d_i^{(alpha)} as a diagonal matrix
        D_i_inv_alpha = np.diag(np.sum(K_alpha, axis=1) ** (-1))
        # Compute k_ij^{(alpha)}/d_i^{(alpha)}
        P = D_i_inv_alpha @ K_alpha

        return P


    @staticmethod
    @njit
    def _fix_vector_orientation(vectors):
        """
        Fix the orientation of eigenvectors.

        Parameters
        ----------
        vectors : array-like, shape (n_samples, n_components)
            Eigenvectors to fix.

        Returns
        -------
        vectors : array-like, shape (n_samples, n_components)
            Eigenvectors with fixed orientation.
        """
        # Fix the first non-zero component of every vector to be positive
        for i in range(vectors.shape[1]):
            first_nonzero = np.nonzero(vectors[:, i])[0][0]
            if vectors[first_nonzero, i] < 0:
                vectors[:, i] *= -1

        return vectors


    @staticmethod
    @njit
    def _spectral_decomposition(A):
        """
        Perform spectral decomposition on matrix A.

        Parameters
        ----------
        A : array-like, shape (n_samples, n_samples)
            Matrix to decompose.

        Returns
        -------
        eigenvalues : array-like, shape (n_samples,)
            Eigenvalues in decreasing order.
        eigenvectors : array-like, shape (n_samples, n_samples)
            Corresponding eigenvectors.
        """
        # Compute the eigenvalues and right eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # Find the order of the eigenvalues (decreasing order)
        order = np.argsort(np.real(eigenvalues))[::-1]
        # Sort eigenvalues and eigenvectors
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        return eigenvalues, eigenvectors


    def fit(self, X, y=None):
        """
        Fit the model with X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X = X
        # Compute the kernel
        K = self.get_kernel(self.X, self.X, self.sigma, self.kernel)
        # Compute the matrix P
        P = self._get_P(K, self.alpha)
        # Get the eigenvalues and eigenvectors of P
        self.lambdas, self.psis = self._spectral_decomposition(P)
        # Fix eigenvectors orientation
        self.psis = DiffusionMaps._fix_vector_orientation(self.psis)
        # Reduce dimension
        lambdas_red = self.lambdas[1:self.n_components + 1]
        psis_red = self.psis[:, 1:self.n_components + 1]
        # Compute the new coordinates
        self.Psi_step = psis_red * (lambdas_red ** self.step)

        return self


    def fit_transform(self, X, y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        X_red : array-like, shape (n_samples, n_components)
            Reduced data.
        """
        self.fit(X)
        X_red = self.Psi_step

        return X_red


    def transform(self, Y):
        """
        Transform Y using the fitted model.
        This function is implemented using the Nyström formula.

        Parameters
        ----------
        Y : array-like, shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        Y_red : array-like, shape (n_samples, n_components)
            Transformed data.
        """
        # Compute the kernel
        K = self.get_kernel(Y, self.X, self.sigma, self.kernel)
        # Compute the matrix P(Y, X)
        P = self._get_P(K, self.alpha)
        # Get the n_components biggest eigenvalues of P(X, X)
        lambdas_red = self.lambdas[1:self.n_components + 1]
        # Apply Nyström formula
        Y_red = (P @ self.Psi_step) / lambdas_red

        return Y_red
