import numpy as np
from numba import njit
from sklearn.base import BaseEstimator, TransformerMixin
from kernels import rbf_kernel, laplacian_kernel


class DiffusionMaps(TransformerMixin, BaseEstimator):
    """
    Diffusion Maps.
    """
    def __init__(self, sigma, n_components, steps=1, alpha=1, kernel='rbf'):
        """
        Initialize the Diffusion Maps instance.
        
        Parameters
        ----------
        sigma : float
            Scale parameter for the kernel.
        n_components : int
            Number of diffusion map components to keep.
        steps : int, optional (default=1)
            Power to which eigenvalues are raised in the diffusion map.
        alpha : float, optional (default=1)
            Normalization factor.
        kernel : str, optional (default='rbf')
            Type of kernel to use ('rbf' or 'laplacian').
        """
        self.sigma = sigma
        self.n_components = n_components
        self.steps = steps
        self.alpha = alpha
        self.kernel = kernel


    @staticmethod
    def get_kernel(X, Y, sigma, kernel, alpha):
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
        alpha : float
            Normalization factor.

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

        
        d_i_alpha = np.sum(K, axis=1)**alpha
        # D_i_alpha_inv = np.diag(d_i ** (-1))
        d_j_alpha = np.sum(K, axis=0)**alpha
        # D_j_alpha_inv = np.diag(d_j ** (-1))
        # Compute k_ij/(d_i^alpha * d_j^alpha)
        K_alpha = K/np.outer(d_i_alpha, d_j_alpha) # D_i_alpha_inv @ K @ D_j_alpha_inv

        return K_alpha


    @staticmethod
    @njit
    def _get_P(K):
        """
        Compute the diffusion matrix P.

        Parameters
        ----------
        K : array-like, shape (n_samples_X, n_samples_Y)
            Kernel matrix.

        Returns
        -------
        P : array-like, shape (n_samples_X, n_samples_Y)
            Diffusion matrix.
        """
        
        d_i = np.sum(K, axis=1)
        # D_i_inv = np.diag(d_i ** (-1))
        # Compute k_ij^{(alpha)}/d_i^{(alpha)}
        P = K / d_i[:, np.newaxis] # D_i_inv @ K 

        return P
    

    @staticmethod
    @njit
    def diffusion_distances(P, d):
        """
        Compute diffusion distances.

        Args:
            P (np.array): Diffusion probability matrix.
            d (np.array): Degree vector.

        Returns:
            np.array: Matrix of diffusion distances.
        """
        D = np.zeros(P.shape)
        # Compute stationary distribution
        pi = d / np.sum(d)
        for i in range(P.shape[0]):
            for j in range(i+1, P.shape[1]):
                # Compute diffusion distance between points i and j
                D_ij = np.sqrt(np.sum(((P[i, :] - P[j, :])**2) / pi))
                # Store the distance (matrix is symmetric)
                D[i, j] = D_ij
                D[j, i] = D_ij

        return D


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
            first_nonzero = np.nonzero(np.real(vectors[:, i]))[0][0]
            if np.real(vectors[first_nonzero, i]) < 0:
                vectors[:, i] *= -1

        return vectors


    @staticmethod
    # @njit
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
        eigenvalues = np.real(eigenvalues) # np.real_if_close(eigenvalues, tol=1e10)
        eigenvectors = np.real(eigenvectors) # np.real_if_close(eigenvectors, tol=1e10)
        # Find the order of the eigenvalues (decreasing order)
        order = np.argsort(eigenvalues)[::-1]
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
        self.K = self.get_kernel(self.X, self.X, self.sigma, self.kernel, self.alpha)
        # Compute the matrix P
        self.P = self._get_P(self.K)
        # Get the eigenvalues and eigenvectors of P
        self.lambdas, self.psis = self._spectral_decomposition(self.P)
        # Fix eigenvectors orientation
        self.psis = DiffusionMaps._fix_vector_orientation(self.psis)
        # Reduce dimension
        lambdas_red = self.lambdas[1:self.n_components + 1]
        psis_red = self.psis[:, 1:self.n_components + 1]
        # Compute the new coordinates
        self.Psi_steps = psis_red * (lambdas_red ** self.steps)

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
        X_red = self.Psi_steps

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
        K = self.get_kernel(Y, self.X, self.sigma, self.kernel, self.alpha)
        # Compute the matrix P(Y, X)
        P = self._get_P(K)
        # Get the n_components biggest eigenvalues of P(X, X)
        lambdas_red = self.lambdas[1:self.n_components + 1]
        # Apply Nyström formula
        Y_red = (P @ self.Psi_steps) / lambdas_red

        return Y_red
