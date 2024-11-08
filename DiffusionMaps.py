import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances


class DiffusionMaps(TransformerMixin, BaseEstimator):
    """
    Diffusion Maps.
    """
    def __init__(self, sigma, n_components, steps=1, alpha=1, kernel='rbf'):
        self.sigma = sigma
        self.gamma = 1/(2*sigma**2)
        self.n_components = n_components
        self.steps = steps
        self.alpha = alpha


    @staticmethod
    def rbf_kernel(X, Y=None, gamma=None):
        gamma = gamma if gamma else 1.0 / X.shape[1]
        distances = pairwise_distances(X, Y, metric='sqeuclidean')
        K = np.exp(-gamma * distances)

        return K


    @staticmethod
    def get_kernel(X, Y, gamma, alpha):
        K = DiffusionMaps.rbf_kernel(X, Y, gamma=gamma)        
        d_alpha = np.sum(K, axis=1)**alpha
        K_alpha = K/np.outer(d_alpha, d_alpha)

        return K_alpha


    @staticmethod
    def _get_P(K):
        d = np.sum(K, axis=1)
        P = K / d[:, np.newaxis]

        return P
    

    @staticmethod
    def diffusion_distances(P, pi):
        D = pairwise_distances(
            P, metric=lambda P_i, P_j: np.sqrt(np.sum(((P_i - P_j)**2) / pi))
        )

        return D
    

    @staticmethod
    def _get_A(K):
        d = np.sum(K, axis=1)
        A = K/np.sqrt(np.outer(d, d))

        return A


    @staticmethod
    def _fix_vector_orientation(vectors):
        # Fix the first non-zero component of every vector to be positive
        for i in range(vectors.shape[1]):
            first_nonzero = np.nonzero(vectors[:, i])[0][0]
            if vectors[first_nonzero, i] < 0:
                vectors[:, i] *= -1

        return vectors


    @staticmethod
    def _spectral_decomposition(A):
        # Compute the eigenvalues and right eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        # Find the order of the eigenvalues (decreasing order)
        order = np.argsort(eigenvalues)[::-1]
        # Sort eigenvalues and eigenvectors
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        return eigenvalues, eigenvectors


    def fit(self, X, y=None):
        self.X = X
        # Compute the kernel
        self.K = self.get_kernel(self.X, self.X, self.gamma, self.alpha)
        # Compute the matrix A
        self.A = self._get_A(self.K)
        # Get the eigenvalues and eigenvectors of P
        self.lambdas, self.phis = self._spectral_decomposition(self.A)
        # Fix eigenvectors orientation
        self.phis = self._fix_vector_orientation(self.phis)
        # Reduce dimension
        self.lambdas = self.lambdas[:self.n_components + 1]
        self.phis = self.phis[:, :self.n_components + 1]
        # Compute degree vector
        d = np.sum(self.K, axis=1)
        # Compute the stationary distribution
        self.pi = d / np.sum(d)
        # Compute P right eigenvectors
        self.psis = self.phis / np.sqrt(self.pi[:, np.newaxis])
        # Compute the new coordinates
        self.Psi_steps = self.psis * (self.lambdas[np.newaxis, :] ** self.steps)

        return self


    def fit_transform(self, X, y=None):
        self.fit(X)
        X_red = self.Psi_steps[:, 1:]

        return X_red


    def _get_K_alpha_approx(self, X, Y):
        K = DiffusionMaps.rbf_kernel(X, Y, gamma=self.gamma)
        K_x = DiffusionMaps.rbf_kernel(X, self.X, gamma=self.gamma)
        K_y = DiffusionMaps.rbf_kernel(Y, self.X, gamma=self.gamma)
        d_x = np.sum(K_x, axis=1)
        d_y = np.sum(K_y, axis=1)
        K_alpha = K / np.outer(d_x, d_y)

        return K_alpha
    

    def _get_A_approx(self, X, Y):
        K_alpha = self._get_K_alpha_approx(X, Y)
        K_alpha_x = self._get_K_alpha_approx(X, self.X)
        K_alpha_y = self._get_K_alpha_approx(Y, self.X)
        d_alpha_x = np.sum(K_alpha_x, axis=1)
        d_alpha_y = np.sum(K_alpha_y, axis=1)
        A = K_alpha / np.sqrt(np.outer(d_alpha_x, d_alpha_y))

        return A


    def transform(self, Y):
        new_A = self._get_A_approx(Y, self.X)
        new_phis = (new_A @ self.phis) / self.lambdas[np.newaxis, :]
        new_lambdas = self.lambdas
        new_pi = new_phis[:, 0]**2
        new_psis = new_phis / np.sqrt(new_pi[:, np.newaxis])
        new_Psi_steps = new_psis * (new_lambdas[np.newaxis, :] ** self.steps)
        Y_red = new_Psi_steps[:, 1:]

        return Y_red
