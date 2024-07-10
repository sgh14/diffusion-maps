import numpy as np
from numba import njit


@njit
def rbf_kernel(X, gamma=None):
    """
    Compute the Radial Basis Function (RBF) kernel between each pair of samples in X.

    The RBF kernel is defined as:
    K(x, y) = exp(-gamma * ||x - y||^2)
    where ||x - y||^2 is the squared Euclidean distance between the vectors x and y.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
        Input data where n_samples is the number of samples and n_features is the number of features.
    gamma : float
        The gamma parameter for the RBF kernel. This is a non-negative parameter that controls 
        the width of the Gaussian kernel. If None, default to 1/n_features.

    Returns
    -------
    K : numpy.ndarray, shape (n_samples, n_samples)
        The RBF kernel matrix.
    """
    # Compute the squared Euclidean distances
    # This uses the identity (a-b)^2 = a^2 + b^2 - 2ab
    X_norm = np.sum(X ** 2, axis=-1)
    X_norm = X_norm[:, np.newaxis] + X_norm[np.newaxis, :] - 2 * np.dot(X, X.T)
    # Apply the RBF kernel function
    K = np.exp(-gamma * X_norm)

    return K


@njit
def laplacian_kernel(X, gamma=None):
    """
    Compute the Laplacian kernel between each pair of samples in X.

    The Laplacian kernel is defined as:
    K(x, y) = exp(-gamma * ||x - y||_1)
    where ||x - y||_1 is the L1 norm (Manhattan distance) between the vectors x and y.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
        Input data where n_samples is the number of samples and n_features is the number of features.
    gamma : float
        The gamma parameter for the Laplacian kernel. This is a non-negative parameter that controls 
        the width of the kernel.

    Returns
    -------
    K : numpy.ndarray, shape (n_samples, n_samples)
        The Laplacian kernel matrix.
    """
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distance = np.sum(np.abs(X[i] - X[j]))
            K[i, j] = K[j, i] = np.exp(-gamma * distance)

    return K
