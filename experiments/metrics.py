import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, pairwise_distances
from sklearn.manifold import trustworthiness


def get_sigma(X, q=0.5):
    distances = pairwise_distances(X)
    distances = distances.flatten()
    sigma = np.quantile(distances[distances > 0], q)

    return sigma


def rbf_kernel(X, Y=None, gamma=None):
    gamma = gamma if gamma else 1.0 / X.shape[1]
    distances = pairwise_distances(X, Y, metric='sqeuclidean')
    K = np.exp(-gamma * distances)

    return K


def get_kernel(X, Y, gamma, alpha):
    K = rbf_kernel(X, Y, gamma=gamma)        
    d_alpha = np.sum(K, axis=1)**alpha
    K_alpha = K/np.outer(d_alpha, d_alpha)

    return K_alpha


def get_P(K):
    d = np.sum(K, axis=1)
    P = K / d[:, np.newaxis]

    return P


def diffusion_distances(X, sigma, steps, alpha):
    gamma = 1/(2*sigma**2)
    # Compute the kernel matrix
    K = get_kernel(X, X, gamma, alpha)
    # Compute diffusion probabilities
    P = get_P(K)
    if steps > 1:
        P = np.linalg.matrix_power(P, steps)
        
    # Compute degree vector
    d = np.sum(K, axis=1)
    # Compute the stationary distribution
    pi = d / np.sum(d)
    # Compute diffusion distances
    D = pairwise_distances(
        P, metric=lambda P_i, P_j: np.sqrt(np.sum(((P_i - P_j)**2) / pi))
    )

    return D


# Function to compute Diffusion error
def mean_diffusion_error(X_orig, X_red, sigma, steps, alpha):
    D_diff = diffusion_distances(X_orig, sigma, steps, alpha)
    D_euc = pairwise_distances(X_red, metric='euclidean')
    mean_diff_err = np.mean(np.abs(D_diff - D_euc))

    return mean_diff_err


# Function to compute rec_error
def mean_reconstruction_error(X_orig, X_rec):
    mean_rec_err = np.mean(np.linalg.norm(X_orig - X_rec, axis=1))

    return mean_rec_err


# Function to compute trustworthiness curves
def trustworthiness_curve(X, X_red, k_vals, metric='euclidean'):
    t_curve = [trustworthiness(X, X_red, n_neighbors=k, metric=metric) for k in k_vals]

    return t_curve


# Function to compute trustworthiness curves
def continuity_curve(X, X_red, k_vals, metric='euclidean'):
    c_curve = trustworthiness_curve(X_red, X, k_vals, metric)

    return c_curve


def clustering_homogeneity_and_completeness(X_red, y):
    n_classes = len(np.unique(y))
    k_means = KMeans(n_clusters=n_classes)
    clusters = k_means.fit_predict(X_red)
    homogeneity = homogeneity_score(y, clusters)
    completeness = completeness_score(y, clusters)

    return homogeneity, completeness