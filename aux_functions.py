import numpy as np
from scipy.spatial.distance import pdist


def get_sigma(X, q=0.5):
    X_flat = X.reshape((X.shape[0], -1))
    distances = pdist(X_flat, metric='euclidean')
    sigma = np.quantile(distances, q)
    
    return sigma