import numpy as np
import h5py
from os import path
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import pairwise_distances

from experiments.metrics import get_sigma, mean_diffusion_error, mean_reconstruction_error, trustworthiness_curve, clustering_homogeneity_and_completeness


root = '/scratch/sgarcia/tfm/DM/experiments/phoneme/results'
titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]
results_file = 'results.h5'
history_file = 'history.h5'

diffusion_weights = np.arange(0.1, 0.99, 0.2)
q_vals = [0.1, 0.1, 0.1, 0.1]
steps_vals = [1, 1, 1, 1]
alpha_vals = [0, 0, 0, 0]

for i in range(len(titles)):
    title = titles[i]
    q, steps, alpha = q_vals[i], steps_vals[i], alpha_vals[i]
    experiment = f'quantile_{q}-steps_{steps}-alpha_{alpha}'
    output_dir = path.join(root, title, experiment)
    print(experiment, '-', title)

    with h5py.File(path.join(output_dir, 'metrics.h5'), "w") as m_f:
        for subset in ('train', 'test'):
            with h5py.File(path.join(output_dir, results_file), 'r') as r_f:
                X_orig = np.array(r_f['/X_' + subset])
                X_red = np.array(r_f['/X_' + subset + '_red'])
                X_rec = np.array(r_f['/X_' + subset + '_rec'])
                y = np.array(r_f['/y_' + subset])
            
            if subset == 'train':
                sigma = get_sigma(X_orig, q)

            
            # distances = pairwise_distances(X_orig, metric='euclidean')
            classes, counts = np.unique(y, return_counts=True)
            min_count = np.min(counts)
            k_vals = list(range(1, round(0.9*min_count), min_count//50))
            graph = kneighbors_graph(X_orig, n_neighbors=min_count//20, mode='distance', include_self=False)
            distances = shortest_path(graph, method='D')
            distances = np.where(np.isinf(distances), 1e10, distances)
            # print(np.max(distances), np.all(np.isfinite(distances)))
            diff_err = mean_diffusion_error(X_orig, X_red, sigma, steps, alpha)
            rec_err = mean_reconstruction_error(X_orig, X_rec)
            t_curve = trustworthiness_curve(distances, X_red, k_vals)
            # c_curve = continuity_curve(distances, X_red, k_vals)
            completeness, homogeneity = clustering_homogeneity_and_completeness(X_red, y)
    
            m_f.create_dataset("diff_err_" + subset, data=diff_err)
            m_f.create_dataset("rec_err_" + subset, data=rec_err)
            m_f.create_dataset("t_curve_" + subset, data=t_curve, compression='gzip')
            # m_f.create_dataset("c_curve_" + subset, data=c_curve, compression='gzip')
            m_f.create_dataset("k_vals_" + subset, data=k_vals, compression='gzip')
            m_f.create_dataset("completeness_" + subset, data=completeness)
            m_f.create_dataset("homogeneity_" + subset, data=homogeneity)
