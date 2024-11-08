import numpy as np
import h5py
from os import path

from experiments.metrics import get_sigma, mean_diffusion_error, mean_reconstruction_error, trustworthiness_curve, continuity_curve, clustering_purity_and_accuracy


root = 'experiments/mnist/results'
titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]
results_file = 'results.h5'
history_file = 'history.h5'

diffusion_weights = np.arange(0.1, 0.99, 0.1)
q_vals = [0.0075, 0.0075, 0.0075, 0.0075]
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
            
            k_vals = list(range(1, round(0.5*len(X_orig)), len(X_orig)//50))
            if subset == 'train':
                sigma = get_sigma(X_orig, q)

            diff_err = mean_diffusion_error(X_orig, X_red, sigma, steps, alpha)
            rec_err = mean_reconstruction_error(X_orig, X_rec)
            t_curve = trustworthiness_curve(X_orig, X_red, k_vals)
            c_curve = continuity_curve(X_orig, X_red, k_vals)
            accuracy, purity = clustering_purity_and_accuracy(X_red, y)
    
            m_f.create_dataset("diff_err_" + subset, data=diff_err)
            m_f.create_dataset("rec_err_" + subset, data=rec_err)
            m_f.create_dataset("t_curve_" + subset, data=t_curve, compression='gzip')
            m_f.create_dataset("c_curve_" + subset, data=c_curve, compression='gzip')
            m_f.create_dataset("k_vals_" + subset, data=k_vals, compression='gzip')
            m_f.create_dataset("accuracy_" + subset, data=accuracy)
            m_f.create_dataset("purity_" + subset, data=purity)
