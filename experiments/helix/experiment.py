import numpy as np
import traceback
from os import path

from DiffusionMaps import DiffusionMaps
from aux_functions import get_sigma
from experiments.helix.plot_results import my_colormap1D, plot_original, plot_projection, plot_history
from experiments.helix.load_data import normalize, get_datasets
from experiments.utils import build_decoder
from experiments.helix.metrics import compute_metrics


root = 'experiments/helix/results'

q_vals = [0.005, 0.25] # [0.01, 0.5, 0.99]
steps_vals = [1, 10, 100]
alpha_vals = [0, 1]
kernel = 'rbf'

titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]
datasets_train, datasets_test = get_datasets(npoints=2000, test_size=0.5, seed=123, noise=0.1)

for (X, y), title in zip(datasets_train, titles):
    plot_original(X, y, title, path.join(root, 'train_orig'))

for (X, y), title in zip(datasets_test, titles):
    plot_original(X, y, title, path.join(root, 'test_orig'))

for q in q_vals:
    for steps in steps_vals:
        for alpha in alpha_vals:
            experiment = f'percentile_{q}-steps_{steps}-alpha_{alpha}'
            for (X_train, y_train), (X_test, y_test), title in zip(datasets_train, datasets_test, titles):
                try:
                    sigma = get_sigma(X_train, q)
                    DM = DiffusionMaps(sigma=sigma, n_components=2, steps=steps, alpha=alpha)
                    X_train_red = DM.fit_transform(X_train)
                    X_test_red = DM.transform(X_test)

                    decoder = build_decoder(output_shape=(X_train.shape[-1],), units=128, n_components=2)
                    decoder.compile(optimizer='adam', loss='mse')
                    history = decoder.fit(X_train_red, X_train, epochs=300, validation_split=0.1, shuffle=True, batch_size=64, verbose=0)
                    X_train_rec = decoder(X_train_red)
                    X_test_rec = decoder(X_test_red)

                    plot_projection(X_train_red, y_train, title, path.join(root, experiment, 'train_red'))
                    plot_original(X_train_rec, y_train, title, path.join(root, experiment, 'train_rec'))
                    plot_projection(X_test_red, y_test, title, path.join(root, experiment, 'test_red'))
                    plot_original(X_test_rec, y_test, title, path.join(root, experiment, 'test_rec'))
                    plot_history(history, path.join(root, experiment, 'histories', title), log_scale=True)

                    compute_metrics(X_test, X_test_rec, title, path.join(root, experiment))        
                    
                    P = DM.P
                    if steps > 1:
                        P = np.linalg.matrix_power(P, steps)
                    
                    D = DM.diffusion_distances(P, np.sum(DM.K, axis=1))
                    point = np.argmin(X_train[:, 2])
                    P_color = np.array([my_colormap1D(x) for x in normalize(P[:, point])])
                    D_color = np.array([my_colormap1D(x) for x in normalize(D[:, point])])

                    plot_original(X_train, P_color, title, path.join(root, experiment, 'train_P'))
                    plot_original(X_train, D_color, title, path.join(root, experiment, 'train_D'))
        
                except Exception as e:
                    # print(f"An error occurred: {e}")
                    traceback.print_exc()
                    pass