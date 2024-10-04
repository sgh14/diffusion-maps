import numpy as np
import time
import tensorflow as tf
from os import path

from DiffusionMaps import DiffusionMaps
from aux_functions import get_sigma
from experiments.swiss_roll.plot_results import my_colormap1D, plot_original, plot_projection, plot_history
from experiments.swiss_roll.load_data import normalize, get_datasets
from experiments.utils import build_decoder
from experiments.swiss_roll.metrics import compute_metrics
    

seed = 123
tf.random.set_seed(seed)
root = 'experiments/swiss_roll/results'
titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]
datasets_train, datasets_test = get_datasets(npoints=2000, test_size=0.5, seed=123, noise=0.75)

q_vals = [5e-3, 5e-3, 2.5e-3, 2.5e-3]
steps_vals = [100, 100, 100, 100]
alpha_vals = [1, 1, 1, 1]
kernel = 'rbf'

for i in range(len(titles)):
    q, steps, alpha = q_vals[i], steps_vals[i], alpha_vals[i]
    experiment = f'percentile_{q}-steps_{steps}-alpha_{alpha}'
    X_train, y_train = datasets_train[i]
    X_test, y_test = datasets_test[i]
    title = titles[i]
    sigma = get_sigma(X_train, q)

    print(experiment, '-', title)
    DM = DiffusionMaps(sigma=sigma, n_components=2, steps=steps, alpha=alpha)
    tic = time.perf_counter()
    X_train_red = DM.fit_transform(X_train)
    tac = time.perf_counter()
    X_test_red = DM.transform(X_test)
    toc = time.perf_counter()

    decoder = build_decoder(output_shape=(X_train.shape[-1],), units=128, n_components=2)
    decoder.compile(optimizer='adam', loss='mse')
    history = decoder.fit(X_train_red, X_train, epochs=500, validation_split=0.1, shuffle=False, batch_size=64, verbose=0)
    X_train_rec = decoder(X_train_red)
    X_test_rec = decoder(X_test_red)

    plot_original(X_train, y_train, path.join(root, title), 'train_orig')
    plot_original(X_test, y_test, path.join(root, title), 'test_orig')
    plot_projection(X_train_red, y_train, path.join(root, title, experiment), 'train_red')
    plot_original(X_train_rec, y_train, path.join(root, title, experiment), 'train_rec')
    plot_projection(X_test_red, y_test, path.join(root, title, experiment), 'test_red')
    plot_original(X_test_rec, y_test, path.join(root, title, experiment), 'test_rec')
    plot_history(history, path.join(root, 'histories', title, experiment), log_scale=True)

    time_in_sample = tac - tic
    time_out_of_sample = toc - tac
    compute_metrics(
        X_train,
        X_train_red,
        X_train_rec,
        X_test,
        X_test_red,
        X_test_rec,
        time_in_sample,
        time_out_of_sample,
        title,
        output_dir=path.join(root, title, experiment)
    )

    P = DM.P
    if steps > 1:
        P = np.linalg.matrix_power(P, steps)
    
    D = DM.diffusion_distances(P, np.sum(DM.K, axis=1))
    point = np.argmin(X_train[:, 2])
    P_color = np.array([my_colormap1D(x) for x in normalize(P[:, point])])
    D_color = np.array([my_colormap1D(x) for x in normalize(D[:, point])])

    plot_original(X_train, P_color, path.join(root, title, experiment), 'train_P')
    plot_original(X_train, D_color, path.join(root, title, experiment), 'train_D')
