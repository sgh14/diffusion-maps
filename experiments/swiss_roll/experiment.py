import time
import numpy as np
import tensorflow as tf
import os
import random
from os import path
import h5py

from DiffusionMaps import DiffusionMaps
from aux_functions import get_sigma
from experiments.swiss_roll.load_data import normalize, my_colormap1D, get_datasets
from experiments.utils import build_decoder
    
# ENSURE REPRODUCIBILITY
seed = 123
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # force the use of CPU


root = 'experiments/swiss_roll/results'
titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]
datasets_train, datasets_test = get_datasets(npoints=2000, test_size=0.5, seed=seed, noise=0.5)

q_vals = [5e-3, 5e-3, 2.5e-3, 2.5e-3]
steps_vals = [100, 100, 100, 100]
alpha_vals = [1, 1, 1, 1]
kernel = 'rbf'

for i in range(len(titles)):
    title = titles[i]
    q, steps, alpha = q_vals[i], steps_vals[i], alpha_vals[i]
    experiment = f'quantile_{q}-steps_{steps}-alpha_{alpha}'
    output_dir = path.join(root, title, experiment)
    X_train, y_train = datasets_train[i]
    X_test, y_test = datasets_test[i]
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

    P = DM._get_P(DM.K)
    if steps > 1:
        P = np.linalg.matrix_power(P, steps)
    
    D = DM.diffusion_distances(P, DM.pi)
    point = np.argmin(X_train[:, 2])
    P_color = np.array([my_colormap1D(x) for x in normalize(P[:, point])])
    D_color = np.array([my_colormap1D(x) for x in normalize(D[:, point])])

    decoder.save(path.join(output_dir, 'decoder.h5'))
    with h5py.File(path.join(output_dir, 'history.h5'), 'w') as file:
        for key, value in history.history.items():
            file.create_dataset(key, data=value)

    with h5py.File(path.join(output_dir, 'results.h5'), "w") as file:
        file.create_dataset("X_train", data=X_train, compression='gzip')
        file.create_dataset("X_train_red", data=X_train_red, compression='gzip')
        file.create_dataset("X_train_rec", data=X_train_rec, compression='gzip')
        file.create_dataset("y_train", data=y_train, compression='gzip')
        file.create_dataset("X_test", data=X_test, compression='gzip')
        file.create_dataset("X_test_red", data=X_test_red, compression='gzip')
        file.create_dataset("X_test_rec", data=X_test_rec, compression='gzip')
        file.create_dataset("y_test", data=y_test, compression='gzip')
        
        file.create_dataset("P_color", data=P_color, compression='gzip')
        file.create_dataset("D_color", data=D_color, compression='gzip')

    time_in_sample = tac - tic
    time_out_of_sample = toc - tac
    times = np.array([time_in_sample, time_out_of_sample])
    np.savetxt(path.join(output_dir, 'times.txt'), times)
