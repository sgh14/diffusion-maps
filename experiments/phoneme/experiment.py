import time
import numpy as np
import tensorflow as tf
import os
import random
from os import path
import h5py

from DiffusionMaps import DiffusionMaps
from aux_functions import get_sigma
from experiments.phoneme.load_data import get_datasets
from experiments.utils import build_seq_decoder

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ENSURE REPRODUCIBILITY
seed = 123
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # force the use of CPU


root = 'experiments/phoneme/results'
titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]

datasets_train, datasets_test = get_datasets(test_size=0.2, seed=seed, noise=0.25)

q_vals = [0.75, 0.75, 0.75, 0.75]
steps_vals = [1, 1, 1, 1]
alpha_vals = [0, 0, 0, 0]


for i in range(len(titles)):
    title = titles[i]
    q, steps, alpha = q_vals[i], steps_vals[i], alpha_vals[i]
    experiment = f'quantile_{q}-steps_{steps}-alpha_{alpha}'
    output_dir = path.join(root, title, experiment)
    os.makedirs(output_dir, exist_ok=True)
    X_train, y_train = datasets_train[i]
    X_test, y_test = datasets_test[i]
    sigma = get_sigma(X_train, q)
    
    print(experiment, '-', title)  
    DM = DiffusionMaps(sigma=sigma, n_components=2, steps=steps, alpha=alpha)
    tic = time.perf_counter()
    X_train_red = DM.fit_transform(X_train.reshape((X_train.shape[0], -1)))
    tac = time.perf_counter()
    X_test_red = DM.transform(X_test.reshape((X_test.shape[0], -1)))
    toc = time.perf_counter()

    decoder = build_seq_decoder(output_shape=X_train.shape[1:], filters=8, n_components=2, cropping=0)
    decoder.compile(optimizer='adam', loss='mse')
    history = decoder.fit(X_train_red, X_train, epochs=100, validation_split=0.1, shuffle=False, batch_size=64, verbose=0)
    X_train_rec = decoder(X_train_red).numpy()
    X_test_rec = decoder(X_test_red).numpy()

    decoder.save(path.join(output_dir, 'decoder.keras'))
    with h5py.File(path.join(output_dir, 'history.h5'), 'w') as file:
        for key, value in history.history.items():
            file.create_dataset(key, data=value)
    
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_train_rec = X_train_rec.reshape(X_train_rec.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_test_rec = X_test_rec.reshape(X_test_rec.shape[0], -1)
    with h5py.File(path.join(output_dir, 'results.h5'), "w") as file:
        file.create_dataset("X_train", data=X_train, compression='gzip')
        file.create_dataset("X_train_red", data=X_train_red, compression='gzip')
        file.create_dataset("X_train_rec", data=X_train_rec, compression='gzip')
        file.create_dataset("y_train", data=y_train, compression='gzip')
        file.create_dataset("X_test", data=X_test, compression='gzip')
        file.create_dataset("X_test_red", data=X_test_red, compression='gzip')
        file.create_dataset("X_test_rec", data=X_test_rec, compression='gzip')
        file.create_dataset("y_test", data=y_test, compression='gzip')

    time_in_sample = tac - tic
    time_out_of_sample = toc - tac
    times = np.array([time_in_sample, time_out_of_sample])
    np.savetxt(path.join(output_dir, 'times.txt'), times)