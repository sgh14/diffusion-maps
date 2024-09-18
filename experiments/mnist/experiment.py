import traceback
from os import path

from DiffusionMaps import DiffusionMaps
from aux_functions import get_sigma
from experiments.mnist.plot_results import plot_original, plot_projection, plot_interpolations, plot_history
from experiments.mnist.load_data import get_datasets
from experiments.mnist.metrics import compute_metrics
from experiments.utils import build_decoder


root = 'experiments/mnist/results'

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

datasets_train, datasets_test = get_datasets(npoints=2000, test_size=0.1, seed=123, noise=0.25)

for (X, y), title in zip(datasets_train, titles):
    plot_original(
        X, y, title, path.join(root, 'train_orig'), images_per_class=2, grid_shape=(3, 4)
    )

for (X, y), title in zip(datasets_test, titles):
    plot_original(
        X, y, title, path.join(root, 'test_orig'), images_per_class=2, grid_shape=(3, 4)
    )

for q in q_vals:
    for steps in steps_vals:
        for alpha in alpha_vals:
            experiment = f'percentile_{q}-steps_{steps}-alpha_{alpha}'
            for (X_train, y_train), (X_test, y_test), title in zip(datasets_train, datasets_test, titles):
                try:
                    img_shape = X_train.shape[1:]
                    X_train = X_train.reshape((X_train.shape[0], -1))
                    X_test = X_test.reshape((X_test.shape[0], -1))
                    
                    sigma = get_sigma(X_train, q)
                    DM = DiffusionMaps(sigma=sigma, n_components=2, steps=steps, alpha=alpha)
                    X_train_red = DM.fit_transform(X_train)
                    X_test_red = DM.transform(X_test)

                    decoder = build_decoder(output_shape=(X_train.shape[-1],), units=128, n_components=2)
                    decoder.compile(optimizer='adam', loss='mse')
                    history = decoder.fit(X_train_red, X_train, epochs=5, validation_split=0.1, shuffle=True, batch_size=64, verbose=0)
                    X_train_rec = decoder(X_train_red)
                    X_test_rec = decoder(X_test_red)
                    X_train_rec = X_train_rec.numpy().reshape((X_train_rec.shape[0], *img_shape))
                    X_test_rec = X_test_rec.numpy().reshape((X_test_rec.shape[0], *img_shape))

                    plot_projection(X_train_red, y_train, title, path.join(root, experiment, 'train_red'))
                    plot_original(
                        X_train_rec, y_train, title,
                        path.join(root, experiment, 'train_rec'),
                        images_per_class=2, grid_shape=(3, 4)
                    )
                    plot_projection(X_test_red, y_test, title, path.join(root, experiment, 'test_red'))
                    plot_original(
                        X_test_rec, y_test, title,
                        path.join(root, experiment, 'test_rec'),
                        images_per_class=2, grid_shape=(3, 4)
                    )
                    plot_interpolations(
                        X_test_red, y_test, title,
                        decoder,
                        path.join(root, experiment, 'test_interp'),
                        img_shape,
                        class_pairs = [(i, i+1) for i in range(0, 6, 2)],
                        n_interpolations=4
                    )
                    plot_history(history, path.join(root, experiment, 'histories', title), log_scale=True)

                    compute_metrics(X_test, X_test_red, X_test_rec, y_test, title, path.join(root, experiment))        

                except Exception as e:
                    # print(f"An error occurred: {e}")
                    traceback.print_exc()
                    pass