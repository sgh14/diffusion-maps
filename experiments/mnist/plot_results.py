import os
from os import path
import h5py
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

from experiments.utils import ConvBlock2D, UpConvBlock2D

plt.style.use('experiments/science.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def set_equal_ranges(ax):
    # Get the current axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Calculate the ranges of the x and y axes
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Find the maximum range between x and y
    max_range = max(x_range, y_range)

    # Set new limits with the same range for both axes
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2

    ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
    ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)

    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    return ax


# Function to sample 2 images per class from the dataset
def sample_images_per_class(X, y, images_per_class=2):
    selected_images = []
    selected_labels = []
    
    n_classes = len(np.unique(y))
    for class_label in range(n_classes):
        class_indices = np.where(y == class_label)[0]
        selected_indices = class_indices[:images_per_class]
        selected_images.extend(X[selected_indices])
        selected_labels.extend(y[selected_indices])
    
    return np.array(selected_images), np.array(selected_labels)


def plot_images(axes, X, y=[]):
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(X[i], cmap='gray')
        if len(y) > 0:
            ax.set_title(y[i])

        ax.axis('off')
    
    return axes


def plot_original(
    X,
    y,
    output_dir,
    filename,
    images_per_class=2,
    grid_shape=(3, 4)
):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(
        grid_shape[0], grid_shape[1],
        figsize=(3, 3),
        gridspec_kw={'wspace': 0.2, 'hspace': 0.2}
    )
    X, y = sample_images_per_class(X, y, images_per_class)
    axes = plot_images(axes, X, y)
    fig.tight_layout()
    for format in ('.pdf', '.png', '.svg'):
        fig.savefig(path.join(output_dir, filename + format))
    
    plt.close(fig)


def plot_projection(X, y, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
    ax.scatter(X[:, 0], X[:, 1], c=[colors[i] for i in y])#y, cmap=cmap)
    # Remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # Remove the tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.set_xlabel(r'$\Psi_1$')
    # ax.set_ylabel(r'$\Psi_2$')
    ax = set_equal_ranges(ax) # ax.set_box_aspect(1)

    # Create a list of handles and labels for the legend
    unique_y = np.unique(y)
    handles = [plt.Line2D([0], [0], linewidth=2, marker='o', color='w', markerfacecolor=colors[val], markersize=10) for val in unique_y]
    labels = [str(val) for val in unique_y]  # Adjust labels based on your case

    # Add the legend below the plot, with ncol=number of unique y values for one-row legend
    # fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1) # loc='lower center', ncol=len(unique_y)//2, bbox_to_anchor=(0.5, -0.1))
    fig.legend(handles, labels, loc='lower center', ncol=len(unique_y), handletextpad=0.2, columnspacing=0.2, bbox_to_anchor=(0.5, -0.12))

    for format in ('.pdf', '.png', '.svg'):
        fig.savefig(path.join(output_dir, filename + format))
    
    plt.close(fig)


def plot_history(history, output_dir, filename, log_scale=False):
    os.makedirs(output_dir, exist_ok=True)

    keys = [key for key in history.keys() if not key.startswith('val_')]
    for key in keys:
        y = np.array([history[key], history['val_' + key]])
        fig, ax = plt.subplots()
        if log_scale:
            ax.semilogy(y[0], label='Training')
            ax.semilogy(y[1], label='Validation')
        else:
            ax.plot(y[0], label='Training')
            ax.plot(y[1], label='Validation')
            ax.ticklabel_format(axis='both', style='sci', scilimits=(-1, 1), useMathText=True)
        
        ax.set_ylabel(key.capitalize())
        ax.set_xlabel('Epoch')
        ax.legend()
        for format in ('.pdf', '.png', '.svg'):
            fig.savefig(path.join(output_dir, filename + '-' + key + format))
        
        plt.close(fig)


# Helper function to compute class centroids in latent space
def compute_centroids(X_red, y):
    n_classes = len(np.unique(y))
    centroids = [np.mean(X_red[y == i], axis=0) for i in range(n_classes)]

    return np.array(centroids)

# Helper function to interpolate between two centroids
def interpolate(x1, x2, n_interpolations=4):
    alphas = np.linspace(0, 1, n_interpolations)
    interpolations = [(1 - alpha) * x1 + alpha * x2 for alpha in alphas]
    
    return np.array(interpolations)


def interpolate_images(X_red, y, decoder, class_pairs, n_interpolations):
    centroids = compute_centroids(X_red, y)
    interpolated_images = []
    for class1, class2 in class_pairs:
        centroid1 = centroids[class1]
        centroid2 = centroids[class2]
        interpolations = interpolate(centroid1, centroid2, n_interpolations)
        interpolated_images.append(decoder(interpolations).numpy())

    interpolated_images = np.vstack(interpolated_images)

    return interpolated_images


# Function to generate and plot interpolations between two classes
def plot_interpolations(
    X_red,
    y,
    decoder,
    output_dir,
    filename,
    class_pairs,
    n_interpolations=4
):
    os.makedirs(output_dir, exist_ok=True)

    grid_shape = (len(class_pairs), n_interpolations)
    fig, axes = plt.subplots(
        grid_shape[0], grid_shape[1],
        figsize=(3, 3),
        gridspec_kw={'wspace': 0, 'hspace': 0}
    )
    X_interp = interpolate_images(
            X_red, y, decoder, class_pairs, n_interpolations
    )
    axes = plot_images(axes, X_interp)
    fig.tight_layout()
    for format in ('.pdf', '.png', '.svg'):
        fig.savefig(path.join(output_dir, filename + format))
    
    plt.close(fig)


root = '/scratch/sgarcia/tfm/DM/experiments/mnist/results'
titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]
results_file = 'results.h5'
history_file = 'history.h5'
decoder_file = 'decoder.keras'
img_shape = (28, 28, 1)

q_vals = [0.005, 0.005, 1e-4, 1e-4]
steps_vals = [1, 1, 1, 1]
alpha_vals = [0, 0, 0, 0]

for i in range(len(titles)):
    title = titles[i]
    q, steps, alpha = q_vals[i], steps_vals[i], alpha_vals[i]
    experiment = f'quantile_{q}-steps_{steps}-alpha_{alpha}'
    output_dir = path.join(root, title, experiment)
    history = {}
    with h5py.File(path.join(output_dir, history_file), 'r') as file:
        for key in file.keys():
            history[key] = np.array(file[key])

    plot_history(history, output_dir, 'history', log_scale=True)

    decoder = keras.models.load_model(
        path.join(output_dir, decoder_file),
        custom_objects={'ConvBlock2D': ConvBlock2D, 'UpConvBlock2D': UpConvBlock2D},
        compile=False
    )
    for subset in ('train', 'test'):
        with h5py.File(path.join(output_dir, results_file), 'r') as file:
            X_orig = np.array(file['/X_' + subset]).reshape(-1, *img_shape)
            X_red = np.array(file['/X_' + subset + '_red'])
            X_rec = np.array(file['/X_' + subset + '_rec']).reshape(-1, *img_shape)
            y = np.array(file['/y_' + subset])
        
        plot_original(X_orig, y, output_dir, subset + '_orig', images_per_class=2, grid_shape=(3, 4))
        plot_projection(X_red, y, output_dir, subset + '_red')
        plot_original(X_rec, y, output_dir, subset + '_rec', images_per_class=2, grid_shape=(3, 4))
        plot_interpolations(X_red, y, decoder, output_dir, subset + '_interp', class_pairs = [(i, i+1) for i in range(0, 5)], n_interpolations=6)
