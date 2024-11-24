import os
from os import path
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

plt.style.use('experiments/science.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
classes = {0: 'aa', 1: 'ao', 2: 'dcl', 3: 'iy', 4: 'sh'}


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


def plot_original(X, y, output_dir, filename, n_samples=30):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
    for i in range(n_samples):
        ax.plot(X[i], color=colors[y[i]], linewidth=1, alpha=0.75)
    
    ax.set_box_aspect(1)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    # Create a list of handles and labels for the legend
    unique_y = np.unique(y)
    handles = [plt.Line2D([0], [0], linewidth=2, marker='o', color='w', markerfacecolor=colors[val], markersize=10) for val in unique_y]
    labels = [classes[val] for val in unique_y]  # Adjust labels based on your case

    # Add the legend below the plot, with ncol=number of unique y values for one-row legend
    fig.legend(handles, labels, loc='lower center', ncol=len(unique_y), handletextpad=0.2, columnspacing=0.2, bbox_to_anchor=(0.5, -0.1))


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
    labels = [classes[val] for val in unique_y]  # Adjust labels based on your case

    # Add the legend below the plot, with ncol=number of unique y values for one-row legend
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


root = '/scratch/sgarcia/tfm/DM/experiments/phoneme/results'
titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]
results_file = 'results.h5'
history_file = 'history.h5'

q_vals = [0.25, 0.25, 0.25, 0.25]
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

    for subset in ('train', 'test'):
        with h5py.File(path.join(output_dir, results_file), 'r') as file:
            X_orig = np.array(file['/X_' + subset])
            X_red = np.array(file['/X_' + subset + '_red'])
            X_rec = np.array(file['/X_' + subset + '_rec'])
            y = np.array(file['/y_' + subset])
        
        plot_original(X_orig, y, output_dir, subset + '_orig')
        plot_projection(X_red, y, output_dir, subset + '_red')
        plot_original(X_rec, y, output_dir, subset + '_rec')