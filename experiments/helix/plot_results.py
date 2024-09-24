import os
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('experiments/science.mplstyle')


def my_colormap1D(x):
    c1=(0.75, 0, 0.75)
    c2=(0, 0.75, 0.75)
    # Calculate the RGB values based on interpolation
    color = np.array(c1) * (1 - x) + np.array(c2) * x

    return color


def plot_original(X, y, title, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={"projection": "3d"})
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, alpha=1)
    ax.set_xlabel('$x$')
    ax.set_xlim([-1, 1])
    ax.set_ylabel('$y$')
    ax.set_ylim([-1, 1])
    ax.set_zlabel('$z$')
    ax.set_zlim([-1, 1])
    ax.view_init(15, -72)
    ax.dist = 12
    fig.tight_layout()

    for format in ('.pdf', '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, title + format))

    plt.close(fig)


def plot_projection(X, y, title, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
    ax.scatter(X[:, 0], X[:, 1], c=y)
    ax.ticklabel_format(axis='both', style='sci', scilimits=(-1, 1), useMathText=True)
    ax.set_xlabel(r'$\Tilde{x}$')
    ax.set_ylabel(r'$\Tilde{y}$')
    ax.set_box_aspect(1)
    # fig.tight_layout()

    for format in ('.pdf', '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, title + format))
    
    plt.close(fig)


def plot_history(history, output_dir, log_scale=False):
    os.makedirs(output_dir, exist_ok=True)

    h = history.history
    keys = [key for key in h.keys() if not key.startswith('val_')]
    for key in keys:
        y = np.array([h[key], h['val_' + key]])
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
            fig.savefig(os.path.join(output_dir, key + format))
        
        plt.close(fig)