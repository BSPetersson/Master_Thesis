import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

def visualize_paths(gt_path, pred_path, output_path, title="VO", show=False):
    gt_path = np.array(gt_path)
    pred_path = np.array(pred_path)

    gt_x, gt_y = gt_path.T
    pred_x, pred_y = pred_path.T

    diff = np.linalg.norm(gt_path - pred_path, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot paths
    ax1.set_title("Paths")
    ax1.plot(gt_x, gt_y, marker='o', linestyle='-', color='blue', label='GT')
    ax1.plot(pred_x, pred_y, marker='o', linestyle='-', color='green', label='Pred')
    for i in range(len(gt_path)):
        ax1.plot([gt_x[i], pred_x[i]], [gt_y[i], pred_y[i]], linestyle='--', color='red')

    ax1.legend()
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # Plot error
    ax2.set_title("Error")
    ax2.plot(diff, marker='o', linestyle='-', color='red', label='Error')
    ax2.set_xlabel("frame")
    ax2.set_ylabel("error")

    plt.suptitle(title)
    
    # Save the plot as an image
    plt.savefig(output_path)
    
    # Close the figure
    plt.close(fig)

    if show:
        plt.show()



def make_residual_plot(x, residual_init, residual_minimized):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.set_title("Initial residuals")
    ax1.plot(x, residual_init)
    ax1.set_xlabel("residual")
    ax1.set_ylabel("")

    change = np.abs(residual_minimized) - np.abs(residual_init)
    ax2.set_title("Optimized residuals")
    ax2.plot(x, residual_minimized)
    ax2.set_xlabel(ax1.get_xlabel())
    ax2.set_ylabel(ax1.get_ylabel())

    ax3.set_title("Change")
    ax3.plot(x, change)
    ax3.set_xlabel(ax1.get_xlabel())
    ax3.set_ylabel(ax1.get_ylabel())

    return fig


def plot_residual_results(qs_small, small_residual_init, small_residual_minimized,
                          qs, residual_init, residual_minimized):
    x = np.arange(2 * qs_small.shape[0])
    fig1 = make_residual_plot(x, small_residual_init, small_residual_minimized)

    x = np.arange(2 * qs.shape[0])
    fig2 = make_residual_plot(x, residual_init, residual_minimized)

    fig1.suptitle("Bundle Adjustment with Reduced Parameters")
    fig2.suptitle("Bundle Adjustment with All Parameters (with sparsity)")

    plt.show()


def plot_sparsity(sparse_mat):
    fig, ax = plt.subplots(figsize=[20, 10])
    plt.title("Sparsity matrix")

    ax.spy(sparse_mat, aspect="auto", markersize=0.02)
    plt.xlabel("Parameters")
    plt.ylabel("Residuals")

    plt.show()