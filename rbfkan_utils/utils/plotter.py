from typing import Literal, Optional, List, Dict, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(
        cm,
        class_names,
        title='Confusion Matrix',
        normalize=False,
        figsize=(6, 4),
        cmap='Blues',
        save_path=None
):
    """
    Plot a confusion matrix with optional normalisation.
    """
    if normalize:
        with np.errstate(divide='ignore', invalid='ignore'):
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm, nan=0.0)
        fmt = '.2f'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                ax=ax, square=True, linewidths=0.5, linecolor='gray')

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_img_comparison(
        gt_img,
        pr_img,
        slice_axis: Literal['x', 'y', 'z', 'h', 'w', 'c', 'i', 'j', 'k'] = 'z',
        n_slices=5,
        figsize=(15, 5),
        cmap='gray',
        save_path=None,
        axis_seq: Literal['xyz', 'chw', 'hwc', 'ijk'] = 'xyz',
):
    """
    Compare ground truth and predicted 3D volumes with slice views.
    """
    axis = axis_seq.lower().index(slice_axis.lower())
    total_slices = gt_img.shape[axis]
    if n_slices == -1:
        n_slices = total_slices
    slice_indices = np.linspace(0, total_slices - 1, n_slices, dtype=int)

    if total_slices in (1, 3, 4):
        fig, axes = plt.subplots(2, n_slices + 1, figsize=figsize)
    else:
        fig, axes = plt.subplots(2, n_slices, figsize=figsize)

    for i, idx in enumerate(slice_indices):
        if axis == 0:
            gt_slice = gt_img[idx, :, :]
            pr_slice = pr_img[idx, :, :]
        elif axis == 1:
            gt_slice = gt_img[:, idx, :]
            pr_slice = pr_img[:, idx, :]
        else:  # axis == 2
            gt_slice = gt_img[:, :, idx]
            pr_slice = pr_img[:, :, idx]

        axes[0, i].imshow(gt_slice.T, cmap=cmap, origin='lower')
        axes[0, i].set_title(f'GT Slice {idx}', fontsize=10)
        axes[0, i].axis('off')

        axes[1, i].imshow(pr_slice.T, cmap=cmap, origin='lower')
        axes[1, i].set_title(f'PR Slice {idx}', fontsize=10)
        axes[1, i].axis('off')

    if total_slices in (1, 3, 4):
        axes[0, -1].imshow(gt_img.T, cmap=cmap, origin='lower')
        axes[0, -1].set_title(f'GT Image', fontsize=10)
        axes[0, -1].axis('off')

        axes[1, -1].imshow(pr_img.T, cmap=cmap, origin='lower')
        axes[1, -1].set_title(f'PR Image', fontsize=10)
        axes[1, -1].axis('off')

    fig.patch.set_facecolor('black')
    plt.suptitle('Ground Truth vs Predicted Images', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes


# ---------- NEW HELPER FUNCTIONS ----------

def plot_loss_curves(
        epochs: np.ndarray,
        train_loss: np.ndarray,
        val_loss: np.ndarray,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        log_scale_auto: bool = True,
        figsize: tuple = (8, 5)
):
    """
    Plot training and validation loss curves.
    If log_scale_auto is True, uses log scale when max/min > 10.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, train_loss, label='Training', color='blue', linestyle='-')
    ax.plot(epochs, val_loss, label='Validation', color='orange', linestyle='-')

    if log_scale_auto:
        max_val = max(train_loss.max(), val_loss.max())
        min_val = min(train_loss.min(), val_loss.min())
        if min_val > 0 and max_val / min_val > 10:
            ax.set_yscale('log')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    if title is None:
        title = 'Training vs Validation Loss'
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax


def plot_metric_curves(
        epochs: np.ndarray,
        train_vals: np.ndarray,
        val_vals: np.ndarray,
        metric_name: str,
        color: str = 'blue',
        title: Optional[str] = None,
        ylabel: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (8, 5)
):
    """
    Plot training and validation curves for a single metric (e.g., Accuracy, F1).
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Use provided colour for both lines or distinguish them
    ax.plot(epochs, train_vals, label='Training', marker='.', linestyle='-', color=color)
    ax.plot(epochs, val_vals, label='Validation', marker='.', linestyle='--', color=color, alpha=0.7)

    ax.set_xlabel('Epoch')
    ylabel = ylabel if ylabel is not None else metric_name
    ax.set_ylabel(ylabel)
    if title is None:
        title = f'{metric_name} over Epochs'
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax


def plot_test_metrics_bar(
        metrics_dict: Dict[str, float],
        color: str = 'skyblue',
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (8, 5)
):
    """
    Create a bar chart of test metrics (single epoch).
    Metrics are sorted alphabetically.
    """
    # Filter out non-scalar entries (e.g., PrecisionRecallCurve)
    scalar_metrics = {k: v for k, v in metrics_dict.items()
                      if isinstance(v, (int, float, np.floating, np.integer))}

    if not scalar_metrics:
        print("No scalar test metrics to plot.")
        return None, None

    names = sorted(scalar_metrics.keys())
    values = [scalar_metrics[n] for n in names]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(names, values, color=color, edgecolor='black')

    # Annotate values on top of bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01 * max(values),
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Metric Value')
    if title is None:
        title = 'Test Set Metrics'
    ax.set_title(title, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax


def plot_lr_schedule(
        epochs: np.ndarray,
        lr_values: np.ndarray,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (8, 5)
):
    """
    Plot learning rate schedule (log scale).
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, lr_values, marker='.', linestyle='-', color='green')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    if title is None:
        title = 'Learning Rate Schedule'
    ax.set_title(title, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax