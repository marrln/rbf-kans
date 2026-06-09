from typing import Literal
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
    if normalize:
        # Normalize confusion matrix, handling zero-sum rows
        with np.errstate(divide='ignore', invalid='ignore'):
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # Replace NaN values (from rows with zero predictions) with 0
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
        slice_axis : Literal['x','y','z','h','w','c','i','j','k'] = 'z', 
        n_slices=5, 
        figsize=(15, 5), 
        cmap='gray', 
        save_path=None,
        axis_seq : Literal['xyz','chw','hwc','ijk'] = 'xyz',
    ):
    axis = axis_seq.lower().index(slice_axis.lower)
    total_slices = gt_img.shape[axis]
    if n_slices == -1 :
        n_slices = total_slices
    slice_indices = np.linspace(0, total_slices - 1, n_slices, dtype=int)
    
    if total_slices in (1,3,4):
        fig, axes = plt.subplots(2, n_slices + 1, figsize=figsize)
    else :
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
        
    if total_slices in (1,3,4):
        axes[0,-1].imshow(gt_img.T, cmap=cmap, origin='lower')
        axes[0,-1].set_title(f'GT Image', fontsize=10)
        axes[0,-1].axis('off')
        
        axes[1,-1].imshow(pr_img.T, cmap=cmap, origin='lower')
        axes[1,-1].set_title(f'PR Image', fontsize=10)
        axes[1,-1].axis('off')

    # Set figure background colour to black
    fig.patch.set_facecolor('black')

    plt.suptitle('Ground Truth vs Predicted Images', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes

