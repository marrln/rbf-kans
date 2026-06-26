#!/usr/bin/env python3
if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(__file__)
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(
        description=f'Extract result statistics and plots for a test configuration.'
    )

    parser.add_argument('-d', '--test-dir', dest='test_dir', default=None, help='The directory to be used as a top directory for training, if None uses dataset-specific default directory.')
    parser.add_argument('--dataset', dest='dataset', type=str, required=True, help=f'Dataset to use (required)')
    parser.add_argument('--hash', dest='hash', type=str, help='The hash value of the configuration.', required=True)
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')
    parser.add_argument('--epoch', dest='epoch', type=str, default='best')

    args = parser.parse_args()
    
    # Add dataset folder to path
    dataset_path = os.path.join(THIS_DIR, args.dataset)
    if dataset_path not in sys.path:
        sys.path.insert(0, dataset_path)
    from prepare_dataset import DATASET_DIR, DATASET_NAME, create_labels # pyright: ignore[reportMissingImports]
    
    # Set default after DATASET_DIR is available
    if args.test_dir is None:
        args.test_dir = os.path.join(DATASET_DIR, 'train')
        
    args.test_version = '_'.join(['test', args.test_version])

    # Use dataset-specific test directory
    if args.dataset != DATASET_NAME:
        dataset_dir = os.path.join(THIS_DIR, args.dataset)
        args.test_dir = os.path.join(dataset_dir, 'train')
    else:
        args.test_dir = os.path.join(DATASET_DIR, 'train') if not os.path.isabs(args.test_dir) else args.test_dir

    # Check argument validity
    if os.path.isdir(args.test_dir) or not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir, exist_ok=True)
    else:
        raise ValueError(f'Destination folder is not a directory; got "{os.path.splitext(args.test_dir)[-1]}"')
    
    args.test_dir = os.path.join(args.test_dir, args.hash, args.test_version)

    if args.hash is None:
        raise ValueError(f'Cannot locate training configuration file.')
    else:
        path = os.path.join(args.test_dir, 'config', 'train.json')
        if os.path.exists(path):
            args.train_config = path
            print(f'-- Using training configuration path "{path}"')
        else:
            raise ValueError(f'Cannot locate training configuration file.')
            
        path = os.path.join(args.test_dir, 'config', 'model.json')
        if os.path.exists(path):
            args.model_config = path
            print(f'-- Using model configuration path "{path}"')
        else:
            raise ValueError(f'Cannot locate model configuration file.')
            
    import pandas as pd
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    from rbfkan_utils.utils import load_checkpoint
    from rbfkan_utils.utils.plotter import (
        plot_confusion_matrix,
        plot_loss_curves,
        plot_metric_curves,
        plot_test_metrics_bar,
        plot_lr_schedule
    )
    from rbfkan_utils.config import *

    # Load configurations
    train_config = load_config(args.train_config, locals=get_locals())
    model_config = load_config(args.model_config, locals=get_locals())

    # Determine which checkpoint to load for history
    model_dir = os.path.join(args.test_dir, 'models')
    last_checkpoint_path = os.path.join(model_dir, 'last.pt')
    if not os.path.exists(last_checkpoint_path):
        raise FileNotFoundError(f"Last checkpoint not found: {last_checkpoint_path}")
    
    print(f"Loading full history from {last_checkpoint_path}")
    checkpoint = load_checkpoint(last_checkpoint_path, load_rng=False)
    history = checkpoint['history']   # contains 'train' and 'val' with all epochs

    # Determine which epoch to use for test metrics / CSV files
    if args.epoch == 'best':
        # Find epoch with minimum validation loss from the full history
        best_val_loss = float('inf')
        best_epoch = None
        for ep, metrics in history['val'].items():
            if metrics['loss'] < best_val_loss:
                best_val_loss = metrics['loss']
                best_epoch = ep
        if best_epoch is None:
            raise RuntimeError("No validation metrics found in history.")
        use_epoch = best_epoch
        print(f"Using best validation epoch: {use_epoch} (loss = {best_val_loss:.5f})")
    else:
        try:
            use_epoch = int(args.epoch)
        except ValueError:
            raise ValueError(f"Invalid epoch value: {args.epoch}. Must be 'best' or an integer.")
        if use_epoch not in history['val']:
            raise KeyError(f"Epoch {use_epoch} not found in validation history. Available: {sorted(history['val'].keys())}")
        print(f"Using specified epoch: {use_epoch}")

    if 'test' in history and use_epoch in history['test']:
        test_metrics = history['test'][use_epoch]
        print(f"Using test metrics from epoch {use_epoch}")
    else:
        print(f"Warning: No test metrics for epoch {use_epoch}. Using validation metrics instead.")
        test_metrics = history['val'][use_epoch]
    
    # Print basic statistics
    print(f'Loss for epoch "{use_epoch}": {test_metrics["loss"]}')
    for key in test_metrics:
        print(f'{key} for epoch "{use_epoch}": {test_metrics[key]}')

    # Create plots directory
    plots_path = os.path.join(args.test_dir, 'plot')
    os.makedirs(plots_path, exist_ok=True)

    # ---------- PLOT 1: Training vs Validation Loss ----------
    epochs = np.asarray(list(history['train'].keys()), dtype=int)
    tr_loss = np.array([history['train'][ep]['loss'] for ep in epochs])
    val_loss = np.array([history['val'][ep]['loss'] for ep in epochs])

    loss_title = f'Training vs Validation Loss - {DATASET_NAME.upper()}'
    loss_save = os.path.join(plots_path, 'tr_vs_val.png')
    plot_loss_curves(epochs, tr_loss, val_loss,
                     title=loss_title, save_path=loss_save,
                     log_scale_auto=False)
    print(f"Training vs Validation diagram saved to: {loss_save}")

    # ---------- PLOT 2: Learning Rate Schedule ----------
    lr_values = []
    for ep in epochs:
        lr = history['train'][ep].get('lr', None)
        if lr is None:
            print(f"Warning: No learning rate found for epoch {ep}. LR plot will be incomplete.")
            lr_values.append(np.nan)
        else:
            lr_values.append(lr)
    mask = ~np.isnan(lr_values)
    if np.any(mask):
        lr_save = os.path.join(plots_path, 'lr_schedule.png')
        plot_lr_schedule(epochs[mask], np.array(lr_values)[mask],
                         title=f'Learning Rate Schedule - {DATASET_NAME.upper()}',
                         save_path=lr_save)
        print(f"Learning rate schedule saved to: {lr_save}")
    else:
        print("No learning rate data found; skipping LR plot.")

    # ---------- PLOT 3: Per‑metric curves over epochs ----------
    # Common metric names to look for in train/val histories
    common_metrics = ['Accuracy', 'F1Score', 'Precision', 'Recall', 'MSE', 'MAE', 'AUROC']
    # Also include any other scalar metric found in test_metrics
    all_possible = set(common_metrics) | set(test_metrics.keys()) # exclude loss since it's already plotted
    all_possible.discard('loss')  # loss already plotted
    # Filter to metrics that exist in both train and val histories
    for metric in all_possible:
        # Check if metric is present in train and val for at least one epoch
        if (metric in history['train'].get(epochs[0], {}) and
                metric in history['val'].get(epochs[0], {})):
            # Extract values for all epochs
            tr_vals = np.array([history['train'][ep].get(metric, np.nan) for ep in epochs])
            val_vals = np.array([history['val'][ep].get(metric, np.nan) for ep in epochs])
            # Remove epochs where either is missing
            valid = ~(np.isnan(tr_vals) | np.isnan(val_vals))
            if np.sum(valid) > 1:  # need at least two points
                metric_save = os.path.join(plots_path, f'{metric}_curve.png')
                # Choose a colour from a palette
                colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink']
                color = colors[list(all_possible).index(metric) % len(colors)]
                plot_metric_curves(epochs[valid], tr_vals[valid], val_vals[valid],
                                   metric_name=metric,
                                   color=color,
                                   title=f'{metric} - {DATASET_NAME.upper()}',
                                   save_path=metric_save)
                print(f"{metric} curve saved to: {metric_save}")

    # ---------- PLOT 4: Bar chart of test metrics (for chosen epoch) ----------
    test_bar_save = os.path.join(plots_path, 'test_metrics_bar.png')
    # Exclude non-scalar entries automatically handled inside function
    plot_test_metrics_bar(test_metrics,
                          color='skyblue',
                          title=f'Test Set Metrics - {DATASET_NAME.upper()} (Epoch {use_epoch})',
                          save_path=test_bar_save)
    print(f"Test metrics bar chart saved to: {test_bar_save}")

    # ---------- PLOT 5: Precision‑Recall Curve (if available) ----------
    if 'PrecisionRecallCurve' in test_metrics:
        import torch
        from rbfkan_utils.config import instantiate

        pr_curve = instantiate(train_config['eval_criteria'], 'PrecisionRecallCurve')
        
        plt_args = [torch.tensor(_).float() for _ in test_metrics['PrecisionRecallCurve']]
        fig, ax = pr_curve.plot(curve = plt_args, score=True)
            
        save_path = os.path.join(plots_path, 'precision_recall_curve.png')
        plt.savefig(save_path)
        plt.close('all')
        print(f"Precision-Recall Curve saved to: {save_path}")

    # ---------- PLOT 6: Confusion Matrices & Regression plots ----------
    # Use the CSV files for the chosen epoch (use_epoch)
    rslt_path = os.path.join(args.test_dir, 'rslt')
    epoch_str = str(use_epoch)

    gt_df = pd.read_csv(os.path.join(rslt_path, 'ground_truth.csv'), index_col='Index')
    pr_df = pd.read_csv(os.path.join(rslt_path, f'{epoch_str}.csv'), index_col='Index')

    categories = create_labels(save=False)
    os.makedirs(os.path.join(plots_path, epoch_str), exist_ok=True)

    # Extract Confusion Matrices for each set of categories
    categorical_cols = []
    for category, types in categories.items():
        class_names = list(types.keys())
        search_prefix = "Label"
        gt_cols = [col for col in gt_df.columns if col == search_prefix or col.startswith(f'{search_prefix}_Is_')]
        pr_cols = [col for col in pr_df.columns if col == search_prefix or col.startswith(f'{search_prefix}_Is_')]
        categorical_cols.extend(gt_cols)
        categorical_cols.extend(pr_cols)
        
        gt_slice = gt_df[gt_cols].copy()
        pr_slice = pr_df[pr_cols].copy()
        
        if len(gt_cols) == 1:
            idx_to_class = {val: key for key, val in types.items()}
            gt_type = gt_slice[gt_cols[0]].map(idx_to_class)
        else:
            gt_slice.columns = [col[col.find('_Is_') + 4:] for col in gt_slice.columns]
            if train_config['task'] == 'multiclass':
                gt_type = gt_slice.aggregate('argmax', axis=1).map(dict(enumerate(gt_slice.columns)))
            else:
                gt_type = gt_slice
                
        if len(pr_cols) == 1:
            idx_to_class = {val: key for key, val in types.items()}
            pr_type = pr_slice[pr_cols[0]].map(idx_to_class)
        else:
            pr_slice.columns = [col[col.find('_Is_')+4:] for col in pr_slice.columns]
            if train_config['task'] == 'multiclass':
                pr_type = pr_slice.apply(np.argmax, axis=1).map(dict(enumerate(pr_slice.columns)))
            else:
                pr_type = pr_slice
                
        cm = confusion_matrix(gt_type.values, pr_type.values, labels=class_names)
        print(f"Confusion matrix \n{cm}")
        
        save_path = os.path.join(plots_path, epoch_str, f'cm_{category}.png')
        plot_confusion_matrix(
            cm, 
            class_names, 
            normalize = False, 
            title     = f'Confusion Matrix : {category} - {DATASET_NAME.upper()}',
            save_path = save_path
        )
        print(f"{category} Confusion matrix saved to: {save_path}")
        plt.close('all')

    # Plot Regression-type Columns
    idx = gt_df.index.values
    for col in gt_df.columns[np.isin(gt_df.columns, categorical_cols, invert=True)]:
        plt.plot(idx, gt_df[col], pr_df[col])

        plt.title(f'{col} - {DATASET_NAME.upper()}')
        plt.legend(['Ground Truth', 'Prediction'])
        plt.xlabel('Index')
        plt.ylabel(col)

        save_path = os.path.join(plots_path, epoch_str, f'{col}.png')
        plt.savefig(save_path)
        print(f"{col} Diagram saved to: {save_path}")
        plt.close('all')