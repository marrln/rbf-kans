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
    from rbfkan_utils.utils.plotter import plot_confusion_matrix
    from rbfkan_utils.config import *

    # Check configuration file validity
    train_config = load_config(args.train_config, locals=get_locals())
    model_config = load_config(args.model_config, locals=get_locals())

    # Determine which checkpoint to load for history
    model_dir = os.path.join(args.test_dir, 'models')
    if args.epoch == 'best':
        checkpoint_path = os.path.join(model_dir, 'best.pt')
    else:
        checkpoint_path = os.path.join(model_dir, 'last.pt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading history from {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, load_rng=False)
    history = checkpoint['history']   # contains 'train' and 'val' keys

    # The script expects history['test'][args.epoch]. 
    # If 'test' key is missing, fall back to validation metrics (with a warning)
    if 'test' not in history:
        print("Warning: No 'test' key found in history. Using validation metrics instead.")
        if args.epoch == 'best':
            # Find best validation loss epoch
            best_val_loss = float('inf')
            best_epoch = None
            for ep, metrics in history['val'].items():
                if metrics['loss'] < best_val_loss:
                    best_val_loss = metrics['loss']
                    best_epoch = ep
            test_metrics = history['val'][best_epoch]
            print(f"Using validation metrics from epoch {best_epoch} (best validation loss: {best_val_loss:.5f})")
        else:
            epoch_int = int(args.epoch)
            if epoch_int in history['val']:
                test_metrics = history['val'][epoch_int]
            else:
                raise KeyError(f"Epoch {args.epoch} not found in validation history. Available epochs: {sorted(history['val'].keys())}")
    else:
        # Original behaviour: get test metrics for the requested epoch
        test = history['test'].get(args.epoch, None)
        if test is None:
            raise KeyError(f"Epoch '{args.epoch}' not found in test history. Available: {list(history['test'].keys())}")
        test_metrics = test
    
    # Print basic statistics
    print(f'Loss for epoch "{args.epoch}": {test_metrics["loss"]}')
    
    for key in ['Accuracy', 'F1Score', 'Precision', 'Recall', 'MSE', 'MAE', 'AUROC']:
        if key in test_metrics.keys():
            print(f'{key} for epoch "{args.epoch}": {test_metrics[key]}')

    # Extract result statistics
    plots_path = os.path.join(args.test_dir, 'plot')
    os.makedirs(plots_path, exist_ok=True)

    # Training vs Validation Loss
    epochs = np.asarray(list(history['train'].keys()), dtype=int)
    tr_loss = np.array([history['train'][ep]['loss'] for ep in epochs])
    val_loss = np.array([history['val'][ep]['loss'] for ep in epochs])

    plt.plot(epochs, tr_loss, val_loss)
    if tr_loss.max()/tr_loss.min() > 10 or val_loss.max()/val_loss.min() > 10:
        plt.yscale('log')

    plt.title(f'Training vs Validation Loss - {DATASET_NAME.upper()}')
    plt.legend(['training','validation'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    save_path = os.path.join(plots_path, 'tr_vs_val.png')
    plt.savefig(save_path)
    plt.close('all')
    print(f"Training vs Validation diagram saved to: {save_path}")

    # Extract learning rates from training history (key 'lr' stored per epoch)
    lr_values = []
    for ep in epochs:
        lr = history['train'][ep].get('lr', None)
        if lr is None:
            print(f"Warning: No learning rate found for epoch {ep}. LR plot will be incomplete.")
            lr_values.append(np.nan)
        else:
            lr_values.append(lr)
    
    # Filter out epochs where lr is missing (if any)
    mask = ~np.isnan(lr_values)
    if np.any(mask):
        plt.figure()
        plt.plot(epochs[mask], np.array(lr_values)[mask], marker='.', linestyle='-', color='green')
        plt.title(f'Learning Rate Schedule - {DATASET_NAME.upper()}')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')  # LR typically changes on log scale
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        save_path_lr = os.path.join(plots_path, 'lr_schedule.png')
        plt.savefig(save_path_lr)
        plt.close('all')
        print(f"Learning rate schedule saved to: {save_path_lr}")
    else:
        print("No learning rate data found in history; skipping LR plot.")
    
    if 'PrecisionRecallCurve' in test_metrics.keys():
        import torch
        import torchmetrics
        from rbfkan_utils.config import *
        
        pr_curve = instantiate(train_config['eval_criteria'], 'PrecisionRecallCurve')
        
        plt_args = [torch.tensor(_).float() for _ in test_metrics['PrecisionRecallCurve']]
        fig, ax = pr_curve.plot(curve = plt_args, score=True)
            
        save_path = os.path.join(plots_path, 'precision_recall_curve.png')
        plt.savefig(save_path)
        plt.close('all')
        print(f"Precision-Recall Curve saved to: {save_path}")
        
    # Read ground truth and predicted values
    rslt_path = os.path.join(args.test_dir, 'rslt')

    gt_df = pd.read_csv(os.path.join(rslt_path, 'ground_truth.csv'), index_col='Index')
    pr_df = pd.read_csv(os.path.join(rslt_path, f'{args.epoch}.csv'), index_col='Index')

    # Read Categories - Works for both CIFAR-10 and CIFAR-100
    categories = create_labels(save=False)
    os.makedirs(os.path.join(plots_path, args.epoch), exist_ok=True)

    # Extract Confusion Matrices for each set of categories
    categorical_cols = []
    for category, types in categories.items():
        class_names = list(types.keys())
        
        # Find columns of the specified category
        # For CIFAR datasets, columns use "Label" as prefix
        search_prefix = "Label"
        gt_cols = [col for col in gt_df.columns if col == search_prefix or col.startswith(f'{search_prefix}_Is_')]
        pr_cols = [col for col in pr_df.columns if col == search_prefix or col.startswith(f'{search_prefix}_Is_')]
        categorical_cols.extend(gt_cols)
        categorical_cols.extend(pr_cols)
        
        # Get DataFrame slices
        gt_slice = gt_df[gt_cols].copy()
        pr_slice = pr_df[pr_cols].copy()
        
        if len(gt_cols) == 1:
            # Single column with class indices - map to class names
            idx_to_class = {val: key for key, val in types.items()}
            gt_type = gt_slice[gt_cols[0]].map(idx_to_class)
        else:
            # Multiple one-hot encoded columns
            # Fix DataFrames
            gt_slice.columns = [col[col.find('_Is_')+4:] for col in gt_slice.columns]
            
            # Find probabilities of the unknown class
            if train_config['task'] == 'multiclass':
                gt_type = gt_slice.aggregate('argmax', axis=1).map(dict(enumerate(gt_slice.columns)))
            else:
                gt_type = gt_slice
                
        if len(pr_cols) == 1:
            # Single column with class indices - map to class names
            idx_to_class = {val: key for key, val in types.items()}
            pr_type = pr_slice[pr_cols[0]].map(idx_to_class)
        else:
            # Multiple probability columns - take argmax
            # Fix DataFrames
            pr_slice.columns = [col[col.find('_Is_')+4:] for col in pr_slice.columns]
            
            # Find probabilities of the unknown class
            if train_config['task'] == 'multiclass':
                pr_type = pr_slice.apply(np.argmax, axis=1).map(dict(enumerate(pr_slice.columns)))
            else:
                pr_type = pr_slice
                
        cm = confusion_matrix(gt_type.values, pr_type.values, labels=class_names)
        save_path = os.path.join(plots_path, args.epoch, f'cm_{category}.png')
        plot_confusion_matrix(
            cm, 
            class_names, 
            normalize = True, 
            figsize   = (0.5*len(pr_slice.columns), 0.5*len(pr_slice.columns)+2),
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
        plt.legend(['Ground Truth','Prediction'])
        plt.xlabel('Index')
        plt.ylabel(col)

        save_path = os.path.join(plots_path, args.epoch, f'{col}.png')
        plt.savefig(save_path)
        print(f"{col} Diagram saved to: {save_path}")
        plt.close('all')