#!/usr/bin/env python3   
import sys, os
from argparse import ArgumentParser
import subprocess

THIS_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

if __name__ == '__main__' :
    parser = ArgumentParser(
        description=f'Compare models across different configurations.'
    )
    parser.add_argument('-d', '--test-dir', dest='test_dir', default=None, help='The directory to be used as a top directory for training. Defaults to the train folder of the dataset.')
    parser.add_argument('-l', '--limit', dest='limit', type=int, default=-1, help='Limit the number of characters of the hashes shown in the figures.')
    parser.add_argument('--dataset', dest='dataset', type=str, help=f'Dataset to use (required)', required=True)
    parser.add_argument('--no-pbar', action='store_true', dest='no_pbar', help='Suppress progress bars when running tests.')

    args = parser.parse_args()

    # Add dataset folder to path
    dataset_path = os.path.join(THIS_DIR, args.dataset)
    if dataset_path not in sys.path:
        sys.path.insert(0, dataset_path)
    from prepare_dataset import DATASET_DIR, DATASET_NAME # pyright: ignore[reportMissingImports]
    
    # Set default after DATASET_DIR is available
    if args.test_dir is None:
        args.test_dir = os.path.join(DATASET_DIR, 'train')
    
    # Check argument validity
    if os.path.isdir(args.test_dir) or not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir, exist_ok=True)
    else:
        raise ValueError(f'Destination folder is not a directory; got "{os.path.splitext(args.test_dir)[-1]}"')

    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    from rbfkan_utils.utils import load_dict

    # ------------------------------------------------------------------
    # Helper function: run test.py for a given config if needed
    # ------------------------------------------------------------------
    def run_test_if_needed(config_dir, hash_val, version_folder, dataset, test_root, no_pbar=False):
        """
        Check if test metrics exist in checkpoint; if not, call test.py.
        Returns the history dictionary after ensuring test metrics are present.
        """
        checkpoint_path = os.path.join(config_dir, 'models', 'best.pt')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint to see if test metrics exist
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        history = checkpoint.get('history', {})
        if 'test' in history and history['test']:
            print(f"  Test metrics already present for {hash_val}/{version_folder}")
            return history
        
        print(f"  No test metrics found for {hash_val}/{version_folder} – running test.py...")
        
        # Extract the numeric version from the folder name (e.g., "test_0" -> "0")
        if version_folder.startswith('test_'):
            test_version = version_folder[5:]  # remove "test_"
        else:
            test_version = version_folder   # fallback
        
        # Build command to call test.py
        test_script = os.path.join(THIS_DIR, 'test_model.py')
        cmd = [
            sys.executable, test_script,
            '--dataset', dataset,
            '--hash', hash_val,
            '--test-version', test_version,
            '--epoch', 'best',
            '--test-dir', test_root,
        ]
        if no_pbar:
            cmd.append('--no-pbar')
        
        # Run the test script
        try:
            subprocess.run(cmd, check=True, capture_output=False)  # show output live
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Test script failed for {hash_val}/{version_folder}: {e}")
        
        # Reload checkpoint to get updated history
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint.get('history', {})
    # ------------------------------------------------------------------

    tests = []
    test_root = args.test_dir
    # Walk directories and look for the best checkpoint
    for root, dirs, files in os.walk(test_root):
        # Check if this directory is a 'models' folder containing best.pt
        if os.path.basename(root) == 'models' and 'best.pt' in files:
            # Parent directory is test_root/<hash>/<version>
            parent_rel = os.path.relpath(os.path.dirname(root), test_root)
            parts = parent_rel.split(os.sep)
            if len(parts) >= 2:
                # Use the last two components as Configuration and Version
                tests.append([parts[-2], parts[-1]])
            elif len(parts) == 1:
                # If only one level (e.g., hash/version?), treat as hash and default version
                tests.append([parts[0], 'default'])
    
    tests = pd.DataFrame(data=tests, columns=['Configuration','Version']).sort_values(['Configuration','Version']).reset_index(drop=True)
    
    # --- Define all hyperparameter columns (model + training) ---
    # Model architecture hyperparams (from model config)
    model_hyperparams = [
        'num_layers', 'hidden_layers', 'mode', 'grids', 'grid_min', 'grid_max', 
        'scale', 'residual', 'dynamic', 'use_v2', 'normalize', 'normalize_rbf',
        'dropout_rate', 'dropout_linear', 'use_logits'
    ]
    # Training hyperparams (from train config)
    train_hyperparams = [
        'seed', 'batch_size', 'lr', 'optimizer', 'weight_decay', 'momentum',
        'clip_limit', 'lr_factor', 'lr_patience', 'resize', 'probability',
        'patience', 'epochs', 'dynamic_dropout'
    ]
    all_hyperparams = model_hyperparams + train_hyperparams
    for col in all_hyperparams:
        tests[col] = None
    
    for idx in tests.index:
        hash_val = tests.loc[idx, 'Configuration']
        version_folder = tests.loc[idx, 'Version']
        config_dir = os.path.join(test_root, hash_val, version_folder)
        
        # Ensure test metrics exist; if not, run test.py
        try:
            history = run_test_if_needed(
                config_dir=config_dir,
                hash_val=hash_val,
                version_folder=version_folder,
                dataset=args.dataset,
                test_root=test_root,
                no_pbar=args.no_pbar
            )
        except Exception as e:
            print(f'Error processing {config_dir}: {e}')
            tests.drop(idx, axis=0, inplace=True)
            continue
        
        # ----- Load and parse model config (architecture) -----
        model_config_path = os.path.join(config_dir, 'config', 'model.json')
        train_config_path = os.path.join(config_dir, 'config', 'train.json')
        
        if os.path.exists(model_config_path):
            model_config = load_dict(model_config_path.replace('.json', ''))
            try:
                # Extract model parameters (same as original)
                model_args = model_config.get('model_args', [{}])[0]
                kan_config = None
                if '_args' in model_args and len(model_args['_args']) > 0:
                    for outer_item in model_args['_args']:
                        if isinstance(outer_item, list):
                            for middle_item in outer_item:
                                if isinstance(middle_item, list) and len(middle_item) >= 2:
                                    if middle_item[0] == 'kan' and isinstance(middle_item[1], dict):
                                        kan_config = middle_item[1].get('_kwargs', {})
                                        break
                        if kan_config:
                            break
                
                if kan_config:
                    hidden_layers = kan_config.get('hidden_layers', [])
                    if hidden_layers:
                        tests.loc[idx, 'num_layers'] = len(hidden_layers) - 1
                        tests.loc[idx, 'hidden_layers'] = str(hidden_layers)
                    
                    tests.loc[idx, 'mode'] = kan_config.get('mode', 'N/A')
                    grids = kan_config.get('num_grids', [])
                    tests.loc[idx, 'grids'] = str(grids) if grids else 'N/A'
                    tests.loc[idx, 'grid_min'] = str(kan_config.get('grid_min', [])) or 'N/A'
                    tests.loc[idx, 'grid_max'] = str(kan_config.get('grid_max', [])) or 'N/A'
                    tests.loc[idx, 'scale'] = str(kan_config.get('inv_denominator', [])) or 'N/A'
                    tests.loc[idx, 'residual'] = str(kan_config.get('residual', 'N/A'))
                    tests.loc[idx, 'dynamic'] = str(kan_config.get('dynamic', 'N/A'))
                    tests.loc[idx, 'use_v2'] = str(kan_config.get('use_v2', 'N/A'))
                    tests.loc[idx, 'normalize'] = str(kan_config.get('normalize', 'N/A'))
                    tests.loc[idx, 'normalize_rbf'] = str(kan_config.get('normalize_rbf', 'N/A'))
                    
                    dropout_rate_val = kan_config.get('dropout_rate', 'N/A')
                    if isinstance(dropout_rate_val, dict) and '_args' in dropout_rate_val:
                        args_list = dropout_rate_val.get('_args', [])
                        if args_list:
                            dropout_rate_val = str(args_list[0])
                    tests.loc[idx, 'dropout_rate'] = str(dropout_rate_val)
                    
                    tests.loc[idx, 'dropout_linear'] = str(kan_config.get('dropout_linear', 'N/A'))
                    
                    # dynamic_dropout is inferred from whether dropout_rate is an UpdatableFloat or a float
                    # We already have the value; we'll parse from train config later.
            except Exception as e:
                print(f'Warning: Could not parse model config for "{config_dir}": {e}')
        
        # ----- Load and parse train config (training hyperparams) -----  
        if os.path.exists(train_config_path):
            train_config = load_dict(train_config_path.replace('.json', ''))
            # use_logits
            tests.loc[idx, 'use_logits'] = 'Yes' if 'BCEWithLogitsLoss' in str(train_config) else 'No' # This is wrong!!!!!!!!!!!!!!!!!!
            
            # Extract optimizer info
            opt = train_config.get('optimizer')
            if isinstance(opt, dict):
                opt_name = opt.get('target_name', 'N/A')
                opt_kwargs = opt.get('_kwargs', {})
                weight_decay = opt_kwargs.get('weight_decay', 'N/A')
                momentum = opt_kwargs.get('momentum', 'N/A')
            else:
                opt_name = str(opt) if opt is not None else 'N/A'
                weight_decay = 'N/A'
                momentum = 'N/A'
            
            # Extract scheduler info
            sched = train_config.get('scheduler')
            if isinstance(sched, dict):
                sched_kwargs = sched.get('_kwargs', {})
                lr_factor = sched_kwargs.get('factor', 'N/A')
                lr_patience = sched_kwargs.get('patience', 'N/A')
            else:
                lr_factor = 'N/A'
                lr_patience = 'N/A'
            
            # Extract other training parameters
            train_params = {
                'seed': train_config.get('seed', 'N/A'),
                'batch_size': train_config.get('batch_size', 'N/A'),
                'lr': train_config.get('lr', 'N/A'),
                'optimizer': opt_name,
                'weight_decay': weight_decay,
                'momentum': momentum,
                'clip_limit': train_config.get('clip_limit', 'N/A'),
                'lr_factor': lr_factor,
                'lr_patience': lr_patience,
                'resize': str(train_config.get('resize', 'N/A')),
                'probability': train_config.get('probability', 'N/A'),
                'patience': train_config.get('patience', 'N/A'),
                'epochs': train_config.get('epochs', 'N/A'),
                'dynamic_dropout': 'Yes' if train_config.get('callbacks', {}).get('train_iter_start', []) else 'No',
            }
            for key, val in train_params.items():
                tests.loc[idx, key] = val if val is not None else 'N/A'
        
        # Extract test metrics from history
        test_metrics = history.get('test', {}).get('best', None)
        if test_metrics is None:
            epochs = list(history.get('test', {}).keys())
            if epochs:
                test_metrics = history['test'][epochs[0]]
            else:
                print(f'Dropped "{config_dir}"; no test metrics found after attempted run.')
                tests.drop(idx, axis=0, inplace=True)
                continue
        
        for metric, value in test_metrics.items():
            if metric not in tests.columns:
                tests[metric] = float('NaN')
            tests.loc[idx, metric] = value
    
    # Continue with the rest of the original script (plotting, top 5, etc.)
    tests['Configuration'] = [_[:args.limit] for _ in tests['Configuration'].values]
    
    # Drop columns that are all NaN (but keep hyperparams even if some are NaN)
    # We'll drop only metric columns that are all NaN; hyperparams are kept.
    config_cols = ['Configuration', 'Version'] + all_hyperparams
    metric_cols = [col for col in tests.columns if col not in config_cols]
    for col in metric_cols:
        if tests[col].isna().all():
            tests.drop(columns=col, inplace=True)
    # Recompute metric_cols after dropping
    metric_cols = [col for col in tests.columns if col not in config_cols]
    
    # --- Print model configurations with all hyperparameters ---
    print("\n" + "="*80)
    print(f"MODEL CONFIGURATIONS - {DATASET_NAME.upper()}")
    print("="*80)
    for idx in tests.index:
        config_name = tests.loc[idx, 'Configuration']
        version = tests.loc[idx, 'Version']
        print(f"\n[{config_name} / {version}]")
        # Model hyperparams
        print(f"  Model:")
        print(f"    Layers: {tests.loc[idx, 'num_layers']} | Hidden: {tests.loc[idx, 'hidden_layers']} | Mode: {tests.loc[idx, 'mode']} | Grids: {tests.loc[idx, 'grids']}")
        print(f"    GridMin: {tests.loc[idx, 'grid_min']} | GridMax: {tests.loc[idx, 'grid_max']} | Scale: {tests.loc[idx, 'scale']} | Logits: {tests.loc[idx, 'use_logits']}")
        print(f"    Residual: {tests.loc[idx, 'residual']} | Dynamic: {tests.loc[idx, 'dynamic']} | UseV2: {tests.loc[idx, 'use_v2']} | Normalize: {tests.loc[idx, 'normalize']} | NormalizeRBF: {tests.loc[idx, 'normalize_rbf']}")
        print(f"    DropoutRate: {tests.loc[idx, 'dropout_rate']} | DropoutLinear: {tests.loc[idx, 'dropout_linear']}")
        # Training hyperparams
        print(f"  Training:")
        print(f"    Seed: {tests.loc[idx, 'seed']} | Batch: {tests.loc[idx, 'batch_size']} | LR: {tests.loc[idx, 'lr']} | Optimizer: {tests.loc[idx, 'optimizer']}")
        print(f"    WeightDecay: {tests.loc[idx, 'weight_decay']} | Momentum: {tests.loc[idx, 'momentum']} | ClipLimit: {tests.loc[idx, 'clip_limit']}")
        print(f"    LRFactor: {tests.loc[idx, 'lr_factor']} | LRPatience: {tests.loc[idx, 'lr_patience']} | Resize: {tests.loc[idx, 'resize']}")
        print(f"    Prob: {tests.loc[idx, 'probability']} | Patience: {tests.loc[idx, 'patience']} | Epochs: {tests.loc[idx, 'epochs']} | DynamicDropout: {tests.loc[idx, 'dynamic_dropout']}")
    
    print("\n" + "="*80)
    print(tests)
    print("="*80)
    
    plt_dir = os.path.join(args.test_dir, 'comparison')
    os.makedirs(plt_dir, exist_ok=True)
    
    # For plotting, we only use metric columns (exclude all hyperparams)
    metric_cols = [col for col in tests.columns if col not in config_cols]
    
    # Global compare
    if len(metric_cols) > 0:
        tests_g = tests.set_index(['Configuration', 'Version'])
        ax = tests_g.plot.barh(
            y        = metric_cols,
            subplots = True,
            legend   = False,
            sharex   = False,
            figsize  = (3 + len(metric_cols), 1.5+0.75*len(tests)),
        )
        plt.xticks(rotation = 45, fontsize=7) 
        [axi.set(xlim=(0.999*tests_g[col].min(), 1.001*tests_g[col].max())) for axi, col in zip(ax, metric_cols)]
        [axi.minorticks_on() for axi in ax]
        [axi.xaxis.grid(True, which='major') for axi in ax]
        [axi.xaxis.grid(True, which='minor', color='gray', linestyle=':', linewidth=0.5) for axi in ax]
        plt.tight_layout()
        plt.savefig(os.path.join(plt_dir,f'global.png'))
        plt.close('all')
        
    # Top 30 by F1Score
    if 'F1Score' in tests.columns and len(metric_cols) > 0:
        tests_top30 = tests.nlargest(30, 'F1Score')
        tests_top30_g = tests_top30.set_index(['Configuration', 'Version'])
        ax = tests_top30_g.plot.barh(
            y        = metric_cols,
            subplots = True,
            legend   = False,
            sharex   = False,
            figsize  = (3 + len(metric_cols), 1.5+0.75*len(tests_top30)),
        )
        plt.xticks(rotation = 45, fontsize=7) 
        [axi.set(xlim=(0.999*tests_top30_g[col].min(), 1.001*tests_top30_g[col].max())) for axi, col in zip(ax, metric_cols)]
        [axi.minorticks_on() for axi in ax]
        [axi.xaxis.grid(True, which='major') for axi in ax]
        [axi.xaxis.grid(True, which='minor', color='gray', linestyle=':', linewidth=0.5) for axi in ax]
        plt.tight_layout()
        plt.savefig(os.path.join(plt_dir,f'top30_f1score.png'))
        plt.close('all')
    
    # Print top 5 for key metrics (include all hyperparams in output)
    print("\n" + "="*80)
    print(f"TOP 5 MODELS BY METRIC - {DATASET_NAME.upper()}")
    print("="*80)
    
    key_metrics = ['Accuracy', 'F1Score', 'AUROC', 'Top3Accuracy', 'Top5Accuracy']
    for metric in key_metrics:
        if metric in tests.columns:
            print(f"\n{'='*80}")
            print(f"TOP 5 - {metric}")
            print('='*80)
            top5_cols = ['Configuration', 'Version', metric] + all_hyperparams
            top5 = tests.nlargest(5, metric)[top5_cols]
            for i, (idx, row) in enumerate(top5.iterrows(), 1):
                print(f"{i}. {row[metric]:.6f} - [{row['Configuration']} / {row['Version']}]")
                # Show all hyperparams in a compact format
                hyper_str = []
                for hp in all_hyperparams:
                    hyper_str.append(f"{hp}={row[hp]}")
                print("   " + " | ".join(hyper_str))
    
    print("\n" + "="*80)
    print("DETAILED METRIC SUMMARY")
    print("="*80)
    # Recompute tests_g (in case we modified metric_cols)
    if len(metric_cols) > 0:
        tests_g = tests.set_index(['Configuration', 'Version'])
        for col in metric_cols:
            print('-- Metric :', col)
            print(f'  -- Max : {tests_g[col].max() :.4f} --  {tests_g.index[tests_g[col].argmax()]}')
            print(f'  -- Min : {tests_g[col].min() :.4f} --  {tests_g.index[tests_g[col].argmin()]}')