#!/usr/bin/env python3   
if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser
    
    THIS_DIR = os.path.dirname(__file__)
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(description=f'Generalized training configuration script.')
    
    # Dataset override
    parser.add_argument('--dataset', dest='dataset', type=str, required=True, help=f'Dataset to use (required)')
    parser.add_argument('-d', '--dest-top-directory', dest='dest_top_dir', default=None, help='The directory to be used as a top directory for training, if None uses dataset-specific default directory.')
    
    # Model architecture flags
    parser.add_argument('--seed', dest='seed', type=int, default=42)
    parser.add_argument('--layers', '--hidden-layers', dest='hidden_layers', action='extend', nargs="+")
    parser.add_argument('--num-grids', dest='num_grids', action='extend', nargs="+")
    parser.add_argument('--grid-min', dest='grid_min', action='extend', nargs="+")
    parser.add_argument('--grid-max', dest='grid_max', action='extend', nargs="+")
    parser.add_argument('--scale','--inv_denominator', dest='scale', action='extend', nargs="+")
    parser.add_argument('--mode', dest='mode', type=str, default='RSWAFF')
    parser.add_argument('--residual', dest='residual', action='store_true')
    parser.add_argument('--dynamic', dest='dynamic', action='store_true')
    parser.add_argument('--use-v2', dest='use_v2', action='store_true', help='Use V2 RBFKAN layers')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false', help='Disable layer norm')
    parser.add_argument('--no-normalize-rbf', dest='normalize_rbf', action='store_false', default=True, help='Disable RBF output normalization')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.5)
    parser.add_argument('--dropout-linear', dest='dropout_linear', type=float, default=None, help='Dropout after linear layer (default: same as --dropout)')
    parser.add_argument('--with-logits', dest='with_logits', action='store_true', help='Use logits output (no final activation) and corresponding loss')
    
    # Training hyperparameters
    parser.add_argument('--patience', dest='patience', default=10)
    parser.add_argument('--epochs', dest='epochs', default=500)
    parser.add_argument('--batch', '--batch-size', dest='batch_size', type=int, default=16)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='Adam')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
    parser.add_argument('--lr-factor', dest='lr_factor', type=float, default=0.5)
    parser.add_argument('--lr-patience', dest='lr_patience', type=int, default=8)
    parser.add_argument('--resize', dest='resize', type=int, nargs=2, metavar=('H', 'W'), help="Resize images to HxW (e.g., --resize 16 16)")
    parser.add_argument('--hash', action='store_true', dest='hash', help="Return the corresponding hash value instead of the full directory")
    parser.add_argument('--export', action='store_true', dest='export', help="Save the model configuration")
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')

    args = parser.parse_args()
    
    # Add dataset folder to path
    dataset_path = os.path.join(THIS_DIR, args.dataset)
    if dataset_path not in sys.path:
        sys.path.insert(0, dataset_path)
    from prepare_dataset import DATASET_DIR, get_dataset_info, get_class_names
    
    # Set default after DATASET_DIR is available
    if args.dest_top_dir is None:
        args.dest_top_dir = os.path.join(DATASET_DIR, 'train')
    
    import torch
    import torchmetrics
    import hashlib
    from collections import OrderedDict

    from rbfkan_utils.config import *
    from rbfkan_utils.metrics import *
    from rbfkan_utils.utils.callbacks import *
    from rbfkan_utils.models import LambdaModule, RBFKAN
    from rbfkan_utils.utils import uses_momentum
    from rbfkan_utils.utils.callbacks import UpdatableFloat, GatherStatistics

    # Get dataset info
    dataset_info = get_dataset_info()
    num_classes = dataset_info['num_classes']
    is_multiclass = dataset_info['task'] == 'multiclass'
    
    # Calculate input features based on image size
    if args.resize is not None:
        h, w = args.resize
        num_features = h * w * 3
    else:
        h, w = dataset_info['input_shape'][:2]
        num_features = h * w * 3
    
    features = [f'pixel_{i}' for i in range(num_features)]
    labels = get_class_names()
    
    # --- Model configuration ---
    model_config = {}
    model_config['input'] = features
    model_config['output'] = labels
    model_config['outputs_logits'] = args.with_logits
    model_config['flatten'] = True
    
    # Determine final activation (only if not outputting logits)
    if is_multiclass:
        if args.with_logits:
            final_act = torch.nn.Identity
        else:
            final_act = torch.nn.Softmax
    else:
        if args.with_logits:
            final_act = torch.nn.Identity
        else:
            final_act = torch.nn.Sigmoid
    
    # Build the KAN model with RBFKAN
    kan_layer = object_to_config(
        RBFKAN,
        hidden_layers     = [
            len(model_config['input']),
            *([] if args.hidden_layers is None else args.hidden_layers),
            (len(model_config['output']) if len(model_config['output']) != 2 else 1),
        ],
        num_grids         = args.num_grids,
        grid_min          = args.grid_min,
        grid_max          = args.grid_max,
        inv_denominator   = args.scale,
        mode              = args.mode,
        residual          = args.residual,
        dynamic           = args.dynamic,
        use_v2            = args.use_v2,
        normalize         = args.normalize,
        normalize_rbf     = args.normalize_rbf,
        dropout_rate      = object_to_config(UpdatableFloat, 0),
        dropout_linear    = args.dropout_linear if args.dropout_linear is not None else args.dropout,
    )
    
    if final_act != torch.nn.Identity:
        model_layers = OrderedDict([
            ('kan', kan_layer),
            ('actf', object_to_config(final_act))
        ])
    else:
        model_layers = OrderedDict([('kan', kan_layer)])
    
    model_config.update(
        object_to_config(
            torch.nn.Sequential,
            object_to_config(OrderedDict, model_layers),
            target_name='model',
        )
    )
    
    # --- Training configuration ---
    train_config = get_default_training_config()
    train_config['task'] = 'multiclass' if is_multiclass else 'multilabel'
    train_config['sampler'] = ['Label']
    train_config['splits'] = [0.8, 0.2]
    
    # Loss function
    if is_multiclass:
        if args.with_logits:
            train_config.update(object_to_config(
                torch.nn.CrossEntropyLoss,
                label_smoothing=0.0,
                target_name='criterion'
            ))
        else:
            print("Warning: Without --with-logits, forcing --with-logits for multiclass.")
            args.with_logits = True
            model_config['outputs_logits'] = True
            train_config.update(object_to_config(
                torch.nn.CrossEntropyLoss,
                target_name='criterion'
            ))
    else:
        if args.with_logits:
            train_config.update(object_to_config(
                torch.nn.BCEWithLogitsLoss,
                target_name='criterion'
            ))
        else:
            train_config.update(object_to_config(
                torch.nn.BCELoss,
                target_name='criterion'
            ))
    
    # Evaluation criteria
    if is_multiclass:
        targ_apply = 'lambda target: target.to(torch.int64).squeeze(-1)'
        train_config['eval_criteria'] = {
            **object_to_config(
                ProcessAndApplyMetric,
                object_to_config(torchmetrics.Accuracy, task='multiclass', num_classes=num_classes),
                targ_apply=targ_apply,
                target_name='Accuracy',
            ),
            **object_to_config(
                ProcessAndApplyMetric,
                object_to_config(torchmetrics.F1Score, task='multiclass', num_classes=num_classes),
                targ_apply=targ_apply,
                target_name='F1Score',
            ),
            **object_to_config(
                ProcessAndApplyMetric,
                object_to_config(torchmetrics.AUROC, task='multiclass', num_classes=num_classes),
                targ_apply=targ_apply,
                target_name='AUROC',
            ),
        }
    else:
        train_config['eval_criteria'] = {
            **object_to_config(torchmetrics.Accuracy, task='binary', target_name='Accuracy'),
            **object_to_config(torchmetrics.F1Score, task='binary', target_name='F1Score'),
            **object_to_config(
                ProcessAndApplyMetric,
                object_to_config(torchmetrics.PrecisionRecallCurve, task='binary', thresholds=100, normalization=False),
                **object_to_config(LambdaModule, 'lambda target: target.to(torch.int8)', target_name='targ_apply'),
                target_name='PrecisionRecallCurve',
            ),
            **object_to_config(torchmetrics.AUROC, task='binary', thresholds=100, target_name='AUROC'),
        }
    
    # Hyperparameters
    train_config['epochs'] = args.epochs
    train_config['patience'] = args.patience
    train_config['lr'] = args.lr
    train_config['seed'] = args.seed
    train_config['batch_size'] = args.batch_size
    train_config['probability'] = 0.25
    if args.resize is not None:
        train_config['resize'] = tuple(args.resize)
    
    # Optimizer
    train_config.update(
        object_to_config(
            getattr(torch.optim, args.optimizer),
            target_name='optimizer',
            weight_decay=args.weight_decay,
            amsgrad=True,
            **({'momentum': args.momentum} if uses_momentum(args.optimizer) else {})
        )
    )
    # Scheduler
    train_config.update(
        object_to_config(
            torch.optim.lr_scheduler.ReduceLROnPlateau,
            factor=args.lr_factor,
            patience=args.lr_patience,
            target_name='scheduler'
        )
    )
    
    # Callbacks for statistics gathering
    train_config['callbacks_arguments'].update({
        **object_to_config(
            GatherStatistics,
            input_cols=model_config['input'],
            output_cols=model_config['output'],
            task=train_config['task'],
            export_path=os.path.join(DATASET_DIR, 'tr_statistics.csv'),
            target_name='train_gatherer',
        ),
        **object_to_config(
            GatherStatistics,
            input_cols=model_config['input'],
            output_cols=model_config['output'],
            task=train_config['task'],
            export_path=os.path.join(DATASET_DIR, 'val_statistics.csv'),
            overwrite=1,
            target_name='val_gatherer',
        ),
    })
    
    # Dropout schedule – **removed unsqueeze callbacks** (caused CrossEntropyLoss error)
    train_config['callbacks']['train_iter_start'].extend([
        object_to_config(
            f'lambda *args, model=None, iteration=0, epoch=0, epochs=1, dataloader=None, **kwargs : model._modules["kan"].dropout_rate.set('
            f'{args.dropout} * torch.sigmoid( torch.tensor( ((epoch + (iteration / len(dataloader)) - {int(args.epochs) / 2}) / {int(args.epochs) / 4}) )).item()'
            ')',
        ),
        # Removed: object_to_config('lambda *args, target=None, _f=apply_to_tensor, **kwargs: _f(target, "unsqueeze_", -1)'),
        object_to_config('lambda *args, train_gatherer=None, **kwargs: train_gatherer(*args,**kwargs)'),
    ])
    train_config['callbacks']['train_end'].extend([
        object_to_config('lambda *args, train_gatherer=None, **kwargs: train_gatherer.finalize(*args,**kwargs)'),
    ])
    train_config['callbacks']['eval_iter_start'].extend([
        # Removed: object_to_config('lambda *args, target=None, _f=apply_to_tensor, **kwargs: _f(target, "unsqueeze_", -1)'),
        object_to_config('lambda *args, val_gatherer=None, **kwargs: val_gatherer(*args,**kwargs)'),
    ])
    train_config['callbacks']['eval_metrics_start'].extend([
        object_to_config('lambda *args, val_gatherer=None, **kwargs: val_gatherer.finalize(*args,**kwargs)'),
    ])
    
    # Helper to build hash and directory
    def build_test_dir(train_config, model_config, top_dir=None, test_version=None):
        pdir = os.path.join(
            '_'.join(['_'.join([key, str(val)]) for key, val in train_config.items()]),
            '_'.join(['_'.join([key, str(val)]) for key, val in model_config.items()]),
        )
        hashed = hashlib.sha1(pdir.encode()).hexdigest()
        pdir = hashed
        if top_dir is not None:
            pdir = os.path.join(top_dir, pdir)
        if test_version is not None:
            pdir = os.path.join(pdir, '_'.join(['test', test_version]))
        return pdir, hashed

    pdir, hashed = build_test_dir(train_config, model_config, top_dir=args.dest_top_dir, test_version=args.test_version)

    if not args.export:
        print(f'Test directory : {pdir}')

    if args.hash:
        print(hashed)

    if args.export:
        path = os.path.join(pdir, 'config', 'train')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        path = save_config(train_config, path)
        
        path = os.path.join(pdir, 'config', 'model')
        path = save_config(model_config, path)
        
    if not args.hash:
        print(os.path.dirname(pdir))