#!/usr/bin/env python3

if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser
    from augmentations import Augmentor # pyright: ignore[reportMissingImports]

    THIS_DIR = os.path.dirname(__file__)
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(
        description='Testing script for Dataset.'
    )

    parser.add_argument('-d', '--test-dir', dest='test_dir', default=None, help='The directory to be used to load the model and save the results. If not specified, it will be set to the "train" folder of the dataset.')
    parser.add_argument('--dataset', dest='dataset', type=str, required=True, help='Dataset to use. Should be the name of a folder in the "dataset" directory.')
    parser.add_argument('--hash', dest='hash', type=str, help='The hash value of the configuration.', required=True)
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')
    parser.add_argument('--epoch', dest='epoch', type=str, default='best')
    parser.add_argument('--no-pbar', action='store_true', dest='no_pbar')

    args = parser.parse_args()
    
    # Add dataset folder to path
    dataset_path = os.path.join(THIS_DIR, args.dataset)
    if dataset_path not in sys.path:
        sys.path.insert(0, dataset_path)
    from prepare_dataset import get_dataset, DATASET_DIR # pyright: ignore[reportMissingImports]
    
    # Add test dir
    if args.test_dir is None:
        args.test_dir = os.path.join(DATASET_DIR, 'train')
    
    from custom_dataset import GenericDataset
    
    args.test_version = '_'.join(['test', args.test_version])

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
    import torch
    import albumentations as A
    from torch.utils.data import DataLoader
    from rbfkan_utils.utils import load_checkpoint, save_checkpoint, set_seed, apply_to_tensor
    from rbfkan_utils.config import *
    from rbfkan_utils.training import evaluate

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check configuration file validity
    train_config = load_config(args.train_config, locals=get_locals())
    model_config = load_config(args.model_config, locals=get_locals())
    set_seed(train_config['seed'])

    # Instantiate model, optimizer, and scheduler (same as in training)
    model     = instantiate(model_config, 'model')
    criterion = instantiate(train_config, 'criterion')
    optimizer = instantiate(train_config, 'optimizer', model.parameters(), lr=train_config['lr'])
    scheduler = instantiate(train_config, 'scheduler', optimizer)

    # Path to the checkpoint (full state)
    checkpoint_path = os.path.join(args.test_dir, 'models', f'{args.epoch}.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")
    # Load full checkpoint: model, optimizer, scheduler, history, etc.
    checkpoint = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        load_rng=False,
        device=device
    )
    history = checkpoint['history']   # contains 'train' and 'val' (and maybe previous 'test')

    # Instantiate callbacks and evaluation criteria
    callbacks = weak_instantiate_all(train_config['callbacks'])
    callbacks_arguments = weak_instantiate_all(train_config['callbacks_arguments'])
    eval_criteria = weak_instantiate_all(train_config['eval_criteria'])
    if 'loss' not in eval_criteria:
        eval_criteria['loss'] = instantiate(train_config, 'criterion')

    # Get test dataset
    data, labels = get_dataset('test')
    
    # Instantiate the augmentor
    augmentor = Augmentor(train_config)
     
    test_dataset = GenericDataset(
        data                    = data, 
        labels                  = labels,
        task                    = train_config['task'],
        return_key              = True,
        return_weights          = train_config['sample_weight'],
        preprocess_data         = augmentor.test,
        preprocess_targ         = None,
        flatten                 = model_config['flatten'],
    )
    test_loader = DataLoader(
        dataset         = test_dataset, 
        batch_size      = train_config['batch_size'],
        num_workers     = os.cpu_count(),
        pin_memory      = device == torch.device('cuda'),
    )

    print('-- Using dataset split :', train_config['splits'])
    print('-- Test Size:', len(test_loader.dataset))

    test_metrics = evaluate(
        model,
        eval_dataloader     = test_loader,
        criteria            = eval_criteria,
        keep_copy           = True,
        checkpoint_path     = checkpoint_path.replace('models', 'rslt'),
        epoch               = args.epoch,
        sample_weight       = train_config['sample_weight'],
        show_pbar           = not args.no_pbar,
        device              = device,
        callbacks           = callbacks,
        callbacks_arguments = {
            'epoch' : args.epoch,
            **callbacks_arguments,
        },
    )

    # Store test metrics in history
    if 'test' not in history:
        history['test'] = {}
    history['test'][args.epoch] = test_metrics

    # Save the updated checkpoint (preserving model, optimizer, scheduler, etc.)
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch=checkpoint['epoch'],
        best_loss=checkpoint['best_loss'],
        history=history,
        filepath=checkpoint_path,
        rng_state=checkpoint.get('rng_state', None)   # keep original RNG state if present
    )

    print(f"Updated checkpoint saved with test metrics for epoch {args.epoch}")

    # Separate ground truth and predicted values (same as original)
    rslt_path = os.path.join(args.test_dir, 'rslt')
    test_df = pd.read_csv(os.path.join(rslt_path, f'{args.epoch}.csv'), index_col='Index')

    gt_df = test_df[[col for col in test_df.columns if 'targ' in col]]
    pr_df = test_df[[col for col in test_df.columns if 'pred' in col]]

    if len(gt_df.columns) == 1: # OR train_config['task'] == 'multiclass'
        gt_df.columns = ['Label']
        gt_df = gt_df['Label'].astype(int)
        if len(pr_df.columns) == 1 and len(model_config['output']) == 2:
            if model_config.get('outputs_logits', False):
                pr_df = pd.DataFrame(
                    data=torch.sigmoid(torch.tensor(pr_df.values)).numpy(),
                    index=pr_df.index,
                    columns=[f'Label_Is_{model_config["output"][1]}']
                )
            else:
                pr_df.columns = [f'Label_Is_{model_config["output"][1]}']
            
        elif len(pr_df.columns) > 1:
            if model_config.get('outputs_logits', False):
                pr_df = pd.DataFrame(
                    data=torch.softmax(torch.tensor(pr_df.values), dim=-1).numpy(),
                    index=pr_df.index,
                    columns=[f'Label_Is_{_}' for _ in model_config['output']]
                )
            else:
                pr_df.columns = [f'Label_Is_{_}' for _ in model_config['output']]
        else:
            pr_df.columns = model_config['output']
    else:
        gt_df.columns = model_config['output']
        if model_config.get('outputs_logits', False):
            pr_df = pd.DataFrame(
                data=torch.sigmoid(torch.tensor(pr_df.values)).numpy(),
                index=pr_df.index,
                columns=model_config['output']
            )
        else:
            pr_df.columns = model_config['output']

    gt_df.to_csv(os.path.join(rslt_path, 'ground_truth.csv'))
    pr_df.to_csv(os.path.join(rslt_path, f'{args.epoch}.csv'))
    test_df.to_csv(os.path.join(rslt_path, 'rslt.csv'))