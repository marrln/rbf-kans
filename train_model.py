#!/usr/bin/env python3   

if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(__file__)
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)


    parser = ArgumentParser(description=f'Training script ')
    parser.add_argument('--dataset', dest='dataset', type=str, required=True, help=f'Dataset to use (required)')
    parser.add_argument('-d', '--test-dir', dest='test_dir', default=None, help='The directory to be used as a top directory for training, if None uses dataset-specific default directory.')
    parser.add_argument('--hash', dest='hash', type=str, help='The hash value of the configuration.', required=True)
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')
    parser.add_argument('--no-pbar', action='store_true', dest='no_pbar')

    args = parser.parse_args()
    
    # Add dataset folder to path
    dataset_path = os.path.join(THIS_DIR, args.dataset)
    if dataset_path not in sys.path:
        sys.path.insert(0, dataset_path)
    from prepare_dataset import get_dataset, DATASET_DIR, DATASET_NAME, build_dataset, get_dataset, get_groups
    
    # Set default after DATASET_DIR is available
    if args.test_dir is None:
        args.test_dir = os.path.join(DATASET_DIR, 'train')
    
    args.test_version = '_'.join(['test', args.test_version])

    # Use dataset-specific directory if provided
    if args.dataset != DATASET_NAME:
        dataset_dir = os.path.join(THIS_DIR, args.dataset)
    else:
        dataset_dir = DATASET_DIR

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
    
    import torch
    from torch.utils.data import DataLoader
    import albumentations as A
    
    from rbfkan_utils.config import *
    from rbfkan_utils.utils.dataset import smart_split_dataset
    from rbfkan_utils.training import train
    from rbfkan_utils.utils import set_seed, apply_to_tensor
    from rbfkan_utils.utils.summary import get_summary
    from custom_dataset import GenericDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check configuration file validity
    train_config = load_config(args.train_config, locals=get_locals())
    model_config = load_config(args.model_config, locals=get_locals())
    set_seed(train_config['seed'])

    # Instantiate models
    model = instantiate(model_config, 'model')
    print('-- Model :', model)

    # Instantiate criterion
    criterion = instantiate(train_config, 'criterion')
    print('-- Criterion :', criterion)

    # Instantiate optimizer
    optimizer = instantiate(train_config, 'optimizer', model.parameters(), lr=train_config['lr'])
    print('-- Optimizer :', optimizer)

    # Instantiate scheduler
    scheduler = instantiate(train_config, 'scheduler', optimizer)
    print('-- Scheduler :', scheduler)

    # Instantiate evaluation criteria
    eval_criteria = weak_instantiate_all(train_config['eval_criteria'])
    print('-- Evaluation Criteria :')
    if len(eval_criteria):
        for key, val in eval_criteria.items():
            print('  --', key, ':', val)
    else:
        print('  No evaluation criteria.')
        
    # Instantiate callbacks
    callbacks = weak_instantiate_all(train_config['callbacks'])
    callbacks_arguments = weak_instantiate_all(train_config['callbacks_arguments'])

    build_dataset() # Build dataset if not already built
    
    data, labels = get_dataset('train_val')
    preprocess_data = A.Compose([
            # NOTE: Here you can edit the data augmentation pipeline as needed
            # A.HorizontalFlip(p=train_config['probability']),
            # A.VerticalFlip(p=train_config['probability']),
            # A.ShiftScaleRotate(scale_limit=(-0.2, 0), rotate_limit=(-90, 90), p=train_config['probability']),
            # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=train_config['probability']),
            # A.SafeRotate(limit=(-90, 90), p=train_config['probability']),
            *([] if 'resize' not in train_config.keys() else [A.Resize(*train_config['resize'])]),
            A.Normalize(normalization='min_max_per_channel'),
            A.ToTensorV2(),
        ],
        # telemetry=False,
        seed=train_config['seed'],
    )
    
    # Create iterable train and val datasets
    train_loader, val_loader = smart_split_dataset(
        splits=train_config['splits'],
        full_dataset=None,
        groups=get_groups(),
        seed=train_config['seed']
    )
    print(f"DEBUG: train indices count = {len(train_loader)}")
    print(f"DEBUG: val indices count = {len(val_loader)}")
    print(f"DEBUG: type(train_idx) = {type(train_loader)}")
    print(f"DEBUG: first few train_idx = {train_loader[:5]}")
    
    train_loader = GenericDataset(
        data[train_loader], labels[train_loader],
        task=train_config['task'],
        return_weights=train_config['sample_weight'],
        # preprocess_data=lambda **kwargs: preprocess_data(**kwargs)[list(kwargs.keys())[0]],
        preprocess_data         = lambda x: preprocess_data(image=x)["image"],
        preprocess_targ=None,
        flatten=model_config['flatten'],
    )
    print(f"DEBUG: train_loader length = {len(train_loader)}")
    print(f"DEBUG: train_loader data shape = {train_loader.data.shape if hasattr(train_loader.data, 'shape') else len(train_loader.data)}")
    
    val_loader = GenericDataset(
        data[val_loader], labels[val_loader],
        task=train_config['task'],
        return_weights=train_config['sample_weight'],
        # preprocess_data=lambda **kwargs: preprocess_data(**kwargs)[list(kwargs.keys())[0]],
        preprocess_data         = lambda x: preprocess_data(image=x)["image"],
        preprocess_targ=None,
        flatten=model_config['flatten'],
    )
    
    os.makedirs(os.path.join(args.test_dir, 'models'), exist_ok=True)
    # data_sample = next(iter(train_loader))[0] BUG: this does not work because train_loader is not a DataLoader yet, but a Dataset. We need to get a sample directly from the dataset.
    data_sample = train_loader[0][0]
    print(
        '-- Model Summary :',
        get_summary(
            model,
            data_sample,
            dest=os.path.join(args.test_dir, 'models', 'summary'),
            depth=5,
        )
    )

    train_loader = DataLoader(
        train_loader,
        batch_size=train_config['batch_size'],
        num_workers=os.cpu_count(),
        persistent_workers=True,
        pin_memory=device == torch.device('cuda'),
    )
    val_loader = DataLoader(
        val_loader,
        batch_size=train_config['batch_size'],
        num_workers=os.cpu_count(),
        pin_memory=device == torch.device('cuda'),
    )
    
    print('-- Using dataset split :', train_config['splits'])
    print('  -- Train      :', len(train_loader.dataset))
    print('  -- Validation :', len(val_loader.dataset))
    print('-- Sample weight:', train_config['sample_weight'])
    
    history = train(
        model,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        criterion=criterion,
        eval_criteria=eval_criteria,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=train_config['epochs'],
        patience=train_config['patience'],
        update_limit=100,
        sample_weight=train_config['sample_weight'],
        clip_limit=0.5,
        top_dirname=args.test_dir,
        device=device,
        evaluate_training=False,
        saving_steps='log',
        show_pbar=None if args.no_pbar else 'external',
        callbacks=callbacks,
        callbacks_arguments=callbacks_arguments,
    )
    print('-- Model :', model)