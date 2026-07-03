#!/usr/bin/env python3   

if __name__ == '__main__':
    import sys, os
    import random
    import numpy as np
    from argparse import ArgumentParser
    from augmentations import Augmentor # pyright: ignore[reportMissingImports]
    
    THIS_DIR = os.path.dirname(__file__)
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(description=f'Training script ')
    parser.add_argument('--dataset', dest='dataset', type=str, required=True, help=f'Dataset to use (required)')
    parser.add_argument('-d', '--test-dir', dest='test_dir', default=None, help='The directory to be used as a top directory for training, if None uses dataset-specific default directory.')
    parser.add_argument('--hash', dest='hash', type=str, help='The hash value of the configuration.', required=True)
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')
    parser.add_argument('--no-pbar', action='store_true', dest='no_pbar')
    parser.add_argument('--no-resume', action='store_true', dest='no_resume', help='Do not resume training even if checkpoint exists')

    args = parser.parse_args()
    
    # Add dataset folder to path
    dataset_path = os.path.join(THIS_DIR, args.dataset)
    if dataset_path not in sys.path:
        sys.path.insert(0, dataset_path)
    from prepare_dataset import get_dataset, DATASET_DIR, DATASET_NAME, build_dataset, get_dataset, get_groups # pyright: ignore[reportMissingImports]
    
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
    
    from rbfkan_utils.config import *
    from rbfkan_utils.utils.dataset import smart_split_dataset
    from rbfkan_utils.training import train
    from rbfkan_utils.utils import set_seed, separate_lr_params
    from rbfkan_utils.utils.summary import get_summary
    from custom_dataset import GenericDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check configuration file validity
    train_config = load_config(args.train_config, locals=get_locals())
    model_config = load_config(args.model_config, locals=get_locals())
    set_seed(train_config['seed'])

    # Instantiate model, criterion, optimizer, scheduler
    model     = instantiate(model_config, 'model')
    criterion = instantiate(train_config, 'criterion')
    param_groups = separate_lr_params(model, train_config['lr'], scale_factor=0.01)
    optimizer = instantiate(train_config, 'optimizer', param_groups, lr=train_config['lr'])
    scheduler = instantiate(train_config, 'scheduler', optimizer)
    
    print('-- Model :',     model)
    print('-- Criterion :', criterion)
    print('-- Optimizer :', optimizer)
    print('-- Scheduler :', scheduler)

    # Instantiate evaluation criteria
    eval_criteria = weak_instantiate_all(train_config['eval_criteria'])

    # Instantiate callbacks
    callbacks = weak_instantiate_all(train_config['callbacks'])
    callbacks_arguments = weak_instantiate_all(train_config['callbacks_arguments'])

    build_dataset() # Build dataset if not already built
    
    augmentor = Augmentor(train_config)
    
    data, labels = get_dataset('train_val')
    train_indices, val_indices = smart_split_dataset(
        splits       = train_config['splits'],
        full_dataset = None,
        groups       = get_groups(),
        seed         = train_config['seed']
    )
    
    train_dataset = GenericDataset(
        data            = data[train_indices], 
        labels          = labels[train_indices],
        task            = train_config['task'],
        return_key      = False,
        return_weights  = train_config['sample_weight'],
        preprocess_data = augmentor.train,
        preprocess_targ = None,
        flatten         = model_config['flatten'],
    )

    val_dataset = GenericDataset(
        data            = data[val_indices], 
        labels          = labels[val_indices],
        task            = train_config['task'],
        return_key      = False,
        return_weights  = train_config['sample_weight'],
        preprocess_data = augmentor.val,
        preprocess_targ = None,
        flatten         = model_config['flatten'],
    )
    
    os.makedirs(os.path.join(args.test_dir, 'models'), exist_ok=True)
    data_sample = train_dataset[0][0]
    print(
        '-- Model Summary :',
        get_summary(
            model,
            data_sample,
            dest=os.path.join(args.test_dir, 'models', 'summary'),
            depth=5,
        )
    )

    # Worker init function to seed each worker's random generators independently
    def worker_init_fn(worker_id):
        base_seed = train_config['seed']
        # Unique seed per worker to avoid duplicate augmentation sequences
        np.random.seed(base_seed + worker_id)
        random.seed(base_seed + worker_id)
        torch.manual_seed(base_seed + worker_id)

    # Dedicated generator for shuffling (decouples shuffle order from global RNG)
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(train_config['seed'])

    train_loader = DataLoader(
        dataset            = train_dataset,
        batch_size         = train_config['batch_size'],
        shuffle            = True,                  # <-- FIXED: Critical for i.i.d. assumption
        generator          = shuffle_generator,    # <-- ADDED: Controls only index order
        worker_init_fn     = worker_init_fn,       # <-- ADDED: Controls augmentations per worker
        num_workers        = os.cpu_count(),
        persistent_workers = True,
        pin_memory         = (device == torch.device('cuda')),
    )

    val_loader = DataLoader(
        dataset            = val_dataset,
        batch_size         = train_config['batch_size'],
        shuffle            = False,  # Explicitly False for clarity
        num_workers        = os.cpu_count(),
        persistent_workers = True,
        pin_memory         = (device == torch.device('cuda')),
    )
    
    print('-- Using dataset split :', train_config['splits'])
    print('  -- Train      :', len(train_loader.dataset))
    print('  -- Validation :', len(val_loader.dataset))
    print('  -- Using Sample weight:', train_config['sample_weight'])
    
    history = train(
        model               = model,
        train_dataloader    = train_loader,
        eval_dataloader     = val_loader,
        criterion           = criterion,
        eval_criteria       = eval_criteria,
        optimizer           = optimizer,
        scheduler           = scheduler,
        epochs              = train_config['epochs'],
        patience            = train_config['patience'],
        resume_training     = not args.no_resume,   # resume by default, use --no-resume to start fresh
        sample_weight       = train_config['sample_weight'],
        clip_limit          = train_config.get('clip_limit', 1.0),
        update_limit        = 500,
        top_dirname         = args.test_dir,
        device              = device,
        evaluate_training   = False,
        saving_steps        = 'log',
        show_pbar           = None if args.no_pbar else 'external',
        callbacks           = callbacks,
        callbacks_arguments = callbacks_arguments,
    )
    print('-- Model :', model)