from typing import Mapping, Optional
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import os, json
import numpy as np
from .callbacks import *
from .dataset import *
from .plotter import *
from .summary import *

def separate_lr_params(model, base_lr, scale_factor=0.01):
    return [
        {
            'params': [p for n, p in model.named_parameters() if 'grid' not in n and 'inv_denominator' not in n],
            'lr': base_lr
        },
        {
            'params': [p for n, p in model.named_parameters() if 'grid' in n],
            'lr': base_lr * scale_factor,
            # 'weight_decay': 0.0
        },
        {
            'params': [p for n, p in model.named_parameters() if 'inv_denominator' in n],
            'lr': base_lr * scale_factor,
            # 'weight_decay': 0.0
        }
    ]

def save_model(model: Module, fname:str, device = torch.device('cpu')):
    if os.path.splitext(fname)[-1] not in ('.pt','.pth'):
        fname = f'{fname}.pt'
    torch.save(
        model.cpu().state_dict(),
        fname
    )
    model.to(device)
    return fname
    
def load_model(model: Module, fname:str, device = torch.device('cpu')):
    if os.path.splitext(fname)[-1] not in ('.pt','.pth'):
        fname = f'{fname}.pt'
    state_dict = torch.load(fname)
    model.cpu().load_state_dict(state_dict)
    return model.to(device)

def save_dict(obj : dict, fname : str):
    if os.path.splitext(fname)[-1] != '.json':
        fname = f'{fname}.json'
    with open(fname, 'w') as fwriter:
        json.dump(obj, fwriter, indent=2)
    return fname

def load_dict(fname) -> dict:
    if os.path.splitext(fname)[-1] != '.json':
        fname = f'{fname}.json'
    with open(fname, 'r') as freader:
        obj = json.load(freader)    
    return obj

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # torch.random.manual_seed(seed)
    np.random.seed(seed)
    
def expand_value(val, size):
    if not hasattr(val, '__iter__'):
        val = [val for _ in range(size)]

    if len(val) < size:
        val = val + [val[-1] for _ in range(size-len(val))]
    val = val[:size]

    assert len(val) == size, f"Size missmatch; expected size {size}; got {len(val)} \n {val}"
    return val

uses_momentum = lambda optimizer : optimizer in ('SGD', 'RMSprop')

def isnested(x):
    return not (hasattr(x,'__iter__') and isinstance(x,torch.Tensor))

def nested2dict(x) :
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, Mapping):
        return {key : nested2dict(val) for key,val in x.items()}
    elif hasattr(x, '__iter__'):
        if isinstance(x[0], torch.Tensor):
            return x
        return {f'arg.{_iter}' : nested2dict(val) for _iter, val in enumerate(x)}
    else :
        raise NotImplementedError(f'Cannot convert type "{type(x)}" to dict')

def apply_to_tensor(
    x,
    method : str,
    *args,
    **kwargs
):
    if isinstance(x, (torch.Tensor, torch.nn.Module)):
        return getattr(x,method)(*args, **kwargs)
    elif isinstance(x, Mapping):
        return type(x)(**{
            key : apply_to_tensor(val, method, *args, **kwargs)
            for key, val in x.items()
        })
    elif hasattr(x, '__iter__'):
        return type(x)([
            apply_to_tensor(val, method, *args, **kwargs)
            for val in x
        ])
    else :
        raise NotImplementedError(f'Function "{method}" does not support type {type(x)}')

def to(
    x,
    device = None,
    dtype = None,
    non_blocking = False,
    copy = False,
    memory_format = None,
):
    kwargs = {
        'device' : device,
        'dtype' : dtype,
        'non_blocking' : non_blocking,
        'copy' : copy,
        'memory_format' : memory_format,
    }
    return apply_to_tensor(x, 'to', **kwargs)

def tolist(x):
    return apply_to_tensor(x, 'tolist')

def cat(x : list[list[torch.Tensor],dict[str,torch.Tensor]], dim: int = 0,):
    if isinstance(x[0], Mapping):
        return type(x[0])(**{
            key : torch.cat([val[key] for val in x], dim=dim) 
                for key in x[0].keys()
        })
    elif isinstance(x[0], torch.Tensor):
        return torch.cat(x, dim=dim)
    elif hasattr(x[0], '__iter__'):
        return type(x[0])(*[
            torch.cat([val[_iter] for val in x], dim=dim)
                for _iter in range(len(x[0]))
        ])
    else :
        raise NotImplementedError(f'Function "cat" does not support type {type(x)}')


def save_checkpoint(
    model: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    best_loss: float,
    history: dict,
    filepath: str,
    rng_state: Optional[torch.Tensor] = None
) -> None:
    """
    Save a full training checkpoint containing model, optimizer, scheduler,
    epoch, best loss, history, and optional RNG state.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
        'history': history,
    }
    if rng_state is not None:
        rng_state = rng_state.detach().clone().to(dtype=torch.uint8, device='cpu')
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: Optional[Module] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    load_rng: bool = False,
    device: torch.device = torch.device('cpu')
) -> dict:
    """
    Load a full training checkpoint.

    Args:
        filepath: Path to checkpoint file.
        model: Model to load state_dict into (optional).
        optimizer: Optimizer to load state_dict into (optional).
        scheduler: Scheduler to load state_dict into (optional).
        load_rng: If True and checkpoint contains 'rng_state', restore torch RNG state.
        device: Device to map tensors to.

    Returns:
        The checkpoint dictionary (contains keys: epoch, model_state_dict,
        optimizer_state_dict, scheduler_state_dict, best_loss, history, maybe rng_state).
    """
    checkpoint = torch.load(filepath, map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if load_rng and 'rng_state' in checkpoint:
        rng_state = checkpoint['rng_state']
        if not isinstance(rng_state, torch.ByteTensor):
            # Convert to ByteTensor on CPU using detach().clone()
            rng_state = rng_state.detach().clone().to(dtype=torch.uint8, device='cpu').contiguous()
        else:
            rng_state = rng_state.cpu().contiguous()
        torch.random.set_rng_state(rng_state)

    return checkpoint