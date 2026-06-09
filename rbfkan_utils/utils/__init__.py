from typing import Mapping
import torch
from torch.nn import Module
import os, json
import numpy as np
from .callbacks import *
from .dataset import *
from .plotter import *
from .summary import *

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
