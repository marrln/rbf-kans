from typing import Iterable
import os
import torch
import torchmetrics
import copy
import collections
import re
from .. import metrics
from .. import models
from ..utils import callbacks
from .. import utils
from ..training import get_callable_basis
from ..utils import save_dict, load_dict

def get_locals(*args):
    locals_dict = {**__builtins__}
    for module in args:
        locals_dict.update(**module.__dict__)
    return locals_dict

__cls_dict = get_locals(
    collections,
    torch.nn,
    torch.optim,
    torch.optim.lr_scheduler,
    torchmetrics,
    torchmetrics.classification,
    torchmetrics.regression,
    torchmetrics.image,
    models,
    metrics,
    callbacks,
    utils,
)

def get_default_model_config() -> dict:
    return {
        'model'                 : models.RBFKAN,
        'model_kwargs'          : {
            'hidden_layers'     : None,
            'num_grids'         : None,
            'grid_min'          : None,
            'grid_max'          : None,
            'inv_denominator'   : None,
        },
        'input'                 : [
        ],
        'output'                : [
        ],
    }

def get_default_training_config() -> dict:
    return {
        'seed'                  : 42,
        'batch_size'            : 16,
        'splits'                : [0.75, 0.09, 0.16],
        'epochs'                : 100,
        'sample_weight'         : False,
        'criterion'             : torch.nn.MSELoss,
        'criterion_args'        : (),
        'optimizer'             : torch.optim.Adagrad,
        'optimizer_kwargs'      : {},
        'lr'                    : 1e-3,
        'scheduler'             : torch.optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_kwargs'      : {
            'factor'            : 0.9,
            'patience'          : 5,
        },
        'eval_criteria'         : {
            'MSELoss'           : torch.nn.MSELoss,
            'MSELoss_args'      : {},
        },
        'pretrained'            : False,
        'callbacks'             : get_callable_basis(),
        'callbacks_arguments'   : {},
    }
    
# def type_to_config(obj_type, target_name=None):
#     if target_name is None:
#         target_name = ''
#     return {
#         target_name : '!' + repr(obj_type),
#     }

def object_to_config(
    obj, 
    *args, 
    target_name = None, 
    **kwargs
):
    if target_name is None:
        target_name = ''
    
    targ_dict = {
        target_name                 : obj,
    }
    if len(args):
        targ_dict.update({
            f'{target_name}_args'   : list(args),
        })
    if len(kwargs):
        targ_dict.update({
            f'{target_name}_kwargs' : kwargs,
        })
    return targ_dict

def parse_config_val(val):
    '''Parse values from a configuration dictionary to a json compatible dictionary.'''
    if isinstance(val, dict):
        return parse_config(val)
    elif isinstance(val, str):
        return val
    elif isinstance(val, type) and issubclass(val, torch.nn.Module):
        return repr(val)
    elif hasattr(val,'__iter__') and not isinstance(val, type):
        return [parse_config_val(val_i) for val_i in val]
    else:
        return repr(val)
    
def parse_config(config):
    '''Parse a configuration dictionary to a json compatible dictionary.
    '''
    config = copy.copy(config)
    
    for key, val in config.items():
        config[key] = parse_config_val(val)
    return config

# def find_object_from_name(val, locals=None):
#     local_dict = __cls_dict
#     if isinstance(locals,dict):
#         local_dict.update(**locals)
#     return local_dict[val]

# def find_class_name(val, locals=None):
#     local_dict = __cls_dict
#     if isinstance(locals,dict):
#         local_dict.update(**locals)
#     elif locals is not None :
#         raise TypeError(f'Expected dict or None for variable "locals"; got {type(locals)}')
#     for key, cls in local_dict.items() :
#         if cls == val:
#             return key
#     else :
#         raise ValueError(f'Class type not found; got type {cls}')

def find_object(cls_repr, locals=None):
    local_dict = __cls_dict
    if isinstance(locals, dict):
        local_dict.update(**locals)
    for cls in local_dict.values() :
        if repr(cls) == cls_repr:
            return cls
    else :
        raise ValueError(f'Class type not found; got class repr {cls_repr}')

def parse_json_val(val, locals=None):
    '''Parse values from a json compatible dictionary to a configuration dictionary.'''
    if   isinstance(val, dict):
        return parse_json(val,  locals)
    elif isinstance(val, str):
        if   val.startswith(('<class ', '<function ')):
            return find_object(val, locals)
        elif re.fullmatch(r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$',val) or val.lower() in ['true','false'] or val.startswith('lambda'):
            return eval(val, globals(), locals)
        else :
            return val
    elif hasattr(val,'__iter__'):
        return [parse_json_val(val_i, locals) for val_i in val]
    else:
        raise TypeError(f'Unknown type {type(val)}.')
        
def parse_json(config, locals=None) -> dict:
    '''Parse a json compatible dictionary to a configuration dictionary.'''
    config = copy.copy(config)
    
    for key, val in config.items():
        config[key] = parse_json_val(val, locals)
        # print(key, val, config[key])
    return config
    
def save_config(config, fname) -> str:
    '''Save a configuration dictionary to the specified json file.'''
    config = parse_config(config)
    return save_dict(config, fname)

def load_config(fname, locals : dict[str, object]=None) -> dict:
    '''Load a configuration dictionary from the specified json file.'''
    config = load_dict(fname)
    
    local_dict = __cls_dict
    if isinstance(locals,dict):
        local_dict.update(**locals)
        
    return parse_json(config, local_dict)

def weak_instantiate_all(config : dict[str,object] | Iterable | object) -> dict[str,object] | Iterable | object:
    if isinstance(config, dict):
        if len(config) <= 3 and '' in config.keys():
            return weak_instantiate(config, '')
        else :
            return {
                key : weak_instantiate(config, key) 
                    for key in config.keys()
                        if not (key.endswith('_args') or key.endswith('_kwargs'))
            }
    elif isinstance(config, (str, )):
        if config.startswith(('!<class ')):
            return find_object(config[1:])
        return config
    elif hasattr(config,'__iter__'):
        return [weak_instantiate_all(_) for _ in config]
    else :
        return weak_instantiate({'' : config}, '')
    
def weak_instantiate(config, key):
    if isinstance(config[key], type):
        return instantiate(config, key)
    elif hasattr(config[key],'__iter__'):
        return weak_instantiate_all(config[key])
    else :
        return config[key]

def instantiate(config, key, *args, **kwargs):
    if not isinstance(type(config[key]), type):
        raise TypeError(f'Unexpected object for key "{key}"; expected "{type}"; got "{config[key]}"')
        
    if f'{key}_args' in config.keys() :
        args += tuple(weak_instantiate_all(config[f'{key}_args']))

    if f'{key}_kwargs' in config.keys() :
        kwargs.update(
            weak_instantiate_all(config[f'{key}_kwargs'])
        )
    return config[key](*args, **kwargs)

# def check_config(config, locals=None) -> dict:
#     '''Check if a configuration dictionary is valid.
    
#     :param config: The configuration dictionary
#     :type config: dict
#     :raises TypeError: If the configuration dictionary is invalid
#     '''
#     config = parse_config(config)
#     return parse_json(config, locals)

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    config_folder = os.path.join('./')
    
    tr_dict = get_default_training_config()
    print('Default config:')
    print(tr_dict)
    
    path = os.path.join(config_folder,'default_config.json')
    save_config(tr_dict, path)
    
    saved_dict = load_config(path)
    print('Saved & Loaded default config:')
    print(saved_dict)
    assert tr_dict == saved_dict
    
    os.remove(path)