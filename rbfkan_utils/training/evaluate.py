from typing import Union, Callable, Literal, Any
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os
from ..utils import to, tolist, cat, apply_to_tensor, isnested, nested2dict

def get_callable_basis() :
    '''Basis for callable categories of callable functions.
    All callables should be of form:
    
        def callable(arg1, ..., argN, **kwargs):
            ...

    
    Available callback stages
    -------------------------
        epoch_start: 
            At the start of an epoch (before training).
        train_iter_start: 
            At the start of an iteration (during training).
        train_iter_end: 
            At the end of an iteration (during training).
        train_end: 
            At the end of the training in an epoch (after training).
        eval_start: 
            At the start of an evaluation (before validation).
        eval_iter_start: 
            At the start of an iteration (during validation).
        eval_iter_end: 
            At the end of an iteration (during validation).
        eval_metrics_start: 
            At the end of model prediction, before calculating metrics (during validation)
        eval_end: 
            At the end of an evaluation (after validation).
        epoch_end: 
            At the end of an epoch (after scheduler).
        exception_raised:
            When an exception is raised. 
        training_finished:
            At the end of all training (after all epochs are executed or when the patience counter reaches the maximum value).
            
    Returns
    -------
        dict[str, (...) -> None]
    '''
    return {
        'epoch_start'           : [],
        'train_iter_start'      : [],
        'train_iter_end'        : [],
        'train_end'             : [],
        'eval_start'            : [],
        'eval_iter_start'       : [],
        'eval_iter_end'         : [],
        'eval_metrics_start'    : [],
        'eval_end'              : [],
        'epoch_end'             : [],
        'exception_raised'      : [], 
        'training_finished'     : [],
    }

def evaluate(
    model : Module,
    eval_dataloader : DataLoader,
    criteria : dict[str:Callable[[torch.Tensor],torch.Tensor]],
    keep_copy = True,
    checkpoint_path = None,
    epoch = None,
    sample_weight : bool = False,
    show_pbar = True,
    device = torch.device('cpu'),
    callbacks = get_callable_basis(),
    callbacks_arguments : dict[str, Any] = {},
) -> dict[str, Union[float,list[float]]]:
    if len(criteria) == 0:
        return {}
    sample_weight = bool(sample_weight)
    model.eval()
    model.to(device)
    
    preds = []
    targs = []
    keys  = []
    if sample_weight:
        weights = []
    
    if show_pbar:
        pbar = tqdm(eval_dataloader)
        
    else :
        pbar = eval_dataloader

    with torch.no_grad():    
        loc_kwargs = {
            'model'            : model,
            'epoch'            : epoch, 
            'eval_dataloader'  : eval_dataloader, 
            'device'           : device,
        }
        loc_kwargs.update(callbacks_arguments)
        for callback in callbacks['eval_start']:
            callback(**loc_kwargs)
            
        for data, target, *key in pbar:
            if sample_weight:
                weight = key[0]
            
            if len(key) > int(sample_weight):
                key = key[int(sample_weight)]
                if isinstance(key, torch.Tensor):
                    key = key.tolist()
            else :
                key = None
                          
            loc_kwargs = {
                'model'         : model,
                'epoch'         : epoch, 
                'data'          : data,
                'target'        : target,
                'key'           : key,
                'dataloader'    : eval_dataloader, 
                'device'        : device,
            }
            loc_kwargs.update(callbacks_arguments)
            for callback in callbacks['eval_iter_start']:
                callback(**loc_kwargs)
                
            data    = to(data, device)
            target  = to(target, device)

            prediction = to(model(data), 'cpu')
            
            loc_kwargs = {
                'model'         : model,
                'epoch'         : epoch, 
                'prediction'    : prediction,
                'target'        : target,
                'key'           : key,
                'dataloader'    : eval_dataloader, 
                'device'        : device,
            }
            loc_kwargs.update(callbacks_arguments)
            for callback in callbacks['eval_iter_end']:
                callback(**loc_kwargs)
                
            preds.append(prediction)
            targs.append(target)
            if sample_weight:
                weights.append(weight)
                
            if key is not None:
                keys.extend(key)
                # print(keys)
        
        del data, target, prediction
        if sample_weight:
            del weight
            
        model.to('cpu')
        prediction = cat(preds)
        target = cat(targs)
        del preds, targs
        
        if sample_weight:
            weight = torch.cat(weights)
            del weights
        try :
            prediction  = to(prediction, device)
            target      = to(target, device)
            if sample_weight:
                weight  = to(weight, device)
        except:
            prediction  = to(prediction, 'cpu')
            target      = to(target, 'cpu')
            if sample_weight:
                weight  = to(weight, 'cpu')
                
        if show_pbar:
            pbar.close()
            
        loc_kwargs = {
            'eval_criteria' : criteria,
            'epoch'         : epoch, 
            'prediction'    : prediction,
            'target'        : target,
            'key'           : keys,
            'dataloader'    : eval_dataloader, 
            'device'        : device,
        }
        loc_kwargs.update(callbacks_arguments)
        for callback in callbacks['eval_metrics_start']:
            callback(**loc_kwargs)
            
        metrics = {}
        for name, criterion in criteria.items():
            try :
                criterion.to(device)
            except :
                pass
            
            try :
                if sample_weight:
                    try :
                        metrics[name] = criterion(prediction, target, weight)
                    except :
                        metrics[name] = criterion(prediction, target)
                else :
                    metrics[name] = criterion(prediction, target)
            except Exception as e:
                # pass
                print(f'Warning -- {name}:',e)
            
            try :
                metrics[name] = tolist(to(metrics[name], device=device, dtype=torch.float64))
            except :
                pass
        
            try :
                criterion.to('cpu')
            except :
                pass

    if keep_copy and len(keys) > 0 and checkpoint_path is not None:
        rslt_path = os.path.join(os.path.dirname(checkpoint_path), "rslt.csv" if epoch is None else f"{epoch}.csv")
        
        os.makedirs(os.path.dirname(rslt_path), exist_ok=True)
        # print(target.shape, prediction.shape)
        
        prediction = apply_to_tensor(to(prediction, device='cpu', dtype=torch.float32), 'flatten', 1)
        try:
            target = apply_to_tensor(to(target    , device='cpu', dtype=torch.float32), 'flatten', 1)
        except:
            target = apply_to_tensor(to(target    , device='cpu', dtype=torch.float32), 'unsqueeze', -1)
            
        if isnested(prediction):
            prediction = nested2dict(prediction)
            target     = nested2dict(target)
            df  = pd.DataFrame.from_dict({
                'Index' : keys,
                **{
                    f"targ_{key}_{_iter}" : val[:,_iter]
                        for key, val in target.items()
                            for _iter in range(val.shape[-1])
                },
                **{
                    f"pred_{key}_{_iter}" : val[:,_iter]
                        for key, val in prediction.items()
                            for _iter in range(val.shape[-1])
                },
            }).set_index('Index').sort_index()
        else :
            df  = pd.DataFrame({
                'Index' : keys,
                **{
                    f"targ_{_iter}" : target[:,_iter] 
                        for _iter in range(target.shape[-1])
                },
                **{
                    f"pred_{_iter}" : prediction[:,_iter] 
                        for _iter in range(prediction.shape[-1])
                },
            }).set_index('Index').sort_index()
        df.to_csv(rslt_path)
        
        print(f'Results written to "{rslt_path}"')
        del df
        
    loc_kwargs = {
        'metrics'          : metrics,
        'epoch'            : epoch, 
        'prediction'       : prediction,
        'target'           : target,
        'key'              : key,
        'eval_dataloader'  : eval_dataloader, 
        'device'           : device,
    }
    loc_kwargs.update(callbacks_arguments)
    for callback in callbacks['eval_end']:
        callback(**loc_kwargs)
        
    return metrics