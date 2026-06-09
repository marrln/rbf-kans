from typing import Union, Callable, Literal, Any
from tqdm import tqdm
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler 
import numpy as np
import os

from .evaluate import evaluate, get_callable_basis
from ..utils import save_model, load_model, save_dict, to, tolist

def train(
    model : Module,
    train_dataloader : DataLoader,
    eval_dataloader : DataLoader,
    criterion : Callable[[torch.Tensor],torch.Tensor],
    eval_criteria : dict[str:Callable[[torch.Tensor],torch.Tensor]],
    optimizer : Optimizer,
    scheduler : LRScheduler,
    epochs : int,
    patience : int = None,
    sample_weight : bool = False,
    clip_limit : Union[float, bool] = None,
    history : dict[str,dict[int,dict[str,Union[float,list[float]]]]] = {},
    start_epoch = 0,
    update_limit : bool | int | float = True,
    top_dirname = './train',
    device = torch.device('cpu'),
    evaluate_training = False,
    saving_steps : int | Literal['log'] = 1,
    show_pbar : Literal[None, 'external','internal'] = 'external',
    callbacks = get_callable_basis(),
    callbacks_arguments : dict[str, Any] = {},
) -> dict[str,dict[int,dict[str,Union[float,list[float]]]]]:

    best_loss = float('inf')
    val_loss = float('inf')
    best_epoch = 0
    if len(history) == 0:
        history = {'train':{}, 'val':{}}
        
    elif len(history['train']) > 0:
        start_epoch = max()
        best_loss = min([value['loss'] for value in history['train'].values()])
            
    model.to(device)
    
    try :
        optimizer.to(device)
    except:
        pass
    
    if 'loss' not in eval_criteria.keys():
        eval_criteria = {
            'loss' : criterion,
            **eval_criteria
        }
    
    patience_counter = 0
    if isinstance(saving_steps, int):
        _saving_steps = saving_steps
        
    
    os.makedirs(top_dirname, exist_ok=True)
    model_dirname = os.path.join(top_dirname, 'models')
    os.makedirs(model_dirname, exist_ok=True)
        
    if show_pbar == 'external':
        pbar_epoch = tqdm(range(start_epoch+1, start_epoch+epochs+1), dynamic_ncols=True)
    else:
        pbar_epoch = range(start_epoch+1, start_epoch+epochs+1)
        
    descr = 'Epoch {epoch} -- Tr Loss {tr_loss:.5f} -- Val Loss {val_loss:.5f} -- Best [{best_epoch}] {best_loss:.5f}'
    
    if update_limit:
        if 0 < update_limit <= 1. :
            update_period = len(train_dataloader) // int(1/update_limit)
        elif isinstance(update_limit, int) and update_limit > 0:
            update_period = len(train_dataloader) // update_limit
        else :
            update_period = len(train_dataloader) // 1000
    else :
        update_period = 1 
    
    try :
        for epoch in pbar_epoch:
            hist_train_epoch = {}
            tr_loss = 0.
            
            model.train().to(device)
            
            if show_pbar == 'internal':
                pbar_iter = tqdm(train_dataloader, dynamic_ncols=True)
            else:
                pbar_iter = train_dataloader
                
            if saving_steps == 'log':
                _saving_steps = int(np.ceil(2*np.log(epoch))+1)
                
            loc_kwargs = {
                'model'            : model,
                'epoch'            : epoch,
                'epochs'           : epochs,
                'best_loss'        : best_loss, 
                'dataloader'       : train_dataloader, 
                'patience'         : patience, 
                'patience_counter' : patience_counter,
                'criterion'        : criterion,
                'optimizer'        : optimizer,
                'scheduler'        : scheduler,
                'device'           : device,
                'history'          : history,
            }
            loc_kwargs.update(callbacks_arguments)
            for callback in callbacks['epoch_start']:
                callback(**loc_kwargs)

            try :
                criterion.to(device)
            except:
                pass

            try :
                if show_pbar == 'external':
                    pbar_epoch.set_description(descr.format(epoch=f"{epoch}[{0}/{len(pbar_iter)}]", tr_loss=tr_loss, val_loss=val_loss, best_epoch=best_epoch, best_loss=best_loss))
                   
                for _iter, (data, target, *weights) in enumerate(pbar_iter, start=1):
                    data    = to(data, device)
                    target  = to(target, device)
                    
                    if sample_weight:
                        if len(weights) == 1:
                            weights = weights[0].to(device)
                        elif isinstance(data, tuple):
                            weights = torch.ones_like(target[0])
                        else :
                            weights = torch.ones_like(target)
                    
                    loc_kwargs = {
                        'model'            : model,
                        'iteration'        : _iter,
                        'epoch'            : epoch, 
                        'epochs'           : epochs,
                        'data'             : data,
                        'target'           : target,
                        'dataloader'       : train_dataloader, 
                        'device'           : device,
                        'history'          : history,
                        'locals'           : locals,
                    }
                    loc_kwargs.update(callbacks_arguments)
                    for callback in callbacks['train_iter_start']:
                        callback(**loc_kwargs)

                    prediction = model(data)
                    
                    if sample_weight:
                        loss : torch.Tensor = criterion(prediction,target,weights)
                    else :
                        loss : torch.Tensor = criterion(prediction,target)
                    
                    if np.isnan(loss.detach().cpu()) :
                        data    = to(data, 'cpu')
                        target  = to(target, 'cpu')
                        prediction  = to(prediction, 'cpu')
                        
                        print(tolist(data), tolist(target), tolist(prediction), loss, sep='\n', file=open(
                            os.path.join(top_dirname,'error.log'),'w'
                        ))
                        save_dict(history, os.path.join(model_dirname,'history'))
                        raise ValueError(f'Encountered NaN value at epoch {epoch}')
                    else :
                        tr_loss += loss.detach().cpu()
                        
                    # Reset grads
                    optimizer.zero_grad()
                    
                    # Calculate new grads
                    loss.backward()
                    
                    # Clip parameters
                    if clip_limit is not None and bool(clip_limit):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_limit))
                            
                    # Update weights
                    optimizer.step()
                    
                    loc_kwargs = {
                        'model'         : model,
                        'iteration'     : _iter,
                        'epoch'         : epoch, 
                        'loss'          : loss.detach().cpu(),
                        'dataloader'    : train_dataloader, 
                        'optimizer'     : optimizer,
                        'device'        : device,
                        'history'       : history,
                    }
                    loc_kwargs.update(callbacks_arguments)
                    for callback in callbacks['train_iter_end']:
                        callback(**loc_kwargs)
                        
                    if _iter % update_period == 0:
                        if show_pbar == 'internal':
                            pbar_iter.set_description(descr.format(epoch=epoch, tr_loss=tr_loss/_iter, val_loss=val_loss, best_epoch=best_epoch, best_loss=best_loss))
                        elif show_pbar == 'external':
                            pbar_epoch.set_description(descr.format(epoch=f"{epoch}[{_iter}/{len(pbar_iter)}]", tr_loss=tr_loss/_iter, val_loss=val_loss, best_epoch=best_epoch, best_loss=best_loss))
                   
                del data, target, prediction, loss
                    
                tr_loss = float(tr_loss / _iter)
                hist_train_epoch.update({'loss' : tr_loss})
                
                history['train'].update({epoch : hist_train_epoch})
                
                loc_kwargs = {
                    'model'         : model,
                    'epoch'         : epoch, 
                    'epochs'        : epochs,
                    'tr_loss'       : tr_loss,
                    'best_loss'     : best_loss, 
                    'dataloader'    : train_dataloader, 
                    'optimizer'     : optimizer,
                    'scheduler'     : scheduler,
                    'device'        : device,
                    'history'       : history,
                }
                loc_kwargs.update(callbacks_arguments)
                for callback in callbacks['train_end']:
                    callback(**loc_kwargs)

                if evaluate_training:
                    history['train'][epoch].update(
                        evaluate(
                            model           = model, 
                            eval_dataloader = train_dataloader, 
                            criteria        = eval_criteria, 
                            sample_weight   = sample_weight,
                            device          = device, 
                            show_pbar       = show_pbar == 'internal',
                            callbacks       = callbacks,
                            callbacks_arguments = {
                                'epoch' : epoch,
                                **callbacks_arguments
                            }
                        )
                    )
                    
                val_metrics = evaluate(
                    model, 
                    eval_dataloader = eval_dataloader, 
                    criteria        = eval_criteria, 
                    sample_weight   = sample_weight,
                    device          = device, 
                    show_pbar       = show_pbar == 'internal',
                    callbacks       = callbacks,
                    callbacks_arguments = {
                        'epoch'  : epoch,
                        'epochs' : epochs,
                        **callbacks_arguments
                    }
                )
                history['val'].update({
                    epoch : val_metrics
                })
                val_loss = val_metrics['loss']
            
                if show_pbar == 'internal':
                    pbar_iter.set_description(descr.format(epoch=epoch, tr_loss=tr_loss, val_loss=val_loss, best_epoch=best_epoch, best_loss=best_loss))
                    pbar_iter.close()
                    
                elif show_pbar == 'external':
                    pbar_epoch.set_description(descr.format(epoch=epoch, tr_loss=tr_loss, val_loss=val_loss, best_epoch=best_epoch, best_loss=best_loss))
                    
                if _saving_steps > 0 and (epoch % _saving_steps == 0):
                    save_model(model, os.path.join(model_dirname,'last'), device)
                    save_dict(history, os.path.join(top_dirname,'history'))
                
                if best_loss > val_loss:
                    best_loss = val_loss
                    save_model(model, os.path.join(model_dirname,'best'), device)   
                    patience_counter = 0
                    best_epoch = epoch
                elif patience is not None and patience > 1:
                    patience_counter += 1
                    
                    if patience_counter >= patience:
                        break
                    
                loc_kwargs = {
                    'model'            : model,
                    'epoch'            : epoch, 
                    'epochs'           : epochs,
                    'tr_loss'          : tr_loss,
                    'val_loss'         : val_loss,
                    'best_loss'        : best_loss, 
                    'train_dataloader' : train_dataloader, 
                    'eval_dataloader'  : eval_dataloader, 
                    'optimizer'        : optimizer,
                    'scheduler'        : scheduler,
                    'device'           : device,
                    'history'          : history,
                }
                loc_kwargs.update(callbacks_arguments)
                for callback in callbacks['epoch_end']:
                    callback(**loc_kwargs)
                    
                scheduler.step(val_loss)
                
            except Exception as e:
                if show_pbar == 'internal':
                    pbar_iter.close()
                raise e
            
        if show_pbar == 'external':
            pbar_epoch.close()
        
    except Exception as e:
        if show_pbar == 'external':
            pbar_epoch.close()
            
        loc_kwargs = {
            'model'            : model,
            'epoch'            : epoch, 
            'epochs'           : epochs,
            'val_loss'         : val_loss,
            'best_loss'        : best_loss, 
            'train_dataloader' : train_dataloader, 
            'eval_dataloader'  : eval_dataloader, 
            'optimizer'        : optimizer,
            'scheduler'        : scheduler,
            'device'           : device,
            'history'          : history,
            'exception'        : e, 
        }
        loc_kwargs.update(callbacks_arguments)
        for callback in callbacks['exception_raised']:
            callback(**loc_kwargs)
        for callback in callbacks['training_finished']:
            callback(**loc_kwargs)
            
        save_model(model, os.path.join(model_dirname,'last'), device)
        try:
            save_dict(history, os.path.join(top_dirname,'history'))
        except Exception as e:
            print(history)
            raise e
        
        if 'data' in locals():
            del data
        if 'target' in locals():
            del target
        if 'prediction' in locals():
            del prediction
        if 'loss' in locals():
            del loss
        
        raise e
    
    loc_kwargs = {
        'model'            : model,
        'epoch'            : epoch, 
        'epochs'           : epochs,
        'tr_loss'          : tr_loss,
        'val_loss'         : val_loss,
        'best_loss'        : best_loss, 
        'train_dataloader' : train_dataloader, 
        'eval_dataloader'  : eval_dataloader, 
        'optimizer'        : optimizer,
        'scheduler'        : scheduler,
        'device'           : device,
        'history'          : history,
    }
    loc_kwargs.update(callbacks_arguments)
    for callback in callbacks['training_finished']:
        callback(**loc_kwargs)
        
    if _saving_steps < 1 or (epoch % _saving_steps != 0):
        save_model(model, os.path.join(model_dirname,'last'), device)
        save_dict(history, os.path.join(top_dirname,'history'))
        
    return history