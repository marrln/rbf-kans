from typing import Union, Callable, Literal, Any, Optional, Dict
from tqdm import tqdm
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import numpy as np
import os
import inspect

from .evaluate import evaluate, get_callable_basis
from ..utils import save_model, load_model, save_dict, to, tolist

def train(
    model: Module,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    criterion: Callable,
    eval_criteria: Dict[str, Callable],
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epochs: int,
    patience: Optional[int] = None,
    sample_weight: bool = False,
    clip_limit: float = 1.0,
    history: Optional[Dict] = None,
    start_epoch: int = 0,
    update_limit: Union[bool, int, float] = True,
    top_dirname: str = './train',
    device: torch.device = torch.device('cpu'),
    evaluate_training: bool = False,
    saving_steps: Union[int, Literal['log']] = 1,
    show_pbar: Literal[None, 'external', 'internal'] = 'external',
    callbacks: Optional[Dict] = None,
    callbacks_arguments: Optional[Dict[str, Any]] = None,
    skip_nan_batch: bool = False,
) -> Dict[str, Dict[int, Dict[str, Union[float, list[float]]]]]:

    # ---------- Initialisation ----------
    if callbacks is None:
        callbacks = get_callable_basis()
    if callbacks_arguments is None:
        callbacks_arguments = {}
    if history is None:
        history = {'train': {}, 'val': {}}

    # Add 'loss' to eval_criteria if missing
    if 'loss' not in eval_criteria:
        eval_criteria = {
            'loss': criterion,
            **eval_criteria
        }

    if len(history['train']) > 0:
        start_epoch = max(history['train'].keys())
        best_loss = min([v['loss'] for v in history['train'].values()])
        
        # BUG:
        # We need to load model and optimizer state to resume training properly
        # We will also need to update the scheduler state if it is not None.
        # Auto-resume from saved checkpoint
        checkpoint_path = os.path.join(model_dirname, 'last.pt')  # or .pth
        if os.path.exists(checkpoint_path):
            print(f"Resuming training from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # if scheduler supports
            # Optionally restore RNG
            if 'rng_state' in checkpoint:
                torch.random.set_rng_state(checkpoint['rng_state'])
    else:
        best_loss = float('inf')

    best_epoch = start_epoch
    patience_counter = 0
    val_loss = float('inf')

    # Move model to device once
    model.to(device)
    if hasattr(criterion, 'to'):
        criterion.to(device)

    # Check if criterion accepts a 'weight' argument (for sample_weight)
    criterion_signature = inspect.signature(criterion)
    accepts_weight = 'weight' in criterion_signature.parameters

    # Prepare saving directories
    os.makedirs(top_dirname, exist_ok=True)
    model_dirname = os.path.join(top_dirname, 'models')
    os.makedirs(model_dirname, exist_ok=True)

    # Progress bars
    if show_pbar == 'external':
        pbar_epoch = tqdm(range(start_epoch + 1, start_epoch + epochs + 1), dynamic_ncols=True)
    else:
        pbar_epoch = range(start_epoch + 1, start_epoch + epochs + 1)

    descr = 'Epoch {epoch} -- Tr Loss {tr_loss:.5f} -- Val Loss {val_loss:.5f} -- Best [{best_epoch}] {best_loss:.5f}'

    # Update period for live logging
    if update_limit:
        if 0 < update_limit <= 1.:
            update_period = len(train_dataloader) // max(1, int(1 / update_limit))
        elif isinstance(update_limit, int) and update_limit > 0:
            update_period = len(train_dataloader) // update_limit
        else:
            update_period = len(train_dataloader) // 1000
    else:
        update_period = 1
    update_period = max(1, update_period)

    if isinstance(saving_steps, int):
        _saving_steps = max(1, saving_steps)

    # ---------- Training loop ----------
    try:
        for epoch in pbar_epoch:
            hist_train_epoch = {}
            tr_loss = 0.0
            model.train()

            if show_pbar == 'internal':
                pbar_iter = tqdm(train_dataloader, dynamic_ncols=True)
            else:
                pbar_iter = train_dataloader

            if saving_steps == 'log':
                _saving_steps = max(1, int(np.ceil(2 * np.log(epoch)) + 1))

            # Epoch start callbacks
            for cb in callbacks.get('epoch_start', []):
                cb(model=model, epoch=epoch, epochs=epochs, best_loss=best_loss,
                   dataloader=train_dataloader, patience=patience, patience_counter=patience_counter,
                   criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                   device=device, history=history, **callbacks_arguments)

            for _iter, batch in enumerate(pbar_iter, start=1):
                # Unpack batch
                if sample_weight and len(batch) == 3:
                    data, target, weight = batch
                    weight = to(weight, device)
                else:
                    data, target = batch[:2]
                    weight = None

                data = to(data, device)
                target = to(target, device)

                # Train iteration start callbacks
                for cb in callbacks.get('train_iter_start', []):
                    cb(model=model, iteration=_iter, epoch=epoch, data=data, target=target,
                       dataloader=train_dataloader, device=device, history=history,
                       **callbacks_arguments)

                # Forward
                prediction = model(data)

                # Compute loss (with optional weight)
                if sample_weight and weight is not None and accepts_weight:
                    loss = criterion(prediction, target, weight=weight)
                else:
                    loss = criterion(prediction, target)

                # NaN/Inf handling
                if torch.isnan(loss) or torch.isinf(loss):
                    if skip_nan_batch:
                        print(f"Warning: NaN/Inf loss at epoch {epoch}, iter {_iter}. Skipping batch.")
                        continue
                    else:
                        with open(os.path.join(top_dirname, 'error.log'), 'w') as f:
                            print(tolist(data), tolist(target), tolist(prediction), loss.item(), file=f)
                        save_dict(history, os.path.join(model_dirname, 'history'))
                        raise ValueError(f"NaN/Inf loss at epoch {epoch}, iter {_iter}")

                tr_loss += loss.detach().cpu().item()

                # Backward
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if clip_limit is not None and clip_limit > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_limit)

                optimizer.step()

                # Train iteration end callbacks
                for cb in callbacks.get('train_iter_end', []):
                    cb(model=model, iteration=_iter, epoch=epoch, loss=loss.item(),
                       dataloader=train_dataloader, optimizer=optimizer, device=device,
                       history=history, **callbacks_arguments)

                # Update progress description
                if _iter % update_period == 0:
                    avg_loss = tr_loss / _iter
                    if show_pbar == 'internal':
                        pbar_iter.set_description(descr.format(epoch=epoch, tr_loss=avg_loss,
                                                               val_loss=val_loss, best_epoch=best_epoch,
                                                               best_loss=best_loss))
                    elif show_pbar == 'external':
                        pbar_epoch.set_description(descr.format(epoch=f"{epoch}[{_iter}/{len(pbar_iter)}]",
                                                                tr_loss=avg_loss, val_loss=val_loss,
                                                                best_epoch=best_epoch, best_loss=best_loss))

            # End of epoch
            tr_loss /= _iter
            hist_train_epoch['loss'] = tr_loss
            history['train'][epoch] = hist_train_epoch

            # Train end callbacks
            for cb in callbacks.get('train_end', []):
                cb(model=model, epoch=epoch, tr_loss=tr_loss, best_loss=best_loss,
                   dataloader=train_dataloader, optimizer=optimizer, scheduler=scheduler,
                   device=device, history=history, **callbacks_arguments)

            # Optional evaluation on training set
            if evaluate_training:
                train_metrics = evaluate(
                    model=model,
                    eval_dataloader=train_dataloader,
                    criteria=eval_criteria,
                    sample_weight=sample_weight,
                    device=device,
                    show_pbar=(show_pbar == 'internal'),
                    callbacks=callbacks,
                    callbacks_arguments={'epoch': epoch, **callbacks_arguments}
                )
                history['train'][epoch].update(train_metrics)

            # Validation
            val_metrics = evaluate(
                model=model,
                eval_dataloader=eval_dataloader,
                criteria=eval_criteria,
                sample_weight=sample_weight,
                device=device,
                show_pbar=(show_pbar == 'internal'),
                callbacks=callbacks,
                callbacks_arguments={'epoch': epoch, **callbacks_arguments}
            )
            history['val'][epoch] = val_metrics
            val_loss = val_metrics['loss']

            # Update progress bars
            if show_pbar == 'internal':
                pbar_iter.set_description(descr.format(epoch=epoch, tr_loss=tr_loss, val_loss=val_loss,
                                                       best_epoch=best_epoch, best_loss=best_loss))
                pbar_iter.close()
            elif show_pbar == 'external':
                pbar_epoch.set_description(descr.format(epoch=epoch, tr_loss=tr_loss, val_loss=val_loss,
                                                        best_epoch=best_epoch, best_loss=best_loss))

            # Save checkpoint
            if _saving_steps > 0 and (epoch % _saving_steps == 0):
                save_model(model, os.path.join(model_dirname, 'last'), device)
                save_dict(history, os.path.join(top_dirname, 'history'))

            # Early stopping and best model saving
            if val_loss < best_loss:
                best_loss = val_loss
                save_model(model, os.path.join(model_dirname, 'best'), device)
                patience_counter = 0
                best_epoch = epoch
            elif patience is not None and patience > 0:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            # Epoch end callbacks
            for cb in callbacks.get('epoch_end', []):
                cb(model=model, epoch=epoch, epochs=epochs, tr_loss=tr_loss, val_loss=val_loss,
                   best_loss=best_loss, train_dataloader=train_dataloader,
                   eval_dataloader=eval_dataloader, optimizer=optimizer, scheduler=scheduler,
                   device=device, history=history, **callbacks_arguments)

            # Step scheduler
            scheduler.step(val_loss)

        if show_pbar == 'external':
            pbar_epoch.close()

    except Exception as e:
        if show_pbar == 'external':
            pbar_epoch.close()
        for cb in callbacks.get('exception_raised', []):
            cb(model=model, epoch=epoch, val_loss=val_loss, best_loss=best_loss,
               train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
               optimizer=optimizer, scheduler=scheduler, device=device, history=history,
               exception=e, **callbacks_arguments)
        save_model(model, os.path.join(model_dirname, 'last'), device)
        save_dict(history, os.path.join(top_dirname, 'history'))
        raise e

    # Final save if needed
    if _saving_steps < 1 or (epoch % _saving_steps != 0):
        save_model(model, os.path.join(model_dirname, 'last'), device)
        save_dict(history, os.path.join(top_dirname, 'history'))

    # Training finished callbacks
    for cb in callbacks.get('training_finished', []):
        cb(model=model, epoch=epoch, tr_loss=tr_loss, val_loss=val_loss,
           best_loss=best_loss, train_dataloader=train_dataloader,
           eval_dataloader=eval_dataloader, optimizer=optimizer, scheduler=scheduler,
           device=device, history=history, **callbacks_arguments)

    return history