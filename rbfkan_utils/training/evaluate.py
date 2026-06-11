from typing import Union, Callable, Dict, List, Any, Optional
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os
from ..utils import to, cat, isnested, nested2dict


def get_callable_basis() -> Dict[str, List[Callable]]:
    """Return default callback dictionary (same as original)."""
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
    model: Module,
    eval_dataloader: DataLoader,
    criteria: Dict[str, Callable[[torch.Tensor], torch.Tensor]],
    keep_copy: bool = True,
    checkpoint_path: Optional[str] = None,
    epoch: Optional[int] = None,
    sample_weight: bool = False,
    show_pbar: bool = True,
    device: torch.device = torch.device('cpu'),
    callbacks: Optional[Dict[str, List[Callable]]] = None,
    callbacks_arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Union[float, List[float]]]:
    """
    Evaluate a model on a dataloader with multiple criteria.
    Returns a dictionary of metric values.
    """
    if not criteria:
        return {}

    # Prepare callbacks (merge user-provided with defaults)
    default_callbacks = get_callable_basis()
    if callbacks is None:
        callbacks = default_callbacks
    else:
        # Ensure all keys exist
        for key in default_callbacks:
            if key not in callbacks:
                callbacks[key] = []
        for key in callbacks:
            if key not in default_callbacks:
                # Unknown callback stage – ignore or warn? We'll keep it.
                pass

    if callbacks_arguments is None:
        callbacks_arguments = {}

    # Store original training mode and device (to restore later)
    original_training = model.training
    model.eval()

    # Determine model's device (use it instead of moving model)
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = device  # fallback if model has no parameters

    # Move model to user-specified device (if different)
    if model_device != device:
        model.to(device)
        model_device = device

    preds = []
    targs = []
    keys = []
    if sample_weight:
        weights = []

    # Progress bar setup
    if show_pbar:
        pbar = tqdm(eval_dataloader, desc=f"Eval epoch {epoch}" if epoch is not None else "Evaluating")
    else:
        pbar = eval_dataloader

    with torch.no_grad():
        # ---- eval_start callbacks ----
        loc_kwargs = {
            'model': model,
            'epoch': epoch,
            'eval_dataloader': eval_dataloader,
            'device': model_device,
        }
        loc_kwargs.update(callbacks_arguments)
        for cb in callbacks['eval_start']:
            cb(**loc_kwargs)

        # ---- Iteration over batches ----
        for batch in pbar:
            # Batch unpacking: assume batch can be (data, target) or (data, target, weight) or (data, target, key) or (data, target, weight, key)
            # We'll handle by index based on sample_weight flag.
            data = batch[0]
            target = batch[1]
            weight = None
            key = None

            idx = 2
            if sample_weight:
                if len(batch) > idx:
                    weight = batch[idx]
                    idx += 1
                else:
                    raise ValueError("sample_weight=True but batch does not contain a weight tensor.")
            if len(batch) > idx:
                key = batch[idx]
                if isinstance(key, torch.Tensor):
                    # Convert tensor of keys to list (assuming each element is a sample key)
                    key = key.tolist()
                elif not isinstance(key, list):
                    # Wrap single key in list for uniform processing
                    key = [key]
            # If key is None or empty, we'll keep it as None (no keys for this batch)

            # Move data to model's device
            data = to(data, model_device)
            target = to(target, model_device)

            # ---- eval_iter_start callbacks ----
            loc_kwargs = {
                'model': model,
                'epoch': epoch,
                'data': data,
                'target': target,
                'key': key,
                'dataloader': eval_dataloader,
                'device': model_device,
            }
            loc_kwargs.update(callbacks_arguments)
            for cb in callbacks['eval_iter_start']:
                cb(**loc_kwargs)

            # Forward pass
            prediction = to(model(data), 'cpu')  # store on CPU to save GPU memory

            # ---- eval_iter_end callbacks ----
            loc_kwargs = {
                'model': model,
                'epoch': epoch,
                'prediction': prediction,
                'target': target.cpu(),  # keep on CPU for storage
                'key': key,
                'dataloader': eval_dataloader,
                'device': model_device,
            }
            loc_kwargs.update(callbacks_arguments)
            for cb in callbacks['eval_iter_end']:
                cb(**loc_kwargs)

            # Store batch results
            preds.append(prediction)
            targs.append(target.cpu())
            if sample_weight and weight is not None:
                weights.append(weight.cpu() if torch.is_tensor(weight) else weight)
            if key is not None:
                keys.extend(key)  # key is already a list

        # ---- After all batches ----
        # Concatenate all stored tensors
        prediction = cat(preds)   # shape (N, ...)
        target = cat(targs)
        del preds, targs

        if sample_weight and weights:
            weight = torch.cat(weights) if torch.is_tensor(weights[0]) else torch.tensor(weights)
            del weights
        else:
            weight = None

        # Move tensors to model device for metric computation (if needed)
        # But keep original CPU copies for CSV writing
        prediction_cpu = prediction
        target_cpu = target
        if weight is not None:
            weight_cpu = weight

        prediction = to(prediction, model_device)
        target = to(target, model_device)
        if weight is not None:
            weight = to(weight, model_device)

        # Close progress bar if it exists and was created by tqdm
        if show_pbar and isinstance(pbar, tqdm):
            pbar.close()

        # ---- eval_metrics_start callbacks ----
        loc_kwargs = {
            'eval_criteria': criteria,
            'epoch': epoch,
            'prediction': prediction,
            'target': target,
            'key': keys,          # now full list of keys
            'dataloader': eval_dataloader,
            'device': model_device,
        }
        loc_kwargs.update(callbacks_arguments)
        for cb in callbacks['eval_metrics_start']:
            cb(**loc_kwargs)

        # ---- Compute metrics ----
        metrics = {}
        for name, criterion in criteria.items():
            # Move criterion to device if it has .to() method (e.g., torch.nn.Module)
            if hasattr(criterion, 'to'):
                criterion.to(model_device)

            try:
                if sample_weight and weight is not None:
                    try:
                        # Try with weight argument
                        value = criterion(prediction, target, weight)
                    except TypeError:
                        # Criterion doesn't accept weight -> fallback to unweighted
                        value = criterion(prediction, target)
                else:
                    value = criterion(prediction, target)
            except Exception as e:
                print(f'Warning -- metric "{name}": {e}')
                continue

            # Keep metric as is (could be scalar tensor, float, or list)
            # Convert to Python float if it's a 0-dim tensor
            if torch.is_tensor(value) and value.numel() == 1:
                value = value.item()
            metrics[name] = value

            # Move criterion back to CPU if it was moved
            if hasattr(criterion, 'to'):
                criterion.to('cpu')

        # ---- Save predictions to CSV if requested ----
        if keep_copy and keys and checkpoint_path is not None:
            rslt_path = os.path.join(os.path.dirname(checkpoint_path), f"rslt.csv" if epoch is None else f"{epoch}.csv")
            os.makedirs(os.path.dirname(rslt_path), exist_ok=True)

            # Ensure prediction and target are at least 2D for DataFrame building
            def ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
                if tensor.dim() == 1:
                    return tensor.unsqueeze(1)  # (N,) -> (N,1)
                return tensor

            pred_2d = ensure_2d(prediction_cpu)
            targ_2d = ensure_2d(target_cpu)

            # Flatten extra dimensions (e.g., (N, D) -> (N, D) is fine; we keep as columns)
            # apply_to_tensor can handle flattening, but we want to avoid errors with 1D case.
            # We'll simply use the already 2D tensors.
            # If nested (e.g., dict of tensors), handle separately.
            try:
                if isnested(pred_2d):
                    pred_dict = nested2dict(pred_2d)
                    targ_dict = nested2dict(targ_2d)
                    df_data = {'Index': keys}
                    for key_name, val in targ_dict.items():
                        # val shape: (N, feature_dim)
                        for i in range(val.shape[-1]):
                            df_data[f"targ_{key_name}_{i}"] = val[:, i].tolist()
                    for key_name, val in pred_dict.items():
                        for i in range(val.shape[-1]):
                            df_data[f"pred_{key_name}_{i}"] = val[:, i].tolist()
                    df = pd.DataFrame(df_data).set_index('Index').sort_index()
                else:
                    # Simple 2D tensor: shape (N, D)
                    df_data = {'Index': keys}
                    for i in range(targ_2d.shape[-1]):
                        df_data[f"targ_{i}"] = targ_2d[:, i].tolist()
                    for i in range(pred_2d.shape[-1]):
                        df_data[f"pred_{i}"] = pred_2d[:, i].tolist()
                    df = pd.DataFrame(df_data).set_index('Index').sort_index()
                df.to_csv(rslt_path)
                print(f'Results written to "{rslt_path}"')
            except Exception as e:
                print(f"Warning: could not save CSV at {rslt_path}: {e}")

        # ---- eval_end callbacks (pass full keys list) ----
        loc_kwargs = {
            'metrics': metrics,
            'epoch': epoch,
            'prediction': prediction_cpu,   # provide CPU version for possible serialization
            'target': target_cpu,
            'key': keys,                    # full list of sample keys
            'eval_dataloader': eval_dataloader,
            'device': model_device,
        }
        loc_kwargs.update(callbacks_arguments)
        for cb in callbacks['eval_end']:
            cb(**loc_kwargs)

    # Restore original training mode (and optionally device if we changed it)
    model.train(original_training)
    # If we moved the model to a different device and want to restore, uncomment below:
    # if model_device != device:
    #     model.to(device)

    return metrics