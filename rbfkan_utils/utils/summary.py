from typing import overload
import os
import torch
from torchinfo import summary, ModelStatistics

@overload
def get_summary(
    model: torch.nn.Module,
    input_data : torch.Tensor,
    dest = None,
    depth : int = 3,
) -> ModelStatistics :
    ...

@overload
def get_summary(
    model: torch.nn.Module,
    input_data : torch.Tensor,
    dest : str,
    depth : int = 3,
) -> str:
    ...
    
def get_summary(
    model: torch.nn.Module,
    input_data : torch.Tensor,
    dest = None,
    depth : int = 3,
):
    model_summary = summary(
        model, 
        input_data  = input_data.clone().unsqueeze(0), 
        verbose     = 0,
        depth       = depth
    )
    
    if dest is None:
        return model_summary
    
    if os.path.splitext(dest)[-1] != '.txt':
        dest += '.txt'
    
    with open(dest, 'w') as fw:
        fw.write(str(model_summary))

    return dest