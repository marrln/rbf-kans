from warnings import warn
import torch
import torch.nn as nn
from torch.nn import functional as F
from ..utils import expand_value
from typing import List, Union, Callable, Optional
from .params import RBF_MODE
from .rbfkan_layers import RBFKANLayer, RBFKANLayerV2, DynamicRBFKANLayer


class RBFKAN(nn.Module):
    def __init__(
        self, hidden_layers: List[int], 
        num_grids: Union[int, List[int]],
        grid_min: float,
        grid_max: float,
        inv_denominator: float,
        mode : Callable | RBF_MODE = 'RSWAF',
        residual : list[bool] = False,
        dynamic : bool = False,
        use_v2 : bool = False,
        normalize : bool = True,
        normalize_rbf : bool = False,
        dropout_rate: Optional[float] = None,
        dropout_linear: Optional[float] = None,
    ):
        super(RBFKAN, self).__init__()

        self.train_grid = True
        self.train_inv_denominator = True
        self.dropout_rate = dropout_rate
        self.dropout_linear = dropout_linear
        self.normalize_rbf = normalize_rbf
        
        num_grids       = expand_value(num_grids,       len(hidden_layers)-1)
        grid_min        = expand_value(grid_min,        len(hidden_layers)-1)
        grid_max        = expand_value(grid_max,        len(hidden_layers)-1)
        inv_denominator = expand_value(inv_denominator, len(hidden_layers)-1)
        residual        = expand_value(residual,        len(hidden_layers)-1)
        
        if dynamic :
            LayerClass = DynamicRBFKANLayer
        elif use_v2 :
            LayerClass = RBFKANLayerV2
        else :
            LayerClass = RBFKANLayer
        
        self.residual   = []
        for _iter, residual_i in enumerate(residual):
            if residual_i :
                if hidden_layers[_iter] == hidden_layers[_iter+1] :
                    self.residual.append(True)
                else :
                    warn(f"Skipped residual connection at layer {_iter}; Number of features do not match ({hidden_layers[_iter]} != {hidden_layers[_iter+1]})")
                    self.residual.append(False)
            else :
                self.residual.append(False)
        
        self.layers = nn.ModuleList([
            LayerClass(
                train_grid            = self.train_grid,
                train_inv_denominator = self.train_inv_denominator,
                input_dim             = in_dim, 
                output_dim            = out_dim, 
                grid_min              = grid_min_i,
                grid_max              = grid_max_i,
                num_grids             = num_grids_i,
                inv_denominator       = inv_denominator_i,
                mode                  = mode,
                dropout_rate          = self.dropout_rate,
                dropout_linear        = self.dropout_linear,
                normalize_rbf         = self.normalize_rbf,
            ) for _iter, (
                num_grids_i, 
                in_dim, 
                out_dim, 
                grid_min_i, 
                grid_max_i, 
                inv_denominator_i, 
            ) in enumerate(zip(
                num_grids, 
                hidden_layers[:-1], 
                hidden_layers[1:],
                grid_min,
                grid_max,
                inv_denominator,
            ))
        ])
        if normalize and len(hidden_layers) >= 2:
            self.normalize = nn.ModuleList([
                nn.LayerNorm(out_dim) for out_dim in hidden_layers[1:-1]
            ])
        else :
            self.normalize = False

    def extra_repr(self) -> str:
        return f"residual={tuple(self.residual)}"

    def eval(self):
        self.is_eval = True
        self.train_grid = False
        self.train_inv_denominator = False
        return super().eval()

    def train(self, mode=True):
        self.is_eval = not mode
        self.train_grid = mode
        self.train_inv_denominator = mode
        return super().train(mode)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _iter, (layer, res) in enumerate(zip(self.layers, self.residual)):
            
            identity = x
            x = layer(x)
            
            if self.normalize and _iter < len(self.layers) - 1:
                x = self.normalize[_iter](x) 
            
            if res:
                x = x + identity
            
            if _iter < len(self.layers) - 1:
                x = F.relu(x)
        return x
