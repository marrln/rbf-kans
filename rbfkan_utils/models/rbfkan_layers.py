
import torch.nn as nn
from typing import Optional
from .params import USE_BIAS_ON_LINEAR, RBF_MODE
from .rbfkan_modules import RBFAuto, RBFAutoV2, DynamicRBFAuto


class RBFKANLayerV2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float,
        grid_max: float,
        num_grids: int,
        inv_denominator: float,
        mode: RBF_MODE = 'RSWAFF',
        dropout_rate: Optional[float] = None,
        dropout_linear: Optional[float] = None,
        train_grid: bool = True,        
        train_inv_denominator: bool = True,
        normalize_rbf: bool = False,
    ) -> None:
        super(RBFKANLayerV2, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize_rbf = normalize_rbf

        self.rbf = RBFAutoV2(
            input_dim=input_dim,
            train_grid=train_grid, 
            train_inv_denominator=train_inv_denominator,
            grid_min=grid_min, 
            grid_max=grid_max, 
            num_grids=num_grids, 
            inv_denominator=inv_denominator, 
            mode=mode
        )
        # Add normalization to stabilize RBF outputs before linear layer
        if normalize_rbf:
            self.rbf_norm = nn.LayerNorm(input_dim * num_grids)
        self.linear = nn.Linear(input_dim * num_grids, output_dim, bias=USE_BIAS_ON_LINEAR) 
        self.drop = nn.Dropout(0 if dropout_rate is None else dropout_rate)
        self.drop_linear = nn.Dropout(0 if dropout_linear is None else dropout_linear)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        spline_basis = self.rbf(x).view(batch_size, -1)
        if self.normalize_rbf:
            spline_basis = self.rbf_norm(spline_basis)  # Normalize before dropout and linear
        spline_basis = self.drop(spline_basis)
        output = self.linear(spline_basis)
        output = self.drop_linear(output)
        return output


class RBFKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float,
        grid_max: float,
        num_grids: int,
        inv_denominator: float,
        mode : RBF_MODE = 'RSWAFF',
        dropout_rate: Optional[float] = None,
        dropout_linear: Optional[float] = None,
        train_grid: bool = True,        
        train_inv_denominator: bool = True,
        normalize_rbf: bool = False,
    ) -> None:
        super(RBFKANLayer,self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize_rbf = normalize_rbf

        self.rbf = RBFAuto(train_grid, train_inv_denominator,grid_min, grid_max, num_grids, inv_denominator, mode=mode)
        # Add normalization to stabilize RBF outputs before linear layer
        if normalize_rbf:
            self.rbf_norm = nn.LayerNorm(input_dim * num_grids)
        self.linear = nn.Linear(input_dim * num_grids, output_dim, bias=USE_BIAS_ON_LINEAR) 
        
        self.drop = nn.Dropout(0 if dropout_rate is None else float(dropout_rate))
        self.drop_linear = nn.Dropout(0 if dropout_linear is None else float(dropout_linear))
        # self.drop = nn.Dropout(0 if dropout_rate is None else dropout_rate)
        # self.drop_linear = nn.Dropout(0 if dropout_linear is None else dropout_linear) 

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        spline_basis = self.rbf(x).view(batch_size, -1)
        if self.normalize_rbf:
            spline_basis = self.rbf_norm(spline_basis)  # Normalize before dropout and linear
        spline_basis = self.drop(spline_basis)
        output = self.linear(spline_basis)
        output = self.drop_linear(output)
        return output


class DynamicRBFKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_grids: int,
        mode : RBF_MODE = 'RSWAFF',
        dropout_rate: Optional[float] = None,
        dropout_linear: Optional[float] = None,
        **kwargs
    ) -> None:
        super(DynamicRBFKANLayer,self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.rbf = DynamicRBFAuto(input_dim, num_grids, mode=mode, dropout_rate=dropout_rate)
        self.linear = nn.Linear(input_dim * num_grids, output_dim, bias=USE_BIAS_ON_LINEAR) 
        self.drop = nn.Dropout(0 if dropout_rate is None else dropout_rate)
        self.drop_linear = nn.Dropout(0 if dropout_linear is None else dropout_linear)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        spline_basis = self.rbf(x).view(batch_size, -1)
        spline_basis = self.drop(spline_basis)
        output = self.linear(spline_basis)
        output = self.drop_linear(output)
        return output
