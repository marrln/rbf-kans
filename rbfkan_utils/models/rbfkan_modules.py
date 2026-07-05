
import torch
import torch.nn as nn
from typing import Optional
from .params import RBF_MODE
from .rbf import RadialBasisFunction, RSWAFFunction


class RBF(nn.Module):
    def __init__(
        self,
        train_grid: bool,        
        train_inv_denominator: bool,
        grid_min: float,
        grid_max: float,
        num_grids: int,
        inv_denominator: float
    ):
        super(RBF,self).__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.scale = inv_denominator
        
        grid = torch.linspace(grid_min, grid_max, num_grids).float()

        self.train_grid = train_grid
        self.train_inv_denominator = train_inv_denominator

        self.grid = torch.nn.Parameter(grid, requires_grad=train_grid)
        self.inv_denominator = torch.nn.Parameter(torch.tensor(inv_denominator).float(), requires_grad=train_inv_denominator)  # Cache the inverse of the denominator

    def extra_repr(self) -> str:
        return f"num_grids={self.num_grids}, grid_min={self.grid_min}, grid_max={self.grid_max}, inv_denominator={self.scale}"

    def forward(self, x):
        return RSWAFFunction.apply(x, self.grid, self.inv_denominator) # returns tanh_diff_derivative


class RBFAuto(nn.Module):
    def __init__(
        self,
        train_grid: bool,        
        train_inv_denominator: bool,
        grid_min: float,
        grid_max: float,
        num_grids: int,
        inv_denominator: float,
        mode : RBF_MODE = 'RSWAF'
    ):
        super(RBFAuto,self).__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.scale = inv_denominator
        
        grid = torch.linspace(grid_min, grid_max, num_grids).float()
        self.grid = torch.nn.Parameter(grid, requires_grad=train_grid)
        self.inv_denominator = torch.nn.Parameter(torch.tensor(inv_denominator).float(), requires_grad=train_inv_denominator)  # Cache the inverse of the denominator
        self.rbf = RadialBasisFunction(mode)

    def extra_repr(self) -> str:
        return f"num_grids={self.num_grids}, grid_min={self.grid_min}, grid_max={self.grid_max}, inv_denominator={self.scale}"

    def forward(self, x):
        diff = (x[..., None] - self.grid).mul(self.inv_denominator) 
        return self.rbf(diff)


class RBFAutoV2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        train_grid: bool,
        train_inv_denominator: bool,
        grid_min: float,
        grid_max: float,
        num_grids: int,
        inv_denominator: float,
        mode: RBF_MODE = 'RSWAF'
    ):
        super(RBFAutoV2, self).__init__()
        self.input_dim = input_dim
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.scale = inv_denominator

        # Create grid: one set of num_grids points per input dimension
        # Initialize by repeating the same linspace for each input
        base_grid = torch.linspace(grid_min, grid_max, num_grids).float()  # shape: (num_grids,)
        grid = base_grid.unsqueeze(0).expand(input_dim, -1).contiguous()   # shape: (input_dim, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=train_grid)

        # Inverse denominator: one scalar per input dimension
        inv_den = torch.full((input_dim,), inv_denominator, dtype=torch.float32)  # shape: (input_dim,)
        self.inv_denominator = nn.Parameter(inv_den, requires_grad=train_inv_denominator)

        # Radial basis function (applied elementwise)
        self.rbf = RadialBasisFunction(mode)

    def extra_repr(self) -> str:
        return (f"input_dim={self.input_dim}, num_grids={self.num_grids}, "
                f"grid_min={self.grid_min}, grid_max={self.grid_max}, "
                f"inv_denominator={self.scale}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        # grid: (input_dim, num_grids) -> unsqueeze(0) to (1, input_dim, num_grids)
        # inv_denominator: (input_dim,) -> unsqueeze(0).unsqueeze(-1) to (1, input_dim, 1)
        diff = (x.unsqueeze(-1) - self.grid.unsqueeze(0)) * self.inv_denominator.unsqueeze(0).unsqueeze(-1)
        return self.rbf(diff)


class DynamicRBFAuto(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_grids: int,
        mode : RBF_MODE = 'RSWAF',
        dropout_rate: Optional[float] = None,
    ):
        super(DynamicRBFAuto,self).__init__()
        
        from .rbfkan_layers import RBFKANLayer
        
        self.mode = mode
        self.params_linear = RBFKANLayer(
            input_dim               = input_dim,
            output_dim              = num_grids + 1,
            num_grids               = 1,
            grid_min                = 0,
            grid_max                = 0,
            inv_denominator         = 1.0,
            train_grid              = True,
            train_inv_denominator   = True,
            mode                    = mode,
            dropout_rate            = dropout_rate,
        )
        self.rbf = RadialBasisFunction(self.mode)

    def extra_repr(self) -> str:
        return f"num_grids={self.params_linear.output_dim-1}"

    def forward(self, x):
        params = self.params_linear(x).unsqueeze(-2)
        grid, scale = params.split([self.params_linear.output_dim - 1, 1], dim=-1)
        diff = (x[..., None] - grid).mul(scale) 
        return self.rbf(diff)
