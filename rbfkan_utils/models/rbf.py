
import torch
import torch.nn as nn
from torch.autograd import Function
from .params import get_rbf_mode


class RadialBasisFunction(nn.Module):
    def __init__(self, mode):
        super(RadialBasisFunction, self).__init__()
        self.rbf, self.mode = get_rbf_mode(mode)

    def extra_repr(self) -> str:
        if hasattr(self, 'mode'):
            return f"mode={self.mode}"
        return ''

    def forward(self, x):
        return self.rbf(x)


class RSWAFFunction(Function):
    @staticmethod
    def forward(ctx, input, grid, inv_denominator):
        diff = (input[..., None] - grid)
        diff_mul = diff.mul(inv_denominator) 
        tanh_diff = torch.tanh(diff_mul)
        tanh_diff_deriviative = 1. - tanh_diff ** 2  # sech^2(x) = 1 - tanh^2(x)

        ctx.save_for_backward(inv_denominator, diff_mul, tanh_diff, tanh_diff_deriviative) # Save tensors for backward pass

        return tanh_diff_deriviative
    
    @staticmethod
    def backward(ctx, grad_output,train_grid: bool = True, train_inv_denominator: bool = True, gradient_boost=10):
        inv_denominator, diff_mul, tanh_diff, tanh_diff_deriviative = ctx.saved_tensors
        grad_grid = grad_inv_denominator = None
        
        deriv = -2 * inv_denominator * tanh_diff * tanh_diff_deriviative * grad_output

        # Compute the backward pass for the input
        grad_input =  deriv.sum(dim=-1)
        ctx.train_grid = train_grid
        ctx.train_inv_denominator = train_inv_denominator

        # Compute the backward pass for grid
        if ctx.train_grid:
            grad_grid = - gradient_boost * deriv.sum(dim=-2) 

        # Compute the backward pass for inv_denominator        
        if ctx.train_inv_denominator:
            grad_inv_denominator = gradient_boost * (diff_mul * deriv).sum(0)

            if inv_denominator.view(-1).size(0) == 1 :
                grad_inv_denominator = grad_inv_denominator.sum()
                
        return grad_input, grad_grid, grad_inv_denominator
