import torch
import torch.nn as nn


class LambdaModule(nn.Module):
    """A module that applies a given function to its input."""
    def __init__(self, func):
        super().__init__()
        if isinstance(func, str):
            self.func = eval(func)
        else:
            self.func = func
        
    def extra_repr(self):
        return f'func={self.func}'
        
    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    
class RSWAFF(nn.Module):
    """RSWAFF activation function: 1 - tanh(x)^2"""
    def __init__(self):
        super(RSWAFF, self).__init__()
        self.tanh = torch.nn.Tanh()
        
    def forward(self, x):
        """x should be x-grid * inv_denominator where grid is the grid point and inv_denominator is the inverse denominator"""
        return torch.ones_like(x) - self.tanh(x) ** 2


class PReLUGlobalParam(nn.Module):
    """PReLU with a single learnable parameter for all channels"""
    def __init__(self, init=0.25):
        super(PReLUGlobalParam, self).__init__()
        self.prelu = nn.PReLU(num_parameters=1, init=init)
        
    def forward(self, x):
        return self.prelu(x)


class Gaussian(torch.nn.Module):
    """"Gaussian activation function: exp(-x^2) """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-x**2)


class Tanh2(torch.nn.Module):
    """"Tanh squared activation function: tanh(x)^2 """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)**2


class Sinc(torch.nn.Module):
    """"Sinc activation function: sin(pi*x)/(pi*x) """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sinc(x)   # note: torch.sinc is πx version; adjust if needed
