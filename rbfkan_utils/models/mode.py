import torch
import torch.nn as nn

class LambdaModule(nn.Module):
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
    def __init__(self):
        super(RSWAFF, self).__init__()
        self.tanh = torch.nn.Tanh()
        
    def forward(self, x):
        return torch.ones_like(x) - self.tanh(x) ** 2

class PReLUGlobalParam(nn.Module):
    """PReLU with a single learnable parameter for all channels"""
    def __init__(self, init=0.25):
        super(PReLUGlobalParam, self).__init__()
        self.prelu = nn.PReLU(num_parameters=1, init=init)
        
    def forward(self, x):
        return self.prelu(x)

class Gaussian(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-x**2)

class Tanh2(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)**2

class Sinc(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sinc(x)   # note: torch.sinc is πx version; adjust if needed
