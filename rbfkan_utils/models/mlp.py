import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
    """
    A single block within the MLP containing a Linear transformation,
    optional Layer Normalization, ReLU activation, Dynamic Dropout, 
    and optional Residual matching.
    """
    def __init__(self, in_dim, out_dim, use_residual=False, use_normalize=True, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.use_residual = use_residual
        
        self.linear = nn.Linear(in_dim, out_dim)
        
        if not self.is_last:
            self.norm = nn.LayerNorm(out_dim) if use_normalize else nn.Identity()
            self.act = nn.ReLU()
            
        if self.use_residual:
            # If input and output features don't match, project input to the output size
            if in_dim != out_dim:
                self.shortcut = nn.Linear(in_dim, out_dim)
            else:
                self.shortcut = nn.Identity()

    def forward(self, x, dropout_rate=0.0):
        identity = x
        x = self.linear(x)
        
        if not self.is_last:
            x = self.norm(x)
            x = self.act(x)
            if dropout_rate > 0.0:
                x = F.dropout(x, p=dropout_rate, training=self.training)
                
        if self.use_residual:
            x = x + self.shortcut(identity)
            
        return x


class MLP(nn.Module):
    """
    Generalized Multi-Layer Perceptron architecture with support for 
    Layer Normalization, Residual paths, and Dynamic Scheduled Dropout.
    """
    def __init__(self, hidden_layers, residual=False, normalize=True, dropout_rate=0.0):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.residual = residual
        self.normalize = normalize
        self.dropout_rate = dropout_rate  # Can safely handle standard float or UpdatableFloat objects
        
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            in_dim = hidden_layers[i]
            out_dim = hidden_layers[i+1]
            is_last = (i == len(hidden_layers) - 2)
            
            self.blocks.append(
                MLPBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    use_residual=residual,
                    use_normalize=normalize,
                    is_last=is_last
                )
            )

    def forward(self, x):
        # Flatten image-like inputs safely down to (Batch, Features)
        if x.dim() > 2:
            x = x.flatten(1)
            
        # Extract dropout value smoothly regardless of if it's a raw float or an object wrapper
        p = 0.0
        if hasattr(self.dropout_rate, 'value'):
            p = float(self.dropout_rate.value)
        elif hasattr(self.dropout_rate, 'val'):
            p = float(self.dropout_rate.val)
        else:
            try:
                p = float(self.dropout_rate)
            except (TypeError, ValueError):
                p = 0.0
                
        for block in self.blocks:
            x = block(x, dropout_rate=p)
            
        return x