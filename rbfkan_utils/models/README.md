# RBF-KAN Models

This submodule provides the core building blocks for **Radial Basis Function Kolmogorov–Arnold Networks (RBF-KAN)**.  
It implements trainable radial basis function layers with various activation modes, supporting both static and dynamically predicted RBF parameters.

## Overview

RBF-KAN replaces traditional linear layers with a combination of:
- A radial basis function applied to scaled differences between inputs and trainable grid points.
- A linear projection that mixes the RBF outputs.

The submodule includes:
- Custom RBF modes (e.g., `RSWAF`, `GAUSSIAN`, `TANH2`).
- Multiple layer variants (`RBFKANLayer`, `RBFKANLayerV2`, `DynamicRBFKANLayer`).
- A full `RBFKAN` model that stacks layers with optional residual connections and normalisation.

## File Descriptions

### `mode.py`
Defines custom activation functions / basis functions usable as RBF modes.

| Class / Function | Description |
|------------------|-------------|
| `LambdaModule` | Wraps an arbitrary callable as a `nn.Module`. |
| `RSWAF` | Returns `1 - tanh(x)^2` (the derivative of tanh, i.e. `sech²(x)`). |
| `PReLUGlobalParam` | PReLU with a single learnable parameter shared across all channels. |
| `tanh2(x)` | Returns `tanh(x)²`. |
| `gaussian(x)` | Returns `exp(-x²)`. |
| `sinc(x, guard=1e-8)` | Normalised sinc function `sin(x+ε)/(x+ε)`. |

### `params.py`
Utility functions and constants for RBF mode configuration.

| Constant / Function | Description |
|----------------------|-------------|
| `USE_BIAS_ON_LINEAR` | Boolean flag (set `False` for FPGA compatibility). |
| `RBF_MODE` | Literal type: `"RSWAF"`, `"PRELU"`, `"TANH2"`, `"GAUSSIAN"`, `"SAMPLE"`. |
| `get_rbf_mode(mode)` | Resolves a mode name or callable into a `(callable, name)` tuple. Supports custom modes and `torch.nn` activations (e.g. `"ReLU"`). |

### `rbf.py`
Implements the base RBF module and a custom autograd function for the `RSWAF` mode.

| Class | Description |
|-------|-------------|
| `RadialBasisFunction` | Applies the resolved RBF callable elementwise. |
| `RSWAFFunction` | Custom `torch.autograd.Function` that computes `1 - tanh²(diff)` and provides manually defined gradients. Used by the `RBF` class in `rbfkan_modules.py`. |

### `rbfkan_modules.py`
Low-level RBF modules that compute scaled differences and apply the radial basis function.

| Class | Description |
|-------|-------------|
| `RBF` | Hardcoded to use `RSWAFFunction`. Manages trainable grid and inverse denominator. |
| `RBFAuto` | Generic version: applies any RBF mode to `(x - grid) * inv_denominator`. |
| `RBFAutoV2` | Like `RBFAuto` but supports per‑input‑dimension grid and inverse denominator. |
| `DynamicRBFAuto` | Uses a small `RBFKANLayer` to predict grid points and scale for each input sample before applying the RBF. |

### `rbfkan_layers.py`
Complete RBF-KAN layers that combine RBF expansion with a linear layer, dropout, and optional normalisation.

| Class | Description |
|-------|-------------|
| `RBFKANLayer` | Standard layer using `RBFAuto`. |
| `RBFKANLayerV2` | Uses `RBFAutoV2` (per‑dimension grids). |
| `DynamicRBFKANLayer` | Uses `DynamicRBFAuto` – grid and scale are predicted dynamically. |

All layers accept:
- `input_dim`, `output_dim`
- `grid_min`, `grid_max`, `num_grids`, `inv_denominator`
- `mode` (RBF mode)
- `dropout_rate`, `dropout_linear`
- `train_grid`, `train_inv_denominator`
- `normalize_rbf` (adds `LayerNorm` on flattened RBF features)

### `rbfkan.py`
High‑level model that stacks multiple RBF-KAN layers.

**Class `RBFKAN`**

| Parameter | Description |
|-----------|-------------|
| `hidden_layers` | List of integers defining input and hidden dimensions, e.g. `[784, 256, 128, 10]`. |
| `num_grids` | Number of RBF grid points per layer (int or list). |
| `grid_min`, `grid_max` | Grid boundaries (float or list). |
| `inv_denominator` | Initial inverse denominator (float or list). |
| `mode` | RBF mode (string, callable, or torch.nn class). |
| `residual` | Boolean or list: enable residual connections where dimensions match. |
| `dynamic` | Use `DynamicRBFKANLayer` if `True`. |
| `use_v2` | Use `RBFKANLayerV2` (per‑dimension grids) if `True`. |
| `normalize` | Apply `LayerNorm` before each layer (except the first). |
| `normalize_rbf` | Apply `LayerNorm` on the flattened RBF features inside each layer. |
| `dropout_rate` | Dropout after RBF expansion. |
| `dropout_linear` | Dropout after the linear layer. |

The model automatically handles:
- Expanding scalar parameters to match the number of layers.
- Disabling training of grid/inv_denominator in `eval()` mode.
- Residual connections (when dimensions match and `residual=True`).

## Usage Example

```python
import torch
from rbfkan_utils.models.rbfkan import RBFKAN

# 3‑layer model: input size 10, hidden 32, output 5
model = RBFKAN(
    hidden_layers=[10, 32, 5],
    num_grids=8,
    grid_min=-2.0,
    grid_max=2.0,
    inv_denominator=1.0,
    mode="GAUSSIAN",
    residual=[False, True],   # residual on the last layer only
    dropout_rate=0.1
)

x = torch.randn(4, 10)   # batch size 4
out = model(x)           # shape (4, 5)
```

## Notes

- *FPGA Compatibility*: To ensure compatibility with FPGA implementations, the `USE_BIAS_ON_LINEAR` constant is set to `False`, meaning that the linear layers in RBF-KAN do not use bias terms. This is because bias parameters can complicate hardware implementation and may not be supported in certain FPGA designs.

- *Dynamic Mode*: When `dynamic=True`, the model uses `DynamicRBFKANLayer`, which predicts the RBF grid and scale for each input sample using a small internal `RBFKANLayer`. This allows the RBF transformation to adapt dynamically based on the input features, potentially improving performance on complex tasks.

- *Gradient Boosting*: In `RSWAFFunction.backward`, the gradients are manually boosted by a factor of 10. This is a heuristic to encourage stronger updates to the grid and inverse denominator parameters, which can be crucial for training stability and convergence in RBF-KAN models.

- *Residual Connections*: The `residual` parameter can be a boolean or a list. If `True`, residual connections are added for all layers where the input and output dimensions match. If a list is provided, it specifies which layers should have residual connections. This flexibility allows for easier experimentation with different architectures.