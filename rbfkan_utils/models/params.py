"""
This module defines utility functions and constants for configuring the RBF modes in the RBFKAN model. 
It includes a mapping of supported RBF modes to their corresponding implementations, as well as a
function to resolve a given mode name or callable into a standardized format for use in the model.
"""

from typing import Callable, Literal
import torch

from .mode import RSWAF, PReLUGlobalParam, Tanh2, Gaussian, Sinc

USE_BIAS_ON_LINEAR = False  # Required for FPGA compatibility

RBF_MODE = Literal[
    "RSWAF",
    "PRELU",
    "TANH2",
    "GAUSSIAN",
    "SAMPLE",
    # Add more modes here as needed
]

_CUSTOM_MODES = {
    "RSWAF": RSWAF,
    "PRELU": PReLUGlobalParam,
    "TANH2": Tanh2,
    "GAUSSIAN": Gaussian,
    "SAMPLE": Sinc,
    # Add more custom modes here as needed
}

RBF_MODES = tuple(_CUSTOM_MODES)

def get_rbf_mode(mode: str | Callable):
    """Resolve the RBF mode to its corresponding implementation."""
    if callable(mode) and not isinstance(mode, str):
        return mode, getattr(mode, "__name__", str(mode))

    if not isinstance(mode, str):
        raise ValueError(f"Unsupported RBF mode: {mode}")

    mode_name = mode.upper()
    custom_mode = _CUSTOM_MODES.get(mode_name)
    if custom_mode is not None:
        return custom_mode(), mode_name 
    
    if hasattr(torch.nn, mode):
        # Return the corresponding PyTorch activation function if it exists
        return getattr(torch.nn, mode)(), mode

    raise ValueError(f"Unsupported RBF mode: {mode}")
