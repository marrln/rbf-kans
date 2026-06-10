"""
This module defines utility functions and constants for configuring the RBF modes in the RBFKAN model. 
It includes a mapping of supported RBF modes to their corresponding implementations, as well as a
function to resolve a given mode name or callable into a standardized format for use in the model.
"""

from typing import Callable, Literal
import torch

from .mode import RSWAFF, PReLUGlobalParam, Tanh2, Gaussian, Sinc

USE_BIAS_ON_LINEAR = False  # Required for FPGA compatibility

RBF_MODE = Literal[
    "RSWAFF",
    "PRELU",
    "TANH2",
    "GAUSSIAN",
    "SAMPLE",
]

_CUSTOM_MODES = {
    "RSWAFF": RSWAFF,
    "PRELU": PReLUGlobalParam,
    "TANH2": Tanh2,
    "GAUSSIAN": Gaussian,
    "SAMPLE": Sinc,
}

RBF_MODES = tuple(_CUSTOM_MODES)

def get_rbf_mode(mode: str | Callable):
    """Resolve a radial basis function mode into a callable and canonical name.

    Parameters
    ----------
    mode : str | Callable
        A mode name to look up in custom RBF modes, a torch.nn activation class
        name, or a callable object to use directly.

    Returns
    -------
    tuple[Callable, str]
        The resolved callable and the normalized mode name.

    Raises
    ------
    ValueError
        If the supplied mode is neither a supported custom mode nor a
        recognized torch.nn activation name.
    """
    if callable(mode) and not isinstance(mode, str):
        return mode, getattr(mode, "__name__", str(mode))

    if not isinstance(mode, str):
        raise ValueError(f"Unsupported RBF mode: {mode}")

    mode_name = mode.upper()
    custom_mode = _CUSTOM_MODES.get(mode_name)
    if custom_mode is not None:
        return custom_mode(), mode_name 
    
    if hasattr(torch.nn, mode):
        return getattr(torch.nn, mode)(), mode

    raise ValueError(f"Unsupported RBF mode: {mode}")
