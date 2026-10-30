from importlib.util import find_spec

from ._ops import ops

__all__ = ["ops"]

if find_spec("torch") is not None:
    from . import layers, torch_ops
    from .functional import RMSNormFunction, rms_norm

    __all__ += ["RMSNormFunction", "rms_norm", "layers", "torch_ops"]
