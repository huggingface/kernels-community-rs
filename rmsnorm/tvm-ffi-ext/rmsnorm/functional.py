from typing import Optional

import torch

from . import torch_ops

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # Torch built without distributed support.
    DTensor = None

CASTING_MODES = ("llama", "gemma", "none")


def _full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    # Gather tensor-parallel inputs to compute the norm on each worker.
    if DTensor is not None and isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


class RMSNormFunction(torch.autograd.Function):
    """
    RMSNorm: `x / rms(x) * (offset + weight)` over the last dimension of `x`.

    Some models shift the weight by a constant `offset`; Gemma, for example,
    uses `offset=1.0`. Models also differ in where they cast to float32, which
    `casting_mode` selects:

    - `"llama"`: only the inverse RMS is computed in float32 (Llama).
    - `"gemma"`: everything is computed in float32, then cast back (Gemma).
    - `"none"`: computed in the input dtype. Slightly faster, less accurate.

    `weight` may be `None` (no weight) and, for 16-bit `x`, float32.

    With `in_place`, the backward pass overwrites the output gradient with the
    input gradient to save memory. Don't use it when the output gradient is
    also needed elsewhere, e.g. when the norm output feeds a residual add.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        eps: float,
        offset: float = 0.0,
        casting_mode: str = "llama",
        in_place: bool = True,
    ) -> torch.Tensor:
        if casting_mode not in CASTING_MODES:
            raise ValueError(
                f"Invalid casting mode {casting_mode!r}, expected one of {CASTING_MODES}"
            )
        x = _full_tensor(x).contiguous()
        if weight is not None:
            weight = weight.contiguous()

        y, rstd = torch_ops.rms_norm_forward(x, weight, float(eps), float(offset), casting_mode)
        ctx.offset = float(offset)
        ctx.casting_mode = casting_mode
        ctx.in_place = in_place
        ctx.save_for_backward(x, weight, rstd)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight, rstd = ctx.saved_tensors
        dy = _full_tensor(dy).contiguous()

        args = (x, weight, rstd, ctx.offset, ctx.casting_mode)
        if ctx.in_place:
            dw = torch_ops.rms_norm_backward_(dy, *args)
            dx = dy
        else:
            dx, dw = torch_ops.rms_norm_backward(dy, *args)
        return dx, (dw if weight is not None else None), None, None, None, None


def rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    offset: float = 0.0,
    casting_mode: str = "llama",
    in_place: bool = True,
) -> torch.Tensor:
    """RMSNorm of `x`, see `RMSNormFunction`."""
    return RMSNormFunction.apply(x, weight, eps, offset, casting_mode, in_place)
