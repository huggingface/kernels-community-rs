from typing import Optional

import torch

from ._ops import ops
from ._ops import torch_add_op_namespace_prefix as add_op_namespace_prefix


def _custom_op(name: str, mutates_args=()):
    return torch.library.custom_op(
        add_op_namespace_prefix(name), mutates_args=mutates_args, device_types="cuda"
    )


@_custom_op("rms_norm_forward")
def rms_norm_forward(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    offset: float,
    casting_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the normalized `x` and the float32 inverse RMS of every row."""
    y = torch.empty_like(x)
    rstd = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    ops.rms_norm_forward(y, rstd, x, weight, eps, offset, casting_mode)
    return y, rstd


@rms_norm_forward.register_fake
def _(x, weight, eps, offset, casting_mode):
    return torch.empty_like(x), x.new_empty(x.shape[:-1], dtype=torch.float32)


# Blocks of the backward kernel per SM. Each block accumulates one row of
# partial weight gradients; Liger uses one block per SM, but more blocks hide
# memory latency better (up to 2.4x faster on an L4 for 16-bit inputs).
_BLOCKS_PER_SM = 4


def _weight_grad_buffers(x, weight):
    """The weight gradient and the partial weight gradients, one row per
    backward block. Empty without a weight."""
    if weight is None:
        return x.new_empty(0), None
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    blocks = _BLOCKS_PER_SM * sms
    dw_partial = torch.empty(blocks, x.shape[-1], dtype=torch.float32, device=x.device)
    return torch.empty_like(weight), dw_partial


def _weight_grad(dw, dw_partial):
    if dw_partial is not None:
        ops.rms_norm_reduce_weight_grad(dw, dw_partial)
    return dw


@_custom_op("rms_norm_backward")
def rms_norm_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    rstd: torch.Tensor,
    offset: float,
    casting_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the gradients of `x` and `weight` (empty without a weight)."""
    dx = torch.empty_like(dy)
    dw, dw_partial = _weight_grad_buffers(x, weight)
    ops.rms_norm_backward(dx, dw_partial, dy, x, weight, rstd, offset, casting_mode)
    return dx, _weight_grad(dw, dw_partial)


@rms_norm_backward.register_fake
def _(dy, x, weight, rstd, offset, casting_mode):
    dw = x.new_empty(0) if weight is None else torch.empty_like(weight)
    return torch.empty_like(dy), dw


@_custom_op("rms_norm_backward_", mutates_args=("dy",))
def rms_norm_backward_(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    rstd: torch.Tensor,
    offset: float,
    casting_mode: str,
) -> torch.Tensor:
    """Like `rms_norm_backward`, but overwrites `dy` with the gradient of `x`.
    Returns the gradient of `weight` (empty without a weight)."""
    dw, dw_partial = _weight_grad_buffers(x, weight)
    ops.rms_norm_backward(dy, dw_partial, None, x, weight, rstd, offset, casting_mode)
    return _weight_grad(dw, dw_partial)


@rms_norm_backward_.register_fake
def _(dy, x, weight, rstd, offset, casting_mode):
    return x.new_empty(0) if weight is None else torch.empty_like(weight)
