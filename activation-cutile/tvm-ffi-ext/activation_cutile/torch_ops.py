"""The tvm-ffi ops registered as Torch custom ops, for `torch.compile` and
`torch.library.opcheck`. Schemas match the upstream activation kernel."""

import torch

from ._ops import ops
from ._ops import torch_add_op_namespace_prefix as add_op_namespace_prefix


def _register(name: str, impl):
    op = torch.library.custom_op(
        add_op_namespace_prefix(name),
        impl,
        mutates_args=("out",),
        device_types="cuda",
    )
    # The ops write into `out` and return nothing.
    op.register_fake(lambda *args: None)
    return op


def _activation_op(name: str):
    native = getattr(ops, name)

    def impl(out: torch.Tensor, input: torch.Tensor) -> None:
        native(out, input)

    return _register(name, impl)


def _fatrelu_and_mul(out: torch.Tensor, input: torch.Tensor, threshold: float) -> None:
    ops.fatrelu_and_mul(out, input, threshold)


silu_and_mul = _activation_op("silu_and_mul")
mul_and_silu = _activation_op("mul_and_silu")
gelu_and_mul = _activation_op("gelu_and_mul")
gelu_tanh_and_mul = _activation_op("gelu_tanh_and_mul")
fatrelu_and_mul = _register("fatrelu_and_mul", _fatrelu_and_mul)
gelu_new = _activation_op("gelu_new")
gelu_fast = _activation_op("gelu_fast")
gelu_quick = _activation_op("gelu_quick")
gelu_tanh = _activation_op("gelu_tanh")
silu = _activation_op("silu")
gelu = _activation_op("gelu")
