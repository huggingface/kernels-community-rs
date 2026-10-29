# Adapted from https://github.com/linkedin/Liger-Kernel/blob/55bc58a84249418e8b0bfd293082e9f6108e4c2d/test/transformers/test_rms_norm.py

import kernels
import pytest
import torch
import torch.nn as nn

rmsnorm = kernels.get_kernel("kernels-community-rs/rmsnorm", version=1)

device = "cuda"

_BLOCK_RMS_NORM_OVERFLOW_ROWS = 2**23 + 16
_BLOCK_RMS_NORM_OVERFLOW_COLS = 256
_BLOCK_RMS_NORM_MIN_FREE_BYTES = (
    2
    * _BLOCK_RMS_NORM_OVERFLOW_ROWS
    * _BLOCK_RMS_NORM_OVERFLOW_COLS
    * torch.tensor([], dtype=torch.bfloat16).element_size()
    + _BLOCK_RMS_NORM_OVERFLOW_ROWS * torch.tensor([], dtype=torch.float32).element_size()
)


def assert_allclose(actual, expected, atol, rtol):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


class BaseRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.elementwise_affine:
            return self.weight * hidden_states.to(input_dtype)
        else:
            return hidden_states.to(input_dtype)


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L112
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, elementwise_affine=True):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.elementwise_affine:
            return self.weight * hidden_states.to(input_dtype)
        else:
            return hidden_states.to(input_dtype)


# https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/gemma/modeling_gemma.py#L122
class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        if self.elementwise_affine:
            output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class KernelRMSNorm(nn.Module):
    """The layer under test, configured like upstream's `LigerRMSNorm`."""

    def __init__(
        self,
        hidden_size,
        eps=1e-6,
        offset=0.0,
        casting_mode="llama",
        in_place=True,
        elementwise_affine=True,
    ):
        super().__init__()
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.variance_epsilon = eps
        self.offset = offset
        self.casting_mode = casting_mode
        self.in_place = in_place

    def forward(self, hidden_states):
        return rmsnorm.rms_norm(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            self.offset,
            self.casting_mode,
            self.in_place,
        )


@pytest.mark.skipif(
    torch.cuda.mem_get_info()[0] < _BLOCK_RMS_NORM_MIN_FREE_BYTES,
    reason="requires 8.6 GB of free CUDA memory for the first overflowing row offset",
)
def test_rms_norm_large_row_offset():
    """The forward pass must address the first row beyond the int32 range."""
    x = torch.zeros(
        _BLOCK_RMS_NORM_OVERFLOW_ROWS,
        _BLOCK_RMS_NORM_OVERFLOW_COLS,
        dtype=torch.bfloat16,
        device="cuda",
    )
    x[-1].fill_(1)
    w = torch.ones(_BLOCK_RMS_NORM_OVERFLOW_COLS, dtype=torch.bfloat16, device="cuda")

    y = rmsnorm.RMSNormFunction.apply(x, w, 1e-6, 0.0, "llama", True)

    assert torch.allclose(y[-1], torch.ones_like(y[-1]), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "bs, sl, hd",
    [
        (2, 128, 512),
        # weird shapes
        (5, 123, 123),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-6),
        (torch.float16, 2e-2, 2e-2),
        (torch.bfloat16, 2e-1, 2e-2),
    ],
)
@pytest.mark.parametrize(
    "reference, offset, casting_mode",
    [
        (LlamaRMSNorm, 0.0, "llama"),
        (GemmaRMSNorm, 1.0, "gemma"),
        (BaseRMSNorm, 0.0, "none"),
    ],
)
@pytest.mark.parametrize(
    "in_place",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "elementwise_affine",
    [
        True,
        False,
    ],
)
def test_correctness(bs, sl, hd, dtype, atol, rtol, reference, offset, casting_mode, in_place, elementwise_affine):
    _tensor = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    h1 = _tensor.clone().requires_grad_(True)
    h2 = _tensor.clone().requires_grad_(True)

    # do
    do = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    # reference (llama or gemma)
    ref_rms = reference(hidden_size=hd, elementwise_affine=elementwise_affine).to(device).to(dtype)
    ref_o = ref_rms(h1)
    ref_o.backward(do)

    # kernel
    kernel_rms = (
        KernelRMSNorm(
            hidden_size=hd,
            offset=offset,
            casting_mode=casting_mode,
            in_place=in_place,
            elementwise_affine=elementwise_affine,
        )
        .to(device)
        .to(dtype)
    )
    kernel_o = kernel_rms(h2)
    # clone since in_place=True lets backward overwrite the grad-output buffer
    kernel_o.backward(do.clone())

    assert_allclose(kernel_o, ref_o, atol=atol, rtol=rtol)
    if elementwise_affine:
        assert_allclose(kernel_rms.weight.grad, ref_rms.weight.grad, atol=atol, rtol=rtol)
    assert_allclose(h2.grad, h1.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "bs, sl, hd",
    [
        (2, 2, 8),
        # weird shapes
        (9, 7, 41),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-6),
        (torch.bfloat16, 2e-1, 2e-2),
    ],
)
@pytest.mark.parametrize(
    "reference, offset, casting_mode",
    [
        (LlamaRMSNorm, 0.0, "llama"),
        (GemmaRMSNorm, 1.0, "gemma"),
    ],
)
@pytest.mark.parametrize(
    "elementwise_affine",
    [
        True,
        False,
    ],
)
def test_correctness_functional(bs, sl, hd, dtype, atol, rtol, reference, offset, casting_mode, elementwise_affine):
    # h
    _tensor = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    h1 = _tensor.clone().requires_grad_(True)
    h2 = _tensor.clone().requires_grad_(True)

    if elementwise_affine:
        w = torch.randn(hd, device=device, dtype=dtype)
    else:
        w = None

    y1 = rmsnorm.rms_norm(x=h1, weight=w, eps=1e-6, offset=offset, casting_mode=casting_mode)
    y2 = rmsnorm.RMSNormFunction.apply(h2, w, 1e-6, offset, casting_mode)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    grad = torch.randn_like(y2)

    # Clone grad since in_place=True lets backward overwrite the grad-output buffer
    y1.backward(grad.clone())
    y2.backward(grad.clone())

    assert torch.allclose(h1.grad, h2.grad, atol=atol, rtol=rtol)
