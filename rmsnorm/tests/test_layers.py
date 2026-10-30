import pytest
import torch
import torch.nn as nn

import kernels

rmsnorm = kernels.get_kernel("kernels-community-rs/rmsnorm", version=1)

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else None


class RMSNorm(nn.Module):
    """Llama-style RMSNorm, the layer `RMSNorm` replaces."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class KernelRMSNorm(RMSNorm):
    forward = rmsnorm.layers.RMSNorm.forward


class GemmaRMSNorm(nn.Module):
    """transformers' `GemmaRMSNorm`, the layer `GemmaRMSNorm` replaces."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class KernelGemmaRMSNorm(GemmaRMSNorm):
    forward = rmsnorm.layers.GemmaRMSNorm.forward


class PostNormBlock(nn.Module):
    """OLMo2-style block: the norm output is added straight into the residual,
    so the gradient flowing into the norm is shared with the residual branch."""

    def __init__(self, norm_cls, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = norm_cls(hidden_size)

    def forward(self, x):
        return x + self.norm(self.proj(x))


@pytest.mark.kernels_ci
@pytest.mark.skipif(DEVICE is None, reason="needs an accelerator")
@pytest.mark.parametrize("compile", [False, True])
@pytest.mark.parametrize(
    "reference, kernel_layer",
    [(RMSNorm, KernelRMSNorm), (GemmaRMSNorm, KernelGemmaRMSNorm)],
    ids=["llama", "gemma"],
)
def test_rms_norm_post_norm_backward(compile, reference, kernel_layer):
    torch.manual_seed(0)
    hidden_size = 256
    ref = PostNormBlock(reference, hidden_size).to(DEVICE)
    with torch.no_grad():
        ref.norm.weight.normal_(1.0, 0.1)
    kernel = PostNormBlock(kernel_layer, hidden_size).to(DEVICE)
    kernel.load_state_dict(ref.state_dict())

    x = torch.randn(2, 128, hidden_size, device=DEVICE)
    x_ref = x.clone().requires_grad_()
    x_kernel = x.clone().requires_grad_()
    dy = torch.randn_like(x)

    out_ref = ref(x_ref)
    out_ref.backward(dy)
    kernel_fwd = torch.compile(kernel) if compile else kernel
    out_kernel = kernel_fwd(x_kernel)
    out_kernel.backward(dy)

    torch.testing.assert_close(out_kernel, out_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x_kernel.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    for (name, p_ref), p_kernel in zip(ref.named_parameters(), kernel.parameters()):
        torch.testing.assert_close(p_kernel.grad, p_ref.grad, atol=1e-4, rtol=1e-4, msg=name)
