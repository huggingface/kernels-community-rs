import kernels
import pytest
import torch

rmsnorm = kernels.get_kernel("kernels-community-rs/rmsnorm", version=1)

DEVICE = "cuda"


def reference(x, weight, eps, offset, casting_mode):
    """transformers-style RMSNorm in float32 for `gemma`, otherwise Llama-style."""
    if casting_mode == "gemma":
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        if weight is not None:
            y = y * (offset + weight.float())
        return y.to(x.dtype)
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    y = y.to(x.dtype)
    if weight is not None:
        y = (offset + weight.float()) * y.float()
    return y.to(x.dtype)


def run(fn, x, weight, dy):
    x = x.clone().requires_grad_()
    weight = None if weight is None else weight.clone().requires_grad_()
    y = fn(x, weight)
    # Clone since in_place=True lets backward overwrite the grad-output buffer.
    y.backward(dy.clone())
    return y, x.grad, None if weight is None else weight.grad


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("casting_mode, offset", [("llama", 0.0), ("gemma", 1.0)])
@torch.inference_mode(False)
def test_float32_weight(dtype, casting_mode, offset):
    torch.manual_seed(0)
    x = torch.randn(4, 33, 1000, device=DEVICE, dtype=dtype)
    weight = torch.randn(1000, device=DEVICE, dtype=torch.float32)
    dy = torch.randn_like(x)

    y, dx, dw = run(
        lambda x, w: rmsnorm.rms_norm(x, w, 1e-6, offset, casting_mode, in_place=False), x, weight, dy
    )
    y_ref, dx_ref, dw_ref = run(lambda x, w: reference(x, w, 1e-6, offset, casting_mode), x, weight, dy)

    assert y.dtype == dtype and dx.dtype == dtype and dw.dtype == torch.float32
    tol = dict(atol=2e-2, rtol=2e-2) if dtype == torch.float16 else dict(atol=2e-1, rtol=2e-2)
    torch.testing.assert_close(y, y_ref, **tol)
    torch.testing.assert_close(dx, dx_ref, **tol)
    torch.testing.assert_close(dw, dw_ref, atol=2e-1, rtol=2e-2)


@pytest.mark.parametrize("hidden_size", [1, 31, 4097, 8192, 65536])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hidden_sizes(hidden_size, dtype):
    torch.manual_seed(0)
    x = torch.randn(3, hidden_size, device=DEVICE, dtype=dtype)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    dy = torch.randn_like(x)

    y, dx, dw = run(lambda x, w: rmsnorm.rms_norm(x, w, 1e-6), x, weight, dy)
    y_ref, dx_ref, dw_ref = run(lambda x, w: reference(x, w, 1e-6, 0.0, "llama"), x, weight, dy)

    tol = dict(atol=1e-4, rtol=1e-5) if dtype == torch.float32 else dict(atol=2e-1, rtol=2e-2)
    torch.testing.assert_close(y, y_ref, **tol)
    torch.testing.assert_close(dx, dx_ref, **tol)
    torch.testing.assert_close(dw, dw_ref, **tol)


@pytest.mark.kernels_ci
def test_in_place_overwrites_output_grad():
    torch.manual_seed(0)
    x = torch.randn(8, 64, device=DEVICE).requires_grad_()
    weight = torch.randn(64, device=DEVICE).requires_grad_()
    dy = torch.randn_like(x)
    dy_in_place = dy.clone()

    rmsnorm.rms_norm(x, weight, 1e-6, in_place=True).backward(dy_in_place)
    dx = x.grad.clone()
    x.grad = None
    rmsnorm.rms_norm(x, weight, 1e-6, in_place=False).backward(dy)

    torch.testing.assert_close(dx, x.grad, atol=0.0, rtol=0.0)
    torch.testing.assert_close(dy_in_place, dx, atol=0.0, rtol=0.0)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("with_weight", [True, False])
def test_opcheck(with_weight):
    torch.manual_seed(0)
    x = torch.randn(4, 7, 96, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(96, device=DEVICE, dtype=torch.bfloat16) if with_weight else None
    _, rstd = rmsnorm.torch_ops.rms_norm_forward(x, weight, 1e-6, 0.0, "llama")
    dy = torch.randn_like(x)

    torch.library.opcheck(rmsnorm.torch_ops.rms_norm_forward, (x, weight, 1e-6, 0.0, "llama"))
    torch.library.opcheck(rmsnorm.torch_ops.rms_norm_backward, (dy, x, weight, rstd, 0.0, "llama"))
    torch.library.opcheck(rmsnorm.torch_ops.rms_norm_backward_, (dy.clone(), x, weight, rstd, 0.0, "llama"))


@pytest.mark.kernels_ci
@pytest.mark.parametrize(
    "x_shape, weight_shape, weight_dtype, casting_mode, error, match",
    [
        ((4, 64), (64,), torch.bfloat16, "llama2", ValueError, "casting mode"),
        ((4, 64), (32,), torch.bfloat16, "llama", ValueError, "weight must have shape"),
        ((4, 64), (64,), torch.float16, "llama", ValueError, "dtype of x or be float32"),
        ((2, 65537), (65537,), torch.bfloat16, "llama", ValueError, "at most 65536"),
        ((), (1,), torch.bfloat16, "llama", ValueError, "at least one dimension"),
    ],
)
def test_rejects_bad_arguments(x_shape, weight_shape, weight_dtype, casting_mode, error, match):
    x = torch.randn(x_shape, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(weight_shape, device=DEVICE, dtype=weight_dtype)
    with pytest.raises(error, match=match):
        rmsnorm.rms_norm(x, weight, 1e-6, casting_mode=casting_mode)


@pytest.mark.kernels_ci
def test_rejects_unsupported_dtype():
    x = torch.randn(4, 64, device=DEVICE, dtype=torch.float64)
    with pytest.raises(ValueError, match="unsupported dtype"):
        rmsnorm.rms_norm(x, None, 1e-6)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("with_weight", [True, False])
def test_empty_input(with_weight):
    x = torch.empty(0, 64, device=DEVICE, requires_grad=True)
    weight = torch.randn(64, device=DEVICE, requires_grad=True) if with_weight else None

    y = rmsnorm.rms_norm(x, weight, 1e-6)
    y.backward(torch.empty_like(y))

    assert y.shape == (0, 64) and x.grad.shape == (0, 64)
    if with_weight:
        torch.testing.assert_close(weight.grad, torch.zeros_like(weight))
