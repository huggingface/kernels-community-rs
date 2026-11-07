---
license: apache-2.0
tags:
  - kernel
---

<!-- Adapted from https://github.com/huggingface/kernels-community/blob/18153cd077b6541e9c4cb98ad9678db4e1253d1f/activation/README.md -->
<!-- Changes: describes the cuTile Rust port and its argument validation, loads kernels-community-rs/activation-cutile. -->

## Activation

Activation kernels from [vLLM](https://github.com/vllm-project/vllm/blob/main/csrc/activation_kernels.cu),
ported from [kernels-community/activation](https://github.com/huggingface/kernels-community/tree/main/activation)
to Rust with [cuTile Rust](https://github.com/NVIDIA/cuda-rust/tree/main/cutile-rs).

Copyright 2023-2024, the vLLM team.

The ops, layers and their results match the upstream kernel, except exact
GELU (`gelu` and `gelu_and_mul`): cuTile has no `erf`, so it uses its own
approximation, within 0.95 ulp of `erf` for all f32 inputs (see
[`scripts/fit_erf.py`](scripts/fit_erf.py)). Results can differ from
PyTorch's by one rounding, for about 4.5% of f32 and under 0.01% of f16/bf16
outputs. Unlike upstream, the ops check that `out` has the expected shape and
the same dtype as the input, so a mismatched `out` raises instead of being
written out of bounds.

Tensors must be contiguous and have fewer than 2^31 elements.

## Usage

```python
# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "kernels"
# ]
# ///
import torch
from kernels import get_kernel

# Setup
torch.manual_seed(42)
activation = get_kernel("kernels-community-rs/activation-cutile", version=1)
device = torch.device("cuda")

# Create test tensor for SwiGLU (SiLU and Mul)
# Input has shape (num_tokens, 2 * hidden_dim)
num_tokens, hidden_dim = 128, 512
input_tensor = torch.randn(
    num_tokens, 2 * hidden_dim, device=device, dtype=torch.float16
)

# Custom kernel SwiGLU
out_shape = input_tensor.shape[:-1] + (hidden_dim,)
out_kernel = torch.empty(out_shape, dtype=input_tensor.dtype, device=device)
out_kernel = activation.silu_and_mul(out_kernel, input_tensor)

print(f"Output: {out_kernel.shape}")
# Output: torch.Size([128, 512])
```

The Torch custom ops, for `torch.compile` and `torch.library.opcheck`, are
available as `activation.torch_ops`.
