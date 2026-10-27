---
license: apache-2.0
tags:
  - kernel
---

<!-- Adapted from https://github.com/huggingface/kernels-community/blob/18153cd077b6541e9c4cb98ad9678db4e1253d1f/activation/README.md -->
<!-- Changes: describes the Rust port and its argument validation, loads kernels-community-rs/activation-rs. -->

## Activation

Activation kernels from [vLLM](https://github.com/vllm-project/vllm/blob/main/csrc/activation_kernels.cu),
ported from [kernels-community/activation](https://github.com/huggingface/kernels-community/tree/main/activation)
to Rust with [cuda-oxide](https://github.com/NVIDIA/cuda-rust). Only the CUDA
backend is ported.

Copyright 2023-2024, the vLLM team.

The ops, layers and their results match the upstream kernel. Unlike upstream,
the ops check that `out` has the expected shape and the same dtype as the
input, so a mismatched `out` raises instead of being written out of bounds.

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
activation = get_kernel("kernels-community-rs/activation-rs", version=1)
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
