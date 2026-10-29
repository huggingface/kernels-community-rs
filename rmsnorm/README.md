---
license: bsd-2-clause
tags:
  - kernel
---

## RMSNorm

RMSNorm, forward and backward, ported from the Triton kernels of
[Liger Kernel](https://github.com/linkedin/Liger-Kernel).

The Liger kernels incorporate code from [Unsloth](https://github.com/unslothai/unsloth).

`y = x / rms(x) * (offset + weight)` over the last dimension of `x`, with
Liger's casting modes:

- `"llama"`: only the inverse RMS is computed in float32 (Llama).
- `"gemma"`: everything is computed in float32, then cast back (Gemma).
- `"none"`: computed in the input dtype.

`x` can be float16, bfloat16 or float32, with a weight of the same dtype, a
float32 weight, or no weight. The hidden size can be up to 65536.

## Usage

```python
import torch
from kernels import get_kernel

rmsnorm = get_kernel("kernels-community-rs/rmsnorm", version=1)

x = torch.randn(4, 128, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
weight = torch.ones(4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)

y = rmsnorm.rms_norm(x, weight, eps=1e-6, offset=0.0, casting_mode="llama", in_place=False)
y.sum().backward()
```

`rms_norm` and `RMSNormFunction.apply` take the same arguments as Liger's
`liger_rms_norm` and `LigerRMSNormFunction`, without `row_mode`. With
`in_place=True` (the default, as in Liger) the backward pass overwrites the
output gradient with the input gradient.

The layers replace the forward of transformers' norms with `kernelize`:

- `layers.RMSNorm`: Llama-style, for modules with `weight` and
  `variance_epsilon` (like Liger's `LigerRMSNorm`).
- `layers.GemmaRMSNorm`: Gemma-style (`offset=1.0`, `"gemma"` casting), for
  modules with `weight` and `eps`.

The Torch custom ops, for `torch.compile` and `torch.library.opcheck`, are in
`rmsnorm.torch_ops`.

## Performance

On an NVIDIA L4, GPU time is on par with or better than Liger's Triton kernels
(`kernels-community/liger-kernels`), with hidden sizes 2048-8192 and
4096-16384 rows:

- Forward: 0.96-1.02x Liger's time, except for 16-bit inputs that fit in L2
  (4096 x 2048), where it is up to 1.12x.
- Forward and backward: 0.83-0.96x, except 1.15x for 16-bit inputs at
  4096 x 2048.

The forward kernel loads and stores 16 bytes at a time when rows are 16-byte
aligned; the backward kernel keeps a simple element-wise loop.
