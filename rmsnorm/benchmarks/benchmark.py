import torch
import torch.nn as nn

from kernels.benchmark import Benchmark


class _Norm(nn.Module):
    def __init__(self, hidden_size, dtype, device):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, device=device, dtype=dtype) * 0.1 + 1)
        self.variance_epsilon = 1e-6


def _reference(x, weight, eps):
    """transformers' `LlamaRMSNorm`."""
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return weight * y.to(x.dtype)


class RMSNormBenchmark(Benchmark):
    """Forward pass of the Llama-style `layers.RMSNorm`, in bfloat16."""

    seed: int = 42

    def _setup(self, tokens, hidden_size):
        layer_cls = type("Norm", (_Norm,), {"forward": self.kernel.layers.RMSNorm.forward})
        self.layer = layer_cls(hidden_size, torch.bfloat16, self.device)
        self.x = torch.randn(tokens, hidden_size, device=self.device, dtype=torch.bfloat16)

    def _run(self):
        with torch.no_grad():
            self.out = self.layer(self.x)

    def _verify(self):
        return _reference(self.x, self.layer.weight.detach(), self.layer.variance_epsilon)

    # Workload: small (4096 tokens, hidden size 2048)
    def setup_small(self):
        self._setup(4096, 2048)

    def benchmark_small(self):
        self._run()

    def verify_small(self) -> torch.Tensor:
        return self._verify()

    # Workload: medium (8192 tokens, hidden size 4096)
    def setup_medium(self):
        self._setup(8192, 4096)

    def benchmark_medium(self):
        self._run()

    def verify_medium(self) -> torch.Tensor:
        return self._verify()

    # Workload: large (16384 tokens, hidden size 8192)
    def setup_large(self):
        self._setup(16384, 8192)

    def benchmark_large(self):
        self._run()

    def verify_large(self) -> torch.Tensor:
        return self._verify()
