import torch
import torch.nn as nn

from .functional import RMSNormFunction


class RMSNorm(nn.Module):
    """Llama-style RMSNorm: `weight * (x / rms(x)).to(x.dtype)`.

    Replaces the forward of modules with a `weight` and `variance_epsilon`,
    such as transformers' `LlamaRMSNorm`.
    """

    weight: nn.Parameter
    variance_epsilon: float

    can_torch_compile = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return RMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            0.0,
            "llama",
            # Not in place: dY can be shared with other branches (e.g. a residual add).
            False,
        )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm: `(x / rms(x) * (1 + weight)).to(x.dtype)`, in float32.

    Replaces the forward of modules with a `weight` and `eps`, such as
    transformers' `GemmaRMSNorm`.
    """

    weight: nn.Parameter
    eps: float

    can_torch_compile = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Not in place, as above.
        return RMSNormFunction.apply(x, self.weight, self.eps, 1.0, "gemma", False)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
