import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize x along dim so each vector has unit norm.

    Thin wrapper around F.normalize with a consistent eps for use throughout
    the codebase (QK-Norm, mean-pool re-normalization, EMA target generation).
    """
    return F.normalize(x, p=2, dim=dim, eps=eps)


class RMSNorm(nn.Module):
    """RMSNorm with a learnable per-dimension scale parameter gamma.

    Formula: output = (x / sqrt(mean(x^2) + eps)) * gamma

    gamma is initialized to ones so the layer starts as the identity
    rescaling. It is the only trainable parameter (no bias).

    Shapes: input (..., dim) → output (..., dim)
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS over the feature dimension, keeping dim for broadcasting
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.gamma


class HybridNorm(nn.Module):
    """Hybrid normalization applied after the final Transformer block.

    Two operations in sequence:
      1. RMSNorm  — learnable gamma, no bias
      2. L2 norm  — no learnable parameters; projects each vector onto S^(d-1)

    The L2 step is gated by l2_enabled:
      - DISABLED during pretraining  (only RMSNorm runs)
      - ENABLED  during SST          (RMSNorm + L2 → all outputs on unit hypersphere)

    Toggle via:
        model.hybrid_norm.l2_enabled = True   # before SST begins
        model.hybrid_norm.l2_enabled = False  # default / pretraining

    When l2_enabled is True, an inline assertion verifies that every output
    vector has ||v|| ≈ 1. Disable assertions in production with `python -O`.

    Shapes: input (..., dim) → output (..., dim)
    """

    def __init__(self, dim: int, eps: float = 1e-8, l2_enabled: bool = False) -> None:
        super().__init__()
        self.rms_norm = RMSNorm(dim, eps=eps)
        self.l2_enabled = l2_enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step C (contract_1_sst_loop §3.1): RMSNorm
        x = self.rms_norm(x)

        if self.l2_enabled:
            # Step D (contract_1_sst_loop §3.1): project onto unit hypersphere
            x = l2_normalize(x, dim=-1)

            # Inline verification: every vector must have unit L2 norm
            assert torch.all(x.norm(dim=-1).sub(1.0).abs() < 1e-4), (
                "HybridNorm: output norms deviate from 1 "
                f"(max deviation: {x.norm(dim=-1).sub(1.0).abs().max().item():.6f})"
            )

        return x
