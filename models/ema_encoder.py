"""
EMA Target Encoder.
Specification: docs/jepa_coder_architecture_v2.md Section 2.3

A single embedding layer whose weights are an EMA of the Reasoner's input
embedding weights. Used ONLY during SST training to produce target vectors.
Not used at inference time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMAEncoder(nn.Module):
    """
    EMA target encoder.

    Wraps a single nn.Embedding whose weights shadow the Reasoner's input
    embedding via exponential moving average. The weights are never updated
    by backpropagation — only by explicit EMA updates.

    Usage in SST training loop:
        # Initialization (once, at start of SST)
        ema_encoder = EMAEncoder.from_embedding(reasoner.embedding)

        # Per-step target generation
        t = ema_encoder.encode_block(block_token_ids)   # (d,)

        # After optimizer.step()
        ema_encoder.update(reasoner.embedding, decay=0.98)
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        self.embedding = nn.Embedding(vocab_size, dim)
        # EMA weights are never trained via backprop
        self.embedding.weight.requires_grad_(False)

    @classmethod
    def from_embedding(cls, source: nn.Embedding) -> 'EMAEncoder':
        """
        Create an EMAEncoder initialized from an existing embedding layer.
        Copies the source weights exactly as the starting EMA state.

        Args:
            source: the Reasoner's input embedding (nn.Embedding)

        Returns:
            EMAEncoder with weights == source.weight.data (detached copy)
        """
        vocab_size, dim = source.weight.shape
        encoder = cls(vocab_size, dim)
        with torch.no_grad():
            encoder.embedding.weight.copy_(source.weight.data)
        return encoder

    def encode_block(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens, mean-pool, and L2-normalize. No gradients.

        Tensor shapes (Section 3.3):
            token_ids   : (L_block,)       integer token IDs
            embedded    : (L_block, d)     one vector per token
            mean-pooled : (d,)             single vector
            normalized  : (d,) ||t|| = 1  training target

        Args:
            token_ids: LongTensor of shape (L_block,)

        Returns:
            t: FloatTensor of shape (d,) with ||t|| = 1
        """
        assert token_ids.ndim == 1, \
            f"Expected 1-D token_ids, got shape {token_ids.shape}"
        assert token_ids.numel() > 0, "token_ids must be non-empty"

        with torch.no_grad():
            embeds = self.embedding(token_ids)          # (L_block, d)
            t = embeds.mean(dim=0)                      # (d,)
            t = F.normalize(t, dim=-1)                  # (d,), ||t|| = 1

        assert t.shape == (self.dim,), \
            f"Expected output shape ({self.dim},), got {t.shape}"
        assert abs(t.norm().item() - 1.0) < 1e-4, \
            f"Output not unit norm: {t.norm().item()}"

        return t

    def update(self, source: nn.Embedding, decay: float = 0.98) -> None:
        """
        EMA weight update (Section 2.3):
            θ_target = decay · θ_target + (1 - decay) · θ_source

        Call AFTER optimizer.step(), outside the computation graph.

        Args:
            source: the Reasoner's input embedding (nn.Embedding)
            decay:  EMA momentum (default 0.98 per spec)
        """
        assert 0.0 < decay < 1.0, f"decay must be in (0, 1), got {decay}"

        with torch.no_grad():
            self.embedding.weight.mul_(decay).add_(
                source.weight.data, alpha=1.0 - decay
            )
