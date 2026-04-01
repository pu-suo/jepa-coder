"""
Reasoner model.
Specification: docs/jepa_coder_architecture_v2.md Section 2.1
               docs/contract_1_sst_loop.md Sections 3.1-3.2
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import HybridNorm, RMSNorm


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with QK-Norm (non-learnable).

    QK-Norm: L2-normalize Q and K per head before computing attention scores.
    Prevents attention logit explosion during the autoregressive latent loop
    (Section 2.1). No learnable parameters in the normalization.

    Input/output shape: (T, d)
    """

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        # Fused QKV + output projection, no bias
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (T, d)
            attn_mask: (T, T) additive mask; -inf at positions to suppress
        Returns:
            (T, d)
        """
        T, d = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)                                              # (T, 3d)
        q, k, v = qkv.split(d, dim=-1)                                      # each (T, d)

        # Reshape to (n_heads, T, head_dim) for per-head operations
        q = q.view(T, self.n_heads, self.head_dim).permute(1, 0, 2)         # (H, T, hd)
        k = k.view(T, self.n_heads, self.head_dim).permute(1, 0, 2)
        v = v.view(T, self.n_heads, self.head_dim).permute(1, 0, 2)

        # QK-Norm: L2-normalize Q and K over head_dim (non-learnable)
        # For unit vectors: q·k stays in [-1, 1]; scale guards magnitude
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale            # (H, T, T)
        if attn_mask is not None:
            attn = attn + attn_mask                                          # broadcasts over H
        attn = F.softmax(attn, dim=-1)

        # Aggregate values and reshape
        out = torch.matmul(attn, v)                                          # (H, T, hd)
        out = out.permute(1, 0, 2).contiguous().view(T, d)                  # (T, d)

        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block:
        x = x + Attn(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    FFN: dim → ffn_dim → dim with GELU, no bias.
    """

    def __init__(self, dim: int, n_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, dim, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (T, d)
            attn_mask: (T, T) additive mask
        Returns:
            (T, d)
        """
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Transformer stack
# ---------------------------------------------------------------------------

class TransformerStack(nn.Module):
    """
    Stack of TransformerBlocks that always applies causal masking.

    For length-1 sequences (the step() path) the upper-triangular mask is
    empty — a trivial no-op. For length-T sequences (encode_problem or
    pretraining) the mask is the standard causal upper triangle.

    Input/output shape: (T, d)  — no batch dimension.
    """

    def __init__(self, blocks: List[TransformerBlock]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, d)
        Returns:
            (T, d)
        """
        T = x.shape[0]

        # Build causal mask: upper triangle = -inf, rest = 0
        attn_mask: Optional[torch.Tensor] = None
        if T > 1:
            attn_mask = torch.triu(
                torch.full((T, T), float('-inf'), device=x.device, dtype=x.dtype),
                diagonal=1,
            )

        for block in self.blocks:
            x = block(x, attn_mask)
        return x


# ---------------------------------------------------------------------------
# Reasoner
# ---------------------------------------------------------------------------

class Reasoner(nn.Module):
    """
    Decoder-only Transformer that reasons autoregressively in continuous
    latent space.

    Architecture (Section 2.1):
        16 layers, 768 dim, 12 heads (64 dim/head), 3072 FFN dim, 1024 context

    Attributes that the SST loop and tests access directly:
        embedding            nn.Embedding — token embeddings
        transformer_blocks   TransformerStack — the 16-layer stack
        hybrid_norm          HybridNorm — RMSNorm [+ L2] after the stack
        rms_norm             property → hybrid_norm.rms_norm (for test compat)
        lm_head              nn.Linear or None — tied to embedding during pretraining

    Training phase toggles:
        Pretraining  hybrid_norm.l2_enabled = False  (default)
                     attach_lm_head() before training
        SST          hybrid_norm.l2_enabled = True
                     detach_lm_head() at phase transition
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 768,
        n_layers: int = 16,
        n_heads: int = 12,
        ffn_dim: int = 3072,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Token and positional embeddings
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim)

        # 16-layer Transformer stack — accessible as an attribute for the
        # SST loop and contract tests (e.g. reasoner.transformer_blocks(h_seq))
        self.transformer_blocks = TransformerStack([
            TransformerBlock(dim, n_heads, ffn_dim) for _ in range(n_layers)
        ])

        # Final normalization — HybridNorm(l2_enabled=False) for pretraining;
        # set l2_enabled=True before SST
        self.hybrid_norm = HybridNorm(dim, l2_enabled=False)

        # LM head placeholder; created and tied in attach_lm_head()
        self.lm_head: Optional[nn.Linear] = None

    # ------------------------------------------------------------------
    # Property: expose rms_norm for contract test compatibility
    # contract_1_sst_loop.md tests call: reasoner.rms_norm(r.unsqueeze(0))
    # ------------------------------------------------------------------

    @property
    def rms_norm(self) -> RMSNorm:
        return self.hybrid_norm.rms_norm

    # ------------------------------------------------------------------
    # SST interface
    # ------------------------------------------------------------------

    def encode_problem(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode a problem statement into a single unit-norm latent vector.
        Runs once per training example. Gradients flow through this path.

        contract_1_sst_loop.md Section 3.1:
            embed + pos_embed → Transformer → HybridNorm → mean pool → L2

        Args:
            token_ids: LongTensor of shape (L_prob,), L_prob ≤ max_seq_len

        Returns:
            h: FloatTensor of shape (d,) with ||h|| = 1
        """
        assert token_ids.ndim == 1, \
            f"Expected 1-D token_ids, got shape {token_ids.shape}"
        L = token_ids.shape[0]
        assert L <= self.max_seq_len, \
            f"Sequence length {L} exceeds max_seq_len {self.max_seq_len}"

        positions = torch.arange(L, device=token_ids.device)

        # Step A: token embeddings + positional embeddings
        prob_embeds = self.embedding(token_ids) + self.pos_embedding(positions)  # (L, d)

        # Step B: Transformer (causal masking applied inside TransformerStack)
        prob_processed = self.transformer_blocks(prob_embeds)                    # (L, d)

        # Step C: HybridNorm — RMSNorm always; L2 per-vector if l2_enabled
        prob_normed = self.hybrid_norm(prob_processed)                           # (L, d)

        # Step D: Mean pool — mean of unit vectors is NOT unit length
        h = prob_normed.mean(dim=0)                                              # (d,)

        # Step E: Re-normalize to unit hypersphere
        h = F.normalize(h, dim=-1)                                               # (d,), ||h||=1

        assert abs(h.norm().item() - 1.0) < 1e-4, \
            f"encode_problem: output not unit norm: {h.norm().item()}"

        return h

    def step(self, h: torch.Tensor) -> torch.Tensor:
        """
        One reasoning step: map a continuous latent state h to the next state r.

        contract_1_sst_loop.md Section 3.2:
            unsqueeze(0) → Transformer → squeeze(0) → RMSNorm → L2 normalize

        Note: self-attention over a length-1 sequence is trivially the identity
        for the attention mechanism; the FFN layers do the real computation.
        No position embedding is added — h is a continuous hidden state, not
        a token embedding.

        Args:
            h: FloatTensor of shape (d,) with ||h|| = 1

        Returns:
            r: FloatTensor of shape (d,) with ||r|| = 1
        """
        assert h.shape == (self.dim,), \
            f"Expected shape ({self.dim},), got {h.shape}"
        assert abs(h.norm().item() - 1.0) < 1e-3, \
            f"step: input not unit norm: {h.norm().item()}"

        # Step A: expand to sequence format required by TransformerStack
        h_seq = h.unsqueeze(0)                                                   # (1, d)

        # Step B: Transformer (length-1; causal mask is empty — no-op)
        r_seq = self.transformer_blocks(h_seq)                                   # (1, d)

        # Step C: squeeze back to vector
        r = r_seq.squeeze(0)                                                     # (d,)

        # Step D: HybridNorm — RMSNorm expects a leading dim, matches contract:
        #   r = reasoner.rms_norm(r.unsqueeze(0)).squeeze(0)
        r = self.hybrid_norm(r.unsqueeze(0)).squeeze(0)                         # (d,)

        # Always explicitly L2-normalize so output is unit norm regardless of
        # whether hybrid_norm.l2_enabled is True or False
        r = F.normalize(r, dim=-1)                                               # (d,), ||r||=1

        assert abs(r.norm().item() - 1.0) < 1e-4, \
            f"step: output not unit norm: {r.norm().item()}"

        return r

    # ------------------------------------------------------------------
    # Pretraining interface
    # ------------------------------------------------------------------

    def attach_lm_head(self) -> None:
        """
        Create the temporary LM head and tie its weights to the input embedding.

        Tied embeddings (Section 2.1): W_head = W_embed^T
        Both share the same underlying parameter tensor. The LM head has no
        bias, consistent with standard tied-embedding LMs.

        Call before pretraining. Call detach_lm_head() when SST begins.
        """
        lm_head = nn.Linear(self.dim, self.vocab_size, bias=False)
        lm_head.weight = self.embedding.weight          # tied: same tensor
        self.lm_head = lm_head

    def detach_lm_head(self) -> None:
        """
        Permanently remove the LM head.

        Call at the pretraining → SST phase transition (Section 2.1).
        After this point, lm_forward() will raise an error if called.
        """
        self.lm_head = None

    def lm_forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Standard LM forward pass for next-token prediction during pretraining.
        L2 normalization is disabled (hybrid_norm.l2_enabled=False) in this phase.

        Args:
            token_ids: LongTensor of shape (T,)

        Returns:
            logits: FloatTensor of shape (T, vocab_size)
        """
        assert self.lm_head is not None, \
            "LM head not attached. Call attach_lm_head() before pretraining."
        assert token_ids.ndim == 1, \
            f"Expected 1-D token_ids, got shape {token_ids.shape}"

        T = token_ids.shape[0]
        positions = torch.arange(T, device=token_ids.device)

        x = self.embedding(token_ids) + self.pos_embedding(positions)           # (T, d)
        x = self.transformer_blocks(x)                                           # (T, d)
        x = self.hybrid_norm(x)                                                  # (T, d)
        logits = self.lm_head(x)                                                 # (T, vocab_size)
        return logits
