"""
Vector Quantization module.
Specification: docs/contract_2_vq_module.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Input:  z of shape (d,) — L2-normalized vector on the unit hypersphere
    Output: (quantized, index, loss)
        quantized: (d,) — exact codebook entry, L2-normalized
        index:     LongTensor scalar — codebook index in [0, codebook_size)
        loss:      scalar — commitment loss (EMA mode: no codebook gradient loss)
    """

    def __init__(self, codebook_size: int = 512, dim: int = 768, commitment_cost: float = 0.25):
        super().__init__()

        # Section 2: Internal state
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(codebook_size, dim)
        self.embedding.weight.requires_grad_(False)  # EMA-updated, not gradient-updated

        # Codebook usage tracking (Section 2 / Section 7)
        self.register_buffer('usage_count', torch.zeros(codebook_size))
        self.register_buffer('total_count', torch.tensor(0))

        # Section 3: Initialization
        self.embedding.weight.data.uniform_(
            -1.0 / self.codebook_size,
             1.0 / self.codebook_size,
        )
        with torch.no_grad():
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, dim=-1)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: FloatTensor of shape (d,) with ||z|| = 1
        Returns:
            quantized_st: FloatTensor of shape (d,) — straight-through quantized vector
            index:        LongTensor scalar — codebook index
            loss:         FloatTensor scalar — commitment loss
        """
        # ============================================
        # STEP 1: Input validation  (Section 4)
        # ============================================
        assert z.shape == (self.dim,), \
            f"Expected shape ({self.dim},), got {z.shape}"
        assert abs(z.norm().item() - 1.0) < 1e-3, \
            f"Input not unit norm: {z.norm().item()}"

        # ============================================
        # STEP 2: Nearest codebook entry via dot product  (Section 4)
        # For unit vectors: argmin ||z - c_k||² = argmax (z · c_k)
        # ============================================
        # self.embedding.weight: (K, d),  z: (d,)  →  dots: (K,)
        dots = torch.matmul(self.embedding.weight, z)
        index = dots.argmax()

        assert 0 <= index.item() < self.codebook_size

        # ============================================
        # STEP 3: Look up quantized vector  (Section 4)
        # ============================================
        quantized = self.embedding.weight[index]
        # Exact codebook row — no interpolation
        assert torch.allclose(quantized, self.embedding.weight[index], atol=1e-6)

        # ============================================
        # STEP 4: Commitment loss  (Section 4 + Section 5 DECISION)
        # Using EMA codebook updates → only commitment part in loss
        #   loss = β · ||z - sg[quantized]||²
        # (codebook_loss is handled by update_codebook_ema, not gradient)
        # ============================================
        commit_loss = F.mse_loss(z, quantized.detach().clone())
        loss = self.commitment_cost * commit_loss

        # ============================================
        # STEP 5: Straight-through estimator  (Section 4)
        # Value  = quantized  (discrete)
        # Gradient = flows through z  (continuous)
        # ============================================
        quantized_st = z + (quantized - z).detach()

        assert torch.allclose(quantized_st, quantized, atol=1e-6)

        # ============================================
        # STEP 6: Update usage tracking  (Section 4 / Section 7)
        # ============================================
        self.usage_count[index] += 1
        self.total_count += 1

        return quantized_st, index, loss

    # ------------------------------------------------------------------
    # Section 5: EMA codebook update
    # ------------------------------------------------------------------

    def update_codebook_ema(self, z: torch.Tensor, index: torch.Tensor, decay: float = 0.99):
        """
        Call AFTER forward pass, OUTSIDE the computation graph.
        Moves the selected codebook entry toward z, then re-normalizes
        to keep it on the unit hypersphere.

        Args:
            z:     (d,) the encoder output that was quantized
            index: LongTensor scalar — the codebook index that was selected
            decay: EMA decay rate (default 0.99)
        """
        with torch.no_grad():
            self.embedding.weight[index] = (
                decay * self.embedding.weight[index] +
                (1 - decay) * z
            )
            self.embedding.weight[index] = F.normalize(
                self.embedding.weight[index], dim=-1
            )

    # ------------------------------------------------------------------
    # Section 7: Utilization tracking
    # ------------------------------------------------------------------

    def utilization(self) -> float:
        """Fraction of codebook entries used at least once."""
        used = (self.usage_count > 0).sum().item()
        return used / self.codebook_size

    # ------------------------------------------------------------------
    # Section 7.2: Dead entry recovery
    # ------------------------------------------------------------------

    def reset_unused_entries(self, threshold: int = 0):
        """
        Replace unused codebook entries with perturbed copies of used entries.
        Call periodically (every 1000 training steps) if utilization < 30%.

        Args:
            threshold: entries with usage_count <= threshold are considered unused
        """
        with torch.no_grad():
            unused_mask = self.usage_count <= threshold
            used_mask = ~unused_mask

            if unused_mask.sum() == 0:
                return  # All entries used

            if used_mask.sum() == 0:
                # Catastrophic collapse — reinitialize entire codebook
                self.embedding.weight.data = F.normalize(
                    torch.randn_like(self.embedding.weight), dim=-1
                )
                self.usage_count.zero_()
                self.total_count.zero_()
                return

            used_indices = torch.where(used_mask)[0]
            unused_indices = torch.where(unused_mask)[0]

            for unused_idx in unused_indices:
                # Random unit vector on the hypersphere.
                # Rationale: when active entries are few and EMA-tuned to tight
                # cluster centroids, any perturbation of those centroids still
                # lands inside the centroid's Voronoi cell and never wins a
                # nearest-neighbor competition. Random probes spread across the
                # full sphere instead, acting as sensors for future Reasoner
                # output directions as SST training diversifies the latent space.
                rand_vec = torch.randn(self.dim, device=self.embedding.weight.device)
                self.embedding.weight[unused_idx] = F.normalize(rand_vec, dim=-1)

            # Reset counts after recovery
            self.usage_count.zero_()
            self.total_count.zero_()
