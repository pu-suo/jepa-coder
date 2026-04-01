"""
Verification tests for the SST training loop components.
Source: docs/contract_1_sst_loop.md Section 5 (exact contract code).

Adaptations from contract to our codebase:
  - Reasoner(dim=768, layers=16)  → Reasoner(vocab_size=49152, dim=768, n_layers=16, n_heads=12, ffn_dim=3072)
  - Reasoner(dim=64, layers=2)    → Reasoner(vocab_size=100,   dim=64,  n_layers=2,  n_heads=4,  ffn_dim=256)
  - VectorQuantizer(16, 64)       → unchanged (codebook_size, dim)
  - hybrid_norm.l2_enabled=True   set on Reasoner instances used in SST paths
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.reasoner import Reasoner
from models.vq import VectorQuantizer


# ---------------------------------------------------------------------------
# Test 1: Norm Preservation Across Steps
# contract_1_sst_loop.md §5 Test 1
# ---------------------------------------------------------------------------

def test_norm_preservation():
    reasoner = Reasoner(vocab_size=49152, dim=768, n_layers=16, n_heads=12, ffn_dim=3072)
    reasoner.hybrid_norm.l2_enabled = True
    reasoner.eval()

    # Random initial state
    h = F.normalize(torch.randn(768), dim=-1)

    for step in range(10):
        h_seq = h.unsqueeze(0)
        r_seq = reasoner.transformer_blocks(h_seq)
        r = r_seq.squeeze(0)
        r = reasoner.rms_norm(r.unsqueeze(0)).squeeze(0)
        r = F.normalize(r, dim=-1)

        norm = r.norm().item()
        assert abs(norm - 1.0) < 1e-4, f"Step {step}: norm = {norm}"
        h = r

    print("PASS: Norm preserved across 10 steps")


# ---------------------------------------------------------------------------
# Test 2: Loss Range Validation
# contract_1_sst_loop.md §5 Test 2
# ---------------------------------------------------------------------------

def test_loss_range():
    # Random unit vectors
    for _ in range(1000):
        r = F.normalize(torch.randn(768), dim=-1)
        t = F.normalize(torch.randn(768), dim=-1)

        cos_sim = torch.dot(r, t)
        loss = 4.0 * (1.0 - cos_sim)

        assert 0 <= loss.item() <= 8.0, f"Loss out of range: {loss.item()}"

        # Cross-check with L2 distance
        l2_loss = 2.0 * (r - t).pow(2).sum()
        assert abs(loss.item() - l2_loss.item()) < 1e-4, \
            f"Cosine/L2 mismatch: {loss.item()} vs {l2_loss.item()}"

    # Self-similarity should give ~0 loss
    r = F.normalize(torch.randn(768), dim=-1)
    self_loss = 4.0 * (1.0 - torch.dot(r, r))
    assert self_loss.item() < 1e-5, f"Self-loss should be ~0: {self_loss.item()}"

    print("PASS: Loss always in [0, 8], cosine/L2 agree, self-loss ≈ 0")


# ---------------------------------------------------------------------------
# Test 3: EMA Convergence Direction
# contract_1_sst_loop.md §5 Test 3
# ---------------------------------------------------------------------------

def test_ema_convergence():
    input_emb = nn.Embedding(1000, 768)
    ema_emb = nn.Embedding(1000, 768)
    ema_emb.load_state_dict(input_emb.state_dict())

    # Perturb input
    with torch.no_grad():
        input_emb.weight.add_(torch.randn_like(input_emb.weight) * 0.1)

    initial_dist = (ema_emb.weight - input_emb.weight).norm().item()

    for _ in range(100):
        with torch.no_grad():
            for p_in, p_ema in zip(input_emb.parameters(), ema_emb.parameters()):
                p_ema.data.mul_(0.98).add_(p_in.data, alpha=0.02)

    final_dist = (ema_emb.weight - input_emb.weight).norm().item()
    assert final_dist < initial_dist, \
        f"EMA diverged: {initial_dist:.4f} → {final_dist:.4f}"

    print(f"PASS: EMA converged: {initial_dist:.4f} → {final_dist:.4f}")


# ---------------------------------------------------------------------------
# Test 4: Gradient Flow Through Loop
# contract_1_sst_loop.md §5 Test 4
# ---------------------------------------------------------------------------

def test_gradient_flow():
    reasoner = Reasoner(vocab_size=100, dim=64, n_layers=2, n_heads=4, ffn_dim=256)
    ema_enc = nn.Embedding(100, 64)
    vq = VectorQuantizer(16, 64)

    # Simulate 3-step reasoning
    h = F.normalize(torch.randn(64), dim=-1)
    total_loss = 0

    for step in range(3):
        h_seq = h.unsqueeze(0)
        r_seq = reasoner.transformer_blocks(h_seq)
        r = r_seq.squeeze(0)
        r = F.normalize(r, dim=-1)

        with torch.no_grad():
            t = F.normalize(torch.randn(64), dim=-1)

        sst_loss = 4.0 * (1.0 - torch.dot(r, t))
        _, _, vq_loss = vq(r)
        total_loss = total_loss + sst_loss + vq_loss
        h = r  # Loop back

    total_loss.backward()

    # Check gradients exist on reasoner parameters
    has_grad = any(p.grad is not None and p.grad.norm() > 0
                   for p in reasoner.parameters())
    assert has_grad, "No gradients reached Reasoner parameters"

    # Check NO gradients on EMA encoder
    has_ema_grad = any(p.grad is not None for p in ema_enc.parameters())
    assert not has_ema_grad, "Gradients leaked into EMA encoder"

    print("PASS: Gradients flow through Reasoner, not through EMA")


# ---------------------------------------------------------------------------
# Test 5: Loss Decreases on Synthetic Data
# contract_1_sst_loop.md §5 Test 5
# ---------------------------------------------------------------------------

def test_synthetic_training():
    reasoner = Reasoner(vocab_size=100, dim=64, n_layers=2, n_heads=4, ffn_dim=256)
    ema_enc = nn.Embedding(100, 64)
    vq = VectorQuantizer(16, 64)
    optimizer = torch.optim.AdamW(reasoner.parameters(), lr=1e-3)

    # Fixed synthetic targets (same targets every time = easy to learn)
    fixed_targets = [F.normalize(torch.randn(64), dim=-1) for _ in range(3)]

    losses = []
    for epoch in range(50):
        optimizer.zero_grad()

        # Start from a fixed "problem" embedding
        h = F.normalize(reasoner.embedding(torch.tensor([0])).squeeze(), dim=-1)

        epoch_loss = 0
        for step, target in enumerate(fixed_targets):
            h_seq = h.unsqueeze(0)
            r_seq = reasoner.transformer_blocks(h_seq)
            r = F.normalize(r_seq.squeeze(0), dim=-1)

            sst_loss = 4.0 * (1.0 - torch.dot(r, target))
            _, _, vq_loss = vq(r)
            epoch_loss = epoch_loss + sst_loss + vq_loss
            h = r

        epoch_loss.backward()
        optimizer.step()

        with torch.no_grad():
            for p_in, p_ema in zip(reasoner.embedding.parameters(),
                                    ema_enc.parameters()):
                p_ema.data.mul_(0.98).add_(p_in.data, alpha=0.02)

        losses.append(epoch_loss.item())

    assert losses[-1] < losses[0] * 0.5, \
        f"Loss didn't decrease enough: {losses[0]:.4f} → {losses[-1]:.4f}"

    print(f"PASS: Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running SST component verification tests...\n")

    print("Test 1: Norm preservation across steps")
    test_norm_preservation()

    print("Test 2: Loss range validation")
    test_loss_range()

    print("Test 3: EMA convergence direction")
    test_ema_convergence()

    print("Test 4: Gradient flow through loop")
    test_gradient_flow()

    print("Test 5: Loss decreases on synthetic data")
    test_synthetic_training()

    print("\nAll 5 SST verification tests PASSED")
