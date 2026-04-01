# CONTRACT 2: Vector Quantization Module

## Purpose
This document specifies the exact behavior of the VQ layer that discretizes the Reasoner's continuous latent states into codebook entries. Every operation, every gradient path, and every edge case is specified here.

---

## 1. Module Interface

```python
class VectorQuantizer(nn.Module):
    """
    Input:  z of shape (d,) — L2-normalized vector on the unit hypersphere
    Output: (quantized, index, loss)
        quantized: (d,) — exact codebook entry, L2-normalized
        index:     int  — codebook index in [0, codebook_size)
        loss:      scalar — commitment loss
    """
```

---

## 2. Internal State

```python
self.codebook_size = 512      # Number of entries (K)
self.dim = 768                # Dimension per entry (d)
self.commitment_cost = 0.25   # β in the commitment loss
self.embedding = nn.Embedding(512, 768)  # The codebook

# Codebook usage tracking
self.register_buffer('usage_count', torch.zeros(512))  # Per-entry usage
self.register_buffer('total_count', torch.tensor(0))   # Total quantizations
```

---

## 3. Initialization

```python
# Uniform initialization
self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 
                                      1.0 / self.codebook_size)

# L2-normalize all entries to the unit hypersphere
with torch.no_grad():
    self.embedding.weight.data = F.normalize(self.embedding.weight.data, dim=-1)

# Reserve index 0 for STOP
# No special initialization needed — the STOP entry will learn its position
# through training. But it MUST be index 0 so inference can check for it.
```

---

## 4. Forward Pass — Exact Operations

```python
def forward(self, z):
    """
    Args:
        z: FloatTensor of shape (d,) with ||z|| = 1
    Returns:
        quantized: FloatTensor of shape (d,) — exact codebook entry
        index: LongTensor scalar — codebook index
        loss: FloatTensor scalar — commitment loss
    """
    
    # ============================================
    # STEP 1: Input validation
    # ============================================
    assert z.shape == (self.dim,), f"Expected shape ({self.dim},), got {z.shape}"
    assert abs(z.norm().item() - 1.0) < 1e-3, f"Input not unit norm: {z.norm().item()}"
    
    # ============================================
    # STEP 2: Compute distances to all codebook entries
    # ============================================
    # z: (d,)
    # self.embedding.weight: (K, d)
    
    # For unit vectors: ||z - c_k||² = 2 - 2·(z · c_k)
    # So argmin distance = argmax dot product
    # Using dot product is numerically more stable
    
    dots = torch.matmul(self.embedding.weight, z)
    # Shape: (K,)  — dot product of z with each codebook entry
    
    index = dots.argmax()
    # Shape: scalar LongTensor — the nearest codebook entry index
    
    # VERIFICATION: index should be in [0, K)
    assert 0 <= index.item() < self.codebook_size
    
    # ============================================
    # STEP 3: Look up the quantized vector
    # ============================================
    quantized = self.embedding.weight[index]
    # Shape: (d,) — exact codebook entry, NO interpolation
    
    # VERIFICATION: quantized must exactly equal a codebook row
    assert torch.allclose(quantized, self.embedding.weight[index], atol=1e-6)
    
    # ============================================
    # STEP 4: Compute commitment loss
    # ============================================
    # Two-part loss from VQ-VAE (van den Oord et al., 2017):
    #   Part 1: Move codebook toward encoder output (codebook learning)
    #   Part 2: Move encoder output toward codebook (commitment)
    
    # Part 1: ||sg[z] - quantized||²
    # sg = stop gradient — gradients don't flow through z here
    codebook_loss = F.mse_loss(z.detach(), quantized)
    
    # Part 2: ||z - sg[quantized]||²
    # Gradients flow through z but not quantized
    commit_loss = F.mse_loss(z, quantized.detach())
    
    loss = codebook_loss + self.commitment_cost * commit_loss
    # Scalar
    
    # ============================================
    # STEP 5: Straight-through estimator
    # ============================================
    # During forward pass: output = quantized (discrete)
    # During backward pass: gradients flow as if output = z (continuous)
    # This is achieved by: output = z + (quantized - z).detach()
    
    quantized_st = z + (quantized - z).detach()
    # Shape: (d,)
    # Value: exactly equal to quantized
    # Gradient: flows through z
    
    # VERIFICATION: quantized_st should have the same VALUE as quantized
    assert torch.allclose(quantized_st, quantized, atol=1e-6)
    
    # ============================================
    # STEP 6: Update usage tracking
    # ============================================
    self.usage_count[index] += 1
    self.total_count += 1
    
    return quantized_st, index, loss
```

---

## 5. Codebook EMA Updates (Alternative to Part 1 of Loss)

Instead of using gradient-based codebook learning (Part 1 of the loss), use EMA updates on the codebook entries. This is more stable in practice.

```python
def update_codebook_ema(self, z, index, decay=0.99):
    """
    Call AFTER forward pass, OUTSIDE the computation graph.
    Moves the selected codebook entry toward z.
    
    Args:
        z: (d,) the encoder output that was quantized
        index: the codebook index that was selected
        decay: EMA decay rate
    """
    with torch.no_grad():
        # Move entry toward z
        self.embedding.weight[index] = (
            decay * self.embedding.weight[index] + 
            (1 - decay) * z
        )
        # Re-normalize to unit hypersphere
        self.embedding.weight[index] = F.normalize(
            self.embedding.weight[index], dim=-1
        )
```

When using EMA codebook updates, remove `codebook_loss` from the loss computation:
```python
loss = self.commitment_cost * commit_loss  # Only commitment part
```

**DECISION**: Use EMA codebook updates (not gradient-based). This matches the VQ-VAE best practice and avoids codebook gradient instability.

---

## 6. STOP Entry (Index 0)

Index 0 is reserved for the STOP signal. There is no special initialization or loss treatment — the STOP entry learns its position through normal training:

- During training, the last block of every example is `{'type': 'STOP', 'code': '<STOP>'}`.
- The EMA target encoder embeds `<STOP>` into a target vector.
- The Reasoner learns to produce a state that quantizes to index 0 when the solution is complete.
- During inference, the loop checks `if index == 0: break`.

**IMPORTANT**: The STOP entry must be free to move in the codebook space. Do NOT freeze it, do NOT initialize it to a special value, do NOT exclude it from EMA updates. Let it learn naturally.

**VERIFICATION**: After training, check that >90% of examples correctly produce index 0 as their final VQ output when the STOP block is the target.

---

## 7. Codebook Utilization Tracking and Recovery

### 7.1 Tracking

```python
def utilization(self):
    """Fraction of codebook entries used at least once."""
    used = (self.usage_count > 0).sum().item()
    return used / self.codebook_size

def utilization_recent(self, window=1000):
    """Fraction used in recent examples (requires per-batch tracking)."""
    # Track unique indices over last `window` forward calls
    pass
```

### 7.2 Dead Entry Recovery

If utilization drops below 30%, reset unused entries:

```python
def reset_unused_entries(self, threshold=0):
    """
    Replace unused codebook entries with perturbed copies of used entries.
    Call periodically (every 1000 training steps).
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
            return
        
        # Get the most-used entries
        used_indices = torch.where(used_mask)[0]
        unused_indices = torch.where(unused_mask)[0]
        
        for unused_idx in unused_indices:
            # Pick a random used entry
            donor_idx = used_indices[torch.randint(len(used_indices), (1,))]
            # Copy with small random perturbation
            self.embedding.weight[unused_idx] = F.normalize(
                self.embedding.weight[donor_idx] + 
                torch.randn(self.dim, device=self.embedding.weight.device) * 0.01,
                dim=-1
            )
        
        # Reset counts
        self.usage_count.zero_()
        self.total_count.zero_()
```

---

## 8. Gradient Flow Diagram

```
z (from Reasoner, has gradients)
│
├──→ Find nearest codebook entry (no gradient through argmax)
│     │
│     ▼
│    quantized (codebook entry, no gradient to z)
│     │
│     ├──→ codebook_loss = MSE(sg[z], quantized)  → updates codebook via EMA
│     │
│     └──→ commit_loss = MSE(z, sg[quantized]) * β  → gradient flows to z
│
└──→ straight_through = z + (quantized - z).detach()
      │
      ├── VALUE = quantized (discrete)
      └── GRADIENT = flows through z (continuous)
           │
           └──→ This is what gets STORED for the Talker (the value)
                The gradient path continues back through z
                to the Reasoner's Transformer blocks
```

**KEY INSIGHT**: The Reasoner receives gradients from two sources:
1. The SST cosine loss (comparing r to EMA target)
2. The VQ commitment loss (pushing r toward the nearest codebook entry)

Both gradients flow through r, which flows through the Transformer blocks back to the Reasoner's parameters.

---

## 9. Verification Tests

### Test 1: Output Is Exact Codebook Entry

```python
def test_exact_codebook_entry():
    vq = VectorQuantizer(512, 768)
    
    for _ in range(100):
        z = F.normalize(torch.randn(768), dim=-1)
        quantized, idx, loss = vq(z)
        
        # quantized must exactly equal the codebook row
        expected = vq.embedding.weight[idx]
        assert torch.allclose(quantized, expected, atol=1e-6), \
            f"Output doesn't match codebook entry {idx}"
    
    print("PASS: All outputs are exact codebook entries")
```

### Test 2: Gradient Flows Through z

```python
def test_gradient_flow():
    vq = VectorQuantizer(16, 64)
    z = F.normalize(torch.randn(64), dim=-1)
    z.requires_grad_(True)
    
    quantized, idx, loss = vq(z)
    
    # Backward through the quantized output (simulating downstream use)
    fake_downstream_loss = quantized.sum()
    fake_downstream_loss.backward()
    
    assert z.grad is not None, "No gradient on z"
    assert z.grad.norm() > 0, "Zero gradient on z"
    
    # Also check commitment loss gradient
    z2 = F.normalize(torch.randn(64), dim=-1)
    z2.requires_grad_(True)
    _, _, loss2 = vq(z2)
    loss2.backward()
    
    assert z2.grad is not None, "No gradient from commitment loss"
    
    print("PASS: Gradients flow through VQ via straight-through estimator")
```

### Test 3: Nearest Neighbor Is Correct

```python
def test_nearest_neighbor():
    vq = VectorQuantizer(512, 768)
    
    for _ in range(100):
        z = F.normalize(torch.randn(768), dim=-1)
        _, idx, _ = vq(z)
        
        # Verify this is actually the nearest entry
        all_distances = torch.cdist(z.unsqueeze(0), vq.embedding.weight).squeeze()
        true_nearest = all_distances.argmin()
        
        assert idx == true_nearest, \
            f"VQ returned index {idx} but nearest is {true_nearest}"
    
    print("PASS: VQ correctly finds nearest codebook entry")
```

### Test 4: Codebook Entries Stay on Hypersphere

```python
def test_codebook_normalization():
    vq = VectorQuantizer(512, 768)
    
    # After initialization
    norms = vq.embedding.weight.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(512), atol=1e-4), \
        f"Codebook not normalized after init: norms range [{norms.min():.4f}, {norms.max():.4f}]"
    
    # After EMA updates
    for _ in range(100):
        z = F.normalize(torch.randn(768), dim=-1)
        _, idx, _ = vq(z)
        vq.update_codebook_ema(z, idx)
    
    norms = vq.embedding.weight.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(512), atol=1e-4), \
        f"Codebook not normalized after EMA: norms range [{norms.min():.4f}, {norms.max():.4f}]"
    
    print("PASS: Codebook entries stay on unit hypersphere")
```

### Test 5: Utilization Tracking Works

```python
def test_utilization():
    vq = VectorQuantizer(512, 768)
    
    # Initially zero utilization
    assert vq.utilization() == 0.0
    
    # After many random inputs, utilization should grow
    for _ in range(10000):
        z = F.normalize(torch.randn(768), dim=-1)
        vq(z)
    
    util = vq.utilization()
    assert util > 0.5, f"Low utilization after 10K random inputs: {util:.2%}"
    
    print(f"PASS: Utilization = {util:.2%} after 10K random inputs")
```

---

## 10. What NOT To Do

| Violation | Consequence |
|-----------|------------|
| Interpolate between codebook entries | Defeats the purpose of discrete representation; Talker receives ambiguous signals |
| Allow gradients to flow through codebook lookup (argmax) | argmax is non-differentiable; attempting it will crash or silently do nothing |
| Skip L2 normalization of codebook entries | Entries drift off hypersphere; distance calculations become inconsistent with the Reasoner's normalized space |
| Use a codebook that's too small (<128) | Not enough entries to represent diverse code patterns |
| Use a codebook that's too large (>4096) | Many entries go unused; codebook collapse becomes likely |
| Initialize STOP entry to a specific value | Biases learning; let it find its natural position |
| Freeze codebook during training | Codebook can't adapt to Reasoner's evolving representations |
| Skip dead entry recovery | Unused entries waste capacity; effective codebook shrinks over time |
