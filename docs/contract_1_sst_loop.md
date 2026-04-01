# CONTRACT 1: SST Training Loop

## Purpose
This document specifies the exact behavior of the Self-Supervised Training loop — the core of the JEPA-Coder project. Every tensor shape, every operation, and every gradient flow decision is specified here. Implementation must match this contract exactly.

---

## 1. Inputs to the SST Loop

Each training example is a tuple:
```
(
    problem_text: str,          # Natural language problem description
    blocks: List[dict],         # AST-segmented solution blocks
                                # Each block: {'type': str, 'code': str}
                                # Last block is always {'type': 'STOP', 'code': '<STOP>'}
)
```

Typical block count: 3-8 meaningful blocks + 1 STOP = 4-9 total steps.

---

## 2. Components Involved

| Component | Trainable? | Parameters |
|-----------|-----------|------------|
| reasoner.embedding | YES | nn.Embedding(vocab_size, 768) |
| reasoner.transformer_blocks | YES | 16 Transformer layers |
| reasoner.rms_norm | YES | RMSNorm with learnable γ |
| reasoner.l2_norm | NO | F.normalize(x, dim=-1) |
| ema_encoder | NO (EMA updated) | nn.Embedding(vocab_size, 768) |
| vq_layer | Codebook via EMA | VectorQuantizer(512, 768) |
| optimizer | - | AdamW over reasoner params only |

---

## 3. The Loop — Step by Step with Tensor Shapes

### 3.1 Problem Encoding (runs once per example)

```python
# Input
problem_tokens: LongTensor of shape (L_prob,)
# L_prob = number of BPE tokens in problem statement, variable per example, ≤ 1024

# Step A: Embed
prob_embeds = reasoner.embedding(problem_tokens)
# Shape: (L_prob, 768)
# These are raw embeddings, NOT unit normalized

# Step B: Process through Transformer
prob_processed = reasoner.transformer_blocks(prob_embeds)
# Shape: (L_prob, 768)
# Standard Transformer forward: self-attention + FFN at each layer
# QK-Norm applied inside each attention block

# Step C: Hybrid normalization
prob_rms = reasoner.rms_norm(prob_processed)
# Shape: (L_prob, 768)
# RMSNorm: x / sqrt(mean(x²) + eps) * γ

prob_l2 = F.normalize(prob_rms, dim=-1)
# Shape: (L_prob, 768)
# Each row now has L2 norm = 1

# Step D: Mean pool to single vector
h = prob_l2.mean(dim=0)
# Shape: (768,)
# WARNING: mean of unit vectors is NOT unit length

# Step E: Re-normalize
h = F.normalize(h, dim=-1)
# Shape: (768,)
# ||h|| = 1 GUARANTEED

# VERIFICATION: assert torch.abs(h.norm() - 1.0) < 1e-4
```

### 3.2 One Reasoning Step (runs once per block)

```python
# Input
h: FloatTensor of shape (768,) with ||h|| = 1

# Step A: Expand to sequence format
h_seq = h.unsqueeze(0)
# Shape: (1, 768) — a sequence of length 1

# Step B: Process through Transformer
r_seq = reasoner.transformer_blocks(h_seq)
# Shape: (1, 768)
# NOTE: Self-attention over length-1 sequence is identity for attention
# The FFN layers do the actual computation here

# Step C: Squeeze back to vector
r = r_seq.squeeze(0)
# Shape: (768,)

# Step D: Hybrid normalization
r = reasoner.rms_norm(r.unsqueeze(0)).squeeze(0)  # RMSNorm expects batch dim
r = F.normalize(r, dim=-1)
# Shape: (768,)
# ||r|| = 1 GUARANTEED

# VERIFICATION: assert torch.abs(r.norm() - 1.0) < 1e-4
```

### 3.3 Target Generation (runs once per block, NO GRADIENTS)

```python
# Input
block_code: str  # The actual code text of this block (or '<STOP>')

with torch.no_grad():
    # Step A: Tokenize
    block_tokens = tokenizer(block_code, return_tensors='pt').input_ids.squeeze(0)
    # Shape: (L_block,) — variable length per block

    # Step B: Embed with EMA encoder
    block_embeds = ema_encoder(block_tokens)
    # Shape: (L_block, 768)
    # NOTE: ema_encoder is nn.Embedding, NOT a Transformer
    # It just looks up embeddings, no attention/FFN processing

    # Step C: Mean pool
    t = block_embeds.mean(dim=0)
    # Shape: (768,)

    # Step D: L2 normalize
    t = F.normalize(t, dim=-1)
    # Shape: (768,)
    # ||t|| = 1 GUARANTEED
```

### 3.4 Loss Computation (runs once per block)

```python
# Inputs
r: FloatTensor of shape (768,) with ||r|| = 1  # Reasoner prediction
t: FloatTensor of shape (768,) with ||t|| = 1  # EMA target

# SST Loss: Scaled cosine distance
cos_sim = torch.dot(r, t)  
# Scalar in [-1, 1]
# EQUIVALENT to F.cosine_similarity since both are unit vectors
# When both unit norm: cos_sim = r · t = r^T t

sst_loss = 4.0 * (1.0 - cos_sim)
# Scalar in [0, 8]
# = 0 when r = t (perfect prediction)
# = 4 when r ⊥ t (orthogonal)
# = 8 when r = -t (opposite)

# NUMERICAL VERIFICATION:
# 4 * (1 - cos_sim) = 4 * (1 - r·t) = 2 * ||r - t||²  (for unit vectors)
# So: sst_loss ≈ 2 * (r - t).pow(2).sum()
# These two computations MUST agree within floating point tolerance

# VQ Loss (see Contract 2 for details)
quantized, idx, vq_loss = vq_layer(r)
# quantized: (768,), exact codebook entry
# idx: integer in [0, 511]
# vq_loss: scalar (commitment loss)

# Total step loss
step_loss = sst_loss + vq_loss
```

### 3.5 Loopback (runs once per block)

```python
# The CONTINUOUS state loops back, NOT the quantized state
h = r
# Shape: (768,) with ||h|| = 1
# This is the input to the next reasoning step

# The quantized index is STORED but does NOT affect the loop
stored_indices.append(idx.item())
```

### 3.6 After All Blocks (runs once per example)

```python
# Accumulate loss across all blocks
total_loss = sum(step_losses)  # Sum of sst_loss + vq_loss for each block

# Backpropagate
total_loss.backward()

# Gradient accumulation check
example_count += 1
if example_count % ACCUMULATION_STEPS == 0:  # ACCUMULATION_STEPS = 16
    torch.nn.utils.clip_grad_norm_(reasoner.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    
    # EMA update (AFTER optimizer step)
    with torch.no_grad():
        for p_input, p_target in zip(reasoner.embedding.parameters(), 
                                      ema_encoder.parameters()):
            p_target.data.mul_(0.98).add_(p_input.data, alpha=0.02)
```

---

## 4. What Must NOT Happen

| Violation | Why It's Fatal |
|-----------|---------------|
| VQ output loops back instead of continuous r | Straight-through gradient noise compounds across steps; training becomes unstable |
| L2 normalization skipped | Vectors grow unbounded across steps; cosine loss becomes meaningless |
| EMA update happens before optimizer.step() | Target encoder sees pre-update weights, creating stale targets |
| Gradients flow through the target | Target should be stable; gradient flow defeats the purpose of EMA |
| Loss computed on non-normalized vectors | Cosine similarity on non-unit vectors is a different function; loss range changes |
| Transformer receives (768,) instead of (1, 768) | PyTorch Transformer expects sequence dimension; will crash or silently broadcast wrong |
| Mean pooling without re-normalization | Mean of unit vectors has norm < 1; breaks hypersphere constraint |

---

## 5. Verification Tests

### Test 1: Norm Preservation Across Steps

```python
def test_norm_preservation():
    reasoner = Reasoner(dim=768, layers=16)
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
```

### Test 2: Loss Range Validation

```python
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
```

### Test 3: EMA Convergence Direction

```python
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
```

### Test 4: Gradient Flow Through Loop

```python
def test_gradient_flow():
    reasoner = Reasoner(dim=64, layers=2)  # Tiny model
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
```

### Test 5: Loss Decreases on Synthetic Data

```python
def test_synthetic_training():
    reasoner = Reasoner(dim=64, layers=2)
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
```

---

## 6. Expected Loss Trajectory During Real Training

| Training stage | Expected SST loss | Notes |
|---------------|-------------------|-------|
| Step 0 | ~4.0 | Random unit vectors have expected cosine similarity ≈ 0 in high dimensions |
| Step 1K | 3.0 - 3.5 | Model starts aligning with targets |
| Step 10K | 2.0 - 2.5 | Meaningful structure emerging |
| Step 50K | 1.0 - 1.5 | Converging |
| Step 100K | 0.5 - 1.0 | Stabilized |

If loss stays above 3.5 after 10K steps, something is wrong (likely normalization or EMA issue).
If loss drops below 0.1, the model may be overfitting to training data.
