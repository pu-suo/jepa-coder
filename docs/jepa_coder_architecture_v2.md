# JEPA-Coder: Architectural Truth Document (v2)

## READ THIS FIRST

This document is the single source of truth for the JEPA-Coder architecture. Every design decision has been deliberated, debated, and resolved. If a decision seems wrong or suboptimal, ASK before changing it. Do not substitute standard implementations for custom components. Do not add fallback paths that bypass the intended architecture.

This document supersedes all previous versions.

---

## 1. What JEPA-Coder Is

JEPA-Coder is a decoupled latent reasoning architecture for code generation. It has three components:

1. **Reasoner**: A modified Transformer that reasons autoregressively in continuous latent space to produce a chain of latent states representing a solution plan
2. **VQ Layer**: Quantizes the Reasoner's output states into discrete codebook entries before the Talker sees them
3. **Talker**: An encoder-decoder Transformer that translates discrete codebook entries + problem statement into executable Python

The core thesis: the Reasoner thinks fluidly in continuous space (maintaining multi-hypothesis states, smooth transitions). The VQ layer crystallizes these thoughts into unambiguous discrete instructions. The Talker translates these instructions into syntactically exact code.

---

## 2. Architecture Details

### 2.1 Reasoner

**Type**: Decoder-only Transformer, custom-built (NOT a HuggingFace pretrained model)

**Specifications**:
| Parameter | Value |
|-----------|-------|
| Layers | 16 |
| Hidden dim (d) | 768 |
| Attention heads | 12 (64 dim per head) |
| FFN dim | 3072 (4× hidden) |
| Context length | 1024 tokens |
| Vocabulary | StarCoder2 BPE tokenizer (~49K tokens) |
| Total params | ~350M |

**Custom components that MUST be implemented exactly**:

**QK-Norm (Non-learnable)**: In every attention block, L2-normalize Q and K vectors before computing attention scores. No learnable parameters. This prevents attention logit explosion during the autoregressive latent loop.
```
q = F.normalize(q, dim=-1)
k = F.normalize(k, dim=-1)
attn_scores = (q @ k.T) * scale
```

**Hybrid Normalization Layer**: Applied after the final Transformer block. Two operations in sequence:
1. RMSNorm (with learnable scale parameter γ)
2. L2 Normalization (no learnable parameters)

The L2 norm projects all output vectors to the unit hypersphere S^(d-1). This is what bounds error propagation across reasoning steps.

**State**: During pretraining, L2 normalization is DISABLED (only RMSNorm runs). During SST, both are ENABLED.

**Tied Embeddings**: During pretraining only, the input embedding matrix and the temporary LM head share weights (W_head = W_embed^T). This creates angular alignment in the embedding space that the SST phase depends on. The LM head is permanently removed when SST begins.

### 2.2 VQ Layer

**Type**: Standard Vector Quantizer (VQ-VAE style)

| Parameter | Value |
|-----------|-------|
| Codebook size | 512 |
| Codebook dim | 768 (matches hidden dim) |
| Commitment cost β | 0.25 |
| Update method | EMA codebook updates |
| STOP entry | Index 0 is reserved for STOP |

**Operations**:
```
Input: z ∈ R^d, L2-normalized (||z|| = 1)
1. distances = ||z - c_k||² for all codebook entries c_k
2. idx = argmin(distances)
3. quantized = codebook[idx]  (exact codebook entry, no interpolation)
4. commitment_loss = ||sg[z] - quantized||² + β·||z - sg[quantized]||²
5. straight_through = z + (quantized - z).detach()  (for gradient flow)
```

**Codebook entries should also be L2-normalized** after each update to stay on the hypersphere.

**Utilization tracking**: Count unique indices per batch. If utilization drops below 30%, reset unused entries to random perturbations of the most-used entries.

### 2.3 EMA Target Encoder

**This is NOT a full model.** It is a single embedding layer (nn.Embedding) whose weights are an EMA of the Reasoner's input embedding layer weights.

**Update rule** (applied after each optimizer step):
```
θ_target = 0.98 · θ_target + 0.02 · θ_input
```

**Purpose**: Takes the BPE tokens of the next code block, embeds them, mean-pools to a single vector, and L2-normalizes. This produces the training target for the Reasoner.

**What it processes**: The actual tokens of the next solution block (during training only). At inference time, the EMA encoder is not used.

### 2.4 Talker

**Type**: Encoder-Decoder Transformer

| Component | Layers | Dim | Heads | Params |
|-----------|--------|-----|-------|--------|
| Encoder | 4 | 768 | 12 | ~50M |
| Decoder | 4 | 768 | 12 | ~50M |
| Embeddings + LM head | - | 768→vocab | - | ~50M |
| **Total** | | | | **~150M** |

**Encoder input**: The concatenation of two sequences:
1. Problem statement tokens, embedded via a learned embedding layer
2. Codebook indices from the Reasoner, re-embedded via a learned embedding table (512 × 768)

These are concatenated into a single sequence: [problem_embeds; plan_embeds]. Position embeddings distinguish the two segments.

**Decoder**: Autoregressive code generator. Cross-attends to the full encoder output. Generates Python code token by token.

**The Talker CANNOT reason independently.** It receives structural instructions (codebook indices) and translates them. If the codebook indices are garbage, the Talker's output must also be garbage. This is verified by ablation tests.

---

## 3. Tensor Shapes Through the SST Training Loop

This section specifies the exact tensor shapes at every point. This is the most critical section of this document.

**Notation**: B = batch size (but see Section 4 — we process one example at a time), L_prob = problem token count, L_block = block token count, d = 768.

### 3.1 Problem Encoding

```
Input:  problem_tokens: (L_prob,)  — integer token IDs
After embedding: (L_prob, d)  — one d-dimensional vector per token
After Transformer blocks: (L_prob, d)  — same shape, contextualized
After hybrid norm: (L_prob, d)  — each vector has unit L2 norm
Mean pool over sequence: (d,)  — single vector, re-L2-normalize after pooling
Result: h_0 of shape (d,) with ||h_0|| = 1
```

### 3.2 One Reasoning Step

```
Input: h_{t-1} of shape (d,) with ||h_{t-1}|| = 1

Expand to sequence: (1, d)  — treat as a single-token sequence
After Transformer blocks: (1, d)  — self-attention over 1 token
Squeeze: (d,)
After hybrid norm: (d,) with ||r_t|| = 1

This is the predicted state r_t.
```

**IMPORTANT ARCHITECTURAL NOTE**: After the first step, the Transformer processes a single-token sequence. Self-attention over a single token is trivially the identity (the token attends only to itself). This means the multi-head attention is not contributing after step 0. The effective computation is: FFN layers + normalization.

This is a known limitation. The Transformer blocks still function (FFN layers do meaningful computation), but the attention mechanism is underutilized. If this proves problematic empirically, a future improvement would be to maintain a KV-cache of previous reasoning states so the Transformer can attend to the full history of reasoning. But for v1, we keep it simple: single-vector loopback.

### 3.3 Target Generation

```
Input: block_tokens: (L_block,)  — integer token IDs of the next code block
After EMA embedding: (L_block, d)  — one vector per token
Mean pool: (d,)  — single vector
L2 normalize: (d,) with ||t_t|| = 1

This is the target t_t.
```

### 3.4 Loss Computation

```
SST loss: L_sst = 4.0 * (1 - cosine_similarity(r_t, t_t))
  - r_t: (d,), unit norm
  - t_t: (d,), unit norm
  - Result: scalar in [0, 8]

VQ loss: quantized, idx, L_vq = VQ(r_t)
  - Input: (d,), unit norm
  - quantized: (d,), exact codebook entry
  - idx: integer in [0, 511]
  - L_vq: scalar (commitment loss)

Step loss: L_step = L_sst + L_vq
```

### 3.5 Loopback

```
h_t = r_t  (the CONTINUOUS state, NOT quantized)
Shape: (d,) with ||h_t|| = 1
```

### 3.6 What Gets Stored for the Talker

```
At each step, store the VQ index: idx_t (integer)
After all steps: plan = [idx_1, idx_2, ..., idx_M] (list of integers)
```

---

## 4. Training Phases

### Phase 1: Pretraining

**Data**: The Stack v2 (Python) + subset of C4/Wikitext (for natural language understanding of problem statements)

**Why natural language data**: The Reasoner must understand problem statements written in English. Pretraining only on code would leave it unable to process the problem descriptions.

**Architecture state**:
- Tied embeddings: ENABLED
- L2 normalization: DISABLED (only RMSNorm)
- LM head: ENABLED (temporary)
- VQ: NOT PRESENT

**Objective**: Standard next-token prediction (cross-entropy loss)

**Hyperparameters**:
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4, cosine schedule, 2000 warmup |
| Effective batch size | 128 (32 × 4 accumulation) |
| Weight decay | 0.1 |
| Precision | BF16 |
| Steps | 100K-150K |

**Output**: `pretrained_reasoner.pt`

### Phase 2: Self-Supervised Training (SST)

**Data**: (problem, solution_blocks) pairs from APPS + TACO + OpenCodeReasoning (filtered to parseable Python solutions)

**Architecture state**:
- Tied embeddings: Weights carried over from pretraining (input embedding trainable, LM head REMOVED)
- L2 normalization: ENABLED
- VQ: ENABLED (for loss computation, NOT in the loop)
- EMA target encoder: ENABLED (initialized from input embedding weights, momentum 0.98)

**Processing**: One example at a time (no batching across examples, use gradient accumulation for effective batch size). This avoids complexity of padding variable-length block sequences.

**The training loop** (pseudocode — implement EXACTLY):

```python
for each (problem_text, [block_0, block_1, ..., block_M, STOP_block]):
    
    # === ENCODE PROBLEM ===
    prob_tokens = tokenizer(problem_text)           # (L_prob,)
    prob_embeds = reasoner.embed(prob_tokens)        # (L_prob, d)
    prob_processed = reasoner.transformer(prob_embeds)  # (L_prob, d)
    prob_normed = reasoner.hybrid_norm(prob_processed)  # (L_prob, d), unit norms
    h = prob_normed.mean(dim=0)                     # (d,)  mean pool
    h = F.normalize(h, dim=-1)                      # (d,)  re-normalize after pooling
    
    total_loss = 0.0
    stored_indices = []
    
    # === REASONING LOOP ===
    for block in [block_0, block_1, ..., block_M, STOP_block]:
        
        # Reasoner forward pass
        h_input = h.unsqueeze(0)                    # (1, d) — single-token sequence
        r_seq = reasoner.transformer(h_input)       # (1, d)
        r = r_seq.squeeze(0)                        # (d,)
        r = reasoner.hybrid_norm_vector(r)          # (d,) with ||r|| = 1
        
        # Target generation (no gradients)
        with torch.no_grad():
            blk_tokens = tokenizer(block['code'])   # (L_block,)
            blk_embeds = ema_encoder(blk_tokens)    # (L_block, d)
            t = blk_embeds.mean(dim=0)              # (d,)
            t = F.normalize(t, dim=-1)              # (d,) with ||t|| = 1
        
        # SST loss
        cos_sim = F.cosine_similarity(r.unsqueeze(0), t.unsqueeze(0))  # (1,)
        sst_loss = 4.0 * (1.0 - cos_sim)           # scalar in [0, 8]
        
        # VQ (NOT in the loop — for loss and index storage only)
        quantized, idx, vq_loss = vq_layer(r)
        stored_indices.append(idx.item())
        
        total_loss = total_loss + sst_loss + vq_loss
        
        # Loop back CONTINUOUS state
        h = r                                       # (d,) with ||h|| = 1
    
    # Backprop and update
    total_loss.backward()
    
    # Gradient accumulation
    if (example_count + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(reasoner.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        update_ema(reasoner.embedding, ema_encoder, decay=0.98)
```

**Hyperparameters**:
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4, cosine schedule, 1000 warmup |
| Gradient accumulation | 16 (effective batch = 16 examples) |
| Cosine scale k | 4 |
| EMA momentum | 0.98 |
| VQ commitment β | 0.25 |
| Max grad norm | 1.0 |
| Steps | 50K-100K examples |

**Monitoring**:
- SST loss: should decrease from ~4 (random unit vectors) toward <1
- VQ loss: should decrease and stabilize
- Codebook utilization: should stay >30%, ideally >50%
- Norm check: assert ||r|| ≈ 1 at every step (disable in production)

**CRITICAL CHECKPOINT after Phase 2**: Inspect the codebook before proceeding. For each codebook entry, find the 20 training examples whose Reasoner output was closest. If entries correspond to recognizable patterns (input parsing, loops, DP, output formatting), proceed. If entries are random, debug before continuing.

**Output**: `sst_reasoner.pt`, `vq_codebook.pt`

### Phase 3: Talker Training

**Data preparation**: Run frozen Reasoner + VQ on all training examples. Store (codebook_indices, problem_text, solution_code) triples.

**Architecture state**:
- Reasoner: FROZEN
- VQ: FROZEN
- Talker: TRAINING

**Talker encoder input construction**:
```python
# Problem tokens embedded with Talker's own embedding layer
prob_embeds = talker.text_embed(problem_tokens)     # (L_prob, d)

# Codebook indices re-embedded with Talker's plan embedding table
plan_embeds = talker.plan_embed(codebook_indices)   # (M, d)

# Concatenate with segment markers
encoder_input = concat([prob_embeds, plan_embeds])  # (L_prob + M, d)
# Add position embeddings and segment type embeddings to distinguish
```

**Objective**: Cross-entropy on code token prediction

**Hyperparameters**:
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Batch size | 32 |
| Steps | 30K-50K |

**Ablation tests (MUST PASS before proceeding)**:
1. Normal input → coherent, correct code ✓
2. Random codebook indices → incoherent garbage ✓
3. Gaussian noise as indices → nonsensical output ✓
4. Indices from wrong problem → code for wrong problem ✓

**Output**: `talker.pt`

---

## 5. Data Pipeline

### 5.1 Datasets

| Dataset | Source | Size | Use |
|---------|--------|------|-----|
| The Stack v2 (Python) | `bigcode/the-stack-v2-dedup` | ~50GB | Pretraining (code) |
| C4 (subset) | `allenai/c4` | ~5GB sample | Pretraining (NL understanding) |
| APPS | `codeparrot/apps` | 5K problems | SST + Talker |
| TACO | `BAAI/TACO` | 25K problems | SST + Talker |
| OpenCodeReasoning | `nvidia/OpenCodeReasoning` | 567K samples | SST + Talker (filtered) |
| LiveCodeBench | `livecodebench/code_generation_lite` | 880+ problems | Evaluation ONLY |

### 5.2 Solution Preprocessing

1. Extract Python solutions from each dataset (APPS: all Python; TACO: filter Python; OCR: use `solution` field)
2. Attempt `ast.parse()` on each solution — skip unparseable ones (~5-10% dropout)
3. Segment using the grouped AST strategy:
   - If single top-level node with body, segment the body
   - Group consecutive simple statements (Assign, Expr, Import)
   - Keep compound statements (For, If, While, FunctionDef) as individual blocks
   - Append STOP block
4. Skip solutions with <2 blocks or >15 blocks (outliers)
5. Tokenize each block with StarCoder2 tokenizer
6. Skip blocks with >512 tokens (very long blocks that won't fit)

### 5.3 Decontamination

Before evaluation, verify NO overlap between training problems and LiveCodeBench problems. Check by problem URL, problem title, and 8-gram text overlap on problem descriptions.

---

## 6. Inference Pipeline

```python
def generate_solution(problem_text, reasoner, vq, talker, tokenizer, max_steps=15):
    # 1. Encode problem
    prob_tokens = tokenizer(problem_text)
    h = reasoner.encode_problem(prob_tokens)  # Returns (d,) unit vector
    
    # 2. Reason
    plan_indices = []
    for step in range(max_steps):
        r = reasoner.step(h)                   # (d,) unit vector
        _, idx, _ = vq(r)
        
        if idx.item() == 0:                    # STOP
            break
        
        plan_indices.append(idx.item())
        h = r                                  # Loop back continuous
    
    # 3. Generate code
    code = talker.generate(plan_indices, prob_tokens, max_length=1024)
    
    return code
```

---

## 7. Evaluation

**Primary benchmark**: LiveCodeBench (pass@1)

**Baselines**:
1. Standard Transformer (~500M) fine-tuned on same data — shows improvement from decoupling
2. Continuous JEPA-Reasoner (~500M) without VQ — shows effect of discretization
3. Small public code LLMs (~1B) — contextualizes against existing models

**Ablations**:
1. VQ vs no VQ (central experiment)
2. Codebook size (256, 512, 1024)
3. AST blocks vs fixed token chunks
4. Codebook analysis (what patterns emerge)

---

## 8. Known Limitations and Honest Uncertainties

**Limitation 1: Single-token attention after step 0.** After the problem is encoded and pooled to a single vector, subsequent Reasoner steps process a single-token sequence. Multi-head attention doesn't contribute meaningfully. The FFN layers do the computation. This may limit the Reasoner's expressivity. A KV-cache of previous states could help but adds complexity.

**Limitation 2: Mean-pooled targets are low-resolution.** The EMA target for each block is just the mean embedding of its tokens. This doesn't capture ordering or structure within the block — a for-loop-then-if and an if-then-for-loop could have similar mean embeddings. The VQ codebook must compensate by learning to distinguish these patterns from the Reasoner's output, not just from the targets.

**Limitation 3: Per-example processing is slow.** Without batching across the reasoning loop, training throughput is limited. Gradient accumulation helps but doesn't parallelize the sequential reasoning steps. Training time estimates may be optimistic.

**Uncertainty 1: Will the Reasoner learn problem→solution reasoning from cosine distance alone?** The training signal is: "make your latent state similar to the average embedding of the next code block." This is an indirect signal for algorithmic reasoning. JEPA-Reasoner showed it works for math. Whether it works for code is the central empirical question.

**Uncertainty 2: Will the VQ codebook learn meaningful code patterns?** This depends on whether the Reasoner's latent states for similar code structures cluster together. If they do, VQ naturally captures the clusters. If they don't, VQ produces meaningless entries. The codebook inspection after Phase 2 is the go/no-go checkpoint.

**Uncertainty 3: Can the Talker produce syntactically valid code from just codebook indices + problem text?** The codebook indices tell the Talker the structure. The problem text provides variable names and constraints. Whether this is enough information for valid Python is testable with oracle experiments (feed ground-truth codebook indices and measure Talker accuracy).

---

## 9. Verification Tests

These tests MUST PASS before proceeding past each phase. They are non-negotiable.

### After implementing models (before any training):

**Test 1 — Norm preservation**: Random input through Transformer + hybrid norm, looped 10 times. Assert ||output|| = 1 ± 1e-4 at every step.

**Test 2 — Loss range**: Cosine loss between random unit vectors. Assert loss ∈ [0, 8]. Assert loss(x, x) < 1e-5.

**Test 3 — EMA convergence**: Perturb input weights, run 100 EMA updates. Assert distance between EMA and input decreases.

**Test 4 — VQ correctness**: VQ output must exactly equal a codebook row. Gradient must flow through straight-through estimator.

**Test 5 — Synthetic SST**: Tiny model, random data, 100 steps. Assert loss decreases.

### After Phase 2 (SST):

**Test 6 — Codebook utilization**: >30% of entries used across 1000 examples.

**Test 7 — Codebook interpretability**: Manual inspection of top-20 neighbors per entry reveals recognizable patterns.

**Test 8 — STOP detection**: >90% of examples correctly produce STOP index after the last block.

### After Phase 3 (Talker):

**Test 9 — Talker ablation**: Four corruption tests produce expected failure modes.

**Test 10 — Parse rate**: >90% of Talker outputs pass `ast.parse()`.

---

## 10. What NOT To Do

- DO NOT use a pretrained HuggingFace model as the Reasoner. Train from scratch.
- DO NOT use cross-entropy loss during SST. Use scaled cosine distance (k=4).
- DO NOT put VQ in the autoregressive training loop. VQ is for loss and output only.
- DO NOT let the Talker see continuous Reasoner states. It sees only codebook indices.
- DO NOT skip hybrid normalization. L2 norm constrains states to the hypersphere.
- DO NOT batch examples with different block counts together. Process one at a time.
- DO NOT skip the codebook inspection checkpoint. It's the go/no-go decision.
- DO NOT train the Talker while the Reasoner is unfrozen. The Reasoner must be frozen.
- DO NOT add skip connections or fallback paths between the Reasoner and Talker.
- DO NOT use the Reasoner's pretrained LM head during SST. It must be removed.
