# CONTRACT 3: Talker Interface

## Purpose
This document specifies the exact interface between the Reasoner's discrete output and the Talker's code generation. The Talker is a translation module — it converts structural plans into Python code. It MUST NOT be able to reason independently. This contract specifies what goes in, what comes out, and how to verify the decoupling.

---

## 1. What the Talker Receives

The Talker receives exactly two inputs:

### Input 1: Plan Indices (from VQ)
```
plan_indices: List[int]
Length: M (variable, typically 3-8, determined by Reasoner's STOP)
Values: integers in [1, 511] (index 0 = STOP, never passed to Talker)
Example: [47, 183, 92, 215]
```

These are the VQ codebook indices produced by the Reasoner during either:
- SST training (from running the frozen Reasoner on training examples)
- Inference (from the live Reasoner loop)

### Input 2: Problem Text (original problem statement)
```
problem_tokens: LongTensor of shape (L_prob,)
The BPE-tokenized problem description.
```

The problem text provides the Talker with:
- Variable names and identifiers from the examples
- Input/output format specifications
- Constraint values (n ≤ 10^5, etc.)
- Domain context ("binary strings", "graph", "array")

---

## 2. What the Talker Does NOT Receive

| Forbidden Input | Why |
|----------------|-----|
| Continuous Reasoner states (r_1, r_2, ...) | Would let Talker bypass the VQ bottleneck |
| The Reasoner's internal activations | Would create a hidden continuous channel |
| Ground truth solution code (during inference) | Obviously — that's what it's generating |
| Block type labels ("For", "If", "Initialization") | These are human annotations; the codebook should encode this information |

---

## 3. Talker Architecture

### 3.1 Components

```
Talker
├── text_embedding: nn.Embedding(vocab_size, 768)      # For problem tokens
├── plan_embedding: nn.Embedding(512, 768)              # For codebook indices
├── segment_embedding: nn.Embedding(2, 768)             # 0=problem, 1=plan
├── position_embedding: nn.Embedding(max_len, 768)      # Positional
├── encoder: TransformerEncoder(4 layers, 768 dim, 12 heads)
├── decoder: TransformerDecoder(4 layers, 768 dim, 12 heads)
└── lm_head: nn.Linear(768, vocab_size)                 # Output logits
```

### 3.2 Encoder Input Construction

```python
def build_encoder_input(self, problem_tokens, plan_indices):
    """
    Construct the encoder input by concatenating problem and plan embeddings.
    
    Args:
        problem_tokens: LongTensor (L_prob,) — problem text token IDs
        plan_indices: LongTensor (M,) — VQ codebook indices
    
    Returns:
        encoder_input: FloatTensor (L_prob + M, 768) — ready for encoder
    """
    # Embed problem tokens
    prob_embeds = self.text_embedding(problem_tokens)   # (L_prob, 768)
    
    # Embed plan indices
    # IMPORTANT: These use a SEPARATE embedding table, NOT the VQ codebook
    # The Talker's plan_embedding learns its own representation of each index
    plan_embeds = self.plan_embedding(plan_indices)     # (M, 768)
    
    # Concatenate
    combined = torch.cat([prob_embeds, plan_embeds], dim=0)  # (L_prob + M, 768)
    
    # Add segment embeddings
    prob_segments = self.segment_embedding(
        torch.zeros(len(problem_tokens), dtype=torch.long, device=combined.device)
    )  # (L_prob, 768) — all zeros = problem segment
    
    plan_segments = self.segment_embedding(
        torch.ones(len(plan_indices), dtype=torch.long, device=combined.device)
    )  # (M, 768) — all ones = plan segment
    
    combined = combined + torch.cat([prob_segments, plan_segments], dim=0)
    
    # Add position embeddings
    positions = torch.arange(combined.shape[0], device=combined.device)
    combined = combined + self.position_embedding(positions)
    
    return combined
    # Shape: (L_prob + M, 768)
```

### 3.3 Decoder (Code Generation)

```python
def generate(self, problem_tokens, plan_indices, max_length=1024):
    """
    Generate Python code autoregressively.
    
    The decoder cross-attends to the encoder output,
    which contains both the problem understanding and the structural plan.
    """
    # Encode
    encoder_input = self.build_encoder_input(problem_tokens, plan_indices)
    encoder_input = encoder_input.unsqueeze(0)  # (1, L_prob + M, 768) — add batch
    memory = self.encoder(encoder_input)         # (1, L_prob + M, 768)
    
    # Autoregressive decoding
    generated = [BOS_TOKEN_ID]  # Start with beginning-of-sequence
    
    for step in range(max_length):
        # Embed generated tokens so far
        gen_tokens = torch.tensor([generated], device=memory.device)  # (1, step+1)
        gen_embeds = self.text_embedding(gen_tokens)                   # (1, step+1, 768)
        gen_embeds = gen_embeds + self.position_embedding(
            torch.arange(len(generated), device=memory.device)
        )
        
        # Causal mask for decoder self-attention
        causal_mask = torch.triu(
            torch.ones(len(generated), len(generated), device=memory.device) * float('-inf'),
            diagonal=1
        )
        
        # Decoder forward
        decoder_out = self.decoder(
            gen_embeds,
            memory,
            tgt_mask=causal_mask
        )
        # Shape: (1, step+1, 768)
        
        # Predict next token
        logits = self.lm_head(decoder_out[:, -1, :])  # (1, vocab_size)
        next_token = logits.argmax(dim=-1).item()      # Greedy decoding
        
        if next_token == EOS_TOKEN_ID:
            break
        
        generated.append(next_token)
    
    return tokenizer.decode(generated[1:])  # Skip BOS, return code string
```

---

## 4. Training Procedure

### 4.1 Data Preparation (runs once, before Talker training)

```python
# Freeze Reasoner and VQ
reasoner.eval()
for p in reasoner.parameters():
    p.requires_grad_(False)
for p in vq.parameters():
    p.requires_grad_(False)

talker_data = []
for problem_text, solution_code, blocks in training_examples:
    # Run Reasoner to get plan indices
    h = reasoner.encode_problem(tokenizer(problem_text))
    indices = []
    for block in blocks[:-1]:  # Exclude STOP
        r = reasoner.step(h)
        _, idx, _ = vq(r)
        indices.append(idx.item())
        h = r
    
    talker_data.append({
        'problem_tokens': tokenizer(problem_text).input_ids,
        'plan_indices': indices,
        'target_code': tokenizer(solution_code).input_ids
    })
```

### 4.2 Training Loop

```python
for batch in talker_dataloader:
    problem_tokens = batch['problem_tokens']     # (B, L_prob) padded
    plan_indices = batch['plan_indices']          # (B, M) padded
    target_code = batch['target_code']            # (B, L_code) padded
    
    # Encoder
    encoder_input = talker.build_encoder_input(problem_tokens, plan_indices)
    memory = talker.encoder(encoder_input)
    
    # Decoder with teacher forcing
    target_input = target_code[:, :-1]    # Input: all tokens except last
    target_labels = target_code[:, 1:]    # Labels: all tokens except first
    
    decoder_input_embeds = talker.text_embedding(target_input)
    decoder_out = talker.decoder(decoder_input_embeds, memory, tgt_mask=causal_mask)
    logits = talker.lm_head(decoder_out)
    
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_labels.reshape(-1),
        ignore_index=PAD_TOKEN_ID
    )
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**CRITICAL: The Reasoner is completely frozen during Talker training.** No gradients flow back to the Reasoner. The Talker adapts to whatever plan indices the Reasoner produces.

---

## 5. Ablation Tests — Verifying the Decoupling

These tests verify that the Talker cannot reason independently and genuinely depends on the Reasoner's plan.

### Test 1: Baseline (Normal Operation)

```python
def test_baseline():
    """Normal input should produce coherent, relevant code."""
    problem = "Given an array of integers, find the two elements that sum to a target."
    
    # Get plan from Reasoner
    plan = reasoner.get_plan(problem)  # e.g., [47, 183, 92]
    
    code = talker.generate(tokenizer(problem), plan)
    
    # Verify code is parseable
    ast.parse(code)  # Must not raise SyntaxError
    
    # Verify code is relevant (contains expected keywords)
    assert any(kw in code for kw in ['for', 'if', 'sum', 'target', 'return'])
    
    print(f"PASS: Generated valid code:\n{code[:200]}...")
```

### Test 2: Random Indices (Garbage In → Garbage Out)

```python
def test_random_indices():
    """Random plan indices should produce incoherent output."""
    problem = "Given an array of integers, find the two elements that sum to a target."
    
    # Random indices instead of Reasoner output
    random_plan = [random.randint(1, 511) for _ in range(5)]
    
    code = talker.generate(tokenizer(problem), random_plan)
    
    # The output should be one of:
    # a) Unparseable (SyntaxError)
    # b) Parseable but nonsensical (doesn't solve the problem)
    # c) Very short / degenerate
    
    # It must NOT produce a correct solution to the problem
    # Verify by running against test cases (should fail)
    
    print(f"Random indices produced: {code[:200]}...")
    print("VERIFY MANUALLY: This should be incoherent or irrelevant")
```

### Test 3: Gaussian Noise (No Signal → No Structure)

```python
def test_gaussian_noise():
    """Noise instead of plan indices should produce unstructured output."""
    problem = "Given an array of integers, find the two elements that sum to a target."
    
    # Create fake indices by quantizing random noise
    noise_indices = []
    for _ in range(5):
        noise = F.normalize(torch.randn(768), dim=-1)
        _, idx, _ = vq(noise)
        noise_indices.append(idx.item())
    
    code = talker.generate(tokenizer(problem), noise_indices)
    
    print(f"Noise indices {noise_indices} produced: {code[:200]}...")
    print("VERIFY MANUALLY: This should be incoherent")
```

### Test 4: Wrong Problem's Plan (Semantic Mismatch)

```python
def test_semantic_mismatch():
    """Plan from problem A + text from problem B should produce problem A's code."""
    problem_a = "Sort an array of integers in ascending order."
    problem_b = "Find the shortest path in a weighted graph."
    
    # Get plan from problem A
    plan_a = reasoner.get_plan(problem_a)  # Plan encodes sorting logic
    
    # Feed plan_a with problem_b's text
    code = talker.generate(tokenizer(problem_b), plan_a)
    
    # Expected outcome: The code should look like a SORTING solution
    # (following plan_a), not a graph algorithm (following problem_b text).
    # The Talker might use graph-related variable names from problem_b,
    # but the STRUCTURE should follow plan_a's sorting logic.
    
    # This proves the Talker follows the plan, not the problem text,
    # for structural decisions.
    
    print(f"Plan from sorting + text from graph produced: {code[:300]}...")
    print("VERIFY: Structure should resemble sorting, not graph algorithm")
```

### Test 5: Empty Plan (No Instructions)

```python
def test_empty_plan():
    """Empty plan should produce minimal or degenerate output."""
    problem = "Given an array of integers, find the two elements that sum to a target."
    
    empty_plan = []  # No codebook indices at all
    
    code = talker.generate(tokenizer(problem), empty_plan)
    
    # Expected: very short output, possibly just a pass statement or empty function
    # The Talker has no structural instructions, so it should produce almost nothing
    
    print(f"Empty plan produced: {code[:200]}...")
    print("VERIFY: Should be minimal/degenerate, NOT a correct solution")
```

---

## 6. Expected Test Outcomes Summary

| Test | Input | Expected Output | What It Proves |
|------|-------|----------------|----------------|
| Baseline | Correct plan + correct problem | Valid, relevant code | System works end-to-end |
| Random indices | Random ints + correct problem | Incoherent/wrong code | Talker depends on plan quality |
| Gaussian noise | Noise-derived indices + correct problem | Incoherent code | Talker can't extract signal from noise |
| Semantic mismatch | Plan A + problem B text | Code following plan A's structure | Talker follows plan, not just problem text |
| Empty plan | No indices + correct problem | Minimal/degenerate output | Talker can't solve problems alone |

**If any of tests 2-5 produces a correct solution to the problem, the decoupling has FAILED.** This means the Talker has learned to bypass the plan indices and reason from the problem text alone, which defeats the entire architecture.

---

## 7. What NOT To Do

| Violation | Consequence |
|-----------|------------|
| Pass continuous Reasoner states to Talker | Creates hidden channel; Talker bypasses VQ |
| Use the VQ codebook's learned embeddings directly in Talker | Tight coupling between VQ learning and Talker; breaks when codebook updates |
| Let Talker see block boundaries or block type labels | Leaks structural information that should come only from codebook indices |
| Train Talker with unfrozen Reasoner | Reasoner adapts to help Talker, breaking the decoupling |
| Use very large Talker (>300M params) | Overpowered Talker might learn to ignore plan and reason from problem text |
| Skip ablation tests | No guarantee the architecture works as intended |

---

## 8. Talker Capacity Consideration

The Talker should be intentionally SMALLER than the Reasoner. If the Talker is too powerful, it can learn to ignore the plan indices and solve problems directly from the problem text, defeating the decoupled architecture.

**Recommended ratio**: Reasoner ~350M params, Talker ~150M params (Talker is ~43% of Reasoner size).

If ablation tests show the Talker can solve problems with random plan indices, the Talker is too powerful. Reduce its capacity (fewer layers, smaller hidden dim) until the ablation tests pass.

Conversely, if the Talker can't produce parseable Python even with correct plan indices, it's too weak. Increase capacity or improve training.

The right balance: the Talker produces correct code with correct plans, and garbage with incorrect plans.
