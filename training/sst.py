"""
training/sst.py — Self-Supervised Training (SST) loop.

Specification:
  docs/contract_1_sst_loop.md              — step-by-step tensor shapes and constraints
  docs/jepa_coder_architecture_v2.md §4.2  — Phase 2 pseudocode

Critical invariants enforced here (contract_1_sst_loop.md §4):
  - Continuous state r loops back, NOT quantized state
  - VQ applied ONLY for loss computation and index storage — NOT in the loop
  - EMA update happens AFTER optimizer.step()
  - No gradients flow through ema_encoder (torch.no_grad enforced in EMAEncoder)
  - All vectors are unit-norm at every step
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.ema_encoder import EMAEncoder
from models.reasoner import Reasoner
from models.vq import VectorQuantizer


# ---------------------------------------------------------------------------
# Core training step
# ---------------------------------------------------------------------------

def sst_train_step(
    reasoner: Reasoner,
    ema_encoder: EMAEncoder,
    vq: VectorQuantizer,
    problem_tokens: torch.Tensor,
    blocks: List[Dict[str, str]],
    tokenizer: Any,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Process one training example and return (total_loss, stored_indices).

    Implements contract_1_sst_loop.md §3.1–§3.5 and the pseudocode from
    jepa_coder_architecture_v2.md §4 Phase 2 exactly.

    Args:
        reasoner:       Reasoner in SST mode (hybrid_norm.l2_enabled=True, lm_head=None)
        ema_encoder:    EMAEncoder — no-grad embedding; weights shadow reasoner.embedding
        vq:             VectorQuantizer — for loss computation and index storage only
        problem_tokens: LongTensor of shape (L_prob,) — pre-tokenized problem statement
        blocks:         list of dicts {'type': str, 'code': str};
                        last block must be {'type': 'STOP', 'code': '<STOP>'}
        tokenizer:      HuggingFace tokenizer, called on block['code'] per block

    Returns:
        total_loss:     scalar tensor with gradients attached; caller calls .backward()
        stored_indices: list[int] of VQ codebook indices, one per block (including STOP)

    What NOT to do (contract_1_sst_loop.md §4):
        - Do NOT feed quantized back into the loop — h = r (continuous), not quantized
        - Do NOT skip the F.normalize re-normalization after mean pooling
        - Do NOT let gradients flow through the EMA target path
        - Do NOT pass (d,) to transformer_blocks — always unsqueeze to (1, d) first
    """
    assert problem_tokens.ndim == 1, \
        f"problem_tokens must be 1-D, got shape {problem_tokens.shape}"
    assert len(blocks) >= 1, "blocks must be non-empty (at least a STOP block)"
    assert blocks[-1]['type'] == 'STOP', \
        f"Last block must have type 'STOP', got '{blocks[-1]['type']}'"

    device = problem_tokens.device

    # ------------------------------------------------------------------
    # §3.1 Problem encoding — runs once per example, gradients flow
    #
    # embed + pos_embed → transformer_blocks → hybrid_norm → mean pool → L2
    # contract: h has shape (d,) and ||h|| = 1
    # ------------------------------------------------------------------
    h = reasoner.encode_problem(problem_tokens)
    # h: (768,), ||h|| = 1

    # Accumulate loss as a Python float first; adding tensors to 0.0 yields a
    # tensor with the correct grad_fn without creating an unnecessary leaf node.
    total_loss: torch.Tensor = torch.tensor(0.0, device=device)
    stored_indices: List[int] = []

    # ------------------------------------------------------------------
    # §3.2–§3.5 Reasoning loop — one iteration per block
    # ------------------------------------------------------------------
    for block in blocks:

        # §3.2 One reasoning step (gradients flow through reasoner)
        #   h.unsqueeze(0) → transformer_blocks → squeeze → hybrid_norm → L2
        r = reasoner.step(h)
        # r: (768,), ||r|| = 1

        # §3.3 Target generation — NO GRADIENTS (enforced inside encode_block)
        block_tokens = tokenizer(
            block['code'],
            return_tensors='pt',
            add_special_tokens=False,
        ).input_ids.squeeze(0).to(device)
        # block_tokens: (L_block,)

        t = ema_encoder.encode_block(block_tokens)
        # t: (768,), ||t|| = 1, no grad_fn

        # §3.4 SST loss: scaled cosine distance
        # Both r and t are unit norm → torch.dot == cosine_similarity
        cos_sim = torch.dot(r, t)               # scalar in [-1, 1]
        sst_loss = 4.0 * (1.0 - cos_sim)        # scalar in [0, 8]

        # VQ: applied for loss and index storage ONLY — NOT looped back
        quantized_st, idx, vq_loss = vq(r)
        stored_indices.append(idx.item())

        # VQ EMA codebook update — outside the gradient graph, after each vector
        with torch.no_grad():
            vq.update_codebook_ema(r.detach(), idx)

        total_loss = total_loss + sst_loss + vq_loss

        # §3.5 Loop back the CONTINUOUS state, not quantized
        h = r                                    # (768,), ||h|| = 1

    return total_loss, stored_indices


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SSTConfig:
    # Paths
    pretrained_checkpoint: str          # path to pretrained_reasoner.pt (from Phase 1 short pretraining)
    output_dir: str                     # directory for SST checkpoints (checkpoints/sst/)

    # Model architecture — must match the pretrained checkpoint exactly
    vocab_size: int                     # StarCoder2 BPE vocab (~49K tokens)
    dim: int = 768
    n_layers: int = 16
    n_heads: int = 12
    ffn_dim: int = 3072
    max_seq_len: int = 1024

    # VQ
    codebook_size: int = 512
    commitment_cost: float = 0.25

    # Optimizer (arch v2 §4.2 hyperparameters)
    lr: float = 1e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0

    # LR schedule: linear warmup → cosine decay
    warmup_optimizer_steps: int = 1000  # warmup over optimizer steps, not examples

    # Training length
    max_examples: int = 100_000         # total training examples

    # Gradient accumulation (arch v2: effective batch = 16)
    accumulation_steps: int = 16

    # EMA
    ema_decay: float = 0.98

    # Checkpointing / logging
    checkpoint_every: int = 2000        # checkpoint every N examples
    log_every: int = 100                # wandb log every N examples
    utilization_check_every: int = 1000 # VQ dead-entry check every N examples

    # Runtime
    device: str = 'cuda'
    wandb_project: Optional[str] = 'jepa-coder-sst'
    wandb_run_name: Optional[str] = None
    seed: int = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_lr_lambda(warmup_steps: int, total_steps: int):
    """
    Returns a LambdaLR-compatible callable: linear warmup → cosine decay.
    Both warmup_steps and total_steps are in optimizer step units.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return lr_lambda


def _save_checkpoint(
    output_dir: str,
    tag: str,
    reasoner: Reasoner,
    vq: VectorQuantizer,
    ema_encoder: EMAEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    example_count: int,
) -> None:
    """Save model weights and training state. Matches arch v2 §4.2 output names."""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        reasoner.state_dict(),
        os.path.join(output_dir, f'sst_reasoner_{tag}.pt'),
    )
    torch.save(
        vq.state_dict(),
        os.path.join(output_dir, f'vq_codebook_{tag}.pt'),
    )
    torch.save(
        {
            'ema_encoder': ema_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'example_count': example_count,
        },
        os.path.join(output_dir, f'sst_train_state_{tag}.pt'),
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_sst_training(config: SSTConfig, dataset: Iterator) -> None:
    """
    Full SST training loop.

    Steps performed (arch v2 §4.2):
      1. Load pretrained Reasoner checkpoint
      2. Remove LM head, enable L2 normalization
      3. Initialize EMAEncoder from Reasoner's embedding weights
      4. Initialize VQ
      5. AdamW over Reasoner parameters only (EMA encoder + VQ never receive grads)
      6. Process one example at a time with gradient accumulation (config.accumulation_steps)
      7. EMA update after every optimizer.step()
      8. Checkpoint every config.checkpoint_every examples
      9. Log SST loss, VQ loss, and codebook utilization to wandb

    Args:
        config:  SSTConfig dataclass
        dataset: iterable of (problem_text: str, blocks: list[dict]) where each
                 block is {'type': str, 'code': str} and the last block is STOP
    """
    import wandb
    from transformers import AutoTokenizer

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    os.makedirs(config.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Tokenizer (StarCoder2 BPE, ~49K vocab)
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        'bigcode/starcoder2-3b',
        trust_remote_code=True,
    )

    # ------------------------------------------------------------------
    # Load pretrained Reasoner
    #
    # The pretrained checkpoint was saved with attach_lm_head() active, so its
    # state dict contains 'lm_head.weight'. Since lm_head.weight was tied to
    # embedding.weight (same tensor), loading with strict=False is correct:
    # embedding.weight receives the trained values; lm_head.weight is ignored.
    # ------------------------------------------------------------------
    reasoner = Reasoner(
        vocab_size=config.vocab_size,
        dim=config.dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        ffn_dim=config.ffn_dim,
        max_seq_len=config.max_seq_len,
    )

    raw = torch.load(config.pretrained_checkpoint, map_location='cpu')
    # Support both bare state_dicts and wrapped {'state_dict': ...} checkpoints
    state_dict = raw.get('model_state_dict', raw.get('state_dict', raw)) if isinstance(raw, dict) else raw
    missing, unexpected = reasoner.load_state_dict(state_dict, strict=False)
    # Only 'lm_head.weight' is expected in unexpected (tied weight, not needed)
    non_lmhead_unexpected = [k for k in unexpected if not k.startswith('lm_head')]
    assert not non_lmhead_unexpected, \
        f"Unexpected keys in checkpoint (non-lm_head): {non_lmhead_unexpected}"
    assert not missing, f"Missing keys in checkpoint: {missing}"

    # Phase transition: remove LM head, enable L2 normalization (arch v2 §4.2)
    reasoner.detach_lm_head()
    reasoner.hybrid_norm.l2_enabled = True

    reasoner = reasoner.to(device)
    reasoner.train()

    # ------------------------------------------------------------------
    # EMA encoder — initialized from Reasoner's embedding weights (arch v2 §2.3)
    # ------------------------------------------------------------------
    ema_encoder = EMAEncoder.from_embedding(reasoner.embedding)
    ema_encoder = ema_encoder.to(device)
    ema_encoder.eval()       # never switches to train(); no backprop ever

    # ------------------------------------------------------------------
    # VQ (arch v2 §2.2)
    # ------------------------------------------------------------------
    vq = VectorQuantizer(
        codebook_size=config.codebook_size,
        dim=config.dim,
        commitment_cost=config.commitment_cost,
    )
    vq = vq.to(device)

    # ------------------------------------------------------------------
    # Optimizer: AdamW over Reasoner parameters ONLY (contract §2)
    # EMA encoder and VQ codebook are never in the optimizer.
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        reasoner.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    total_optimizer_steps = config.max_examples // config.accumulation_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_cosine_lr_lambda(config.warmup_optimizer_steps, total_optimizer_steps),
    )

    # ------------------------------------------------------------------
    # Wandb
    # ------------------------------------------------------------------
    if config.wandb_project:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=dataclasses.asdict(config),
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    example_count = 0
    optimizer_step_count = 0
    optimizer.zero_grad()

    # Accumulators for windowed logging
    window_total_loss = 0.0
    window_sst_loss = 0.0
    window_vq_loss = 0.0

    for problem_text, blocks in dataset:
        if example_count >= config.max_examples:
            break

        # Tokenize problem statement (truncated to max_seq_len)
        problem_tokens = tokenizer(
            problem_text,
            return_tensors='pt',
            add_special_tokens=True,
            truncation=True,
            max_length=config.max_seq_len,
        ).input_ids.squeeze(0).to(device)

        # Forward pass — one example at a time (arch v2: no cross-example batching)
        total_loss, stored_indices = sst_train_step(
            reasoner=reasoner,
            ema_encoder=ema_encoder,
            vq=vq,
            problem_tokens=problem_tokens,
            blocks=blocks,
            tokenizer=tokenizer,
        )

        # Accumulate for logging — approximate SST/VQ split via VQ commitment cost
        # The exact per-block breakdown is not tracked here to avoid overhead;
        # total_loss is the authoritative metric.
        total_val = total_loss.item()
        window_total_loss += total_val

        # Backprop (accumulate gradients across examples)
        total_loss.backward()

        example_count += 1

        # Gradient accumulation: optimizer step every accumulation_steps examples
        # contract_1_sst_loop.md §3.6: clip → step → zero → EMA update
        if example_count % config.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(reasoner.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_step_count += 1

            # EMA update MUST happen after optimizer.step() — contract §3.6
            # Violation (updating before step) would create stale targets.
            ema_encoder.update(reasoner.embedding, decay=config.ema_decay)

        # VQ utilization check — reset dead entries if utilization < 30%
        # (arch v2 §2.2: reset unused entries to perturbed copies of used ones)
        if example_count % config.utilization_check_every == 0:
            util = vq.utilization()
            if util < 0.30:
                vq.reset_unused_entries()

        # Wandb logging
        if example_count % config.log_every == 0 and config.wandb_project:
            avg_total = window_total_loss / config.log_every
            util = vq.utilization()
            wandb.log(
                {
                    'sst/total_loss': avg_total,
                    'sst/codebook_utilization': util,
                    'sst/lr': scheduler.get_last_lr()[0],
                    'sst/example_count': example_count,
                    'sst/optimizer_step': optimizer_step_count,
                },
                step=example_count,
            )
            window_total_loss = 0.0

        # Checkpoint every N examples
        if example_count % config.checkpoint_every == 0:
            _save_checkpoint(
                config.output_dir,
                f'{example_count:08d}',
                reasoner, vq, ema_encoder,
                optimizer, scheduler, example_count,
            )

    # Final checkpoint (arch v2 §4.2 output: sst_reasoner.pt, vq_codebook.pt)
    _save_checkpoint(
        config.output_dir, 'final',
        reasoner, vq, ema_encoder,
        optimizer, scheduler, example_count,
    )

    if config.wandb_project:
        wandb.finish()
