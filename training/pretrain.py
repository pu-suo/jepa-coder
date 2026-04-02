"""
training/pretrain.py — Phase 1: Pretraining

Specification: docs/jepa_coder_architecture_v2.md §4 Phase 1

Architecture state:
    tied embeddings   ENABLED   (reasoner.attach_lm_head())
    L2 normalization  DISABLED  (hybrid_norm.l2_enabled=False, the default)
    LM head           ENABLED   (temporary; removed before SST begins)

Objective: standard next-token prediction (cross-entropy)

Data (§5.1):
    bigcode/the-stack-v2-dedup  (Python subset, streaming)
    allenai/c4                  (en, streaming)

Hyperparameters (§4 Phase 1):
    Optimizer    AdamW
    LR           3e-4, cosine decay, 2000-step linear warmup
    Eff. batch   128  (32 seqs × 4 accumulation steps)
    Weight decay 0.1
    Precision    BF16
    Steps        100K (configurable)
    Output       checkpoints/pretrain/pretrained_reasoner.pt

Logging:
    CSV file at <checkpoint_dir>/training_logs.csv
    Columns: step, loss, lr, grad_norm, total_tokens
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from models.reasoner import Reasoner


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PretrainConfig:
    # Model (must match architecture spec §2.1)
    dim: int = 768
    n_layers: int = 16
    n_heads: int = 12
    ffn_dim: int = 3072
    context_length: int = 1024

    # Optimizer (§4 Phase 1)
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100_000

    # Batching: eff. batch = seqs_per_accum * accum_steps = 32 * 4 = 128
    seqs_per_accum: int = 32
    accum_steps: int = 4

    # Data
    stack_language_filter: str = "Python"   # field value to match in the Stack
    c4_mix_ratio: float = 0.2              # fraction of samples drawn from C4
    data_seed: int = 42

    # Checkpointing
    checkpoint_dir: str = "checkpoints/pretrain"
    checkpoint_every: int = 2000

    # Logging
    log_every: int = 10


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def _text_stream(
    stack_ds,
    c4_ds,
    c4_mix_ratio: float,
    seed: int,
) -> Iterator[str]:
    """
    Interleave The Stack and C4 at the given ratio.
    The Stack items expose their code under 'content'; C4 items under 'text'.
    """
    import random
    rng = random.Random(seed)
    stack_iter = iter(stack_ds)
    c4_iter = iter(c4_ds)

    while True:
        if rng.random() < c4_mix_ratio:
            try:
                item = next(c4_iter)
                yield item.get("text", "")
            except StopIteration:
                c4_iter = iter(c4_ds)
        else:
            try:
                item = next(stack_iter)
                # The Stack v2 uses 'content'; fall back to 'text' if absent
                yield item.get("content", item.get("text", ""))
            except StopIteration:
                stack_iter = iter(stack_ds)


def _token_chunks(
    text_iter: Iterator[str],
    tokenizer,
    context_length: int,
) -> Iterator[torch.Tensor]:
    """
    Tokenize a stream of documents, append EOS between them, and pack into
    fixed-length chunks of context_length tokens.  Yields LongTensor (context_length,).
    Leftover tokens that don't fill a full chunk are discarded.
    """
    buffer: list[int] = []
    eos_id: int = tokenizer.eos_token_id

    for text in text_iter:
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(eos_id)
        buffer.extend(ids)

        while len(buffer) >= context_length:
            yield torch.tensor(buffer[:context_length], dtype=torch.long)
            buffer = buffer[context_length:]


def build_data_iterator(config: PretrainConfig, tokenizer) -> Iterator[torch.Tensor]:
    """
    Build a streaming token-chunk iterator over the mixed pretraining corpus.
    Yields LongTensor of shape (context_length,).

    The Stack v2 is filtered to Python files via the 'programming_language'
    (or 'lang') metadata field — exact field name depends on the HF snapshot.
    """
    stack_ds = load_dataset(
        "bigcode/the-stack-v2-dedup",
        split="train",
        streaming=True,
    )
    # Filter to Python — the field name may be 'programming_language' or 'lang'
    stack_ds = stack_ds.filter(
        lambda x: (
            x.get("programming_language", x.get("lang", "")).lower()
            == config.stack_language_filter.lower()
        )
    )

    c4_ds = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
    )

    text_iter = _text_stream(stack_ds, c4_ds, config.c4_mix_ratio, config.data_seed)
    return _token_chunks(text_iter, tokenizer, config.context_length)


# ---------------------------------------------------------------------------
# Learning rate schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def _cosine_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float = 0.0,
) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    ckpt_dir: str,
    step: int,
    total_tokens: int,
    reasoner: Reasoner,
    optimizer: torch.optim.Optimizer,
    tag: str = "",
) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "step": step,
        "total_tokens": total_tokens,
        "model_state_dict": reasoner.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    # Numbered checkpoint for milestone steps
    name = f"step_{step:07d}.pt" if not tag else f"{tag}.pt"
    torch.save(payload, os.path.join(ckpt_dir, name))
    # "latest.pt" is always overwritten so resume can find the most recent state
    torch.save(payload, os.path.join(ckpt_dir, "latest.pt"))


def load_checkpoint(
    ckpt_dir: str,
    reasoner: Reasoner,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    """
    Load the latest checkpoint if one exists.
    Returns (step, total_tokens).  Returns (0, 0) if no checkpoint is found.
    """
    latest = os.path.join(ckpt_dir, "latest.pt")
    if not os.path.isfile(latest):
        return 0, 0

    ckpt = torch.load(latest, map_location=device)
    reasoner.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["step"], ckpt["total_tokens"]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def pretrain(config: PretrainConfig) -> None:
    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # ── Tokenizer ───────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
    vocab_size = len(tokenizer)  # 49152

    # ── Model ───────────────────────────────────────────────────────────────
    reasoner = Reasoner(
        vocab_size=vocab_size,
        dim=config.dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        ffn_dim=config.ffn_dim,
        max_seq_len=config.context_length,
    )
    # Phase 1 state: tied embeddings on, L2 norm off (default)
    reasoner.attach_lm_head()
    assert not reasoner.hybrid_norm.l2_enabled, \
        "L2 normalization must be disabled during pretraining (§4 Phase 1)"
    reasoner = reasoner.to(device)

    # ── Optimizer ───────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        reasoner.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    # ── Resume ──────────────────────────────────────────────────────────────
    start_step, total_tokens = load_checkpoint(
        config.checkpoint_dir, reasoner, optimizer, device
    )
    if start_step > 0:
        print(f"Resumed from step {start_step:,} ({total_tokens:,} tokens seen)")

    # Restore LR to where the schedule left off
    _set_lr(optimizer, _cosine_lr(start_step, config.warmup_steps, config.max_steps, config.lr))

    # ── Data ────────────────────────────────────────────────────────────────
    data_iter = build_data_iterator(config, tokenizer)

    # For streaming datasets there is no efficient seek, so on a resumed run
    # the iterator starts fresh.  The Stack v2 is large enough that duplicate
    # coverage is negligible across a typical 100K-step run.

    # ── CSV logger ──────────────────────────────────────────────────────────
    log_path = os.path.join(config.checkpoint_dir, "training_logs.csv")
    log_file_existed = os.path.isfile(log_path)
    log_fh = open(log_path, "a", newline="")
    log_writer = csv.writer(log_fh)
    if not log_file_existed:
        log_writer.writerow(["step", "loss", "lr", "grad_norm", "total_tokens"])
        log_fh.flush()

    # ── Training ────────────────────────────────────────────────────────────
    seqs_per_step = config.seqs_per_accum * config.accum_steps  # 128

    reasoner.train()
    optimizer.zero_grad()

    print("Warming up data pipeline — first shard download may take 1–2 min...")
    pbar = tqdm(range(start_step, config.max_steps),
                initial=start_step, total=config.max_steps,
                desc="pretrain", unit="step", dynamic_ncols=True)
    for step in pbar:
        lr = _cosine_lr(step, config.warmup_steps, config.max_steps, config.lr)
        _set_lr(optimizer, lr)

        step_loss = 0.0

        # Outer accumulation loop (4 macro-steps)
        for _ in range(config.accum_steps):
            # Inner sequence loop (32 sequences per macro-step)
            for _ in range(config.seqs_per_accum):
                tokens = next(data_iter).to(device)  # (context_length,)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    logits = reasoner.lm_forward(tokens)      # (T, vocab_size)
                    # Next-token prediction: predict token[i+1] from logits[i]
                    loss = F.cross_entropy(
                        logits[:-1].float(),   # cast to fp32 for numerical stability
                        tokens[1:],
                    )

                # Normalize so gradients average over the full effective batch
                (loss / seqs_per_step).backward()
                step_loss += loss.item()

        # Gradient clipping + optimizer step
        grad_norm = torch.nn.utils.clip_grad_norm_(reasoner.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_tokens += seqs_per_step * config.context_length
        completed = step + 1

        # ── Logging ──
        if completed % config.log_every == 0:
            avg_loss = step_loss / seqs_per_step
            grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            log_writer.writerow([completed, f"{avg_loss:.6f}", f"{lr:.8e}",
                                  f"{grad_norm_val:.6f}", total_tokens])
            log_fh.flush()
            print(
                f"step {completed:>7,} | loss {avg_loss:.4f} | lr {lr:.2e} "
                f"| grad_norm {grad_norm_val:.3f} | tokens {total_tokens:,}"
            )
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{lr:.2e}",
                tok=f"{total_tokens / 1e9:.2f}B",
            )

        # ── Checkpoint ──
        if completed % config.checkpoint_every == 0:
            save_checkpoint(
                config.checkpoint_dir,
                completed,
                total_tokens,
                reasoner,
                optimizer,
            )

    # ── Final save ──────────────────────────────────────────────────────────
    # Numbered milestone + clean model file for the SST phase
    save_checkpoint(
        config.checkpoint_dir,
        config.max_steps,
        total_tokens,
        reasoner,
        optimizer,
        tag="pretrained_reasoner",
    )
    pbar.close()
    log_fh.close()
    final_path = os.path.join(config.checkpoint_dir, "pretrained_reasoner.pt")
    print(f"Pretraining complete. Model saved to {final_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> PretrainConfig:
    parser = argparse.ArgumentParser(description="JEPA-Coder Phase 1 Pretraining")
    parser.add_argument("--checkpoint_dir", default="checkpoints/pretrain",
                        help="Directory for checkpoints and final model")
    parser.add_argument("--max_steps", type=int, default=100_000,
                        help="Total optimizer steps (100K–150K per spec)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--c4_mix_ratio", type=float, default=0.2,
                        help="Fraction of samples drawn from C4 (rest from The Stack)")
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    return PretrainConfig(
        checkpoint_dir=args.checkpoint_dir,
        max_steps=args.max_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        c4_mix_ratio=args.c4_mix_ratio,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    pretrain(_parse_args())
