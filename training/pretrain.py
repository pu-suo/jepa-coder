"""
training/pretrain.py — Phase 1: Short Pretraining Run (10K-15K steps)

Specification: docs/jepa_coder_architecture_v2.md §4 Phase 1

Architecture state:
    tied embeddings   ENABLED   (reasoner.attach_lm_head())
    L2 normalization  DISABLED  (hybrid_norm.l2_enabled=False, the default)
    LM head           ENABLED   (temporary; removed before SST begins)

Objective: standard next-token prediction (cross-entropy)

Data:
    APPS + TACO + OpenCodeReasoning (local JSONL, pre-tokenized by data pipeline)
    Interleaves problem text (English) and solution code (Python) to establish
    angular alignment in the embedding space before SST.

Hyperparameters (§4 Phase 1):
    Optimizer    AdamW
    LR           3e-4, cosine decay, 1000-step linear warmup
    Eff. batch   128  (32 seqs × 4 accumulation steps)
    Weight decay 0.1
    Precision    BF16
    Steps        15K (stop on loss convergence; monitor via W&B)
    Output       checkpoints/pretrain/pretrained_reasoner.pt

Logging:
    Weights & Biases (project: jepa-coder-pretrain)
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import math
import os
import random
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
import wandb
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

    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 15_000

    # Batching: eff. batch = seqs_per_accum * accum_steps = 128
    seqs_per_accum: int = 32
    accum_steps: int = 4

    # Data — reads from local JSONL produced by data prep pipeline
    data_dir: str = "/workspace/jepa-coder-data/data"
    data_seed: int = 42

    # Checkpointing
    checkpoint_dir: str = "checkpoints/pretrain"
    checkpoint_every: int = 1000
    save_total_limit: int = 2   # keep only the N most recent numbered ckpts

    # Logging — W&B only
    log_every: int = 10
    wandb_project: str = "jepa-coder-pretrain"
    wandb_run_name: Optional[str] = None

    # Runtime
    device: str = "cuda"
    seed: int = 42


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def build_pretrain_data_iterator(
    config: PretrainConfig,
    tokenizer,
) -> Iterator[torch.Tensor]:
    """
    Build a token-chunk iterator over APPS + TACO + OpenCodeReasoning data.

    Reads from config.data_dir/extracted_solutions.jsonl (produced by data prep).
    Interleaves problem text and solution code so the embeddings learn
    angular alignment across English and Python.

    Yields LongTensor of shape (context_length,).
    """
    jsonl_path = os.path.join(config.data_dir, "extracted_solutions.jsonl")
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(
            f"Data file not found: {jsonl_path}\n"
            "Run scripts/run_data_prep.sh first."
        )

    eos_id: int = tokenizer.eos_token_id
    rng = random.Random(config.data_seed)

    # Shuffle buffer: holds up to 10K tokenized records and yields them in
    # random order. Much cheaper than shuffling the whole file.
    SHUF_BUF = 10_000

    def _record_stream():
        epoch = 0
        while True:
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    yield json.loads(line)
            epoch += 1
            print(f"[data] completed epoch {epoch}, restarting stream")

    records_iter = _record_stream()
    shuffle_buf: list[dict] = []
    token_buf: list[int] = []

    while True:
        # Refill shuffle buffer
        while len(shuffle_buf) < SHUF_BUF:
            try:
                shuffle_buf.append(next(records_iter))
            except StopIteration:
                break
        if not shuffle_buf:
            return
        rng.shuffle(shuffle_buf)

        # Drain half the buffer, tokenize, pack into context_length chunks
        drain = shuffle_buf[: SHUF_BUF // 2]
        shuffle_buf = shuffle_buf[SHUF_BUF // 2 :]

        for record in drain:
            problem  = record.get("problem", "")
            solution = record.get("solution", "")
            for text in (problem, solution):
                if not text:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                ids.append(eos_id)
                token_buf.extend(ids)
                while len(token_buf) >= config.context_length:
                    yield torch.tensor(token_buf[: config.context_length], dtype=torch.long)
                    token_buf = token_buf[config.context_length :]


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
    save_total_limit: int = 2,
    tag: str = "",
) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "step": step,
        "total_tokens": total_tokens,
        "model_state_dict": reasoner.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    name = f"step_{step:07d}.pt" if not tag else f"{tag}.pt"
    torch.save(payload, os.path.join(ckpt_dir, name))
    torch.save(payload, os.path.join(ckpt_dir, "latest.pt"))

    # Rotation: keep only the most recent `save_total_limit` numbered ckpts.
    # Tagged checkpoints (e.g. "pretrained_reasoner.pt") and "latest.pt" are
    # never rotated — they're the artifacts you actually want to keep.
    if not tag and save_total_limit > 0:
        numbered = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
        excess = len(numbered) - save_total_limit
        for old in numbered[:max(0, excess)]:
            try:
                os.remove(old)
                print(f"[ckpt] rotated out {os.path.basename(old)}")
            except OSError:
                pass


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
    data_iter = build_pretrain_data_iterator(config, tokenizer)

    # ── W&B ─────────────────────────────────────────────────────────────────
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=dataclasses.asdict(config),
    )

    # ── Training ────────────────────────────────────────────────────────────
    seqs_per_step = config.seqs_per_accum * config.accum_steps  # 128

    reasoner.train()
    optimizer.zero_grad()

    print("Loading data from local JSONL...")
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
            wandb.log({
                "pretrain/loss": avg_loss,
                "pretrain/lr": lr,
                "pretrain/grad_norm": grad_norm_val,
                "pretrain/tokens_seen": total_tokens,
                "pretrain/step": completed,
            }, step=completed)
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
                save_total_limit=config.save_total_limit,
            )

    # ── Final save ──────────────────────────────────────────────────────────
    # Numbered milestone + clean model file for the SST phase.
    # save_total_limit=0 disables rotation so this final artifact is never deleted.
    save_checkpoint(
        config.checkpoint_dir,
        config.max_steps,
        total_tokens,
        reasoner,
        optimizer,
        save_total_limit=0,
        tag="pretrained_reasoner",
    )
    pbar.close()
    wandb.finish()
    final_path = os.path.join(config.checkpoint_dir, "pretrained_reasoner.pt")
    print(f"Pretraining complete. Model saved to {final_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> PretrainConfig:
    parser = argparse.ArgumentParser(description="JEPA-Coder Phase 1 Short Pretraining Run")
    parser.add_argument("--checkpoint_dir", default="checkpoints/pretrain",
                        help="Directory for checkpoints and final model")
    parser.add_argument("--data_dir", default="/workspace/jepa-coder-data/data",
                        help="Directory containing extracted_solutions.jsonl")
    parser.add_argument("--max_steps", type=int, default=15000,
                        help="Total optimizer steps (10K-15K; stop on loss convergence)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--wandb_project", default="jepa-coder-pretrain")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Keep only the N most recent numbered checkpoints (0 = keep all)")
    args = parser.parse_args()

    return PretrainConfig(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        max_steps=args.max_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        save_total_limit=args.save_total_limit,
    )


if __name__ == "__main__":
    pretrain(_parse_args())
