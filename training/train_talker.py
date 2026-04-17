"""
training/train_talker.py — Phase 3: Talker Training.

Specification: docs/contract_3_talker_interface.md §4

Trains the Talker encoder-decoder on pre-generated (problem, plan, code)
triples produced by training/prepare_talker_data.py.

Critical invariants (contract_3 §4, §7):
  - Reasoner and VQ are COMPLETELY FROZEN — no parameters loaded, no gradients
  - Only the Talker's parameters receive gradients
  - Training uses teacher forcing on the decoder with cross-entropy loss
  - PAD tokens are excluded from the loss via ignore_index

Usage (Vast.ai):
    python -m training.train_talker \\
        --dataset_path /workspace/jepa-coder-data/data/talker_dataset \\
        --output_dir   /workspace/jepa-coder-data/checkpoints/talker \\
        --wandb_project jepa-coder-talker
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.talker import Talker


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TalkerDataset(Dataset):
    """
    Wraps the Arrow dataset produced by prepare_talker_data.py.

    Each row contains:
        problem_token_ids:  list[int]  — tokenized problem text
        plan_indices:       list[int]  — VQ codebook indices (no STOP)
        solution_token_ids: list[int]  — tokenized solution code
    """

    def __init__(
        self,
        dataset_path: str,
        max_prob_len: int = 512,
        max_code_len: int = 1024,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> None:
        from datasets import load_from_disk, load_dataset

        try:
            self.ds = load_from_disk(dataset_path)
        except Exception:
            self.ds = load_dataset("arrow", data_dir=dataset_path, split="train")

        self.max_prob_len = max_prob_len
        self.max_code_len = max_code_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[idx]

        # Truncate problem tokens
        prob_ids = list(row["problem_token_ids"])[: self.max_prob_len]
        plan_ids = list(row["plan_indices"])

        # Target code: prepend BOS, append EOS, truncate
        sol_ids = list(row["solution_token_ids"])
        target = [self.bos_token_id] + sol_ids[: self.max_code_len - 2] + [self.eos_token_id]

        return {
            "problem_token_ids": prob_ids,
            "plan_indices": plan_ids,
            "target_code": target,
        }


PLAN_PAD_ID = 512  # Must match Talker.plan_pad_id — avoids contaminating VQ index 0


def collate_fn(batch: List[dict], pad_token_id: int = 0) -> dict:
    """
    Pad variable-length sequences to the longest in the batch.

    Returns dict with:
        problem_tokens:       (B, L_prob)  LongTensor
        plan_indices:         (B, M)       LongTensor
        target_code:          (B, L_code)  LongTensor
        src_key_padding_mask: (B, L_prob + M) BoolTensor — True at pad positions
        tgt_key_padding_mask: (B, L_code - 1) BoolTensor — True at pad positions
    """

    def _pad(seqs: List[List[int]], pad_val: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(s) for s in seqs)
        padded = []
        mask = []
        for s in seqs:
            pad_len = max_len - len(s)
            padded.append(s + [pad_val] * pad_len)
            mask.append([False] * len(s) + [True] * pad_len)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    prob_ids = [item["problem_token_ids"] for item in batch]
    plan_ids = [item["plan_indices"] for item in batch]
    target_ids = [item["target_code"] for item in batch]

    prob_padded, prob_mask = _pad(prob_ids, pad_token_id)
    plan_padded, plan_mask = _pad(plan_ids, PLAN_PAD_ID)
    tgt_padded, tgt_mask = _pad(target_ids, pad_token_id)

    # src_key_padding_mask covers [problem; plan] concatenated
    src_mask = torch.cat([prob_mask, plan_mask], dim=1)

    # tgt_key_padding_mask covers decoder input (target[:, :-1])
    tgt_input_mask = tgt_mask[:, :-1]

    return {
        "problem_tokens": prob_padded,
        "plan_indices": plan_padded,
        "target_code": tgt_padded,
        "src_key_padding_mask": src_mask,
        "tgt_key_padding_mask": tgt_input_mask,
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TalkerTrainConfig:
    # Paths
    dataset_path: str
    output_dir: str

    # Model architecture (contract_3 §3.1, §8: ~150M params)
    vocab_size: int = 49152
    dim: int = 768
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    n_heads: int = 12
    ffn_dim: int = 3072
    max_seq_len: int = 1024

    # Data limits
    max_prob_len: int = 512
    max_code_len: int = 1024

    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0

    # LR schedule
    warmup_steps: int = 500

    # Training
    batch_size: int = 16
    max_epochs: int = 20
    max_steps: Optional[int] = None

    # Checkpointing / logging
    checkpoint_every: int = 2000
    log_every: int = 50
    eval_every: int = 1000
    save_total_limit: int = 3

    # Runtime
    device: str = "cuda"
    num_workers: int = 4
    wandb_project: Optional[str] = "jepa-coder-talker"
    wandb_run_name: Optional[str] = None
    seed: int = 42

    # Resume
    resume_from: Optional[str] = None


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _cosine_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return lr_lambda


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _save_checkpoint(
    output_dir: str,
    tag: str,
    talker: Talker,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    epoch: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        talker.state_dict(),
        os.path.join(output_dir, f"talker_{tag}.pt"),
    )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
        },
        os.path.join(output_dir, f"talker_train_state_{tag}.pt"),
    )


def _rotate_checkpoints(output_dir: str, save_total_limit: int) -> None:
    """Keep only the most recent numbered checkpoints."""
    import glob as _glob

    if save_total_limit <= 0:
        return
    numbered = sorted(_glob.glob(os.path.join(output_dir, "talker_step_*.pt")))
    excess = len(numbered) - save_total_limit
    for old in numbered[: max(0, excess)]:
        try:
            os.remove(old)
            # Also remove matching train state
            state_path = old.replace("talker_step_", "talker_train_state_step_")
            if os.path.exists(state_path):
                os.remove(state_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_talker_training(config: TalkerTrainConfig) -> None:
    """
    Full Talker training loop.

    Contract_3 §4.2:
      - Encoder processes [problem; plan] embeddings
      - Decoder with teacher forcing on target code
      - Cross-entropy loss with PAD ignore
      - NO Reasoner or VQ parameters involved at all
    """
    import wandb

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    os.makedirs(config.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Tokenizer — needed only for special token IDs and vocab_size
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "bigcode/starcoder2-3b",
        trust_remote_code=True,
    )
    vocab_size = tokenizer.vocab_size
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # ------------------------------------------------------------------
    # Dataset and DataLoader
    # ------------------------------------------------------------------
    dataset = TalkerDataset(
        dataset_path=config.dataset_path,
        max_prob_len=config.max_prob_len,
        max_code_len=config.max_code_len,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )
    print(f"Talker dataset: {len(dataset):,} examples")

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=pad_token_id),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Model — Talker only, NO Reasoner or VQ
    # ------------------------------------------------------------------
    talker = Talker(
        vocab_size=vocab_size,
        dim=config.dim,
        n_encoder_layers=config.n_encoder_layers,
        n_decoder_layers=config.n_decoder_layers,
        n_heads=config.n_heads,
        ffn_dim=config.ffn_dim,
        max_seq_len=config.max_seq_len,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    n_params = sum(p.numel() for p in talker.parameters())
    print(f"Talker: {n_params:,} params (~{n_params / 1e6:.1f}M)")

    talker = talker.to(device)
    talker.train()

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_step = 0
    start_epoch = 0
    if config.resume_from and os.path.isfile(config.resume_from):
        print(f"Resuming from {config.resume_from}")
        ckpt = torch.load(config.resume_from, map_location="cpu")
        talker.load_state_dict(ckpt)

        # Try loading training state
        state_path = config.resume_from.replace("talker_", "talker_train_state_")
        if os.path.isfile(state_path):
            state = torch.load(state_path, map_location="cpu")
            start_step = state.get("step", 0)
            start_epoch = state.get("epoch", 0)
            print(f"  Resumed at step {start_step}, epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Optimizer — only Talker parameters
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        talker.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    total_steps = config.max_steps or (config.max_epochs * len(dataloader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_cosine_lr_lambda(config.warmup_steps, total_steps),
    )

    # Restore optimizer/scheduler state if resuming
    if config.resume_from:
        state_path = config.resume_from.replace("talker_", "talker_train_state_")
        if os.path.isfile(state_path):
            state = torch.load(state_path, map_location="cpu")
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                scheduler.load_state_dict(state["scheduler"])

    # ------------------------------------------------------------------
    # AMP scaler for CUDA
    # ------------------------------------------------------------------
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ------------------------------------------------------------------
    # W&B
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
    global_step = start_step
    window_loss = 0.0
    window_count = 0

    for epoch in range(start_epoch, config.max_epochs):
        talker.train()

        for batch in dataloader:
            if config.max_steps and global_step >= config.max_steps:
                break

            problem_tokens = batch["problem_tokens"].to(device)
            plan_indices = batch["plan_indices"].to(device)
            target_code = batch["target_code"].to(device)
            src_key_padding_mask = batch["src_key_padding_mask"].to(device)
            tgt_key_padding_mask = batch["tgt_key_padding_mask"].to(device)

            # Forward — contract_3 §4.2
            optimizer.zero_grad()

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                logits = talker(
                    problem_tokens,
                    plan_indices,
                    target_code,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                )
                # logits: (B, L_code - 1, vocab_size)

                # Labels: target_code shifted by 1 (teacher forcing)
                target_labels = target_code[:, 1:]  # (B, L_code - 1)

                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    target_labels.reshape(-1),
                    ignore_index=pad_token_id,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(talker.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            window_loss += loss.item()
            window_count += 1

            # Logging
            if global_step % config.log_every == 0 and config.wandb_project:
                avg_loss = window_loss / window_count
                wandb.log(
                    {
                        "talker/loss": avg_loss,
                        "talker/lr": scheduler.get_last_lr()[0],
                        "talker/epoch": epoch,
                        "talker/step": global_step,
                    },
                    step=global_step,
                )
                print(
                    f"step {global_step:>7,} | epoch {epoch} | "
                    f"loss {avg_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
                )
                window_loss = 0.0
                window_count = 0

            # Checkpoint
            if global_step % config.checkpoint_every == 0:
                tag = f"step_{global_step:07d}"
                _save_checkpoint(
                    config.output_dir, tag,
                    talker, optimizer, scheduler,
                    global_step, epoch,
                )
                _rotate_checkpoints(config.output_dir, config.save_total_limit)

        if config.max_steps and global_step >= config.max_steps:
            break

    # Final checkpoint
    _save_checkpoint(
        config.output_dir, "final",
        talker, optimizer, scheduler,
        global_step, epoch,
    )

    if config.wandb_project:
        wandb.finish()

    print(f"Talker training complete. {global_step:,} steps, {epoch + 1} epochs.")
    print(f"Checkpoint: {os.path.join(config.output_dir, 'talker_final.pt')}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JEPA-Coder Phase 3: Talker Training (contract_3 §4)"
    )

    # Paths
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to talker Arrow dataset (output of prepare_talker_data.py)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints/talker",
        help="Directory for Talker checkpoints",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Path to talker checkpoint to resume from (e.g. talker_step_0005000.pt)",
    )

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Logging
    parser.add_argument("--wandb_project", type=str, default="jepa-coder-talker")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--checkpoint_every", type=int, default=2000)

    # Data
    parser.add_argument("--max_prob_len", type=int, default=512)
    parser.add_argument("--max_code_len", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b", trust_remote_code=True)

    config = TalkerTrainConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        vocab_size=_tok.vocab_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        max_prob_len=args.max_prob_len,
        max_code_len=args.max_code_len,
        num_workers=args.num_workers,
    )

    run_talker_training(config)
