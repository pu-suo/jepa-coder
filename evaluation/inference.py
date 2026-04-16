"""
evaluation/inference.py — End-to-end inference pipeline.

Specification: docs/jepa_coder_architecture_v2.md §6

Pipeline:
  1. Encode problem text → unit-norm latent vector h
  2. Reasoner loop: step → VQ index, loop back CONTINUOUS state (not quantized)
     Stop on VQ index 0 (STOP) or max_steps
  3. Talker generates Python code from (problem_tokens, plan_indices)

CRITICAL INVARIANTS (from architecture doc):
  - Loopback uses the CONTINUOUS state r, NOT the quantized vector
  - VQ is for OUTPUT only (index collection), not in the reasoning loop
  - STOP is VQ index 0
  - Plan indices are in [1, 511] (STOP excluded)

Usage:
    python -m evaluation.inference \
        --talker_checkpoint /workspace/jepa-coder-data/checkpoints/talker/talker_final.pt \
        --sst_checkpoint_dir /workspace/jepa-coder-data/checkpoints/sst \
        --problem "Given an array of integers, find two elements that sum to a target value."

    python -m evaluation.inference \
        --talker_checkpoint /workspace/jepa-coder-data/checkpoints/talker/talker_final.pt \
        --sst_checkpoint_dir /workspace/jepa-coder-data/checkpoints/sst \
        --problems_file problems.txt
"""

from __future__ import annotations

import argparse
import ast
import sys
import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from models.reasoner import Reasoner
from models.talker import Talker
from models.vq import VectorQuantizer
from training.prepare_talker_data import load_frozen_models


# ---------------------------------------------------------------------------
# Core inference function (architecture doc §6)
# ---------------------------------------------------------------------------

STOP_INDEX = 0


@torch.no_grad()
def generate_solution(
    problem_text: str,
    reasoner: Reasoner,
    vq: VectorQuantizer,
    talker: Talker,
    tokenizer,
    max_steps: int = 15,
    max_code_length: int = 1024,
) -> Tuple[str, List[int], int]:
    """
    Full inference pipeline: problem text → Python code.

    Implements the three-stage pipeline from architecture doc §6:
      1. Encode problem → h  (unit-norm latent)
      2. Reasoning loop with STOP detection
      3. Talker code generation

    Args:
        problem_text:    The problem description string.
        reasoner:        Frozen Reasoner in eval mode.
        vq:              Frozen VQ module.
        talker:          Trained Talker in eval mode.
        tokenizer:       StarCoder2 tokenizer.
        max_steps:       Maximum reasoning steps before forced stop (default 15).
        max_code_length: Maximum tokens for Talker generation (default 1024).

    Returns:
        code:         Generated Python code string.
        plan_indices: List of VQ codebook indices (STOP excluded).
        n_steps:      Number of reasoning steps taken (including STOP if hit).
    """
    device = next(talker.parameters()).device

    # ------------------------------------------------------------------
    # Stage 1: Encode problem
    # ------------------------------------------------------------------
    prob_token_ids = tokenizer.encode(problem_text, add_special_tokens=False)
    prob_tokens = torch.tensor(prob_token_ids, dtype=torch.long, device=device)

    # Truncate to Reasoner's max sequence length
    if prob_tokens.shape[0] > reasoner.max_seq_len:
        prob_tokens = prob_tokens[: reasoner.max_seq_len]

    h = reasoner.encode_problem(prob_tokens)  # (d,) unit-norm

    # ------------------------------------------------------------------
    # Stage 2: Reasoning loop
    #   - VQ for output (index collection) only
    #   - Loop back the CONTINUOUS state r, NOT quantized
    #   - STOP = VQ index 0
    # ------------------------------------------------------------------
    plan_indices: List[int] = []
    n_steps = 0

    for _ in range(max_steps):
        r = reasoner.step(h)  # (d,) unit-norm

        _, idx, _ = vq(r)
        n_steps += 1

        if idx.item() == STOP_INDEX:
            break

        plan_indices.append(idx.item())
        h = r  # Continuous loopback — NOT the quantized vector

    # ------------------------------------------------------------------
    # Stage 3: Talker generation
    # ------------------------------------------------------------------
    plan_tensor = torch.tensor(plan_indices, dtype=torch.long, device=device)
    generated_ids = talker.generate(prob_tokens, plan_tensor, max_length=max_code_length)
    code = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return code, plan_indices, n_steps


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def load_models(
    talker_checkpoint: str,
    sst_checkpoint_dir: str,
    sst_checkpoint_tag: str = "final",
    device: Optional[torch.device] = None,
) -> Tuple[Reasoner, VectorQuantizer, Talker, object]:
    """Load all three models and the tokenizer."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        "bigcode/starcoder2-3b", trust_remote_code=True,
    )
    vocab_size = tokenizer.vocab_size

    # Reasoner + VQ (frozen)
    reasoner, vq = load_frozen_models(
        checkpoint_dir=sst_checkpoint_dir,
        tag=sst_checkpoint_tag,
        vocab_size=vocab_size,
        device=device,
    )

    # Talker
    talker = Talker(
        vocab_size=vocab_size,
        dim=768,
        n_encoder_layers=4,
        n_decoder_layers=4,
        n_heads=12,
        ffn_dim=3072,
        max_seq_len=1024,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    ckpt = torch.load(talker_checkpoint, map_location="cpu", weights_only=True)
    talker.load_state_dict(ckpt)
    talker.to(device).eval()

    return reasoner, vq, talker, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JEPA-Coder inference (architecture doc §6).",
    )
    parser.add_argument(
        "--talker_checkpoint", type=str, required=True,
        help="Path to talker_final.pt",
    )
    parser.add_argument(
        "--sst_checkpoint_dir", type=str, required=True,
        help="Directory with sst_reasoner_*.pt and vq_codebook_*.pt",
    )
    parser.add_argument(
        "--sst_checkpoint_tag", type=str, default="final",
    )
    parser.add_argument(
        "--problem", type=str, default=None,
        help="Single problem text to solve",
    )
    parser.add_argument(
        "--problems_file", type=str, default=None,
        help="Path to a text file with one problem per paragraph (blank-line separated)",
    )
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--max_code_length", type=int, default=1024)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    if not args.problem and not args.problems_file:
        parser.error("Provide --problem or --problems_file")

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device(args.device) if args.device else None

    # ── Load models ───────────────────────────────────────────────────────
    print("Loading models...", flush=True)
    reasoner, vq, talker, tokenizer = load_models(
        talker_checkpoint=args.talker_checkpoint,
        sst_checkpoint_dir=args.sst_checkpoint_dir,
        sst_checkpoint_tag=args.sst_checkpoint_tag,
        device=device,
    )
    actual_device = next(talker.parameters()).device
    print(f"Models loaded on {actual_device}", flush=True)

    # ── Collect problems ──────────────────────────────────────────────────
    problems: List[str] = []
    if args.problem:
        problems.append(args.problem)
    if args.problems_file:
        with open(args.problems_file) as f:
            text = f.read()
        # Split on double newlines (paragraph-separated)
        for paragraph in text.split("\n\n"):
            paragraph = paragraph.strip()
            if paragraph:
                problems.append(paragraph)

    # ── Run inference ─────────────────────────────────────────────────────
    for i, problem_text in enumerate(problems):
        print(f"\n{'=' * 65}")
        print(f"PROBLEM {i + 1}/{len(problems)}")
        print(f"{'=' * 65}")
        print(problem_text[:300])
        if len(problem_text) > 300:
            print("...")

        t0 = time.time()
        code, plan_indices, n_steps = generate_solution(
            problem_text=problem_text,
            reasoner=reasoner,
            vq=vq,
            talker=talker,
            tokenizer=tokenizer,
            max_steps=args.max_steps,
            max_code_length=args.max_code_length,
        )
        elapsed = time.time() - t0

        stopped = n_steps < args.max_steps or (
            n_steps == args.max_steps and len(plan_indices) < n_steps
        )
        parseable = False
        try:
            ast.parse(code)
            parseable = True
        except SyntaxError:
            pass

        print(f"\nPlan: {plan_indices}  ({n_steps} steps, "
              f"{'STOP hit' if stopped else 'max steps reached'})")
        print(f"Parseable: {parseable}  |  {elapsed:.2f}s")
        print(f"\n--- Generated Code ---")
        print(code)
        print(f"--- End ---")


if __name__ == "__main__":
    main()
