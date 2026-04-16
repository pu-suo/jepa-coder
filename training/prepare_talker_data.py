"""
training/prepare_talker_data.py — Phase 3 data preparation.

Loads frozen Reasoner + VQ (from SST checkpoint), runs every SST training
example through the Reasoner loop, collects VQ indices at each step, and
saves (plan_indices, problem_token_ids, solution_token_ids) triples as an
Arrow dataset ready for Talker training.

Contract: docs/contract_3_talker_interface.md §4.1
Architecture: docs/jepa_coder_architecture_v2.md §4 Phase 3

Key decisions:
  - STOP block is excluded from the plan (contract §1: index 0 never passed
    to Talker). Loop runs over blocks[:-1], matching contract §4.1 pseudocode.
  - VQ index lookup is done via direct codebook dot-product (no forward()
    side effects on the frozen VQ buffers).
  - Solution tokens = concatenation of all code block token_ids in order.
    The training script adds BOS/EOS during batching.
  - Output is an Arrow dataset (HuggingFace datasets) to match the existing
    sst_dataset format and enable efficient streaming to train_talker.py.

Usage (Vast.ai):
    python training/prepare_talker_data.py \\
        --checkpoint_dir checkpoints/sst \\
        --checkpoint_tag final \\
        --dataset_path  /workspace/jepa-coder-data/data/sst_dataset \\
        --output_dir    /workspace/jepa-coder-data/data/talker_dataset

Usage (local dry-run):
    python training/prepare_talker_data.py \\
        --checkpoint_dir checkpoints/sst \\
        --checkpoint_tag final \\
        --dataset_path  ../jepa-coder-data/data/sst_dataset \\
        --output_dir    ../jepa-coder-data/data/talker_dataset \\
        --limit 500
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Tuple

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from models.reasoner import Reasoner
from models.vq import VectorQuantizer


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_frozen_models(
    checkpoint_dir: str,
    tag: str,
    vocab_size: int,
    dim: int = 768,
    n_layers: int = 16,
    n_heads: int = 12,
    ffn_dim: int = 3072,
    max_seq_len: int = 1024,
    codebook_size: int = 512,
    commitment_cost: float = 0.25,
    device: torch.device = torch.device("cpu"),
) -> tuple[Reasoner, VectorQuantizer]:
    """
    Load Reasoner and VQ from an SST checkpoint, freeze both for inference.

    Checkpoint file names (produced by training/sst.py _save_checkpoint):
        sst_reasoner_{tag}.pt  — Reasoner state dict
        vq_codebook_{tag}.pt   — VQ state dict

    Returns:
        (reasoner, vq) — both eval(), all requires_grad=False
    """
    reasoner_path = os.path.join(checkpoint_dir, f"sst_reasoner_{tag}.pt")
    vq_path = os.path.join(checkpoint_dir, f"vq_codebook_{tag}.pt")

    if not os.path.exists(reasoner_path):
        raise FileNotFoundError(f"Reasoner checkpoint not found: {reasoner_path}")
    if not os.path.exists(vq_path):
        raise FileNotFoundError(f"VQ checkpoint not found: {vq_path}")

    # --- Reasoner ---
    reasoner = Reasoner(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
    )
    raw = torch.load(reasoner_path, map_location="cpu", weights_only=True)
    state_dict = (
        raw.get("model_state_dict", raw.get("state_dict", raw))
        if isinstance(raw, dict)
        else raw
    )
    missing, unexpected = reasoner.load_state_dict(state_dict, strict=False)
    non_lmhead_unexpected = [k for k in unexpected if not k.startswith("lm_head")]
    assert not non_lmhead_unexpected, \
        f"Unexpected keys in Reasoner checkpoint: {non_lmhead_unexpected}"
    assert not missing, f"Missing keys in Reasoner checkpoint: {missing}"

    # SST inference mode: L2 enabled, no LM head (arch v2 §4.2)
    reasoner.hybrid_norm.l2_enabled = True
    reasoner.lm_head = None

    reasoner.to(device).eval()
    for p in reasoner.parameters():
        p.requires_grad_(False)

    # --- VQ ---
    vq = VectorQuantizer(
        codebook_size=codebook_size,
        dim=dim,
        commitment_cost=commitment_cost,
    )
    raw_vq = torch.load(vq_path, map_location="cpu", weights_only=True)
    vq_state = (
        raw_vq.get("state_dict", raw_vq)
        if isinstance(raw_vq, dict) and "state_dict" in raw_vq
        else raw_vq
    )
    vq.load_state_dict(vq_state)

    vq.to(device).eval()
    for p in vq.parameters():
        p.requires_grad_(False)

    return reasoner, vq


# ---------------------------------------------------------------------------
# Single-example inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_plan_indices(
    reasoner: Reasoner,
    vq: VectorQuantizer,
    problem_tokens: torch.Tensor,
    blocks: List[dict],
) -> List[int]:
    """
    Run the Reasoner loop on one example and return VQ plan indices.

    Implements contract_3_talker_interface.md §4.1 exactly:
        - Encode problem → h
        - For each block in blocks[:-1]  (STOP block excluded):
              r = reasoner.step(h)
              idx = argmax(VQ_codebook @ r)   (nearest codebook entry)
              indices.append(idx)
              h = r                           (continuous loopback)

    The STOP step is excluded: index 0 is STOP and is never passed to the
    Talker (contract §1). If the Reasoner emits index 0 on a non-terminal
    step, it is still included here — the Talker data reflects exactly what
    the trained Reasoner produces.

    VQ index lookup uses direct dot-product rather than vq.forward() to
    avoid mutating the frozen VQ's usage-tracking buffers.

    Args:
        reasoner:       frozen Reasoner (eval, no_grad)
        vq:             frozen VQ (eval, no_grad)
        problem_tokens: LongTensor (L_prob,) — truncated to max_seq_len
        blocks:         list of block dicts; last element must have type='STOP'

    Returns:
        list[int] of VQ indices, length = len(blocks) - 1
    """
    h = reasoner.encode_problem(problem_tokens)   # (d,), ||h||=1

    indices: List[int] = []
    for block in blocks[:-1]:                      # exclude STOP
        r = reasoner.step(h)                       # (d,), ||r||=1

        # Direct nearest-codebook lookup — matches VQ forward() argmax exactly
        # vq.embedding.weight: (K, d),  r: (d,) → dots: (K,)
        dots = torch.matmul(vq.embedding.weight, r)
        idx = int(dots.argmax().item())
        indices.append(idx)

        h = r                                      # continuous loopback

    return indices


# ---------------------------------------------------------------------------
# Dataset iterator (mirrors training/sst.py)
# ---------------------------------------------------------------------------

def sst_data_iterator(dataset_path: str) -> Iterator[tuple[list, list]]:
    """Yield (problem_token_ids, blocks) from the SST Arrow dataset."""
    from datasets import load_from_disk, load_dataset

    try:
        ds = load_from_disk(dataset_path)
    except Exception:
        ds = load_dataset("arrow", data_dir=dataset_path, split="train")

    for row in ds:
        yield row["problem_token_ids"], json.loads(row["blocks_json"])


# ---------------------------------------------------------------------------
# Parallel worker
#
# Model objects are stored as module-level globals so forked workers inherit
# them via copy-on-write. Passing the model through pool.map args would pickle
# it per chunk (~400 MB for the Reasoner) — which is what caused the hang.
# ---------------------------------------------------------------------------

_WORKER_REASONER: "Reasoner | None" = None
_WORKER_VQ: "VectorQuantizer | None" = None
_WORKER_MAX_SEQ_LEN: int = 0
_WORKER_DEVICE: torch.device = torch.device("cpu")


def _process_chunk(
    chunk: List[Tuple[list, list]],
) -> Tuple[List[dict], List[int], List[int], int]:
    """Process a chunk of examples in a forked worker using module-level globals."""
    reasoner = _WORKER_REASONER
    vq = _WORKER_VQ
    max_seq_len = _WORKER_MAX_SEQ_LEN
    device = _WORKER_DEVICE

    rows: List[dict] = []
    plan_lengths: List[int] = []
    all_indices: List[int] = []
    n_skipped = 0

    for prob_ids, blocks in chunk:
        if not blocks or blocks[-1].get("type") != "STOP":
            n_skipped += 1
            continue
        code_blocks = [b for b in blocks if b.get("type") == "CODE"]
        if not code_blocks:
            n_skipped += 1
            continue

        problem_tokens = torch.tensor(
            prob_ids[:max_seq_len], dtype=torch.long, device=device,
        )
        plan_indices = collect_plan_indices(reasoner, vq, problem_tokens, blocks)

        solution_token_ids: List[int] = []
        for blk in code_blocks:
            solution_token_ids.extend(blk["token_ids"])

        plan_lengths.append(len(plan_indices))
        all_indices.extend(plan_indices)
        rows.append({
            "problem_token_ids": list(prob_ids),
            "plan_indices": plan_indices,
            "solution_token_ids": solution_token_ids,
            "n_plan_steps": len(plan_indices),
        })

    return rows, plan_lengths, all_indices, n_skipped


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _dist_line(label: str, data: List[int]) -> str:
    if not data:
        return f"  {label}: (no data)"
    return (
        f"  {label}: n={len(data):,}  mean={statistics.mean(data):.2f}  "
        f"median={statistics.median(data):.1f}  "
        f"min={min(data)}  max={max(data)}"
    )


def print_statistics(
    plan_lengths: List[int],
    all_indices: List[int],
    codebook_size: int,
) -> None:
    """Print plan length distribution and codebook index usage to stdout."""
    total_examples = len(plan_lengths)
    total_plan_tokens = len(all_indices)

    print("\n" + "=" * 65)
    print("PREPARE TALKER DATA — STATISTICS")
    print("=" * 65)
    print(f"Examples written         : {total_examples:>10,}")
    print(f"Total plan tokens        : {total_plan_tokens:>10,}")

    # ── Plan length distribution ──────────────────────────────────────────
    print(f"\nPlan length distribution:")
    print(_dist_line("plan steps", plan_lengths))

    length_counter = Counter(plan_lengths)
    top_val = max(length_counter.values()) if length_counter else 1
    print("\n  Plan length histogram:")
    for k in sorted(length_counter):
        bar = "█" * (length_counter[k] * 40 // top_val)
        print(f"    {k:>2} steps: {length_counter[k]:>8,}  {bar}")

    # ── Codebook index usage ──────────────────────────────────────────────
    unique_indices = set(all_indices)
    utilization = len(unique_indices) / codebook_size * 100

    print(f"\nCodebook utilization:")
    print(f"  Unique indices used   : {len(unique_indices):>4} / {codebook_size}  "
          f"({utilization:.1f}%)")

    if all_indices:
        index_counter = Counter(all_indices)

        print(f"\n  Top 20 most frequent indices:")
        for idx, cnt in index_counter.most_common(20):
            frac = cnt / total_plan_tokens
            bar = "█" * max(1, int(frac * 200))
            print(f"    idx={idx:>3}: {cnt:>8,} ({frac * 100:.2f}%)  {bar}")

        if len(index_counter) > 20:
            print(f"\n  Least frequent 10 used indices:")
            for idx, cnt in sorted(index_counter.items(), key=lambda x: x[1])[:10]:
                print(f"    idx={idx:>3}: {cnt:>8,}")

    zero_usage = [i for i in range(codebook_size) if i not in unique_indices]
    if zero_usage:
        sample = zero_usage[:30]
        suffix = f"... ({len(zero_usage) - 30} more)" if len(zero_usage) > 30 else ""
        print(f"\n  Unused entries ({len(zero_usage)}): {sample}{suffix}")

    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    import time
    from datasets import Dataset, Features, Sequence, Value
    from transformers import AutoTokenizer

    # ── Prevent OMP/MKL thread explosion across forked workers ────────────
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # ── Device ────────────────────────────────────────────────────────────
    # CUDA auto-detected. With --workers > 1 we force CPU (fork + CUDA unsafe).
    if args.device:
        device = torch.device(args.device)
    elif args.workers != 1 and args.workers != 0:
        # Multiprocessing path: CPU only (fork + CUDA is unsafe)
        device = torch.device("cpu")
    elif args.workers == 0 and (os.cpu_count() or 1) > 1 and not torch.cuda.is_available():
        # Auto mode on CPU-only machine → multiprocessing on CPU
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Tokenizer (needed only for vocab_size) ────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        "bigcode/starcoder2-3b",
        trust_remote_code=True,
    )
    vocab_size = tokenizer.vocab_size

    # ── Load frozen Reasoner + VQ ─────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint_dir}  tag={args.checkpoint_tag}")
    reasoner, vq = load_frozen_models(
        checkpoint_dir=args.checkpoint_dir,
        tag=args.checkpoint_tag,
        vocab_size=vocab_size,
        device=device,
    )
    n_params = sum(p.numel() for p in reasoner.parameters())
    print(f"Reasoner: {n_params:,} params (frozen)")
    print(f"VQ: codebook {vq.codebook_size}×{vq.dim} (frozen)")

    # ── Output directory ──────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Materialize dataset ───────────────────────────────────────────────
    print("Loading dataset into memory...", flush=True)
    all_examples = []
    for i, (prob_ids, blocks) in enumerate(sst_data_iterator(args.dataset_path)):
        if args.limit is not None and i >= args.limit:
            break
        all_examples.append((prob_ids, blocks))
        if (i + 1) % 100000 == 0:
            print(f"  materialized {i + 1:,} examples", flush=True)
    print(f"Loaded {len(all_examples):,} examples", flush=True)

    # ── Resolve worker count ──────────────────────────────────────────────
    n_workers = args.workers
    if n_workers == 0:
        n_workers = min(os.cpu_count() or 1, 10)
    print(f"Workers: {n_workers}")

    # ── Install models as module globals for worker fork inheritance ──────
    global _WORKER_REASONER, _WORKER_VQ, _WORKER_MAX_SEQ_LEN, _WORKER_DEVICE
    _WORKER_REASONER = reasoner
    _WORKER_VQ = vq
    _WORKER_MAX_SEQ_LEN = reasoner.max_seq_len
    _WORKER_DEVICE = device

    # ── Merge buffers ─────────────────────────────────────────────────────
    rows: List[dict] = []
    plan_lengths: List[int] = []
    all_indices: List[int] = []
    n_skipped = 0

    # ── Chunk the work ────────────────────────────────────────────────────
    # Keep chunks small enough that we get periodic progress updates.
    # Target: ~40 chunks per worker → frequent progress + low IPC overhead.
    chunk_size = max(100, math.ceil(len(all_examples) / (n_workers * 40)))
    chunks = [
        all_examples[i : i + chunk_size]
        for i in range(0, len(all_examples), chunk_size)
    ]
    print(f"Chunks: {len(chunks)} × {chunk_size} examples each")

    t_start = time.time()
    n_done = 0

    # ── Process ───────────────────────────────────────────────────────────
    if n_workers == 1:
        # Sequential path (for debugging)
        for chunk in chunks:
            chunk_rows, chunk_pl, chunk_idx, chunk_skip = _process_chunk(chunk)
            rows.extend(chunk_rows)
            plan_lengths.extend(chunk_pl)
            all_indices.extend(chunk_idx)
            n_skipped += chunk_skip
            n_done += len(chunk)
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(all_examples) - n_done) / rate if rate > 0 else 0
            print(
                f"  {n_done:>8,} / {len(all_examples):,}  "
                f"({100*n_done/len(all_examples):.1f}%)  |  "
                f"rate={rate:.1f}/s  eta={eta/60:.1f}m",
                flush=True,
            )
    else:
        mp.set_start_method("fork", force=True)
        with mp.Pool(n_workers) as pool:
            for chunk_rows, chunk_pl, chunk_idx, chunk_skip in pool.imap_unordered(
                _process_chunk, chunks
            ):
                rows.extend(chunk_rows)
                plan_lengths.extend(chunk_pl)
                all_indices.extend(chunk_idx)
                n_skipped += chunk_skip
                n_done += len(chunk_rows) + chunk_skip
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(all_examples) - n_done) / rate if rate > 0 else 0
                print(
                    f"  {n_done:>8,} / {len(all_examples):,}  "
                    f"({100*n_done/len(all_examples):.1f}%)  |  "
                    f"rate={rate:.1f}/s  eta={eta/60:.1f}m",
                    flush=True,
                )

    print(f"\nFinished: {len(rows):,} examples written, {n_skipped} skipped")

    # ── Save Arrow dataset ────────────────────────────────────────────────
    features = Features(
        {
            "problem_token_ids": Sequence(Value("int32")),
            "plan_indices": Sequence(Value("int32")),
            "solution_token_ids": Sequence(Value("int32")),
            "n_plan_steps": Value("int32"),
        }
    )
    ds = Dataset.from_list(rows, features=features)
    ds.save_to_disk(str(output_dir))
    print(f"Saved talker dataset → {output_dir}  ({len(ds):,} rows)")

    # ── Statistics ────────────────────────────────────────────────────────
    print_statistics(plan_lengths, all_indices, vq.codebook_size)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Prepare Talker training data from frozen Reasoner + VQ (Phase 3)."
    )
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing SST checkpoints (e.g. checkpoints/sst)",
    )
    p.add_argument(
        "--checkpoint_tag",
        type=str,
        default="final",
        help="Checkpoint tag to load: 'final' or a step count like '00050000' (default: final)",
    )
    p.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the SST Arrow dataset (output of data/prepare_sst_data.py)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write the talker Arrow dataset",
    )
    p.add_argument(
        "--device",
        type=str,
        default="",
        help="Device override: cuda / cpu / mps  (auto-detected if not set)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Dry-run: process only the first N SST examples",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (0 = auto, 1 = sequential for debugging)",
    )

    # Convenience: resolve standard paths automatically when not overridden
    args = p.parse_args()

    # If paths weren't provided explicitly, try Vast.ai first then relative
    if not args.dataset_path:
        vast = "/workspace/jepa-coder-data/data/sst_dataset"
        local = "../jepa-coder-data/data/sst_dataset"
        args.dataset_path = vast if os.path.exists(vast) else local

    if not args.output_dir:
        vast = "/workspace/jepa-coder-data/data/talker_dataset"
        local = "../jepa-coder-data/data/talker_dataset"
        args.output_dir = vast if os.path.exists(vast) else local

    main(args)
