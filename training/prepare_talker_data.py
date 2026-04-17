"""
training/prepare_talker_data.py — Phase 3 data preparation.

H100-optimized batched version. Runs the frozen Reasoner + VQ over the SST
dataset and writes (problem_token_ids, plan_indices, solution_token_ids)
triples as an Arrow dataset.

Optimizations:
  - Bucket-batching: sort examples by problem length so batches pad minimally
  - Batched encode_problem: (B, T, d) with key-padding mask
  - Batched step loop: all B examples take a step synchronously each iteration;
    finished examples are masked out of the index-collection
  - F.scaled_dot_product_attention (Flash Attention 2 on H100)
  - bf16 autocast for compute, fp32 for VQ argmax to match the contract

Contract: docs/contract_3_talker_interface.md §4.1
Architecture: docs/jepa_coder_architecture_v2.md §4 Phase 3

Key correctness decisions (identical to sequential version):
  - STOP block is excluded from the plan: indices collected for blocks[:-1]
  - VQ lookup via argmax(codebook @ r) with no side-effects on VQ buffers
  - Solution tokens = concatenation of all CODE block token_ids in order
  - Output Arrow schema matches the sequential version exactly

Usage (Vast.ai H100):
    python -m training.prepare_talker_data \\
        --checkpoint_dir /workspace/jepa-coder-data/checkpoints/sst \\
        --checkpoint_tag final \\
        --dataset_path   /workspace/jepa-coder-data/data/sst_dataset \\
        --output_dir     /workspace/jepa-coder-data/data/talker_dataset \\
        --batch_size     512

Usage (local dry-run, small limit):
    python -m training.prepare_talker_data \\
        --checkpoint_dir ../jepa-coder-data/checkpoints/sst \\
        --checkpoint_tag final \\
        --dataset_path   ../jepa-coder-data/data/sst_dataset \\
        --output_dir     ../jepa-coder-data/data/talker_dataset \\
        --limit 500 --batch_size 64
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Tuple

import torch
import torch.nn.functional as F

from models.reasoner import Reasoner
from models.vq import VectorQuantizer

# Use orjson if available (2-3x faster than stdlib json for our payload)
try:
    import orjson
    _json_loads = orjson.loads
except ImportError:
    _json_loads = json.loads


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
    """Load Reasoner and VQ from an SST checkpoint, freeze both for inference."""
    reasoner_path = os.path.join(checkpoint_dir, f"sst_reasoner_{tag}.pt")
    vq_path = os.path.join(checkpoint_dir, f"vq_codebook_{tag}.pt")

    if not os.path.exists(reasoner_path):
        raise FileNotFoundError(f"Reasoner checkpoint not found: {reasoner_path}")
    if not os.path.exists(vq_path):
        raise FileNotFoundError(f"VQ checkpoint not found: {vq_path}")

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

    # SST inference mode: L2 enabled, no LM head (arch v2 §4.2).
    # Disable the inline unit-norm assertion inside HybridNorm — it holds in
    # fp32 but breaks under bf16 autocast (we re-normalize below anyway).
    reasoner.hybrid_norm.l2_enabled = False
    reasoner.lm_head = None

    reasoner.to(device).eval()
    for p in reasoner.parameters():
        p.requires_grad_(False)

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
# Batched transformer forward (uses Reasoner internals directly).
# Operates on (B, T, d) with an optional (B, T) key-padding mask.
# Always applies causal masking (consistent with sequential Reasoner).
# ---------------------------------------------------------------------------

@torch.inference_mode()
def batched_transformer_forward(
    reasoner: Reasoner,
    x: torch.Tensor,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Args:
        x: (B, T, d)
        key_padding_mask: (B, T) bool, True = pad position to mask out.
            Pass None when T == 1 (no padding, no causal mask needed).
    Returns:
        (B, T, d)
    """
    B, T, d = x.shape

    # Build combined causal + padding mask as an additive float tensor.
    # Only needed when T > 1 (causal matters) or padding is present.
    attn_mask: torch.Tensor | None = None
    if T > 1 or key_padding_mask is not None:
        # Causal mask: upper triangle (above diagonal) is masked
        if T > 1:
            causal = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1,
            )  # (T, T) True = masked
            mask = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, T, T)  # (B, 1, T, T)
        else:
            mask = torch.zeros(B, 1, T, T, device=x.device, dtype=torch.bool)

        if key_padding_mask is not None:
            # key_padding_mask: (B, T), True = pad. Apply to KEYS dimension.
            pad = key_padding_mask.unsqueeze(1).unsqueeze(1).expand(B, 1, T, T)
            mask = mask | pad

        attn_mask = torch.zeros(B, 1, T, T, device=x.device, dtype=x.dtype)
        attn_mask = attn_mask.masked_fill(mask, float("-inf"))

    for block in reasoner.transformer_blocks.blocks:
        # ── Attention sublayer (with QK-Norm, no dropout) ────────────────
        x_in = block.norm1(x)
        attn = block.attn

        qkv = attn.qkv_proj(x_in)                                       # (B, T, 3d)
        q, k, v = qkv.split(d, dim=-1)
        q = q.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)   # (B, H, T, hd)
        k = k.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
        v = v.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)

        q = F.normalize(q, dim=-1)  # QK-Norm (matches sequential version)
        k = F.normalize(k, dim=-1)

        # F.scaled_dot_product_attention on H100 uses FlashAttention-2.
        # Passing attn_mask explicitly (not is_causal) because we combine
        # causal + key-padding.
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=attn.scale,
        )                                                               # (B, H, T, hd)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, d)
        attn_out = attn.out_proj(attn_out)
        x = x + attn_out

        # ── FFN sublayer ─────────────────────────────────────────────────
        x = x + block.ffn(block.norm2(x))

    return x


# ---------------------------------------------------------------------------
# Batched encode_problem
# ---------------------------------------------------------------------------

@torch.inference_mode()
def batched_encode_problem(
    reasoner: Reasoner,
    token_ids_list: List[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Batched version of Reasoner.encode_problem.

    Args:
        token_ids_list: list of 1-D LongTensors (already on `device`)
    Returns:
        (B, d) unit-norm tensor.
    """
    B = len(token_ids_list)
    lengths = torch.tensor(
        [t.shape[0] for t in token_ids_list], device=device, dtype=torch.long,
    )
    max_len = int(lengths.max().item())

    # Pad token ids. Pad value can be anything — masked out in attention &
    # in the mean-pool — use 0 to keep embed lookup cheap.
    padded = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, t in enumerate(token_ids_list):
        padded[i, : t.shape[0]] = t

    positions = torch.arange(max_len, device=device)                   # (T,)
    key_padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)  # (B, T) True=pad

    pos = positions.unsqueeze(0).expand(B, max_len)                    # (B, T)
    x = reasoner.embedding(padded) + reasoner.pos_embedding(pos)       # (B, T, d)

    x = batched_transformer_forward(reasoner, x, key_padding_mask)     # (B, T, d)
    x = reasoner.hybrid_norm(x)                                        # (B, T, d)

    # Mean-pool over non-pad positions
    pool_mask = (~key_padding_mask).to(x.dtype).unsqueeze(-1)          # (B, T, 1)
    x_sum = (x * pool_mask).sum(dim=1)                                 # (B, d)
    h = x_sum / lengths.to(x.dtype).unsqueeze(-1)                      # (B, d)

    return F.normalize(h, dim=-1)                                      # (B, d) unit norm


# ---------------------------------------------------------------------------
# Batched step
# ---------------------------------------------------------------------------

@torch.inference_mode()
def batched_step(reasoner: Reasoner, h: torch.Tensor) -> torch.Tensor:
    """(B, d) -> (B, d) unit-norm, matches Reasoner.step but batched."""
    x = h.unsqueeze(1)                                                 # (B, 1, d)
    x = batched_transformer_forward(reasoner, x, key_padding_mask=None)  # (B, 1, d)
    x = x.squeeze(1)                                                   # (B, d)
    x = reasoner.hybrid_norm(x)                                        # (B, d)
    return F.normalize(x, dim=-1)                                      # (B, d)


# ---------------------------------------------------------------------------
# Batched VQ lookup
# ---------------------------------------------------------------------------

@torch.inference_mode()
def batched_vq_argmax(vq: VectorQuantizer, r: torch.Tensor) -> torch.Tensor:
    """(B, d) -> (B,) long tensor of argmax codebook indices."""
    # codebook @ r^T for nearest-by-dot-product (unit-norm ⇒ same as cosine sim)
    # Use fp32 for the final argmax to match the sequential path exactly.
    dots = torch.matmul(r.float(), vq.embedding.weight.float().t())     # (B, K)
    return dots.argmax(dim=-1)                                          # (B,)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _parse_blocks_batch(batch: List[str]) -> List[dict | None]:
    """
    Worker: parse a batch of blocks_json strings. Returns one dict per input
    (or None for invalid rows). Only returns fields derivable from blocks_json
    — problem_token_ids is looked up separately to avoid IPC overhead.
    """
    out: List[dict | None] = []
    for s in batch:
        blocks = _json_loads(s)
        if not blocks or blocks[-1].get("type") != "STOP":
            out.append(None); continue
        code_blocks = [b for b in blocks if b.get("type") == "CODE"]
        if not code_blocks:
            out.append(None); continue
        # Concatenate block token_ids with a newline token between blocks.
        # ast.unparse() (used by segment_ast.py) does NOT emit trailing
        # newlines, so without this separator the Talker trains on
        # "passif x:" instead of "pass\nif x:".
        NEWLINE_TOKEN = 222  # StarCoder2 BPE token for '\n'
        solution: List[int] = []
        for b in code_blocks:
            if solution:
                solution.append(NEWLINE_TOKEN)
            solution.extend(b["token_ids"])
        out.append({
            "solution_token_ids": solution,
            "plan_length": len(blocks) - 1,  # exclude STOP
        })
    return out


def load_examples(
    dataset_path: str,
    limit: int | None,
    n_workers: int = 0,
) -> List[dict]:
    """
    Load and pre-filter the SST dataset into a list of example dicts.

    Uses multiprocessing for JSON parsing (the bottleneck). Each worker gets
    a batch of blocks_json strings and returns the extracted fields.
    """
    from datasets import load_from_disk, load_dataset

    try:
        ds = load_from_disk(dataset_path)
    except Exception:
        ds = load_dataset("arrow", data_dir=dataset_path, split="train")

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    N = len(ds)

    # Extract the two columns as Python lists — one Arrow materialization
    # instead of N per-row accesses. This alone is ~5-10x faster.
    print("  extracting columns from Arrow...", flush=True)
    t0 = time.time()
    blocks_json_col = ds["blocks_json"]
    problem_ids_col = ds["problem_token_ids"]
    print(f"  extracted {N:,} rows in {time.time() - t0:.1f}s", flush=True)

    if n_workers == 0:
        n_workers = min(os.cpu_count() or 1, 16)
    print(f"  parsing JSON with {n_workers} workers (orjson={_json_loads is not json.loads})", flush=True)

    # Batch JSON strings into chunks for worker dispatch
    batch_size = 2000
    batches = [
        blocks_json_col[i : i + batch_size]
        for i in range(0, N, batch_size)
    ]

    examples: List[dict] = []
    n_skipped = 0
    t0 = time.time()
    processed = 0

    with mp.Pool(n_workers) as pool:
        for batch_idx, batch_results in enumerate(
            pool.imap(_parse_blocks_batch, batches, chunksize=1)
        ):
            base = batch_idx * batch_size
            for j, parsed in enumerate(batch_results):
                if parsed is None:
                    n_skipped += 1
                    continue
                examples.append({
                    "problem_token_ids": list(problem_ids_col[base + j]),
                    "solution_token_ids": parsed["solution_token_ids"],
                    "plan_length": parsed["plan_length"],
                })
            processed += len(batch_results)
            if processed // 200_000 > (processed - len(batch_results)) // 200_000:
                dt = time.time() - t0
                print(
                    f"  parsed {processed:>9,} / {N:,}  ({processed / dt:.0f}/s)",
                    flush=True,
                )

    print(
        f"  loaded {len(examples):,} valid examples "
        f"({n_skipped:,} skipped) in {time.time() - t0:.1f}s",
        flush=True,
    )
    return examples


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
    total_examples = len(plan_lengths)
    total_plan_tokens = len(all_indices)

    print("\n" + "=" * 65)
    print("PREPARE TALKER DATA — STATISTICS")
    print("=" * 65)
    print(f"Examples written         : {total_examples:>10,}")
    print(f"Total plan tokens        : {total_plan_tokens:>10,}")

    print(f"\nPlan length distribution:")
    print(_dist_line("plan steps", plan_lengths))

    length_counter = Counter(plan_lengths)
    top_val = max(length_counter.values()) if length_counter else 1
    print("\n  Plan length histogram:")
    for k in sorted(length_counter):
        bar = "█" * (length_counter[k] * 40 // top_val)
        print(f"    {k:>2} steps: {length_counter[k]:>8,}  {bar}")

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
    from datasets import Dataset, Features, Sequence, Value
    from transformers import AutoTokenizer

    # ── Device ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Autocast dtype: bf16 on H100, fp16 on older CUDA, fp32 on CPU/MPS
    if device.type == "cuda":
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        autocast_dtype = torch.float32

    print(f"Device: {device}  |  autocast: {autocast_dtype}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Enable TF32 for any fp32 matmuls that remain
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── Tokenizer (for vocab_size) ────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        "bigcode/starcoder2-3b", trust_remote_code=True,
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

    # ── Load dataset ──────────────────────────────────────────────────────
    print("Loading dataset...")
    examples = load_examples(args.dataset_path, args.limit)
    if not examples:
        print("No examples to process.")
        return

    # ── Bucket-sort by problem length to minimize padding waste ───────────
    examples.sort(key=lambda e: len(e["problem_token_ids"]))
    print(
        f"Problem length: min={len(examples[0]['problem_token_ids'])}  "
        f"max={len(examples[-1]['problem_token_ids'])}"
    )

    # ── Output directory ──────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Shard writer setup ────────────────────────────────────────────────
    # Write rows to numbered parquet shards on disk as we go so a crash near
    # the end doesn't lose hours of GPU work. At the end, concatenate into
    # a single HF Arrow dataset.
    features = Features({
        "problem_token_ids": Sequence(Value("int32")),
        "plan_indices": Sequence(Value("int32")),
        "solution_token_ids": Sequence(Value("int32")),
        "n_plan_steps": Value("int32"),
    })
    shard_dir = output_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: skip shards already on disk
    existing_shards = sorted(shard_dir.glob("shard_*.parquet"))
    shard_idx = len(existing_shards)
    examples_already_done = 0
    if existing_shards:
        # Each shard's length = args.shard_size except possibly the last.
        # To keep things simple, just assume full shards (we shard in fixed
        # row counts from the top of the sorted examples list).
        examples_already_done = shard_idx * args.shard_size
        print(
            f"Found {shard_idx} existing shard(s) → resuming from example "
            f"{examples_already_done:,}",
            flush=True,
        )

    def flush_shard(buffer: List[dict]) -> int:
        nonlocal shard_idx
        if not buffer:
            return 0
        path = shard_dir / f"shard_{shard_idx:05d}.parquet"
        tmp = shard_dir / f".shard_{shard_idx:05d}.parquet.tmp"
        ds_shard = Dataset.from_list(buffer, features=features)
        ds_shard.to_parquet(str(tmp))
        os.replace(tmp, path)  # atomic: either fully there or not at all
        n = len(ds_shard)
        shard_idx += 1
        return n

    # ── Main loop ─────────────────────────────────────────────────────────
    row_buffer: List[dict] = []
    plan_lengths_all: List[int] = []
    all_indices_all: List[int] = []

    B = args.batch_size
    max_seq_len = reasoner.max_seq_len
    t_start = time.time()
    n_done = examples_already_done

    for start in range(examples_already_done, len(examples), B):
        batch = examples[start : start + B]
        bs = len(batch)

        token_tensors = [
            torch.tensor(
                e["problem_token_ids"][:max_seq_len],
                dtype=torch.long, device=device,
            )
            for e in batch
        ]
        plan_lengths = torch.tensor(
            [e["plan_length"] for e in batch], device=device, dtype=torch.long,
        )
        max_plan = int(plan_lengths.max().item())

        with torch.autocast(
            device_type=device.type, dtype=autocast_dtype,
            enabled=(device.type == "cuda"),
        ):
            h = batched_encode_problem(reasoner, token_tensors, device)
            indices_per_step: List[torch.Tensor] = []
            for step_idx in range(max_plan):
                r = batched_step(reasoner, h)
                idx = batched_vq_argmax(vq, r)
                indices_per_step.append(idx)
                h = r

        if indices_per_step:
            idx_matrix = torch.stack(indices_per_step, dim=0).t().cpu()
        else:
            idx_matrix = torch.empty(bs, 0, dtype=torch.long)

        plan_lengths_cpu = plan_lengths.cpu().tolist()

        for i, example in enumerate(batch):
            pl = plan_lengths_cpu[i]
            plan_indices = idx_matrix[i, :pl].tolist()
            row_buffer.append({
                "problem_token_ids": example["problem_token_ids"],
                "plan_indices": plan_indices,
                "solution_token_ids": example["solution_token_ids"],
                "n_plan_steps": pl,
            })
            plan_lengths_all.append(pl)
            all_indices_all.extend(plan_indices)

        n_done += bs

        # Flush a shard once the buffer hits shard_size
        if len(row_buffer) >= args.shard_size:
            n_flushed = flush_shard(row_buffer[: args.shard_size])
            row_buffer = row_buffer[args.shard_size :]
            print(
                f"  ✓ flushed shard {shard_idx - 1} "
                f"({n_flushed:,} rows) → {shard_dir}",
                flush=True,
            )

        elapsed = time.time() - t_start
        rate = (n_done - examples_already_done) / elapsed if elapsed > 0 else 0
        eta_s = (len(examples) - n_done) / rate if rate > 0 else 0
        if (start // B) % 10 == 0 or n_done == len(examples):
            print(
                f"  {n_done:>9,} / {len(examples):,}  "
                f"({100 * n_done / len(examples):.1f}%)  |  "
                f"rate={rate:.0f}/s  eta={eta_s / 60:.1f}m",
                flush=True,
            )

    # Flush the remainder
    if row_buffer:
        flush_shard(row_buffer)
        row_buffer = []

    print(f"\nFinished inference: {n_done:,} examples in {(time.time() - t_start)/60:.1f}m")
    print(f"Shards written to: {shard_dir}")

    # ── Concatenate shards into a single Arrow dataset ────────────────────
    print("Concatenating shards into final Arrow dataset...", flush=True)
    shard_paths = sorted(shard_dir.glob("shard_*.parquet"))
    ds = Dataset.from_parquet([str(p) for p in shard_paths], features=features)
    ds.save_to_disk(str(output_dir))
    print(f"Saved talker dataset → {output_dir}  ({len(ds):,} rows)")

    # Keep shards until the user verifies the final dataset, then they can
    # manually `rm -rf /workspace/jepa-coder-data/data/talker_dataset/_shards`

    print_statistics(plan_lengths_all, all_indices_all, vq.codebook_size)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Prepare Talker training data from frozen Reasoner + VQ (Phase 3, H100-optimized)."
    )
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--checkpoint_tag", type=str, default="final")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="",
                   help="Device override: cuda / cpu / mps (auto-detected)")
    p.add_argument("--limit", type=int, default=None,
                   help="Dry-run: process only the first N SST examples")
    p.add_argument("--batch_size", type=int, default=512,
                   help="Mega-batch size for encode_problem + step loop (default: 512)")
    p.add_argument("--shard_size", type=int, default=100_000,
                   help="Rows per on-disk parquet shard (default: 100,000)")
    args = p.parse_args()
    main(args)
