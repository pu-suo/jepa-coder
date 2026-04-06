"""
Prepare the final SST training dataset.

Key changes vs. the original:
  * NO intermediate sst_dataset.jsonl. We write straight to Arrow shards via
    datasets.Dataset.from_generator, which halves peak disk usage.
  * Multiprocessing works on FILE OFFSETS, not pickled records: workers each
    open the input JSONL, seek to their assigned byte range, and process
    their slice independently. Zero large objects cross process boundaries.
  * Each block stores EITHER code OR token_ids, not both. We keep token_ids
    (what the SST trainer actually consumes) and drop the source text —
    saving ~40% of dataset size. Set --keep_code if you want the source for
    debugging.
  * extracted_solutions.jsonl is deleted at the end (opt out with --keep_input).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from segment_ast import segment_solution_grouped  # noqa: E402

TOKENIZER_ID  = "bigcode/starcoder2-3b"
MAX_BLOCK_TOK = 512

# ---------------------------------------------------------------------------
# Worker: each worker opens the file independently and processes a byte range
# ---------------------------------------------------------------------------
#
# This is the trick that saturates the CPU. Instead of the parent reading
# lines and pickling them out to workers (bottleneck: parent's pickle speed),
# every worker reads its own slice of the input file directly from disk.
# The parent process only ships (path, start, end) — a handful of bytes per
# worker — and receives (shard_path, stats_dict) back.

# These are set once per worker process by the pool initializer, not passed
# through every task. Initialization cost (loading the tokenizer) is paid
# once per worker, not once per example.
_WORKER_TOK = None
_WORKER_KEEP_CODE = False


def _worker_init(tokenizer_id: str, keep_code: bool) -> None:
    global _WORKER_TOK, _WORKER_KEEP_CODE
    _WORKER_TOK = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    _WORKER_KEEP_CODE = keep_code


def _worker_process_range(args: tuple[str, int, int, str]) -> dict:
    """Process [start, end) bytes of input_path; write processed rows to
    shard_path as JSONL. Returns a stats dict (all small scalars / Counters)."""
    input_path, start, end, shard_path = args
    tok = _WORKER_TOK
    keep_code = _WORKER_KEEP_CODE

    stats = {
        "total_raw": 0,
        "dropped_segment": 0,
        "dropped_block_len": 0,
        "accepted": 0,
        "by_source": Counter(),
        "block_counts": [],
        "prob_lens": [],
        "block_len_sum": 0,
        "block_len_n": 0,
        "block_len_max": 0,
    }

    with open(input_path, "rb") as fin, open(shard_path, "w", encoding="utf-8") as fout:
        # If we're not at byte 0, skip the partial line we landed in the
        # middle of — the previous worker owns it.
        if start > 0:
            fin.seek(start - 1)
            fin.readline()
        else:
            fin.seek(0)

        while True:
            pos = fin.tell()
            if pos >= end:
                break
            raw = fin.readline()
            if not raw:
                break
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue

            stats["total_raw"] += 1
            problem  = entry["problem"]
            solution = entry["solution"]
            source   = entry.get("source", "unknown")

            blocks = segment_solution_grouped(solution)
            if blocks is None:
                stats["dropped_segment"] += 1
                continue

            prob_ids = tok.encode(problem, add_special_tokens=False)

            enriched = []
            too_long = False
            for blk in blocks:
                if blk["type"] == "STOP":
                    enriched.append({"type": "STOP", "token_ids": []})
                    continue
                ids = tok.encode(blk["code"], add_special_tokens=False)
                n = len(ids)
                if n > MAX_BLOCK_TOK:
                    too_long = True
                    break
                entry_blk = {"type": "CODE", "token_ids": ids}
                if keep_code:
                    entry_blk["code"] = blk["code"]
                enriched.append(entry_blk)
                stats["block_len_sum"] += n
                stats["block_len_n"]   += 1
                if n > stats["block_len_max"]:
                    stats["block_len_max"] = n

            if too_long:
                stats["dropped_block_len"] += 1
                continue

            n_code = sum(1 for b in enriched if b["type"] == "CODE")
            stats["accepted"] += 1
            stats["by_source"][source] += 1
            stats["block_counts"].append(n_code)
            stats["prob_lens"].append(len(prob_ids))

            row = {
                "problem_token_ids": prob_ids,
                "blocks_json":       json.dumps(enriched, ensure_ascii=False),
                "n_blocks":          n_code,
                "source":            source,
            }
            if keep_code:
                row["problem"] = problem
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"shard_path": shard_path, "stats": stats}


# ---------------------------------------------------------------------------
# Parent: partition input file into byte ranges, launch workers, merge shards
# ---------------------------------------------------------------------------

def _partition_file(path: Path, n_parts: int) -> list[tuple[int, int]]:
    size = path.stat().st_size
    step = size // n_parts
    ranges = []
    for i in range(n_parts):
        start = i * step
        end = size if i == n_parts - 1 else (i + 1) * step
        ranges.append((start, end))
    return ranges


def _merge_stats(results: list[dict]) -> dict:
    merged = {
        "total_raw": 0,
        "dropped_segment": 0,
        "dropped_block_len": 0,
        "accepted": 0,
        "by_source": Counter(),
        "block_counts": [],
        "prob_lens": [],
        "block_len_sum": 0,
        "block_len_n": 0,
        "block_len_max": 0,
    }
    for r in results:
        s = r["stats"]
        for k in ("total_raw", "dropped_segment", "dropped_block_len",
                  "accepted", "block_len_sum", "block_len_n"):
            merged[k] += s[k]
        merged["block_len_max"] = max(merged["block_len_max"], s["block_len_max"])
        merged["by_source"].update(s["by_source"])
        merged["block_counts"].extend(s["block_counts"])
        merged["prob_lens"].extend(s["prob_lens"])
    return merged


def _print_stats(stats: dict) -> None:
    total = max(stats["total_raw"], 1)
    print("\n" + "=" * 65)
    print("PREPARE SST DATA — FINAL STATISTICS")
    print("=" * 65)
    print(f"Total raw pairs read    : {stats['total_raw']:>10,}")
    print(f"Dropped (segmentation)  : {stats['dropped_segment']:>10,}  ({stats['dropped_segment']/total*100:.1f}%)")
    print(f"Dropped (block > 512t)  : {stats['dropped_block_len']:>10,}  ({stats['dropped_block_len']/total*100:.1f}%)")
    print(f"Accepted examples       : {stats['accepted']:>10,}  ({stats['accepted']/total*100:.1f}%)")
    print("\nBy source:")
    for src, cnt in sorted(stats["by_source"].items()):
        print(f"  {src:<12}: {cnt:>10,}")

    def _dist(label, data):
        if not data:
            print(f"  {label}: (no data)")
            return
        print(f"  {label}: mean={statistics.mean(data):.1f}  "
              f"median={statistics.median(data):.1f}  "
              f"min={min(data)}  max={max(data)}")

    print("\nDistributions:")
    _dist("block count (CODE)", stats["block_counts"])
    _dist("problem len (tokens)", stats["prob_lens"])
    if stats["block_len_n"]:
        print(f"  block len   (tokens): mean={stats['block_len_sum']/stats['block_len_n']:.1f}  "
              f"max={stats['block_len_max']}  n={stats['block_len_n']:,}")

    if stats["block_counts"]:
        dist = Counter(stats["block_counts"])
        print("\nBlock count histogram:")
        top = max(dist.values())
        for k in sorted(dist):
            bar = "█" * (dist[k] * 40 // top)
            print(f"  {k:>2} blocks: {dist[k]:>8,}  {bar}")
    print("=" * 65)


def main(
    input_path: Path,
    output_dir: Path,
    n_workers: int,
    keep_code: bool,
    keep_input: bool,
    limit: int | None,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir.parent / f"{output_dir.name}__shards_tmp"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # If --limit is set, we write a truncated copy and work from that.
    # This is only for quick dry runs; production runs use the full file.
    work_path = input_path
    if limit is not None:
        work_path = shard_dir / "_limited_input.jsonl"
        with open(input_path, encoding="utf-8") as fin, open(work_path, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if i >= limit:
                    break
                fout.write(line)
        print(f"[limit] wrote {limit:,} lines to {work_path}")

    ranges = _partition_file(work_path, n_workers)
    print(f"Partitioned {work_path.name} ({work_path.stat().st_size/1e9:.1f} GB) into {n_workers} ranges")

    tasks = [
        (str(work_path), start, end, str(shard_dir / f"shard_{i:03d}.jsonl"))
        for i, (start, end) in enumerate(ranges)
    ]

    # mp.Pool gives us initializer support (ProcessPoolExecutor has it too in
    # recent Python but mp.Pool is more predictable here).
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(TOKENIZER_ID, keep_code),
    ) as pool:
        results = pool.map(_worker_process_range, tasks)

    stats = _merge_stats(results)

    # ── Build Arrow dataset directly from the shard files ──────────────────
    # datasets.Dataset.from_json reads JSONL files in parallel and writes Arrow
    # without ever holding all rows in RAM. It streams shard → Arrow on disk.
    print(f"\nBuilding Arrow dataset from {len(tasks)} shards → {output_dir}")

    shard_files = [t[3] for t in tasks if Path(t[3]).stat().st_size > 0]

    # Schema: fix the types so Arrow doesn't need to scan the whole dataset.
    feats = {
        "problem_token_ids": Sequence(Value("int32")),
        "blocks_json":       Value("string"),
        "n_blocks":          Value("int32"),
        "source":            Value("string"),
    }
    if keep_code:
        feats["problem"] = Value("string")
    features = Features(feats)

    ds = Dataset.from_json(shard_files, features=features)
    ds.save_to_disk(str(output_dir))
    print(f"Saved Arrow dataset → {output_dir}  ({len(ds):,} rows)")

    # ── Cleanup: delete shards and optionally the input JSONL ───────────────
    for sf in shard_files:
        try:
            os.remove(sf)
        except OSError:
            pass
    try:
        shard_dir.rmdir()
    except OSError:
        pass  # limited_input file or similar; leave it

    if not keep_input and limit is None:
        size = input_path.stat().st_size
        input_path.unlink()
        print(f"[cleanup] removed {input_path.name} (freed {size/1e9:.1f} GB)")

    _print_stats(stats)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="Path to extracted_solutions.jsonl")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Directory to write the Arrow dataset to")
    p.add_argument("--n_workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--keep_code", action="store_true",
                   help="Keep source code strings in the dataset (debug only)")
    p.add_argument("--keep_input", action="store_true",
                   help="Don't delete extracted_solutions.jsonl after building")
    p.add_argument("--limit", type=int, default=None,
                   help="Dry-run: process only the first N lines")
    args = p.parse_args()
    main(args.input, args.output_dir, args.n_workers,
         args.keep_code, args.keep_input, args.limit)
