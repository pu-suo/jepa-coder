"""
Prepare the final SST training dataset.

Pipeline (per architecture spec Section 5.2):
  1. Load data/extracted_solutions.jsonl
  2. Segment each solution via segment_ast.segment_solution_grouped()
  3. Tokenize problem text and each code block with bigcode/starcoder2-3b
  4. Drop examples where any single block tokenises to > 512 tokens
  5. Save to data/sst_dataset/ (HuggingFace Dataset) + data/sst_dataset.jsonl
  6. Print statistics

Run with:
    python data/prepare_sst_data.py [--limit N]   (--limit for a dry-run subset)
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

# Resolve sibling module without installing the package
sys.path.insert(0, str(Path(__file__).parent))
from segment_ast import segment_solution_grouped  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────
INPUT_JSONL   = Path(__file__).parent / "extracted_solutions.jsonl"
OUTPUT_DIR    = Path(__file__).parent / "sst_dataset"
OUTPUT_JSONL  = Path(__file__).parent / "sst_dataset.jsonl"
TOKENIZER_ID  = "bigcode/starcoder2-3b"
MAX_BLOCK_TOK = 512          # spec Section 5.2: drop blocks > 512 tokens

# ── Counters ─────────────────────────────────────────────────────────────────
class Stats:
    def __init__(self):
        self.total_raw           = 0   # lines read from extracted_solutions.jsonl
        self.dropped_segment     = 0   # failed segment (<2 or >15 blocks, or unparse)
        self.dropped_block_len   = 0   # had at least one block > MAX_BLOCK_TOK tokens
        self.accepted            = 0
        self.block_counts        = []  # per-example block count (CODE blocks only)
        self.prob_lens           = []  # problem token lengths
        self.block_lens          = []  # all individual block token lengths
        self.by_source: Counter  = Counter()


def _load_tokenizer() -> AutoTokenizer:
    print(f"Loading tokenizer {TOKENIZER_ID} …")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    return tok


def _tokenize(tok: AutoTokenizer, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False)


def _process_stream(tok: AutoTokenizer, limit: int | None, stats: Stats, input_path: Path):
    """Generator: yield fully-processed example dicts one at a time."""
    with open(input_path, encoding="utf-8") as f:
        for raw_line in f:
            if limit is not None and stats.total_raw >= limit:
                break

            stats.total_raw += 1
            if stats.total_raw % 50_000 == 0:
                print(
                    f"  … {stats.total_raw:,} read | "
                    f"{stats.accepted:,} accepted | "
                    f"{stats.dropped_segment:,} dropped(seg) | "
                    f"{stats.dropped_block_len:,} dropped(len)"
                )

            entry    = json.loads(raw_line)
            problem  = entry["problem"]
            solution = entry["solution"]
            source   = entry.get("source", "unknown")

            # ── Segment ──────────────────────────────────────────────────────
            blocks = segment_solution_grouped(solution)
            if blocks is None:
                stats.dropped_segment += 1
                continue

            # ── Tokenize problem ─────────────────────────────────────────────
            prob_ids = _tokenize(tok, problem)

            # ── Tokenize each block; enforce MAX_BLOCK_TOK ───────────────────
            enriched_blocks = []
            too_long = False
            for blk in blocks:
                if blk["type"] == "STOP":
                    enriched_blocks.append({"type": "STOP", "code": "<STOP>", "token_ids": []})
                    continue
                ids = _tokenize(tok, blk["code"])
                if len(ids) > MAX_BLOCK_TOK:
                    too_long = True
                    break
                enriched_blocks.append({"type": "CODE", "code": blk["code"], "token_ids": ids})
                stats.block_lens.append(len(ids))

            if too_long:
                stats.dropped_block_len += 1
                continue

            # ── Accept ───────────────────────────────────────────────────────
            n_code_blocks = sum(1 for b in enriched_blocks if b["type"] == "CODE")
            stats.accepted += 1
            stats.block_counts.append(n_code_blocks)
            stats.prob_lens.append(len(prob_ids))
            stats.by_source[source] += 1

            yield {
                "problem":          problem,
                "problem_token_ids": prob_ids,
                # Store blocks as JSON string — HF Dataset handles nested lists
                # awkwardly; we keep them as a serialised field and also expose
                # the raw lists for fast access during training.
                "blocks_json":      json.dumps(enriched_blocks, ensure_ascii=False),
                "n_blocks":         n_code_blocks,
                "source":           source,
            }


def _print_stats(stats: Stats) -> None:
    def _dist(label: str, data: list[int]) -> None:
        if not data:
            print(f"  {label}: (no data)")
            return
        print(
            f"  {label}: mean={statistics.mean(data):.1f}  "
            f"median={statistics.median(data):.1f}  "
            f"min={min(data)}  max={max(data)}"
        )

    print("\n" + "=" * 65)
    print("PREPARE SST DATA — FINAL STATISTICS")
    print("=" * 65)
    print(f"Total raw pairs read    : {stats.total_raw:>10,}")
    print(f"Dropped (segmentation)  : {stats.dropped_segment:>10,}  "
          f"({stats.dropped_segment / max(stats.total_raw, 1) * 100:.1f}%)")
    print(f"Dropped (block > 512t)  : {stats.dropped_block_len:>10,}  "
          f"({stats.dropped_block_len / max(stats.total_raw, 1) * 100:.1f}%)")
    print(f"Accepted examples       : {stats.accepted:>10,}  "
          f"({stats.accepted / max(stats.total_raw, 1) * 100:.1f}%)")
    print()
    print("By source:")
    for src, cnt in sorted(stats.by_source.items()):
        print(f"  {src:<12}: {cnt:>10,}")
    print()
    print("Distributions (accepted examples):")
    _dist("block count  (CODE)", stats.block_counts)
    _dist("problem len  (tokens)", stats.prob_lens)
    _dist("block len    (tokens)", stats.block_lens)

    if stats.block_counts:
        dist = Counter(stats.block_counts)
        print("\nBlock count histogram:")
        for k in sorted(dist):
            bar = "█" * (dist[k] * 40 // max(dist.values()))
            print(f"  {k:>2} blocks: {dist[k]:>8,}  {bar}")
    print("=" * 65)


def main(limit: int | None = None, input_path: Path = INPUT_JSONL, output_jsonl: Path = OUTPUT_JSONL) -> None:
    tok   = _load_tokenizer()
    stats = Stats()

    # ── Stream directly to JSONL to avoid holding 1.7M rows in memory ────────
    print(f"Processing {input_path} …")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for row in _process_stream(tok, limit, stats, input_path):
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved JSONL    → {output_jsonl}  ({stats.accepted:,} examples)")

    # ── Build HF Dataset from the JSONL file (no full in-memory list) ────────
    print(f"Building HuggingFace Dataset from JSONL …")
    output_dir = output_jsonl.with_suffix("")  # e.g. sst_dataset/ alongside sst_dataset.jsonl
    ds = Dataset.from_json(str(output_jsonl))
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    print(f"Saved HF dataset → {output_dir}/")

    _print_stats(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=INPUT_JSONL,
        help="Path to extracted_solutions.jsonl produced by extract_python_solutions.py.",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_JSONL,
        help="Path for the output sst_dataset.jsonl file.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N raw pairs (useful for a quick dry-run).",
    )
    args = parser.parse_args()
    main(limit=args.limit, input_path=args.input, output_jsonl=args.output)
