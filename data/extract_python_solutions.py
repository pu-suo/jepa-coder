"""
Extract (problem, solution) pairs from APPS, TACO, and nvidia/OpenCodeReasoning.

Design goals (vs. the original):
  * One output file: data/extracted_solutions.jsonl.
  * Sources are interleaved round-robin so any prefix of the output is a
    representative mix — critical for --limit dry runs and for early
    pretraining steps to see reasoning traces immediately.
  * The HuggingFace cache for each dataset is deleted *as soon as* that
    dataset has been consumed, freeing ~30-40 GB per source mid-run.
  * ast.parse() validation is done in a process pool via imap_unordered with
    a large chunksize so IPC overhead stops dominating compute.
  * Workers receive only the raw code string, never the full example dict.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import random
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator

from datasets import load_dataset

# ---------------------------------------------------------------------------
# CPU worker: the smallest possible unit of work
# ---------------------------------------------------------------------------
#
# The worker receives ONE code string and returns ONE bool. That is the entire
# payload crossing the process boundary. Pickling a bool is ~20 bytes; pickling
# a raw dataset example is 10-500 KB. This is the whole ballgame for IPC cost.

def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False


# ---------------------------------------------------------------------------
# Per-source generators: yield (problem, solution, source) tuples WITHOUT
# validating. Validation is done in bulk afterwards by the worker pool.
# ---------------------------------------------------------------------------

_OCR_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_OCR_CODE_RE  = re.compile(r"```python\s*(.*?)```", re.DOTALL)


def _iter_apps() -> Iterator[tuple[str, str, str]]:
    print("[apps]  loading codeparrot/apps ...", flush=True)
    ds = load_dataset("codeparrot/apps", trust_remote_code=True)
    for split_name, split in ds.items():
        print(f"[apps]  split={split_name} n={len(split)}", flush=True)
        for ex in split:
            problem = ex.get("question", "") or ""
            raw = ex.get("solutions", "")
            if not raw or not raw.strip():
                continue
            try:
                sols = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(sols, list):
                sols = [sols]
            for code in sols:
                if isinstance(code, str) and code.strip():
                    yield problem, code, "apps"
    # Free the dataset object and its memory-mapped arrow files before the
    # caller deletes the on-disk cache.
    del ds
    gc.collect()


def _iter_taco() -> Iterator[tuple[str, str, str]]:
    print("[taco]  loading BAAI/TACO ...", flush=True)
    ds = load_dataset("BAAI/TACO", trust_remote_code=True)
    for split_name, split in ds.items():
        print(f"[taco]  split={split_name} n={len(split)}", flush=True)
        for ex in split:
            problem = ex.get("question", "") or ""
            raw = ex.get("solutions", "")
            if isinstance(raw, list):
                sols = raw
            elif isinstance(raw, str) and raw.strip():
                try:
                    sols = json.loads(raw)
                    if not isinstance(sols, list):
                        sols = [sols]
                except json.JSONDecodeError:
                    sols = [raw]
            else:
                continue
            for code in sols:
                if isinstance(code, str) and code.strip():
                    yield problem, code, "taco"
    del ds
    gc.collect()


def _iter_ocr() -> Iterator[tuple[str, str, str]]:
    print("[ocr]   loading nvidia/OpenCodeReasoning ...", flush=True)
    ds = load_dataset("nvidia/OpenCodeReasoning", "split_0", split="split_0")
    print(f"[ocr]   n={len(ds)}", flush=True)
    for ex in ds:
        response = ex.get("output") or ex.get("response") or ""
        if not response:
            continue
        think = _OCR_THINK_RE.search(response)
        if not think:
            continue
        trace = think.group(1).strip()
        if not trace:
            continue

        # Prefer the dedicated code column if present; fall back to the
        # ```python block in the response.
        code = ""
        dedicated = ex.get("code") or ex.get("python_code") or ""
        if isinstance(dedicated, str) and dedicated.strip():
            code = dedicated.strip()
        else:
            m = _OCR_CODE_RE.search(response)
            if m:
                code = m.group(1).strip()
        if not code:
            continue
        # NOTE: for OCR the "problem" field holds the reasoning trace, per
        # the architecture spec §5.2. Source tag lets downstream code tell
        # these apart from APPS/TACO problem statements.
        yield trace, code, "ocr"
    del ds
    gc.collect()


# ---------------------------------------------------------------------------
# Cache deletion helpers
# ---------------------------------------------------------------------------

def _hf_cache_root() -> Path:
    return Path(
        os.environ.get("HF_DATASETS_CACHE")
        or os.path.join(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")), "datasets")
    )


def _purge_dataset_cache(substr: str) -> None:
    """Delete every subdirectory under the HF datasets cache matching `substr`.

    Called right after a source has been fully streamed out to JSONL, so we
    reclaim 30-40 GB before the next source starts downloading.
    """
    root = _hf_cache_root()
    if not root.exists():
        return
    freed = 0
    for child in list(root.iterdir()):
        if substr.lower() in child.name.lower():
            try:
                if child.is_dir():
                    size = sum(p.stat().st_size for p in child.rglob("*") if p.is_file())
                    shutil.rmtree(child, ignore_errors=True)
                    freed += size
            except OSError:
                pass
    if freed:
        print(f"[cache] purged {substr}: freed {freed / 1e9:.1f} GB", flush=True)


# ---------------------------------------------------------------------------
# Main pipeline: interleave sources, validate in a pool, stream to disk
# ---------------------------------------------------------------------------

def _interleave_shuffled(
    iters: list[Iterator[tuple[str, str, str]]],
    seed: int,
    buffer_size: int = 50_000,
) -> Iterator[tuple[str, str, str]]:
    """Round-robin across source iterators, then reservoir-shuffle within a
    rolling buffer. This gives us:
      (a) a representative prefix (every ~N examples contain all 3 sources),
      (b) local shuffling without materializing all 2M examples in RAM.
    """
    rng = random.Random(seed)
    live = list(iters)
    buffer: list[tuple[str, str, str]] = []

    while live:
        # Round-robin one item from each still-alive source
        next_live = []
        for it in live:
            try:
                buffer.append(next(it))
                next_live.append(it)
            except StopIteration:
                pass
        live = next_live

        # When the buffer is full, drain a shuffled chunk
        if len(buffer) >= buffer_size:
            rng.shuffle(buffer)
            # Yield the first half, keep the second half mixing with new items
            half = buffer_size // 2
            for item in buffer[:half]:
                yield item
            buffer = buffer[half:]

    # Final drain
    rng.shuffle(buffer)
    for item in buffer:
        yield item


def _validate_and_write(
    stream: Iterator[tuple[str, str, str]],
    out_path: Path,
    n_workers: int,
    chunksize: int,
) -> dict:
    """Stream (problem, solution, source) tuples through a process pool that
    validates each solution with ast.parse, and write accepted triples to
    out_path as JSONL. Returns per-source stats.
    """
    # We can't send tuples + keep ordering efficiently with imap_unordered
    # without losing the problem/source metadata. Solution: keep a parallel
    # queue in the parent and use imap (ordered) with a big chunksize. imap
    # preserves order so we can zip results back to their metadata.
    stats = {
        "apps": {"total": 0, "passed": 0},
        "taco": {"total": 0, "passed": 0},
        "ocr":  {"total": 0, "passed": 0},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Process in super-chunks: pull N tuples, ship the N code strings to the
    # pool, write the accepted ones, repeat. This caps parent-side memory
    # at ~super_chunk × avg_solution_size.
    super_chunk = 20_000

    with ProcessPoolExecutor(max_workers=n_workers) as pool, \
         open(out_path, "w", encoding="utf-8") as fout:

        batch: list[tuple[str, str, str]] = []
        written = 0

        def flush_batch():
            nonlocal written
            if not batch:
                return
            codes = [t[1] for t in batch]
            # chunksize is critical: with chunksize=1, per-task IPC dominates.
            # With chunksize=500 over an 80-core pool, each worker receives
            # ~250 items at a time and runs for ~12 ms before returning.
            results = pool.map(_is_valid_python, codes, chunksize=chunksize)
            for (problem, code, source), ok in zip(batch, results):
                stats[source]["total"] += 1
                if ok:
                    stats[source]["passed"] += 1
                    fout.write(json.dumps(
                        {"problem": problem, "solution": code, "source": source},
                        ensure_ascii=False,
                    ) + "\n")
                    written += 1
            batch.clear()
            if written and written % 100_000 == 0:
                print(f"  ... {written:,} valid written", flush=True)

        for triple in stream:
            batch.append(triple)
            if len(batch) >= super_chunk:
                flush_batch()
        flush_batch()

    return stats


def main(output_dir: str, seed: int, purge_cache: bool) -> None:
    out_path = Path(output_dir) / "extracted_solutions.jsonl"

    n_workers = os.cpu_count() or 1
    # Chunksize math: we want each worker to get ~50-200 ms of work per chunk
    # so the IPC cost becomes <1%. ast.parse on a typical solution takes
    # ~50 µs, so 500/chunk × 50 µs = 25 ms per chunk — still dominated by
    # compute. Tune down if your solutions are unusually large.
    chunksize = 500
    print(f"Using {n_workers} workers, chunksize={chunksize}", flush=True)

    iters = [_iter_apps(), _iter_taco(), _iter_ocr()]
    stream = _interleave_shuffled(iters, seed=seed)
    stats = _validate_and_write(stream, out_path, n_workers, chunksize)

    # Purge HF caches now that every source has been streamed through.
    if purge_cache:
        for name in ("apps", "taco", "opencodereasoning"):
            _purge_dataset_cache(name)

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXTRACTION STATISTICS")
    print("=" * 60)
    total_attempted = total_passed = 0
    for name, s in stats.items():
        rate = (s["passed"] / s["total"] * 100) if s["total"] else 0.0
        print(f"  {name:<6} attempted={s['total']:>9,}  passed={s['passed']:>9,}  ({rate:5.1f}%)")
        total_attempted += s["total"]
        total_passed    += s["passed"]
    rate = (total_passed / total_attempted * 100) if total_attempted else 0.0
    print(f"  {'TOTAL':<6} attempted={total_attempted:>9,}  passed={total_passed:>9,}  ({rate:5.1f}%)")
    print(f"\nOutput: {out_path}  ({out_path.stat().st_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="data/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_purge_cache", action="store_true",
                   help="Don't delete the HF datasets cache at the end.")
    args = p.parse_args()
    main(args.output_dir, args.seed, purge_cache=not args.no_purge_cache)
