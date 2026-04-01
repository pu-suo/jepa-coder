"""
Extract (problem_text, solution_code) pairs from APPS and TACO datasets.
Filters for syntactically valid Python using ast.parse().
Saves results to data/extracted_solutions.jsonl.

Per architecture spec Section 5.2:
  - Extract Python solutions from APPS and TACO
  - ast.parse() filter (~5-10% dropout expected)
"""

import ast
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset


OUTPUT_PATH = Path(__file__).parent / "extracted_solutions.jsonl"


def try_parse(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def extract_apps() -> tuple[list[dict], dict]:
    """Load codeparrot/apps (all splits), parse solutions JSON array."""
    print("Loading codeparrot/apps ...")
    stats = {"total": 0, "passed": 0, "failed": 0}
    pairs = []

    dataset = load_dataset("codeparrot/apps", trust_remote_code=True)

    for split_name, split in dataset.items():
        print(f"  Processing APPS split: {split_name} ({len(split)} examples)")
        for example in split:
            problem_text = example.get("question", "")
            raw_solutions = example.get("solutions", "")

            if not raw_solutions or not raw_solutions.strip():
                continue

            try:
                solutions = json.loads(raw_solutions)
            except json.JSONDecodeError:
                continue

            if not isinstance(solutions, list):
                solutions = [solutions]

            for code in solutions:
                if not isinstance(code, str) or not code.strip():
                    continue
                stats["total"] += 1
                if try_parse(code):
                    stats["passed"] += 1
                    pairs.append({"problem": problem_text, "solution": code, "source": "apps"})
                else:
                    stats["failed"] += 1

    return pairs, stats


def extract_taco() -> tuple[list[dict], dict]:
    """Load BAAI/TACO (all splits), extract solutions."""
    print("Loading BAAI/TACO ...")
    stats = {"total": 0, "passed": 0, "failed": 0}
    pairs = []

    dataset = load_dataset("BAAI/TACO", trust_remote_code=True)

    for split_name, split in dataset.items():
        print(f"  Processing TACO split: {split_name} ({len(split)} examples)")
        for example in split:
            problem_text = example.get("question", "")

            # TACO solutions field: may be a list or a JSON-encoded string
            raw_solutions = example.get("solutions", "")

            if isinstance(raw_solutions, list):
                solutions = raw_solutions
            elif isinstance(raw_solutions, str) and raw_solutions.strip():
                try:
                    solutions = json.loads(raw_solutions)
                    if not isinstance(solutions, list):
                        solutions = [solutions]
                except json.JSONDecodeError:
                    # Treat the raw string as a single solution
                    solutions = [raw_solutions]
            else:
                continue

            for code in solutions:
                if not isinstance(code, str) or not code.strip():
                    continue
                stats["total"] += 1
                if try_parse(code):
                    stats["passed"] += 1
                    pairs.append({"problem": problem_text, "solution": code, "source": "taco"})
                else:
                    stats["failed"] += 1

    return pairs, stats


def main():
    apps_pairs, apps_stats = extract_apps()
    taco_pairs, taco_stats = extract_taco()

    all_pairs = apps_pairs + taco_pairs

    print(f"\nWriting {len(all_pairs)} valid pairs to {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # ── Statistics ──────────────────────────────────────────────────────────
    total_attempts = apps_stats["total"] + taco_stats["total"]
    total_passed   = apps_stats["passed"] + taco_stats["passed"]
    total_failed   = apps_stats["failed"] + taco_stats["failed"]

    pass_rate = (total_passed / total_attempts * 100) if total_attempts else 0.0
    fail_rate = (total_failed / total_attempts * 100) if total_attempts else 0.0

    print("\n" + "=" * 60)
    print("EXTRACTION STATISTICS")
    print("=" * 60)
    print(f"{'Dataset':<12} {'Attempted':>10} {'Passed':>10} {'Failed':>10} {'Pass %':>8}")
    print("-" * 60)
    for name, s in [("APPS", apps_stats), ("TACO", taco_stats)]:
        rate = (s["passed"] / s["total"] * 100) if s["total"] else 0.0
        print(f"{name:<12} {s['total']:>10,} {s['passed']:>10,} {s['failed']:>10,} {rate:>7.1f}%")
    print("-" * 60)
    print(f"{'TOTAL':<12} {total_attempts:>10,} {total_passed:>10,} {total_failed:>10,} {pass_rate:>7.1f}%")
    print("=" * 60)
    print(f"\nValid pairs saved : {len(all_pairs):,}")
    print(f"  from APPS       : {len(apps_pairs):,}")
    print(f"  from TACO       : {len(taco_pairs):,}")
    print(f"Output file       : {OUTPUT_PATH}")

    if len(all_pairs) < 20_000:
        print("\nWARNING: fewer than 20K pairs — check dataset loading or filters.")
    elif len(all_pairs) > 40_000:
        print("\nNOTE: more than 40K pairs (within expected range per spec).")
    else:
        print("\nPair count within expected range (20K-40K). ✓")


if __name__ == "__main__":
    main()
