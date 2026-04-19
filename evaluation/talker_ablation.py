"""
evaluation/talker_ablation.py — Talker ablation tests.

Runs the five ablation tests from contract_3_talker_interface.md §5 to verify
that the Talker genuinely depends on the Reasoner's plan indices and cannot
reason independently from the problem text alone.

Tests:
  1. Baseline       — correct plan + correct problem  → valid code
  2. Random indices  — random ints in [1,511]          → incoherent output
  3. Gaussian noise  — noise quantized via VQ           → incoherent output
  4. Semantic mismatch — plan from problem A + text B  → follows plan A structure
  5. Empty plan      — no plan indices at all           → minimal/degenerate output

If tests 2-5 produce correct solutions, the decoupling has FAILED and the
Talker is bypassing the plan bottleneck.

Usage (Vast.ai):
    python -m evaluation.talker_ablation \
        --talker_checkpoint /workspace/jepa-coder-data/checkpoints/talker/talker_final.pt \
        --sst_checkpoint_dir /workspace/jepa-coder-data/checkpoints/sst \
        --dataset_path /workspace/jepa-coder-data/data/talker_dataset \
        --n_samples 50

Usage (local dry-run):
    python -m evaluation.talker_ablation \
        --talker_checkpoint ../jepa-coder-data/checkpoints/talker/talker_final.pt \
        --sst_checkpoint_dir ../jepa-coder-data/checkpoints/sst \
        --dataset_path ../jepa-coder-data/data/talker_dataset \
        --n_samples 10
"""

from __future__ import annotations

import argparse
import ast
import random
import time
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from models.talker import Talker
from models.vq import VectorQuantizer
from data.prepare_talker_data import load_frozen_models


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    test_name: str
    n_samples: int
    parse_successes: int = 0
    generated_codes: List[str] = field(default_factory=list)

    @property
    def parse_rate(self) -> float:
        return self.parse_successes / self.n_samples if self.n_samples > 0 else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def try_ast_parse(code: str) -> bool:
    """Return True if code is valid Python (ast.parse succeeds)."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


MAX_PROB_LEN = 512  # Must match training (train_talker.py max_prob_len)


def generate_code(
    talker: Talker,
    tokenizer,
    problem_ids: torch.Tensor,
    plan_indices: torch.Tensor,
    device: torch.device,
    max_length: int = 1024,
) -> str:
    """Run Talker generation and decode to string."""
    # Truncate problem tokens to match training limit — the Talker's
    # position_embedding has max_seq_len=1024 entries and must cover
    # L_prob + M positions in the encoder.
    problem_ids = problem_ids[:MAX_PROB_LEN].to(device)
    plan_indices = plan_indices.to(device)
    token_ids = talker.generate(problem_ids, plan_indices, max_length=max_length)
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def print_code_sample(code: str, max_chars: int = 400) -> None:
    """Print a truncated code sample with a border."""
    display = code[:max_chars]
    if len(code) > max_chars:
        display += "\n... (truncated)"
    print(f"    {'-' * 60}")
    for line in display.split("\n"):
        print(f"    | {line}")
    print(f"    {'-' * 60}")


def print_result_summary(result: AblationResult) -> None:
    """Print summary line for one ablation test."""
    status = "PASS" if result.test_name == "baseline" else "INFO"
    print(
        f"  [{status}] {result.test_name}: "
        f"ast.parse success = {result.parse_successes}/{result.n_samples} "
        f"({result.parse_rate * 100:.1f}%)"
    )


# ---------------------------------------------------------------------------
# Ablation tests
# ---------------------------------------------------------------------------

def test_baseline(
    talker: Talker,
    tokenizer,
    examples: list,
    device: torch.device,
    max_length: int,
) -> AblationResult:
    """Test 1: Correct plan + correct problem -> valid, relevant code."""
    result = AblationResult(test_name="baseline", n_samples=len(examples))
    print("\n" + "=" * 65)
    print("TEST 1: BASELINE (correct plan + correct problem)")
    print("=" * 65)

    for i, ex in enumerate(examples):
        problem_ids = torch.tensor(ex["problem_token_ids"], dtype=torch.long)
        plan_idx = torch.tensor(ex["plan_indices"], dtype=torch.long)
        code = generate_code(talker, tokenizer, problem_ids, plan_idx, device, max_length)
        result.generated_codes.append(code)

        parseable = try_ast_parse(code)
        if parseable:
            result.parse_successes += 1

        if i < 3:  # Show first 3 samples
            problem_text = tokenizer.decode(ex["problem_token_ids"], skip_special_tokens=True)
            print(f"\n  Sample {i + 1} | plan={ex['plan_indices']} | parseable={parseable}")
            print(f"  Problem: {problem_text[:120]}...")
            print_code_sample(code)

    print_result_summary(result)
    return result


def test_random_indices(
    talker: Talker,
    tokenizer,
    examples: list,
    device: torch.device,
    max_length: int,
) -> AblationResult:
    """Test 2: Random plan indices -> incoherent output."""
    result = AblationResult(test_name="random_indices", n_samples=len(examples))
    print("\n" + "=" * 65)
    print("TEST 2: RANDOM INDICES (garbage in -> garbage out)")
    print("=" * 65)

    for i, ex in enumerate(examples):
        problem_ids = torch.tensor(ex["problem_token_ids"], dtype=torch.long)
        # Random indices in [1, 511], same length as the real plan
        plan_len = len(ex["plan_indices"]) if ex["plan_indices"] else 5
        random_plan = torch.tensor(
            [random.randint(1, 511) for _ in range(plan_len)], dtype=torch.long,
        )
        code = generate_code(talker, tokenizer, problem_ids, random_plan, device, max_length)
        result.generated_codes.append(code)

        parseable = try_ast_parse(code)
        if parseable:
            result.parse_successes += 1

        if i < 3:
            print(f"\n  Sample {i + 1} | random_plan={random_plan.tolist()} | parseable={parseable}")
            print_code_sample(code)

    print_result_summary(result)
    return result


def test_gaussian_noise(
    talker: Talker,
    tokenizer,
    vq: VectorQuantizer,
    examples: list,
    device: torch.device,
    max_length: int,
) -> AblationResult:
    """Test 3: Gaussian noise quantized via VQ -> incoherent output."""
    result = AblationResult(test_name="gaussian_noise", n_samples=len(examples))
    print("\n" + "=" * 65)
    print("TEST 3: GAUSSIAN NOISE (noise-derived indices)")
    print("=" * 65)

    for i, ex in enumerate(examples):
        problem_ids = torch.tensor(ex["problem_token_ids"], dtype=torch.long)
        plan_len = len(ex["plan_indices"]) if ex["plan_indices"] else 5

        # Quantize random noise through VQ to get noise-derived indices
        noise_indices = []
        for _ in range(plan_len):
            noise = F.normalize(torch.randn(vq.dim, device=device), dim=-1)
            # Use dot-product argmax (same as VQ forward path)
            dots = torch.matmul(vq.embedding.weight, noise)
            idx = dots.argmax().item()
            noise_indices.append(idx)

        noise_plan = torch.tensor(noise_indices, dtype=torch.long)
        code = generate_code(talker, tokenizer, problem_ids, noise_plan, device, max_length)
        result.generated_codes.append(code)

        parseable = try_ast_parse(code)
        if parseable:
            result.parse_successes += 1

        if i < 3:
            print(f"\n  Sample {i + 1} | noise_indices={noise_indices} | parseable={parseable}")
            print_code_sample(code)

    print_result_summary(result)
    return result


def test_semantic_mismatch(
    talker: Talker,
    tokenizer,
    examples: list,
    device: torch.device,
    max_length: int,
) -> AblationResult:
    """Test 4: Plan from problem A + text from problem B -> follows plan A."""
    result = AblationResult(test_name="semantic_mismatch", n_samples=len(examples))
    print("\n" + "=" * 65)
    print("TEST 4: SEMANTIC MISMATCH (plan from wrong problem)")
    print("=" * 65)

    n = len(examples)
    for i in range(n):
        # Use problem text from example i, but plan from a different example
        j = (i + n // 2) % n  # Pick a maximally distant example
        if j == i:
            j = (i + 1) % n

        problem_ids = torch.tensor(examples[i]["problem_token_ids"], dtype=torch.long)
        wrong_plan = torch.tensor(examples[j]["plan_indices"], dtype=torch.long)

        code = generate_code(talker, tokenizer, problem_ids, wrong_plan, device, max_length)
        result.generated_codes.append(code)

        parseable = try_ast_parse(code)
        if parseable:
            result.parse_successes += 1

        if i < 3:
            prob_i = tokenizer.decode(examples[i]["problem_token_ids"], skip_special_tokens=True)
            prob_j = tokenizer.decode(examples[j]["problem_token_ids"], skip_special_tokens=True)
            print(f"\n  Sample {i + 1} | parseable={parseable}")
            print(f"  Text from:  {prob_i[:100]}...")
            print(f"  Plan from:  {prob_j[:100]}...")
            print(f"  Plan indices: {examples[j]['plan_indices']}")
            print_code_sample(code)

    print_result_summary(result)
    return result


def test_empty_plan(
    talker: Talker,
    tokenizer,
    examples: list,
    device: torch.device,
    max_length: int,
) -> AblationResult:
    """Test 5: Empty plan -> minimal/degenerate output."""
    result = AblationResult(test_name="empty_plan", n_samples=len(examples))
    print("\n" + "=" * 65)
    print("TEST 5: EMPTY PLAN (no plan indices)")
    print("=" * 65)

    for i, ex in enumerate(examples):
        problem_ids = torch.tensor(ex["problem_token_ids"], dtype=torch.long)
        empty_plan = torch.tensor([], dtype=torch.long)

        code = generate_code(talker, tokenizer, problem_ids, empty_plan, device, max_length)
        result.generated_codes.append(code)

        parseable = try_ast_parse(code)
        if parseable:
            result.parse_successes += 1

        if i < 3:
            print(f"\n  Sample {i + 1} | parseable={parseable} | len={len(code)} chars")
            print_code_sample(code)

    print_result_summary(result)
    return result


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def print_final_report(results: List[AblationResult]) -> None:
    """Print a summary table of all ablation results."""
    print("\n" + "=" * 65)
    print("ABLATION SUMMARY")
    print("=" * 65)
    print(f"  {'Test':<22} {'Parse Rate':>12} {'Parseable':>10} {'Total':>8}")
    print(f"  {'-' * 22} {'-' * 12} {'-' * 10} {'-' * 8}")

    for r in results:
        print(
            f"  {r.test_name:<22} {r.parse_rate * 100:>11.1f}% "
            f"{r.parse_successes:>10} {r.n_samples:>8}"
        )

    print(f"\n  {'=' * 54}")

    # Interpret results
    baseline = results[0]
    degraded = results[1:]

    print("\n  INTERPRETATION:")
    if baseline.parse_rate < 0.5:
        print("  WARNING: Baseline parse rate < 50%. Talker may be undertrained.")
    else:
        print(f"  Baseline parse rate: {baseline.parse_rate * 100:.1f}% (healthy)")

    for r in degraded:
        drop = baseline.parse_rate - r.parse_rate
        if r.parse_rate >= baseline.parse_rate * 0.9 and baseline.parse_rate > 0.5:
            print(
                f"  CONCERN: {r.test_name} parse rate ({r.parse_rate * 100:.1f}%) "
                f"is close to baseline ({baseline.parse_rate * 100:.1f}%). "
                f"Talker may be bypassing the plan bottleneck."
            )
        elif drop > 0.1:
            print(
                f"  OK: {r.test_name} shows {drop * 100:.1f}pp drop vs baseline "
                f"-> Talker depends on plan quality."
            )

    # Overall verdict
    bypass_detected = any(
        r.parse_rate >= baseline.parse_rate * 0.9 and baseline.parse_rate > 0.5
        for r in degraded
    )
    if bypass_detected:
        print("\n  VERDICT: POTENTIAL DECOUPLING FAILURE — review Talker capacity.")
    else:
        print("\n  VERDICT: Decoupling looks healthy. Talker depends on plan indices.")

    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Talker ablation tests (contract_3 Section 5)."
    )
    parser.add_argument(
        "--talker_checkpoint", type=str, required=True,
        help="Path to talker checkpoint (e.g. talker_final.pt)",
    )
    parser.add_argument(
        "--sst_checkpoint_dir", type=str, required=True,
        help="Directory with sst_reasoner_*.pt and vq_codebook_*.pt (for Gaussian noise test)",
    )
    parser.add_argument(
        "--sst_checkpoint_tag", type=str, default="final",
        help="Checkpoint tag for SST models (default: final)",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to talker_dataset (Arrow format from prepare_talker_data)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=50,
        help="Number of samples per test (default: 50)",
    )
    parser.add_argument(
        "--max_length", type=int, default=1024,
        help="Max generation length in tokens (default: 1024)",
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="Device override: cuda / cpu / mps (auto-detected)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # ── Seed ──────────────────────────────────────────────────────────────
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Device ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        "bigcode/starcoder2-3b", trust_remote_code=True,
    )
    vocab_size = tokenizer.vocab_size

    # ── Load Talker ───────────────────────────────────────────────────────
    print(f"Loading Talker from {args.talker_checkpoint}")
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
    ckpt = torch.load(args.talker_checkpoint, map_location="cpu", weights_only=True)
    talker.load_state_dict(ckpt)
    talker.to(device).eval()
    n_params = sum(p.numel() for p in talker.parameters())
    print(f"Talker: {n_params:,} params")

    # ── Load VQ (for Gaussian noise test) ─────────────────────────────────
    print(f"Loading VQ from {args.sst_checkpoint_dir} (tag={args.sst_checkpoint_tag})")
    _, vq = load_frozen_models(
        checkpoint_dir=args.sst_checkpoint_dir,
        tag=args.sst_checkpoint_tag,
        vocab_size=vocab_size,
        device=device,
    )
    print(f"VQ: codebook {vq.codebook_size}x{vq.dim}")

    # ── Load test examples ────────────────────────────────────────────────
    print(f"Loading dataset from {args.dataset_path}")
    from datasets import load_from_disk, load_dataset

    try:
        ds = load_from_disk(args.dataset_path)
    except Exception:
        ds = load_dataset("arrow", data_dir=args.dataset_path, split="train")

    # Sample n_samples examples, spread across the dataset
    n = min(args.n_samples, len(ds))
    if n < len(ds):
        step = len(ds) // n
        indices = [i * step for i in range(n)]
        ds_subset = ds.select(indices)
    else:
        ds_subset = ds

    examples = [
        {
            "problem_token_ids": list(row["problem_token_ids"]),
            "plan_indices": list(row["plan_indices"]),
            "solution_token_ids": list(row["solution_token_ids"]),
        }
        for row in ds_subset
    ]
    print(f"Loaded {len(examples)} test examples")

    # ── Run ablation tests ────────────────────────────────────────────────
    t0 = time.time()
    results = []

    results.append(test_baseline(talker, tokenizer, examples, device, args.max_length))
    results.append(test_random_indices(talker, tokenizer, examples, device, args.max_length))
    results.append(test_gaussian_noise(talker, tokenizer, vq, examples, device, args.max_length))
    results.append(test_semantic_mismatch(talker, tokenizer, examples, device, args.max_length))
    results.append(test_empty_plan(talker, tokenizer, examples, device, args.max_length))

    elapsed = time.time() - t0
    print(f"\nAll tests completed in {elapsed / 60:.1f} minutes")

    # ── Final report ──────────────────────────────────────────────────────
    print_final_report(results)


if __name__ == "__main__":
    main()
