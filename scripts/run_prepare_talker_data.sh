#!/usr/bin/env bash
# JEPA-Coder: Phase 3 — Talker Data Preparation
# Runs frozen Reasoner + VQ over the SST dataset to generate
# (problem, plan_indices, solution) triples for Talker training.
#
# Prerequisites:
#   - SST checkpoint (sst_reasoner_final.pt, vq_codebook_final.pt)
#   - SST dataset (Arrow format)
#   - extracted_solutions.jsonl (for original solution text)
#
# Usage: bash scripts/run_prepare_talker_data.sh

set -euo pipefail

# Load credentials if present
if [[ -f /workspace/jepa-coder-data/.env ]]; then
    set -a && source /workspace/jepa-coder-data/.env && set +a
fi

cd /workspace/jepa-coder

SST_CKPT="/workspace/jepa-coder-data/checkpoints/sst"
SST_DATA="/workspace/jepa-coder-data/data/sst_dataset"
SOLUTIONS="/workspace/jepa-coder-data/data/extracted_solutions.jsonl"
TALKER_DATA="/workspace/jepa-coder-data/data/talker_dataset"

# --- Prerequisite checks ---
if [ ! -f "$SST_CKPT/sst_reasoner_final.pt" ]; then
    echo "ERROR: SST Reasoner checkpoint not found at $SST_CKPT/sst_reasoner_final.pt"
    echo "Run SST training first (scripts/run_sst.sh)"
    exit 1
fi

if [ ! -d "$SST_DATA" ]; then
    echo "ERROR: SST dataset not found at $SST_DATA"
    echo "Run SST data prep first (scripts/run_sst_data_prep.sh)"
    exit 1
fi

if [ ! -f "$SOLUTIONS" ]; then
    echo "ERROR: extracted_solutions.jsonl not found at $SOLUTIONS"
    echo "This file is needed for original solution text (correct indentation)."
    echo "Re-run extraction: python data/extract_python_solutions.py --output_dir /workspace/jepa-coder-data/data"
    exit 1
fi

echo "=== Phase 3: Talker Data Preparation ==="
echo "SST checkpoint: $SST_CKPT"
echo "SST dataset:    $SST_DATA"
echo "Solutions JSONL: $SOLUTIONS"
echo "Output:         $TALKER_DATA"
echo ""

python -m data.prepare_talker_data \
    --checkpoint_dir "$SST_CKPT" \
    --checkpoint_tag final \
    --dataset_path "$SST_DATA" \
    --solutions_jsonl "$SOLUTIONS" \
    --output_dir "$TALKER_DATA" \
    --batch_size 512

echo ""
echo "=== Talker data preparation complete ==="
echo "Dataset: $TALKER_DATA"
echo "Next: bash scripts/run_talker.sh"
