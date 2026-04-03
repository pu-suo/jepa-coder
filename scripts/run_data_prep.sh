#!/usr/bin/env bash
# JEPA-Coder: Data preparation pipeline
# Extracts and segments solutions from APPS, TACO, and OpenCodeReasoning.
# Usage: bash scripts/run_data_prep.sh

set -euo pipefail

# Load credentials
set -a && source /workspace/jepa-coder-data/.env && set +a
cd /workspace/jepa-coder

echo "=== Step 1: Extract Python solutions ==="
python data/extract_python_solutions.py \
    --output_dir /workspace/jepa-coder-data/data

echo "=== Step 2: Segment solutions into AST blocks ==="
python data/prepare_sst_data.py \
    --input /workspace/jepa-coder-data/data/extracted_solutions.jsonl \
    --output /workspace/jepa-coder-data/data/sst_dataset.jsonl

echo "=== Data preparation complete ==="
echo "Files written to /workspace/jepa-coder-data/data/"
ls -lh /workspace/jepa-coder-data/data/
