#!/usr/bin/env bash
# JEPA-Coder: Data preparation pipeline (rewritten)
#
# Peak disk usage with the new pipeline:
#   Step 1 (extract):  ~90 GB HF cache + 21 GB JSONL     = ~111 GB peak
#     then purges HF cache → 21 GB
#   Step 2 (segment):  21 GB JSONL + ~40 GB Arrow shards = ~82 GB peak
#     then deletes JSONL → 40 GB
#
# Final resting state: 40 GB Arrow dataset. No intermediate sst_dataset.jsonl.

set -euo pipefail

# Load credentials if present
if [[ -f /workspace/jepa-coder-data/.env ]]; then
    set -a && source /workspace/jepa-coder-data/.env && set +a
fi

# Pin HF caches to the data volume so they're easy to inspect and purge
export HF_HOME=/workspace/jepa-coder-data/hf_home
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_DATASETS_CACHE}"

cd /workspace/jepa-coder

echo "=== Disk before run ==="
df -h /workspace

echo ""
echo "=== Step 1: Extract Python solutions ==="
python data/extract_python_solutions.py \
    --output_dir /workspace/jepa-coder-data/data \
    --seed 42

echo ""
echo "=== Disk after extract (HF cache should be purged) ==="
df -h /workspace

echo ""
echo "=== Step 2: Segment solutions into AST blocks ==="
python data/prepare_sst_data.py \
    --input  /workspace/jepa-coder-data/data/extracted_solutions.jsonl \
    --output_dir /workspace/jepa-coder-data/data/sst_dataset

echo ""
echo "=== Disk after segment (JSONL should be deleted) ==="
df -h /workspace
ls -lh /workspace/jepa-coder-data/data/

echo ""
echo "=== Data preparation complete ==="
