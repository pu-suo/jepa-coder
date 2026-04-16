#!/usr/bin/env bash
# JEPA-Coder: Phase 3 — Talker Training
# Trains the Talker on pre-generated (problem, plan, code) triples.
# Prerequisites: run prepare_talker_data.py first (Phase 3 data prep).
# Usage: tmux new -s talker && bash scripts/run_talker.sh
# Monitor: W&B dashboard (jepa-coder-talker project)

set -euo pipefail

# Load credentials
set -a && source /workspace/jepa-coder-data/.env && set +a
cd /workspace/jepa-coder

TALKER_DATA="/workspace/jepa-coder-data/data/talker_dataset"

if [ ! -d "$TALKER_DATA" ]; then
    echo "ERROR: Talker dataset not found at $TALKER_DATA"
    echo "Run prepare_talker_data.py first:"
    echo "  python -m training.prepare_talker_data \\"
    echo "      --checkpoint_dir /workspace/jepa-coder-data/checkpoints/sst \\"
    echo "      --checkpoint_tag final \\"
    echo "      --dataset_path  /workspace/jepa-coder-data/data/sst_dataset \\"
    echo "      --output_dir    $TALKER_DATA"
    exit 1
fi

echo "=== Phase 3: Talker Training ==="
echo "Dataset: $TALKER_DATA"
echo "Monitor on W&B — stop when loss stabilizes"
echo ""

python -m training.train_talker \
    --dataset_path "$TALKER_DATA" \
    --output_dir /workspace/jepa-coder-data/checkpoints/talker \
    --batch_size 16 \
    --max_epochs 20 \
    --lr 3e-4 \
    --warmup_steps 500 \
    --wandb_project jepa-coder-talker \
    --log_every 50 \
    --checkpoint_every 2000

echo "=== Talker training complete ==="
echo "Checkpoint: /workspace/jepa-coder-data/checkpoints/talker/talker_final.pt"
