#!/usr/bin/env bash
# JEPA-Coder: Phase 1 — Short Pretraining Run
# Establishes angular alignment in embeddings (10K-15K steps).
# Usage: tmux new -s pretrain && bash scripts/run_pretrain.sh
# Monitor: W&B dashboard (jepa-coder-pretrain project)

set -euo pipefail

# Load credentials
set -a && source /workspace/jepa-coder-data/.env && set +a
cd /workspace/jepa-coder

echo "=== Phase 1: Short Pretraining Run ==="
echo "Target: 10K-15K steps, stop on loss convergence"
echo "Monitor loss curve on W&B — stop when it flattens"
echo ""

python -m training.pretrain \
    --checkpoint_dir /workspace/jepa-coder-data/checkpoints/pretrain \
    --data_dir /workspace/jepa-coder-data/data \
    --max_steps 15000 \
    --wandb_project jepa-coder-pretrain \
    --log_every 10

echo "=== Pretraining run complete ==="
echo "Checkpoint: /workspace/jepa-coder-data/checkpoints/pretrain/pretrained_reasoner.pt"
