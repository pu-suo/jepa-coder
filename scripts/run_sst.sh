#!/usr/bin/env bash
# JEPA-Coder: Phase 2 — Self-Supervised Training (SST)
# Core training run: 40-60 hours on H100.
# Usage: tmux new -s sst && bash scripts/run_sst.sh
# Monitor: W&B dashboard (jepa-coder-sst project)

set -euo pipefail

# Load credentials
set -a && source /workspace/jepa-coder-data/.env && set +a
cd /workspace/jepa-coder

PRETRAINED_CKPT="/workspace/jepa-coder-data/checkpoints/pretrain/pretrained_reasoner.pt"

if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "ERROR: Pretrained checkpoint not found at $PRETRAINED_CKPT"
    echo "Run Phase 1 (scripts/run_pretrain.sh) first."
    exit 1
fi

echo "=== Phase 2: Self-Supervised Training (SST) ==="
echo "Pretrained checkpoint: $PRETRAINED_CKPT"
echo "Expected loss trajectory: ~4.0 → 0.5-1.0"
echo "Monitor on W&B — stop when SST loss stabilizes"
echo ""

python -m training.sst \
    --pretrained_checkpoint "$PRETRAINED_CKPT" \
    --output_dir /workspace/jepa-coder-data/checkpoints/sst \
    --wandb_project jepa-coder-sst

echo "=== SST complete ==="
echo "Checkpoints: /workspace/jepa-coder-data/checkpoints/sst/"
