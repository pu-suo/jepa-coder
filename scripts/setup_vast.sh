#!/usr/bin/env bash
# JEPA-Coder: Vast.ai instance setup script
# Run once after provisioning a fresh instance.
#
# Usage (token passed via env var):
#   HF_TOKEN=hf_xxx WANDB_API_KEY=xxx bash scripts/setup_vast.sh
#
# Usage (token read from .env file at /workspace/jepa-coder-data/.env):
#   bash scripts/setup_vast.sh
#
# .env format:
#   HF_TOKEN=hf_your_token_here
#   WANDB_API_KEY=your_key_here

set -euo pipefail

echo ""
echo "=========================================="
echo "  JEPA-Coder Vast.ai Bootstrap"
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
echo "[1/8] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq git python3 python3-pip tmux htop nvtop
echo "      System packages OK."

# ---------------------------------------------------------------------------
# 2. Python dependencies — huggingface_hub must be present (and upgraded)
#    before any hf operations below, so install it early.
# ---------------------------------------------------------------------------
echo "[2/8] Installing Python base dependencies..."
pip install --upgrade --quiet pip
pip install --quiet --upgrade huggingface_hub wandb
echo "      Python base deps OK."

# ---------------------------------------------------------------------------
# 3. Load credentials (env var takes priority over .env file)
# ---------------------------------------------------------------------------
echo "[3/8] Loading credentials..."

ENV_FILE="/workspace/jepa-coder-data/.env"

# Source .env only if the env vars are not already set
if [ -z "${HF_TOKEN:-}" ] || [ -z "${WANDB_API_KEY:-}" ]; then
    if [ -f "$ENV_FILE" ]; then
        echo "      Reading credentials from $ENV_FILE"
        set -a && source "$ENV_FILE" && set +a
    else
        echo ""
        echo "ERROR: Neither HF_TOKEN/WANDB_API_KEY env vars nor $ENV_FILE found."
        echo ""
        echo "Either:"
        echo "  1. Pass them inline:  HF_TOKEN=hf_xxx WANDB_API_KEY=xxx bash scripts/setup_vast.sh"
        echo "  2. Create $ENV_FILE:"
        echo "       mkdir -p /workspace/jepa-coder-data"
        echo "       echo \"HF_TOKEN=hf_your_token_here\"  > $ENV_FILE"
        echo "       echo \"WANDB_API_KEY=your_key_here\" >> $ENV_FILE"
        exit 1
    fi
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is empty. Check your .env file or env var."
    exit 1
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY is empty. Check your .env file or env var."
    exit 1
fi

echo "      Credentials loaded."

# ---------------------------------------------------------------------------
# 4. Authenticate
# ---------------------------------------------------------------------------
echo "[4/8] Authenticating with Hugging Face and W&B..."
python -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=True)"
echo "      HuggingFace auth OK."

wandb login "$WANDB_API_KEY"
echo "      W&B auth OK."

# ---------------------------------------------------------------------------
# 5. Directory structure
# ---------------------------------------------------------------------------
echo "[5/8] Creating directory structure..."

mkdir -p /workspace/jepa-coder
mkdir -p /workspace/jepa-coder-data/checkpoints/pretrain
mkdir -p /workspace/jepa-coder-data/checkpoints/sst
mkdir -p /workspace/jepa-coder-data/checkpoints/talker
mkdir -p /workspace/jepa-coder-data/data/sst_dataset
mkdir -p /workspace/jepa-coder-data/data/talker_dataset

echo "      Directories created:"
echo "        /workspace/jepa-coder"
echo "        /workspace/jepa-coder-data/checkpoints/pretrain"
echo "        /workspace/jepa-coder-data/checkpoints/sst"
echo "        /workspace/jepa-coder-data/checkpoints/talker"
echo "        /workspace/jepa-coder-data/data/sst_dataset"
echo "        /workspace/jepa-coder-data/data/talker_dataset"

# ---------------------------------------------------------------------------
# 6. Clone / update codebase
# ---------------------------------------------------------------------------
echo "[6/8] Pulling codebase..."

REPO_URL="https://github.com/pu-suo/jepa-coder.git"   # <-- update if repo URL changes

if [ ! -d /workspace/jepa-coder/.git ]; then
    git clone "$REPO_URL" /workspace/jepa-coder
    echo "      Cloned $REPO_URL -> /workspace/jepa-coder"
else
    echo "      Repo already present, pulling latest..."
    git -C /workspace/jepa-coder pull
fi

# Install project Python dependencies
echo "      Installing project requirements..."
pip install --quiet -r /workspace/jepa-coder/requirements.txt
echo "      Project requirements OK."

# ---------------------------------------------------------------------------
# 7. Hugging Face data pulls
# ---------------------------------------------------------------------------
echo "[7/8] Downloading datasets and checkpoints from Hugging Face..."

# --- Dataset -----------------------------------------------------------------
echo "      Downloading dataset pusuo2026/jepa-coder-dataset ..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('pusuo2026/jepa-coder-dataset', repo_type='dataset',
                  local_dir='/workspace/jepa-coder-data/data/')
"
echo "      Dataset download complete."

# --- Pretrain checkpoints ----------------------------------------------------
echo "      Downloading pretrain checkpoints pusuo2026/jepa-coder-checkpoints ..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('pusuo2026/jepa-coder-checkpoints', repo_type='model',
                  local_dir='/workspace/jepa-coder-data/checkpoints/pretrain/')
"
echo "      Pretrain checkpoint download complete."

# --- SST checkpoints ---------------------------------------------------------
echo "      Downloading SST checkpoints pusuo2026/jepa-coder-sst-checkpoint ..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('pusuo2026/jepa-coder-sst-checkpoint', repo_type='model',
                  local_dir='/workspace/jepa-coder-data/checkpoints/sst/')
"
echo "      SST checkpoint download complete."

# --- Talker checkpoints ------------------------------------------------------
echo "      Downloading Talker checkpoints pusuo2026/jepa-coder-talker-checkpoint ..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('pusuo2026/jepa-coder-talker-checkpoint', repo_type='model',
                  local_dir='/workspace/jepa-coder-data/checkpoints/talker/')
"
echo "      Talker checkpoint download complete."

# --- Talker dataset ----------------------------------------------------------
echo "      Downloading Talker dataset pusuo2026/jepa-coder-talker-dataset ..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('pusuo2026/jepa-coder-talker-dataset', repo_type='dataset',
                  local_dir='/workspace/jepa-coder-data/data/talker_dataset/')
" || echo "      WARNING: Talker dataset not found on HF. Run prepare_talker_data.py to generate it."
echo "      Talker dataset step complete."

# ---------------------------------------------------------------------------
# 8. Sanity checks
# ---------------------------------------------------------------------------
echo "[8/8] Running sanity checks..."

python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — check driver/runtime'
print(f'      GPU detected: {torch.cuda.get_device_name(0)}')
print(f'      CUDA version: {torch.version.cuda}')
"

echo ""
echo "=========================================="
echo "  Setup complete."
echo ""
echo "  Next steps:"
echo "    tmux new -s train"
echo "    cd /workspace/jepa-coder"
echo "    bash scripts/run_sst.sh        # resume SST"
echo "    bash scripts/run_pretrain.sh   # resume pretrain"
echo "=========================================="
echo ""
