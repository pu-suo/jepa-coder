#!/usr/bin/env bash
# JEPA-Coder: Vast.ai instance setup script
# Run once after provisioning an H100 instance.
# Usage: bash scripts/setup_vast.sh

set -euo pipefail

echo "=== JEPA-Coder Vast.ai Setup ==="

# 1. System dependencies
apt-get update && apt-get install -y tmux htop nvtop git

# 2. Create workspace
mkdir -p /workspace/jepa-coder-data/{data,checkpoints/pretrain,checkpoints/sst}

# 3. Clone repo (or pull if exists)
if [ ! -d /workspace/jepa-coder ]; then
    git clone https://github.com/pu-suo/jepa-coder.git /workspace/jepa-coder
else
    cd /workspace/jepa-coder && git pull
fi

# 4. Install Python dependencies
cd /workspace/jepa-coder
pip install --upgrade pip
pip install -r requirements.txt

# 5. Load environment variables
ENV_FILE="/workspace/jepa-coder-data/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found."
    echo "Create it with:"
    echo '  echo "HF_TOKEN=hf_your_token_here" > /workspace/jepa-coder-data/.env'
    echo '  echo "WANDB_API_KEY=your_key_here" >> /workspace/jepa-coder-data/.env'
    exit 1
fi
set -a && source "$ENV_FILE" && set +a

# 6. Verify authentication
python -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False); print('HuggingFace auth OK')"
wandb login "$WANDB_API_KEY" && echo "W&B auth OK"

# 7. Verify GPU
python -c "import torch; assert torch.cuda.is_available(), 'No GPU!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo "=== Setup complete. Start tmux and run training scripts. ==="
