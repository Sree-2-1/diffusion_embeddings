#!/bin/bash
#SBATCH --job-name=tlpp_vae_all
#SBATCH --account=marksta0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module load python
source "$HOME/tlpp_venv/bin/activate"
cd "$HOME/TLPP_embedding"
mkdir -p logs

EPOCHS="${EPOCHS:-50}"
WORKERS="${WORKERS:-6}"
MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-100}"
RUN_DIR="${RUN_DIR:-/nfs/turbo/coe-marksta/h9_fullscale/vae_runs/combined_subscale_fullscale_128x128_latent128_run1}"

python -m tlpp_embedding.cli train-combined \
  --run-dir "$RUN_DIR" \
  --epochs "$EPOCHS" \
  --num-workers "$WORKERS" \
  --max-runtime-minutes "$MAX_RUNTIME_MINUTES" \
  --device cuda
