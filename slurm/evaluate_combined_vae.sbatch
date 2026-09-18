#!/bin/bash
#SBATCH --job-name=tlpp_vae_eval
#SBATCH --account=marksta0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module load python
source "$HOME/tlpp_venv/bin/activate"
cd "$HOME/TLPP_embedding"
mkdir -p logs

GALLERY_COUNT="${GALLERY_COUNT:-50}"
python -m tlpp_embedding.cli evaluate-combined \
  --device cuda \
  --gallery-count "$GALLERY_COUNT"
