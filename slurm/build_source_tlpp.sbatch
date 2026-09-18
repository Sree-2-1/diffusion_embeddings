#!/bin/bash
#SBATCH --job-name=tlpp_split_build
#SBATCH --account=marksta0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
module load python
source "$HOME/tlpp_venv/bin/activate"
cd "$HOME/TLPP_embedding"
mkdir -p logs

: "${SOURCE:?SOURCE is required}"
: "${DATASET:?DATASET is required}"

SHARD_SIZE="${SHARD_SIZE:-5000}"
READ_BLOCK="${READ_BLOCK:-16}"
WORK_DIR="${WORK_DIR:-$(dirname "$SOURCE")/.${DATASET}_TLPP_work}"

python universal_tlpp.py build-source-shard "$SOURCE" \
  --dataset-name "$DATASET" \
  --work-dir "$WORK_DIR" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --shard-size "$SHARD_SIZE" \
  --read-block "$READ_BLOCK"
