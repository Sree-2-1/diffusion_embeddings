#!/bin/bash
#SBATCH --job-name=tlpp_split_finalize
#SBATCH --account=marksta0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module load python
source "$HOME/tlpp_venv/bin/activate"
cd "$HOME/TLPP_embedding"
mkdir -p logs

: "${SOURCE:?SOURCE is required}"
: "${DATASET:?DATASET is required}"

SHARD_SIZE="${SHARD_SIZE:-5000}"
DEST_DIR="${DEST_DIR:-$(dirname "$SOURCE")}"
WORK_DIR="${WORK_DIR:-$(dirname "$SOURCE")/.${DATASET}_TLPP_work}"
SCRATCH_ROOT="${TLPP_SCRATCH:-/scratch/marksta_root/marksta0/$USER}"
STAGE="$SCRATCH_ROOT/${DATASET}_TLPP_finalize"

PLAIN="${DATASET}_TLPPs.h5"
COMP="${DATASET}_TLPPs_compressed.h5"

rm -rf "$STAGE"
mkdir -p "$STAGE"

python universal_tlpp.py finalize-source "$SOURCE" \
  --dataset-name "$DATASET" \
  --work-dir "$WORK_DIR" \
  --output-dir "$STAGE" \
  --shard-size "$SHARD_SIZE"

python universal_tlpp.py verify-pair \
  "$STAGE/$PLAIN" "$STAGE/$COMP"

python universal_tlpp.py deep-verify "$STAGE/$COMP" --block 512

for name in "$PLAIN" "$COMP"; do
  if [[ -e "$DEST_DIR/$name" ]]; then
    echo "Refusing to overwrite existing final: $DEST_DIR/$name" >&2
    exit 1
  fi

  temp="$DEST_DIR/.${name}.copying"
  rm -f "$temp"

  rsync -ah --info=progress2 --partial \
    "$STAGE/$name" "$temp"

  src_hash=$(sha256sum "$STAGE/$name" | awk '{print $1}')
  dst_hash=$(sha256sum "$temp" | awk '{print $1}')

  if [[ "$src_hash" != "$dst_hash" ]]; then
    echo "SHA256 mismatch for $name" >&2
    exit 1
  fi

  mv "$temp" "$DEST_DIR/$name"
  echo "PUBLISHED $DEST_DIR/$name sha256=$src_hash"
done

rm -rf "$STAGE"
echo "FINALIZATION COMPLETE: $DATASET"
