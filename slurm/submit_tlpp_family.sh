#!/bin/bash
# Submit TLPP shard arrays + finalizers for a family of source HDF5 files.
#
# Example:
#   SOURCE_ROOT=/path/to/raw \
#   OUTPUT_ROOT=/path/to/tlpps \
#   DATASETS="h9_train h9_val_small h9_val_large" \
#   bash slurm/submit_tlpp_family.sh
set -euo pipefail

cd "$HOME/TLPP_embedding"
mkdir -p logs
module load python
source "$HOME/tlpp_venv/bin/activate"

: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${DATASETS:?DATASETS is required (space-separated dataset stems)}"

SHARD_SIZE="${SHARD_SIZE:-5000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-12}"
mkdir -p "$OUTPUT_ROOT"

for dataset in $DATASETS; do
  source="$SOURCE_ROOT/${dataset}.h5"
  work="$OUTPUT_ROOT/.${dataset}_TLPP_work"
  shards=$(python - "$source" "$SHARD_SIZE" <<'PY'
import sys
from pathlib import Path
import universal_tlpp as u
print(u.source_shard_count(Path(sys.argv[1]), int(sys.argv[2])))
PY
)
  last=$((shards - 1))

  echo "Submitting $dataset: source=$source shards=$shards"
  build_job=$(sbatch --parsable \
    --array="0-${last}%${MAX_CONCURRENT}" \
    --export="ALL,SOURCE=$source,DATASET=$dataset,WORK_DIR=$work,SHARD_SIZE=$SHARD_SIZE" \
    slurm/build_source_tlpp.sbatch)

  finalize_job=$(sbatch --parsable \
    --dependency="afterok:${build_job}" \
    --export="ALL,SOURCE=$source,DATASET=$dataset,WORK_DIR=$work,DEST_DIR=$OUTPUT_ROOT,SHARD_SIZE=$SHARD_SIZE" \
    slurm/finalize_source_tlpp.sbatch)

  echo "  build array: $build_job"
  echo "  finalizer:   $finalize_job"
done
