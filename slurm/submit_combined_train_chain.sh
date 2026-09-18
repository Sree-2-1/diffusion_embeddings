#!/bin/bash
set -euo pipefail
cd "$HOME/TLPP_embedding"
mkdir -p logs

# By the way I made it so that the training is split across multiple jobs because the
# jobs have a 2 hour time limit I believe.
# Each segment exits cleanly before the Slurm wall time; the next segment resumes
# from last.pt.  Increase SEGMENTS if the queue/wall-time combination needs it.
SEGMENTS="${SEGMENTS:-12}"
previous="${START_AFTER_JOB:-}"

if [[ -n "$previous" ]]; then
  echo "Training starts only after job $previous succeeds."
fi

for i in $(seq 1 "$SEGMENTS"); do
  if [[ -z "$previous" ]]; then
    job=$(sbatch --parsable slurm/train_combined_vae.sbatch)
  else
    job=$(sbatch --parsable --dependency=afterok:"$previous" slurm/train_combined_vae.sbatch)
  fi
  echo "segment $i -> job $job"
  previous="$job"
done

echo "Final dependency-chain job: $previous"
