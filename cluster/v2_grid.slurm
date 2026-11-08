#!/usr/bin/env bash
# v2 training grid as a Slurm array: 3 curricula x 3 seeds x 3 held-out price years = 27 runs,
# each a cluster/run_v2.sh call inside the Apptainer image (cluster/powersched-cpu.def).
#
# Array index i: curriculum = i / 9 (paper, de, e), seed = (i / 3) % 3 (1, 2, 3), test year = i % 3.
#
# The repository (with data-internal/allusers-main-30.log), the image and the outputs must be on a
# filesystem the worker nodes see (e.g. /lustre/rz/$USER/powersched); submit from that directory,
# since the job runs in $SLURM_SUBMIT_DIR and the --output path below is relative to it.
# Submit from the repository root (fill in partition/account for the cluster):
#   apptainer build powersched-cpu.sif cluster/powersched-cpu.def
#   mkdir -p sessions/v2/slurm                                      # Slurm needs the log directory
#   sbatch --partition=<partition> cluster/v2_grid.slurm             # all 27
#   sbatch --partition=<partition> --array=4,13 cluster/v2_grid.slurm  # rerun / resume some
# A run that hits the time limit resumes from its last checkpoint when its index is resubmitted.
#
# Environment: SIF (default ./powersched-cpu.sif), OUTPUT_DIR, N_ENVS, SMOKE (see run_v2.sh).
#SBATCH --job-name=powersched-v2
#SBATCH --array=0-26
#SBATCH --cpus-per-task=9
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=sessions/v2/slurm/%A_%a.out
set -euo pipefail

CURRICULA=(paper de e)
SEEDS=(1 2 3)
YEARS=(2023 2024 2025)
i="${SLURM_ARRAY_TASK_ID:?run with sbatch --array}"
CURRICULUM="${CURRICULA[$((i / 9))]}"
SEED="${SEEDS[$(((i / 3) % 3))]}"
YEAR="${YEARS[$((i % 3))]}"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
SIF="${SIF:-$REPO/powersched-cpu.sif}"
# The main process and the envs take turns (rollout vs. PPO update), so one core each is enough.
export N_ENVS="${N_ENVS:-$((SLURM_CPUS_PER_TASK - 1))}"
echo "task $i: curriculum=$CURRICULUM seed=$SEED test_year=$YEAR n_envs=$N_ENVS on $(hostname)"

# Record the code version on the host (the image has no git).
git -C "$REPO" log -1 --format='git %h %s' || true

# --no-home: worker nodes may not see the home directory; only the repository is bound.
# Matplotlib then needs a writable config directory.
exec apptainer exec --no-home --bind "$REPO" --pwd "$REPO" \
    --env OUTPUT_DIR="${OUTPUT_DIR:-sessions/v2}",N_ENVS="$N_ENVS",SMOKE="${SMOKE:-0}",MPLCONFIGDIR=/tmp/mpl-$USER \
    "$SIF" bash cluster/run_v2.sh "$CURRICULUM" "$SEED" "$YEAR"
