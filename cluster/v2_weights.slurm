#!/usr/bin/env bash
# v2 weights round as a Slurm array: curriculum de, 4 weight combinations x 3 seeds x 3 held-out
# price years = 36 runs, each a cluster/run_v2.sh call inside the Apptainer image. e4p4s2 is not in
# the list: round 1's de runs (cluster/v2_grid.slurm) are that cell.
#
# Array index i: weights = i / 9 (WEIGHTS below), seed = (i / 3) % 3 (1, 2, 3), test year = i % 3.
#   0-8 e3p5s2, 9-17 e2p2s6, 18-26 e3p3s4, 27-35 e2p6s2
#
# Same requirements as cluster/v2_grid.slurm: repository, trace, image and outputs on a filesystem
# the worker nodes see; submit from the repository root.
#   mkdir -p sessions/v2/slurm
#   sbatch --partition=long cluster/v2_weights.slurm                  # all 36
#   sbatch --partition=long --array=9-17 cluster/v2_weights.slurm     # one weight combination
# A run that hits the time limit resumes from its last checkpoint when its index is resubmitted,
# e.g. with a follow-up array: sbatch --partition=long --dependency=afterany:<jobid> cluster/v2_weights.slurm
#
# Environment: SIF (default ./powersched-cpu.sif), OUTPUT_DIR, N_ENVS, SMOKE (see run_v2.sh).
#SBATCH --job-name=powersched-v2w
#SBATCH --array=0-35
#SBATCH --cpus-per-task=9
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=sessions/v2/slurm/%A_%a.out
set -euo pipefail

WEIGHTS=(e3p5s2 e2p2s6 e3p3s4 e2p6s2)
SEEDS=(1 2 3)
YEARS=(2023 2024 2025)
i="${SLURM_ARRAY_TASK_ID:?run with sbatch --array}"
W="${WEIGHTS[$((i / 9))]}"
SEED="${SEEDS[$(((i / 3) % 3))]}"
YEAR="${YEARS[$((i % 3))]}"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
SIF="${SIF:-$REPO/powersched-cpu.sif}"
# The main process and the envs take turns (rollout vs. PPO update), so one core each is enough.
export N_ENVS="${N_ENVS:-$((SLURM_CPUS_PER_TASK - 1))}"
echo "task $i: curriculum=de weights=$W seed=$SEED test_year=$YEAR n_envs=$N_ENVS on $(hostname)"

# Record the code version on the host (the image has no git).
git -C "$REPO" log -1 --format='git %h %s' || true

# --no-home: worker nodes may not see the home directory; only the repository is bound.
# Matplotlib then needs a writable config directory.
exec apptainer exec --no-home --bind "$REPO" --pwd "$REPO" \
    --env OUTPUT_DIR="${OUTPUT_DIR:-sessions/v2}",N_ENVS="$N_ENVS",SMOKE="${SMOKE:-0}",MPLCONFIGDIR=/tmp/mpl-$USER \
    "$SIF" bash cluster/run_v2.sh de "$SEED" "$YEAR" "$W"
