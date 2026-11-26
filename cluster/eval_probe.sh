#!/usr/bin/env bash
# Observation ablation of finished de runs (cluster/run_v2.sh): the best validated checkpoint of the
# last stage (no wait limit) or the last checkpoint (STEP) of every run, on its held-out price year, with one part of the
# observation altered (analyze_obs_probe.py, closed loop), over the evaluation seeds.
#
#   cluster/eval_probe.sh [sessions dir, default sessions/v2]
#   VARIANTS="full prices_roll12 prices_const time_const aggregates_zero queue_summary_zero"
#   SCALES="1.0"  arrival scales
#   SEEDS="100 200 301"  evaluation seeds
#   STEP=best  checkpoint: best (the selected one) or last (8M)
#   PARALLEL=16  evaluations at once (one process each, about 2 min per evaluation)
#
# Results: <session>/eval/probe/m<step>_x<scale>_<variant>_s<seed>.json; summarise with
# analyze_probe.py. Existing results are skipped, so rerun after an interruption.
set -euo pipefail

OUTPUT_DIR="${1:-sessions/v2}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
if [[ -n "${APPTAINER_CONTAINER:-}" ]]; then PY=python; else PY="$REPO/venv/bin/python"; fi
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
VARIANTS="${VARIANTS:-full prices_roll12 prices_const time_const aggregates_zero queue_summary_zero}"
SCALES="${SCALES:-1.0}"
SEEDS="${SEEDS:-100 200 301}"
PARALLEL="${PARALLEL:-16}"
STEP="${STEP:-best}"
[[ "$STEP" == best || "$STEP" == last ]] || { echo "STEP must be best or last" >&2; exit 1; }

jobs_file="$(mktemp)"
trap 'rm -f "$jobs_file"' EXIT
for session_dir in "$OUTPUT_DIR"/de_*; do
    session="$(basename "$session_dir")"
    [[ "$session" =~ ^de(_e([0-9])p([0-9])s([0-9]))?_s[0-9]+_test(20[0-9]{2})$ ]] || continue
    year="${BASH_REMATCH[5]}"
    if [[ -n "${BASH_REMATCH[1]}" ]]; then
        weights="0.${BASH_REMATCH[2]},0.${BASH_REMATCH[3]},0.0,0.${BASH_REMATCH[4]},0.0"
        models_dir="$session_dir/models/e0.${BASH_REMATCH[2]}_p0.${BASH_REMATCH[3]}_i0.0_a0.${BASH_REMATCH[4]}_d0.0"
    else
        weights="0.4,0.4,0.0,0.2,0.0"
        models_dir="$session_dir/models/e0.4_p0.4_i0.0_a0.2_d0.0"
    fi
    [[ "$STEP" == last ]] && step=8000000 || step="$("$PY" -c "
from src.validation import load_history, best_per_budget, last_stage_start
e = best_per_budget(load_history('$models_dir'), after_step=last_stage_start('$session_dir'))['any']
print(e['step'] if e else 8000000)")"
    [[ -f "$models_dir/$step.zip" ]] || { echo "$session: no checkpoint $step" >&2; continue; }
    mkdir -p "$session_dir/eval/probe"
    for scale in $SCALES; do for variant in $VARIANTS; do for seed in $SEEDS; do
        out="$session_dir/eval/probe/m${step}_x${scale}_${variant}_s${seed}.json"
        [[ -f "$out" ]] || echo "$models_dir/$step.zip $weights $year $scale $variant $seed $out" >> "$jobs_file"
    done; done; done
done
echo "$(wc -l < "$jobs_file") evaluations, $PARALLEL at a time"

run_one() {
    local model="$1" weights="$2" year="$3" scale="$4" variant="$5" seed="$6" out="$7"
    "$PY" analyze_obs_probe.py --model "$model" --weights "$weights" --prices "data-external/prices_$year.csv" \
        --variant "$variant" --seed "$seed" --job-arrival-scale "$scale" --out "$out" > "${out%.json}.log" 2>&1 \
        && rm -f "${out%.json}.log" \
        || echo "[$(date '+%F %T')] FAILED $out (see its log)"
}
export -f run_one
export PY
xargs -r -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _ < "$jobs_file"
echo "[$(date '+%F %T')] finished"
