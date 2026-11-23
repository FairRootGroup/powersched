#!/usr/bin/env bash
# Evaluate finished de runs (cluster/run_v2.sh) on a further price year that no run trained or
# validated on: the best validated checkpoint of the last stage (no wait limit) and the last one,
# at arrival scales 1.5 and 1.0, over evaluation seeds 100/200/301, as in the held-out evaluation.
#
#   cluster/eval_year.sh 2022 [sessions dir, default sessions/v2]
#   PARALLEL=7  evaluations at once (3 processes each)
#   PRICES=<csv>  other prices for that year (e.g. rescaled), with TAG=<name> to keep their results apart
#   SCALES="1.5 1.0"  arrival scales
#
# Results: <session>/eval/m<step>_x<scale>_<TAG>.json (TAG default new<year>). Existing results are skipped, so rerun after an interruption.
set -euo pipefail

YEAR="$1"
OUTPUT_DIR="${2:-sessions/v2}"
PARALLEL="${PARALLEL:-7}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
if [[ -n "${APPTAINER_CONTAINER:-}" ]]; then PY=python; else PY="$REPO/venv/bin/python"; fi
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PRICES="${PRICES:-data-external/prices_$YEAR.csv}"
TAG="${TAG:-new$YEAR}"
SCALES="${SCALES:-1.5 1.0}"
[[ -f "$PRICES" ]] || { echo "no $PRICES" >&2; exit 1; }

jobs_file="$(mktemp)"
trap 'rm -f "$jobs_file"' EXIT
for session_dir in "$OUTPUT_DIR"/de_*; do
    session="$(basename "$session_dir")"
    [[ "$session" =~ ^de(_e([0-9])p([0-9])s([0-9]))?_s[0-9]+_test(20[0-9]{2})$ ]] || continue
    [[ "${BASH_REMATCH[5]}" == "$YEAR" ]] && { echo "$session trained on $YEAR? skipped" >&2; continue; }
    if [[ -n "${BASH_REMATCH[1]}" ]]; then
        W_EFF="0.${BASH_REMATCH[2]}"; W_PRICE="0.${BASH_REMATCH[3]}"; W_AGE="0.${BASH_REMATCH[4]}"
    else
        W_EFF=0.4; W_PRICE=0.4; W_AGE=0.2
    fi
    models_dir="$session_dir/models/e${W_EFF}_p${W_PRICE}_i0.0_a${W_AGE}_d0.0"
    best="$("$PY" -c "
from src.validation import load_history, best_per_budget, last_stage_start
e = best_per_budget(load_history('$models_dir'), after_step=last_stage_start('$session_dir'))['any']
print(e['step'] if e else '')")"
    for step in $(printf '%s\n' "$best" 8000000 | grep -v '^$' | sort -nu); do
        [[ -f "$models_dir/$step.zip" ]] || { echo "$session: no checkpoint $step" >&2; continue; }
        for scale in $SCALES; do
            out="$session_dir/eval/m${step}_x${scale}_${TAG}.json"
            [[ -f "$out" ]] && continue
            echo "$session $step $scale $W_EFF $W_PRICE $W_AGE $out" >> "$jobs_file"
        done
    done
done
echo "$(wc -l < "$jobs_file") evaluations on $YEAR ($PRICES), $PARALLEL at a time"

run_one() {
    local session="$1" step="$2" scale="$3" w_eff="$4" w_price="$5" w_age="$6" out="$7"
    echo "[$(date '+%F %T')] start $session m$step x$scale"
    "$PY" analyze_eval_seeds.py --seeds 100,200,301 --parallel 3 --tag "m${step}_${TAG}_x${scale}" \
        --out "$out" --log-dir "$OUTPUT_DIR/$session/eval/logs" -- \
        --session "$session" --output-dir "$OUTPUT_DIR" \
        --efficiency-weight "$w_eff" --price-weight "$w_price" --idle-weight 0.0 --job-age-weight "$w_age" --drop-weight 0.0 \
        --model "$step" --prices "$PRICES" --hourly-jobs data-internal/allusers-main-30.log \
        --job-arrival-scale "$scale" --eval-months 13 --oracle > "$OUTPUT_DIR/$session/eval/${TAG}_m${step}_x${scale}.log" 2>&1 \
        && echo "[$(date '+%F %T')] done  $session m$step x$scale" \
        || echo "[$(date '+%F %T')] FAILED $session m$step x$scale (see its log)"
}
export -f run_one
export PY OUTPUT_DIR YEAR PRICES TAG
xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _ < "$jobs_file"
