#!/usr/bin/env bash
# Evaluate finished de runs (cluster/run_v2.sh) on a further price year that no run trained or
# validated on: the best validated checkpoint of the last stage (no wait limit) and the last one,
# at arrival scales 1.5 and 1.0, over evaluation seeds 100/200/301 (SEEDS), as in the held-out evaluation.
#
#   cluster/eval_year.sh 2022 [sessions dir, default sessions/v2]
#   cluster/eval_year.sh heldout [...]  each run on its own held-out year (e.g. to add metrics to the
#                                       main evaluation; TAG is required then)
#   PARALLEL=7  evaluations at once (3 processes each)
#   PRICES=<csv>  other prices for that year (e.g. rescaled), with TAG=<name> to keep their results apart
#   SCALES="1.5 1.0"  arrival scales
#   STEPS=both  checkpoints: best (the pick), last (8M) or both
#   TRACE=<log>  workload trace (default: the training trace, data-internal/allusers-main-30.log)
#   SEEDS=100,200,301  evaluation seeds
#   CURRICULA=de  curricula whose runs are evaluated (e.g. "de e paper")
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
SCALES="${SCALES:-1.5 1.0}"
STEPS="${STEPS:-both}"
SEEDS="${SEEDS:-100,200,301}"
CURRICULA="${CURRICULA:-de}"
TRACE="${TRACE:-data-internal/allusers-main-30.log}"
[[ -f "$TRACE" ]] || { echo "no $TRACE" >&2; exit 1; }
if [[ "$YEAR" == heldout ]]; then
    [[ -n "${TAG:-}" && -z "${PRICES:-}" ]] || { echo "heldout needs TAG and no PRICES" >&2; exit 1; }
else
    PRICES="${PRICES:-data-external/prices_$YEAR.csv}"
    TAG="${TAG:-new$YEAR}"
    [[ -f "$PRICES" ]] || { echo "no $PRICES" >&2; exit 1; }
fi

jobs_file="$(mktemp)"
trap 'rm -f "$jobs_file"' EXIT
for session_dir in "$OUTPUT_DIR"/*; do
    session="$(basename "$session_dir")"
    [[ "$session" =~ ^(de|e|paper)(_e([0-9])p([0-9])s([0-9]))?_s[0-9]+_test(20[0-9]{2})$ ]] || continue
    [[ " $CURRICULA " == *" ${BASH_REMATCH[1]} "* ]] || continue
    test_year="${BASH_REMATCH[6]}"
    if [[ "$YEAR" == heldout ]]; then
        prices="data-external/prices_$test_year.csv"
    else
        [[ "$test_year" == "$YEAR" ]] && { echo "$session trained on $YEAR? skipped" >&2; continue; }
        prices="$PRICES"
    fi
    if [[ -n "${BASH_REMATCH[2]}" ]]; then
        W_EFF="0.${BASH_REMATCH[3]}"; W_PRICE="0.${BASH_REMATCH[4]}"; W_AGE="0.${BASH_REMATCH[5]}"
    else
        W_EFF=0.4; W_PRICE=0.4; W_AGE=0.2
    fi
    models_dir="$session_dir/models/e${W_EFF}_p${W_PRICE}_i0.0_a${W_AGE}_d0.0"
    best="$("$PY" -c "
from src.validation import load_history, best_per_budget, last_stage_start
e = best_per_budget(load_history('$models_dir'), after_step=last_stage_start('$session_dir'))['any']
print(e['step'] if e else '')")"
    case "$STEPS" in
        best) candidates="${best:-8000000}" ;;
        last) candidates=8000000 ;;
        both) candidates="$best 8000000" ;;
        *) echo "STEPS must be best, last or both" >&2; exit 1 ;;
    esac
    for step in $(printf '%s\n' $candidates | grep -v '^$' | sort -nu); do
        [[ -f "$models_dir/$step.zip" ]] || { echo "$session: no checkpoint $step" >&2; continue; }
        for scale in $SCALES; do
            out="$session_dir/eval/m${step}_x${scale}_${TAG}.json"
            [[ -f "$out" ]] && continue
            echo "$session $step $scale $W_EFF $W_PRICE $W_AGE $out $prices" >> "$jobs_file"
        done
    done
done
echo "$(wc -l < "$jobs_file") evaluations on $YEAR (${PRICES:-the held-out prices of each run}; $TRACE), $PARALLEL at a time"

run_one() {
    local session="$1" step="$2" scale="$3" w_eff="$4" w_price="$5" w_age="$6" out="$7" prices="$8"
    echo "[$(date '+%F %T')] start $session m$step x$scale"
    "$PY" analyze_eval_seeds.py --seeds "$SEEDS" --parallel "$(tr ',' '\n' <<< "$SEEDS" | wc -l)" --tag "m${step}_${TAG}_x${scale}" \
        --out "$out" --log-dir "$OUTPUT_DIR/$session/eval/logs" -- \
        --session "$session" --output-dir "$OUTPUT_DIR" \
        --efficiency-weight "$w_eff" --price-weight "$w_price" --idle-weight 0.0 --job-age-weight "$w_age" --drop-weight 0.0 \
        --model "$step" --prices "$prices" --hourly-jobs "$TRACE" \
        --job-arrival-scale "$scale" --eval-months 13 --oracle > "$OUTPUT_DIR/$session/eval/${TAG}_m${step}_x${scale}.log" 2>&1 \
        && echo "[$(date '+%F %T')] done  $session m$step x$scale" \
        || echo "[$(date '+%F %T')] FAILED $session m$step x$scale (see its log)"
}
export -f run_one
export PY OUTPUT_DIR YEAR TAG TRACE SEEDS
xargs -r -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _ < "$jobs_file"
