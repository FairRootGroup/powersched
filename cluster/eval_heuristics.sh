#!/usr/bin/env bash
# Evaluate rule-based controllers (src/heuristics.py) like the trained policies: each price year in
# YEARS, the training trace, 364 days, at each arrival scale, over the evaluation seeds.
#
#   cluster/eval_heuristics.sh "demand:headroom=0 price:quantile=0.5,max_wait=12" [output dir, default sessions/heuristics]
#   YEARS="2023 2024 2025"  price years
#   SCALES="1.5 1.0"  arrival scales
#   SEEDS=100,200,301  evaluation seeds
#   TAG=test  result tag
#   PARALLEL=6  evaluations at once (one process per seed each)
#
# Results: <output dir>/test<year>/eval/<spec>_x<scale>_<TAG>.json (spec with ':' ',' '=' replaced).
# Existing results are skipped, so rerun after an interruption.
set -euo pipefail

SPECS="$1"
OUTPUT_DIR="${2:-sessions/heuristics}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
if [[ -n "${APPTAINER_CONTAINER:-}" ]]; then PY=python; else PY="$REPO/venv/bin/python"; fi
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
YEARS="${YEARS:-2023 2024 2025}"
SCALES="${SCALES:-1.5 1.0}"
SEEDS="${SEEDS:-100,200,301}"
TAG="${TAG:-test}"
PARALLEL="${PARALLEL:-6}"
TRACE="${TRACE:-data-internal/allusers-main-30.log}"

jobs_file="$(mktemp)"
trap 'rm -f "$jobs_file"' EXIT
for spec in $SPECS; do
    name="$(tr ':,=' '__-' <<< "$spec")"
    for year in $YEARS; do
        mkdir -p "$OUTPUT_DIR/test$year/eval/logs"
        for scale in $SCALES; do
            out="$OUTPUT_DIR/test$year/eval/${name}_x${scale}_${TAG}.json"
            [[ -f "$out" ]] || echo "$spec $name $year $scale $out" >> "$jobs_file"
        done
    done
done
echo "$(wc -l < "$jobs_file") evaluations (seeds $SEEDS), $PARALLEL at a time"

run_one() {
    local spec="$1" name="$2" year="$3" scale="$4" out="$5"
    local n_seeds; n_seeds="$(tr ',' '\n' <<< "$SEEDS" | wc -l)"
    "$PY" analyze_eval_seeds.py --seeds "$SEEDS" --parallel "$n_seeds" --tag "${name}_${TAG}_x${scale}" \
        --out "$out" --log-dir "$OUTPUT_DIR/test$year/eval/logs" -- \
        --session "test$year" --output-dir "$OUTPUT_DIR" --heuristic "$spec" \
        --prices "data-external/prices_$year.csv" --hourly-jobs "$TRACE" \
        --job-arrival-scale "$scale" --eval-months 13 > "$OUTPUT_DIR/test$year/eval/${name}_x${scale}_${TAG}.log" 2>&1 \
        && echo "[$(date '+%F %T')] done  $name $year x$scale" \
        || echo "[$(date '+%F %T')] FAILED $name $year x$scale (see its log)"
}
export -f run_one
export PY OUTPUT_DIR TAG TRACE SEEDS
xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _ < "$jobs_file"
