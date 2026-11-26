#!/usr/bin/env bash
# All evaluations behind the paper's results: the final checkpoint (8M steps) of every run, evaluation
# seeds 101/201/301, 364 days. Results carry the tag prefix f101_ next to earlier evaluations.
#
#   cluster/eval_final.sh <part> [sessions dir, default sessions/v2]
#   parts:
#     main        training trace: held-out years (curricula de, e, paper; x1.0, x1.5), 2022 (x1.0, x1.5),
#                 2022 rescaled to the training price level (x1.5), training years of each run (x1.0)
#     trace       second trace (data/allusers-main-30-2026-11.log): held-out years and 2022, x0.6/0.9/1.0/1.5
#     probe       observation ablation on the held-out years (x1.0)
#     heuristics  rule-based controller settings of the paper (held-out years, x1.0, x1.5)
#     all         all of the above, in this order
#   PARALLEL=7  evaluations at once (3 processes each; the probe part runs 3 * PARALLEL single runs)
#
# Parts write separate files, so they can run on different machines that share the sessions directory.
# Existing results are skipped, so rerun after an interruption.
set -euo pipefail

PART="$1"
OUTPUT_DIR="${2:-sessions/v2}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PARALLEL="${PARALLEL:-7}" SEEDS=101,201,301 STEPS=last
SECOND_TRACE=data/allusers-main-30-2026-11.log
SCALED_2022="$OUTPUT_DIR/prices_2022_scaled.csv"  # 2022 prices / 2.686: mean of the training years

run_main() {
    CURRICULA="de e paper" TAG=f101_heldout cluster/eval_year.sh heldout "$OUTPUT_DIR"
    TAG=f101_new2022 cluster/eval_year.sh 2022 "$OUTPUT_DIR"
    PRICES="$SCALED_2022" TAG=f101_scaled2022 SCALES=1.5 cluster/eval_year.sh 2022 "$OUTPUT_DIR"
    for year in 2023 2024 2025; do  # runs trained on this year (its held-out runs are skipped)
        TAG="f101_in$year" SCALES=1.0 cluster/eval_year.sh "$year" "$OUTPUT_DIR"
    done
}
run_trace() {
    TRACE="$SECOND_TRACE" SCALES="0.6 0.9 1.0 1.5" TAG=f101_trace2611 cluster/eval_year.sh heldout "$OUTPUT_DIR"
    TRACE="$SECOND_TRACE" SCALES="0.6 0.9 1.0 1.5" TAG=f101_trace2611_new2022 cluster/eval_year.sh 2022 "$OUTPUT_DIR"
}
run_probe() {
    STEP=last SEEDS="101 201 301" PARALLEL=$((3 * PARALLEL)) cluster/eval_probe.sh "$OUTPUT_DIR"
}
run_heuristics() {
    local specs="demand:headroom=0 price:quantile=0.5,max_wait=12 price:quantile=0.25,max_wait=12"
    specs="$specs price:quantile=0.25,max_wait=48 price:quantile=0,max_wait=24 price:quantile=0,max_wait=48"
    TAG=f101 cluster/eval_heuristics.sh "$specs"
}

case "$PART" in
    main) run_main ;;
    trace) run_trace ;;
    probe) run_probe ;;
    heuristics) run_heuristics ;;
    all) run_main; run_trace; run_probe; run_heuristics ;;
    *) echo "part must be main, trace, probe, heuristics or all" >&2; exit 1 ;;
esac
