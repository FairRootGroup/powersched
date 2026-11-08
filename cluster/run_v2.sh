#!/usr/bin/env bash
# One v2 training run: a curriculum for one seed and one held-out price year, then the final
# evaluation on the held-out year. Runs in the venv or inside the Apptainer image (cluster/v2_grid.slurm).
#
# Usage: cluster/run_v2.sh <paper|de|e> <seed> <test-year: 2023|2024|2025>
#   paper  A-E as in the paper: A flat 150 jobs/h, B flat 1,000, C flat 600 + 5% small bursts,
#          D main trace x1.5 (logic prices), E main trace x1.5 on the two training years
#   de     D then E, same total steps
#   e      E only, arrival scale drawn per episode from 0.5-2.0
# Stage ends are cumulative iterations (1 iteration = 100K steps); 8M steps in total.
#
# Environment:
#   OUTPUT_DIR  base for sessions (default: sessions/v2)
#   N_ENVS      parallel environments per run (default: 8; the job needs N_ENVS + 1 cores)
#   SMOKE=1     short stages (1-6 iterations), validation every iteration, 1-month evaluation
#
# Resumable: train.py continues from the latest checkpoint, finished stages are skipped, and the
# final evaluation is skipped if its result exists. Rerun the same arguments after a time-out.
set -euo pipefail

CURRICULUM="$1"
SEED="$2"
TEST_YEAR="$3"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

OUTPUT_DIR="${OUTPUT_DIR:-sessions/v2}"
N_ENVS="${N_ENVS:-8}"
SMOKE="${SMOKE:-0}"
if [[ -n "${APPTAINER_CONTAINER:-}" ]]; then
    PY=python
else
    PY="$REPO/venv/bin/python"
fi
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

case "$TEST_YEAR" in 2023|2024|2025) ;; *) echo "test year must be 2023, 2024 or 2025" >&2; exit 1 ;; esac
TRAIN_PRICES=""
for year in 2023 2024 2025; do
    [[ "$year" == "$TEST_YEAR" ]] && continue
    TRAIN_PRICES="${TRAIN_PRICES:+$TRAIN_PRICES,}data-external/prices_$year.csv"
done
TEST_PRICES="data-external/prices_$TEST_YEAR.csv"
TRACE_FILE=data-internal/allusers-main-30.log

SESSION="${CURRICULUM}_s${SEED}_test${TEST_YEAR}"
[[ "$SMOKE" == "1" ]] && SESSION+="_smoke"
SESSION_DIR="$OUTPUT_DIR/$SESSION"
mkdir -p "$SESSION_DIR"

WEIGHTS=(--efficiency-weight 0.4 --price-weight 0.4 --idle-weight 0.0 --job-age-weight 0.2 --drop-weight 0.0)
MODELS_DIR="$SESSION_DIR/models/e0.4_p0.4_i0.0_a0.2_d0.0"

if [[ "$SMOKE" == "1" ]]; then
    ENDS=(1 2 3 4 6); VAL_EVERY=1; EVAL_MONTHS=1; TOTAL_STEPS=600000
else
    ENDS=(10 20 30 40 80); VAL_EVERY=5; EVAL_MONTHS=13; TOTAL_STEPS=8000000
fi

COMMON=(--session "$SESSION" --output-dir "$OUTPUT_DIR" --seed "$SEED" "${WEIGHTS[@]}"
        --net-arch 64,64 --device cpu --n-envs "$N_ENVS"
        --n-steps $((2048 / N_ENVS)) --batch-size 256 --n-epochs 5 --target-kl 0.02
        --learning-rate 3e-4 --ent-coef 0.002 --decay-steps "$TOTAL_STEPS" --normalize-reward
        --flush-after-drop-streak 3
        --val-every "$VAL_EVERY" --val-hourly-jobs "$TRACE_FILE" --val-prices "$TRAIN_PRICES")
TRACE=(--hourly-jobs "$TRACE_FILE" --job-arrival-scale 1.5)
LOGIC=(--prices "")              # synthetic logic prices
REAL=(--prices "$TRAIN_PRICES")  # the two training years

{
    echo "[$(date '+%F %T')] start curriculum=$CURRICULUM seed=$SEED test=$TEST_YEAR smoke=$SMOKE n_envs=$N_ENVS host=$(hostname)"
    if command -v git > /dev/null && git rev-parse --git-dir > /dev/null 2>&1; then
        echo "  git=$(git rev-parse --short HEAD) uncommitted-diff=$(git diff HEAD -- src train.py | sha256sum | cut -c1-12)"
        git diff HEAD -- src train.py > "$SESSION_DIR/uncommitted.diff"
    fi
} >> "$SESSION_DIR/curriculum.log"

run_stage() {
    local name="$1" end="$2"; shift 2
    echo "[$(date '+%F %T')] stage $name (to iteration $end): $*" >> "$SESSION_DIR/curriculum.log"
    "$PY" ./train.py "${COMMON[@]}" --iter-limit "$end" "$@" >> "$SESSION_DIR/$name.log" 2>&1  # append: a rerun resumes
}

case "$CURRICULUM" in
    paper)
        run_stage stageA "${ENDS[0]}" "${LOGIC[@]}" --workload-gen flat --wg-flat-targets4 150,1,1,2
        run_stage stageB "${ENDS[1]}" "${LOGIC[@]}" --workload-gen flat --wg-flat-targets4 1000,1,1,2
        run_stage stageC "${ENDS[2]}" "${LOGIC[@]}" --workload-gen flat --wg-flat-targets4 600,1,1,2 --wg-burst-small-prob 0.05
        run_stage stageD "${ENDS[3]}" "${LOGIC[@]}" "${TRACE[@]}"
        run_stage stageE "${ENDS[4]}" "${REAL[@]}" "${TRACE[@]}"
        ;;
    de)
        run_stage stageD "${ENDS[0]}" "${LOGIC[@]}" "${TRACE[@]}"
        run_stage stageE "${ENDS[4]}" "${REAL[@]}" "${TRACE[@]}"
        ;;
    e)
        run_stage stageE "${ENDS[4]}" "${REAL[@]}" --hourly-jobs "$TRACE_FILE" --job-arrival-scale-range 0.5,2.0
        ;;
    *)
        echo "unknown curriculum: $CURRICULUM (expected paper, de or e)" >&2
        exit 1
        ;;
esac

# Final evaluation on the held-out year: the best validated checkpoint and the last one,
# at arrival scales 1.5 (the training load) and 1.0, over three evaluation seeds.
BEST_STEP="$("$PY" -c "import json; print(json.load(open('$MODELS_DIR/validation_best.json'))['step'])" 2>/dev/null || true)"
LAST_STEP=$((ENDS[4] * 100000))
mkdir -p "$SESSION_DIR/eval"
for step in $(printf '%s\n' "$BEST_STEP" "$LAST_STEP" | grep -v '^$' | sort -u); do
    for scale in 1.5 1.0; do
        out="$SESSION_DIR/eval/m${step}_x${scale}_test${TEST_YEAR}.json"
        [[ -f "$out" ]] && continue
        echo "[$(date '+%F %T')] evaluate step $step at x$scale on $TEST_YEAR" >> "$SESSION_DIR/curriculum.log"
        "$PY" analyze_eval_seeds.py --seeds 100,200,301 --parallel 3 --tag "test${TEST_YEAR}_x${scale}" \
            --out "$out" --log-dir "$SESSION_DIR/eval/logs" -- \
            --session "$SESSION" --output-dir "$OUTPUT_DIR" "${WEIGHTS[@]}" --model "$step" \
            --prices "$TEST_PRICES" --hourly-jobs "$TRACE_FILE" --job-arrival-scale "$scale" \
            --eval-months "$EVAL_MONTHS" --oracle >> "$SESSION_DIR/eval.log" 2>&1
    done
done
echo "[$(date '+%F %T')] done (best validated step: ${BEST_STEP:-none feasible})" >> "$SESSION_DIR/curriculum.log"
