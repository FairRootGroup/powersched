# Curriculum Setup

This file is a fill-in manual for running the curriculum step by step.
Run it as an operator checklist: fill in the exact commands, then execute the stages in order.

## Structure

Use the same flow for every stage:

1. train the stage,
2. save the current models as a checkpoint in a separate directory,
3. evaluate the behaviour during training,
4. remove all models from the active working directory that should not continue,
5. Continue to next stage with all surviving models (use --continue-existing-only).

For proof of concept we used human eye and intuition based on TensorBoard behaviour.
More dedicated in-between evaluations can be introduced later.

Important repo-specific rules:

- Keep `--session`, `--output-dir`, and the weight sweep definition fixed across stages.
- `--continue-existing-only` only continues combinations that already have a saved model in the active session directory.
- For stages using `--workload-gen`, do not combine it with `--job-arrival-scale`.
- Promotion is therefore controlled by which checkpoints remain in the active session directory before the next stage starts.

## Shared Setup

Fill in the variables below once; every stage command reuses them.

```bash
cd /path/to/powersched
source venv/bin/activate

# --- fill these in ---
SESSION="curriculum_v1"
OUTPUT_DIR="sessions"
SEED=10
HOURLY_JOBS="/path/to/allusers-main-30.log"
ARRIVAL_SCALE=2.0
# ---------------------

COMMON_TRAIN_ARGS="
  --fix-weights efficiency,price,idle,job-age,drop
  --fix-values 0.3,0.5,0.0,0.2,0.0
  --session $SESSION
  --output-dir $OUTPUT_DIR
  --parallel 10
  --plot-dashboard
  --seed $SEED
  --net-arch 64,64
  --flush-after-drop-streak 3
"
# --iter-limit-per-step is cumulative (counts total iters across all stages),
# so each stage passes its own value; see per-stage commands below.

# --model <timestep> loads a specific checkpoint; omit to use the latest
COMMON_EVAL_ARGS="--session $SESSION --output-dir $OUTPUT_DIR --seed $SEED"
```

### Shared checkpoint management

These are the manual intervention points between stages.

```bash
# backup models after a stage (substitute STAGE_NAME)
cp -r $OUTPUT_DIR/$SESSION/models  $OUTPUT_DIR/${SESSION}_stage_STAGENAME_backup

# prune non-promising models from the active session directory
rm $OUTPUT_DIR/$SESSION/models/<weights_prefix>/<timestep>.zip

# next stage continues with survivors only (--continue-existing-only is already in stage B+ commands)
```

### Stage execution template

Repeat this for every stage, substituting the stage-specific arguments:

1. Train:
   ```bash
   python train_iter.py $COMMON_TRAIN_ARGS $STAGE_ARGS
   ```

2. Evaluate (optional):
   ```bash
   python train.py $COMMON_EVAL_ARGS $STAGE_ARGS --evaluate-savings
   ```

3. Promote:
   ```bash
   # backup checkpoints
   # prune rejected checkpoints from active session
   ```

## Stage A: Flat Arrivals + Logic Prices

1. Goal: learn the basic defer-then-clear timing under simple price phases.
2. Steps: 1M (cumulative: 1M = 10 iters)
3. Commands:
   ```bash
   STAGE_ARGS="--workload-gen flat --wg-flat-targets4 150,1,1,2 --wg-burst-small-prob 0.0 --wg-burst-heavy-prob 0.0"

   python train_iter.py $COMMON_TRAIN_ARGS --iter-limit-per-step 10 $STAGE_ARGS
   python train.py $COMMON_EVAL_ARGS $STAGE_ARGS --evaluate-savings
   ```

## Stage B: High-Load Flat Arrivals + Logic Prices

1. Goal: keep the same timing behavior, but under less slack.
2. Steps: 1M (cumulative: 2M = 20 iters)
3. Commands:
   ```bash
   STAGE_ARGS="--workload-gen flat --wg-flat-targets4 1200,1,1,2 --wg-burst-small-prob 0.0 --wg-burst-heavy-prob 0.0"

   python train_iter.py $COMMON_TRAIN_ARGS --iter-limit-per-step 20 --continue-existing-only $STAGE_ARGS
   python train.py $COMMON_EVAL_ARGS $STAGE_ARGS --evaluate-savings
   ```

## Stage C: Bursty Arrivals + Logic Prices

1. Goal: test queue-spike robustness while preserving the defer-then-clear pattern.
2. Steps: 1M (cumulative: 3M = 30 iters)
3. Commands:
   ```bash
   STAGE_ARGS="--workload-gen flat --wg-flat-targets4 600,1,1,2 --wg-burst-small-prob 0.05 --wg-burst-heavy-prob 0.0"

   python train_iter.py $COMMON_TRAIN_ARGS --iter-limit-per-step 30 --continue-existing-only $STAGE_ARGS
   python train.py $COMMON_EVAL_ARGS $STAGE_ARGS --evaluate-savings
   ```

## Stage D: Main Arrivals + Logic Prices

1. Goal: move to the real workload structure while keeping simple price phases.
2. Steps: 2M+ (cumulative: 5M+ = 50+ iters)
3. Note: trained on `ARRIVAL_SCALE=2.0`, but staged scaling such as `1.0 -> 2.0` is also possible.
4. Commands:
   ```bash
   STAGE_ARGS="--hourly-jobs $HOURLY_JOBS --job-arrival-scale $ARRIVAL_SCALE"

   python train_iter.py $COMMON_TRAIN_ARGS --iter-limit-per-step 50 --continue-existing-only $STAGE_ARGS
   python train.py $COMMON_EVAL_ARGS $STAGE_ARGS --evaluate-savings
   ```

## Stage E: Main Arrivals + Noisy Logic Prices (Optional)

1. Goal: keep the learned policy while adding moderate price irregularity.
2. Note: usually skipped — another run with higher job scale is often used instead. The idea remains valid but does not change much in practice.
3. Commands:
   ```bash
   STAGE_ARGS="[fill in noisy-logic-price setup]"

   python train_iter.py $COMMON_TRAIN_ARGS --iter-limit-per-step [cumulative] --continue-existing-only $STAGE_ARGS
   python train.py $COMMON_EVAL_ARGS $STAGE_ARGS --evaluate-savings
   ```

## Stage F: Main Arrivals + Real Prices

1. Goal: final fine-tuning on the full target setup.
2. Steps: 5M+ up to 10M (cumulative: 100–150 iters from stage D baseline)
3. Commands:
   ```bash
   STAGE_ARGS="--hourly-jobs $HOURLY_JOBS --prices data/prices_2023.csv --job-arrival-scale $ARRIVAL_SCALE"

   python train_iter.py $COMMON_TRAIN_ARGS --iter-limit-per-step 100 --continue-existing-only $STAGE_ARGS
   python train.py $COMMON_EVAL_ARGS $STAGE_ARGS --evaluate-savings

   # final checkpoint backup
   cp -r $OUTPUT_DIR/$SESSION/models  $OUTPUT_DIR/${SESSION}_final_backup
   # final evaluation / comparison command
   ```

## Per-Stage Notes

Use this small checklist after each stage:

- session:
- checkpoint used:
- checkpoint promoted:
- main metrics checked:
- go / no-go decision:
