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

### Working directory

```bash
# cd /path/to/powersched
```

### Session / output naming

Use one session name for the full curriculum run.

```bash
# session / output conventions
```

### Shared training arguments

Arguments that should stay the same across all stages.

```bash
# python train_iter.py [COMMON_ARGS] [STAGE_ARGS]
```

```bash
# COMMON_ARGS:
# --fix-weights idle,drop
# --fix-values 0.0,0.0
# --iter-limit-per-step 10
# --session [SESSION_NAME]
# --output-dir [OUTPUT_DIR]
# --parallel 10
# --plot-dashboard
# --seed 10
# --flush-after-drop-streak 3
```

### Shared evaluation arguments

Use the same pattern for stage evaluation.

```bash
# python train.py [COMMON_EVAL_ARGS] [STAGE_ARGS] --evaluate-savings
```

```bash
# COMMON_EVAL_ARGS:
# --session [SESSION_NAME]
# --output-dir [OUTPUT_DIR]
# --seed 10
# --model [MODEL_ID]
```

### Shared checkpoint management

These are the manual intervention points between stages.

```bash
# backup models / model directories after each stage
```

```bash
# prune non-promising models from the active session directory
```

### Stage execution template

Repeat this for every stage, substituting the stage-specific arguments:

1. Train:
   ```bash
   # python train_iter.py [COMMON_ARGS] [STAGE_X_ARGS]
   ```

2. Evaluate (optional):
   ```bash
   # python train.py [COMMON_EVAL_ARGS] [STAGE_X_ARGS] --evaluate-savings
   ```

3. Promote:
   ```bash
   # backup checkpoints
   # prune rejected checkpoints from active session
   # next stage continues with --continue-existing-only
   ```

## Stage A: Flat Arrivals + Logic Prices

1. Goal: learn the basic defer-then-clear timing under simple price phases.
2. Steps: 1M
3. Stage-specific arguments:
   ```bash
   # --workload-gen flat
   # --wg-flat-targets4 150,1,1,2
   # --wg-burst-small-prob 0.0
   # --wg-burst-heavy-prob 0.0
   # --prices ""
   ```
4. Follow the [stage execution template](#stage-execution-template).

## Stage B: High-Load Flat Arrivals + Logic Prices

1. Goal: keep the same timing behavior, but under less slack.
2. Steps: 1M
3. Stage-specific arguments:
   ```bash
   # --workload-gen flat
   # --wg-flat-targets4 1200,1,1,2
   # --wg-burst-small-prob 0.0
   # --wg-burst-heavy-prob 0.0
   # --prices ""
   ```
4. Follow the [stage execution template](#stage-execution-template).

## Stage C: Bursty Arrivals + Logic Prices

1. Goal: test queue-spike robustness while preserving the defer-then-clear pattern.
2. Steps: 1M
3. Stage-specific arguments:
   ```bash
   # --workload-gen flat
   # --wg-flat-targets4 600,1,1,2
   # --wg-burst-small-prob 0.05
   # --wg-burst-heavy-prob 0.0
   # --prices ""
   ```
4. Follow the [stage execution template](#stage-execution-template).


## Stage D: Main Arrivals + Logic Prices

1. Goal: move to the real workload structure while keeping simple price phases.
2. Steps: 2M+
3. Stage-specific arguments:
   ```bash
   # --hourly-jobs [PATH_TO_MAIN_LOG]
   # --prices ""
   # --job-arrival-scale [SCALE]
   ```
   Note: trained on `2.0`, but staged scaling such as `1.0 -> 2.0` is also possible.
4. Follow the [stage execution template](#stage-execution-template).

## Stage E: Main Arrivals + Noisy Logic Prices (Optional)

1. Goal: keep the learned policy while adding moderate price irregularity.
2. Note: usually skipped — another run with higher job scale is often used instead. The idea remains valid but does not change much in practice.
3. Stage-specific arguments:
   ```bash
   # [fill in noisy-logic-price setup]
   ```
4. Follow the [stage execution template](#stage-execution-template).

## Stage F: Main Arrivals + Real Prices

1. Goal: final fine-tuning on the full target setup.
2. Steps: 5M+ (up to 10M)
3. Stage-specific arguments:
   ```bash
   # --hourly-jobs [PATH_TO_MAIN_LOG]
   # --prices "data/prices_2023.csv"
   # --job-arrival-scale [SCALE]
   ```
4. Follow the [stage execution template](#stage-execution-template), then for the final selection:
   ```bash
   # final checkpoint backup
   # final evaluation / comparison command
   ```

## Per-Stage Notes

Use this small checklist after each stage:

- session:
- checkpoint used:
- checkpoint promoted:
- main metrics checked:
- go / no-go decision:
