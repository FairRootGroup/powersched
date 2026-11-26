# cluster/

Training and evaluation scripts for the v2 runs. Each script's header has the full options.

## Runtime

- **`powersched-cpu.def`**: Apptainer recipe for a CPU-only image. It holds Python and the dependencies only. The repository is bind-mounted at run time, so code changes need no rebuild.
  `apptainer build powersched-cpu.sif cluster/powersched-cpu.def`

## Training

- **`run_v2.sh <paper|de|e> <seed> <test-year> [weights]`**: one run. It trains the curriculum (8M steps), validates every 500K steps on the two training years, then evaluates the best validated checkpoint per wait budget and the last checkpoint on the held-out year (seeds 100/200/301, ×1.5 and ×1.0). Rerunning resumes. It runs in the venv or inside the image; `SMOKE=1` gives a short version.
- **`v2_grid.slurm`**: round 1 as a Slurm array: 3 curricula × 3 seeds × 3 held-out years = 27 runs.
- **`v2_weights.slurm`**: round 2: curriculum `de`, 4 further weight combinations × 3 seeds × 3 years = 36 runs. Round 1's `de` runs supply e4p4s2.

The repository, the trace, the image and the outputs must be on a filesystem the worker nodes see. Submit from the repository root. A run that hits the time limit resumes when its array index is resubmitted.

## Evaluation of finished runs

These run locally, in the venv, with several evaluations in parallel (`PARALLEL`). Results that already exist are skipped, so rerun after an interruption.

- **`eval_year.sh <year|heldout> [sessions dir]`**: evaluates the `de` runs on another price year (e.g. 2022), on rescaled prices (`PRICES`, `TAG`), or on another workload trace (`TRACE`). `STEPS` picks the checkpoints: `best`, `last` or `both`. Results go to `<session>/eval/m<step>_x<scale>_<TAG>.json`.
- **`eval_probe.sh [sessions dir]`**: the observation ablation. It runs each run's selected checkpoint on its held-out year with one part of the observation altered (`analyze_obs_probe.py`). Results go to `<session>/eval/probe/`; summarise them with `analyze_probe.py`.
- **`eval_heuristics.sh "<specs>" [output dir]`**: evaluates the rule-based controllers (`src/heuristics.py`, e.g. `price:quantile=0.25,max_wait=12`) on the price years like the policies. Results go to `sessions/heuristics/test<year>/eval/`; summarise them with `analyze_heuristics.py`.

- **`eval_final.sh <main|trace|probe|heuristics|all> [sessions dir]`**: every evaluation behind the paper's results, on the final checkpoints (8M steps) with evaluation seeds 101/201/301, by calling the three scripts above. Results carry the tag prefix `f101_`. The parts write separate files, so they can run on different machines that share the sessions directory.

Summaries of the training runs themselves: `analyze_grid.py` and `analyze_validation.py` in the repository root.
