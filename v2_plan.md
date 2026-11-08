# PowerSched v2: training setup plan

Branch: `v2`. Status as of 8 Nov 2026.

## Why v2

Findings from the paper revision (details in `data-internal/paper/revision_findings.md`, not in git):

- **Training is seed-sensitive.** Retraining the paper recipe with other seeds gave one collapsed run and two runs saving about 0–1% on 2024 prices, against 6.4% for the published e4p4s2. The PPO settings (64-step rollouts, 10 passes over each, no entropy bonus, unscaled rewards) let policies collapse into "switch everything off" or "keep everything on".
- **The policies barely use the 10,000-value flattened queue.** A masking probe showed two of three published models ignore it, and the third depends on slot order. All rely on 5 aggregate statistics.
- **The 1–335 node-count action breaks argmax evaluation** when trained with an entropy bonus: the count stays nearly uniform, and its most likely value is arbitrary, for example "switch off 300 nodes".
- **Savings halve on held-out prices** (2023 → 2024), although 2024 offers more headroom. Training used a single price year.
- **Workload and prices ran on different clocks** in training, offset by a seed-dependent number of hours.

## Done (on `v2`)

| commit | change |
|---|---|
| `00548ec` | One `JobQueue` for all waiting jobs (up to 52,500, arrival = oldest-first order) for agent and baselines; backlog and refill action removed; queue no longer observed. Vectorized `dispatch_jobs`, identical decisions to the old loops (`test/test_job_queue.py`), same env speed. `easy` is the default dispatcher. |
| `74f67e6` | Action magnitude indexes `NODE_CHANGE_STEPS` (1, 2, 4, … 256 nodes): 27 actions instead of 1,005. |
| `befe1de` | Observation adds queue summaries (counts and core-hours per age bin, counts per node-request bin, oldest age, overdue count) and time features (sin/cos hour of day, weekday). Arrivals and time features follow the price clock (`prices.load_price_csv`: local timestamps incl. DST). |
| `b55eda5` | Observation scaled by fixed transforms to roughly [-1, 1] and returned as a copy of `env.state` (no running statistics to save). |
| `4de22d5` | `--n-envs` parallel environments (`SubprocVecEnv`, forked after loading the trace once); TensorBoard metrics from every env via `info["episode_metrics"]`, averaged per log; `train_iter.py` forwards `--n-envs` and `--normalize-reward`. |
| `2d5944a` | `--n-epochs`, `--target-kl`, `--decay-steps` (linear decay of learning rate and `ent_coef` over the whole run); `train_iter.py` forwards them. |
| `90748a1` | `--prices` takes several comma-separated CSVs, concatenated in order with their own timestamps (`load_price_csv`); passes unchanged through `train_iter.py` and the analyze scripts. |
| `3673ee1` | Checkpoint validation (`--val-every`, `src/validation.py`): deterministic runs on fixed windows of the target workload, feasibility vs baseline, best checkpoint in `validation_best.json`. |
| `5cf2ca1` | `--eval-months` 13 by default, annual savings rate over the evaluated days, `eval_summary.json`, `--eval-tag`, `analyze_eval_seeds.py` (mean ± std over evaluation seeds). |
| (pending) | `--job-arrival-scale-range LOW,HIGH`: per-episode arrival scale for training (grid arm c). Validation also records average wait and the queue left at window ends. |
| (pending) | `cluster/`: CPU-only Apptainer image, `run_v2.sh` (curriculum, validation, final evaluation), Slurm array `v2_grid.slurm`. |

Already on master before v2: numpy backlog speed-up, `--normalize-reward`, `--ent-coef` kept on resume, `analyze_obs_probe.py`.

## Still to do

1. ~~Parallel environments~~ — done. Scaling is sub-linear because a lockstep batch waits for its slowest env and step time is very uneven (median 6 ms, p90 35 ms, depending on arrivals): 1 env 65 steps/s, 4 → 107, 8 → 157, 16 → 231. Use `--n-envs` only to fill cores when there are fewer runs than cores; e.g. 27 runs × 8 envs on a 256-core node ≈ 14 h for 8M steps.
2. ~~PPO settings for v2~~ — done (`2d5944a`; values in `cluster/run_v2.sh`). Options added: `--n-epochs`, `--target-kl`, `--decay-steps` (linear decay of learning rate and `ent_coef` to 0 over the whole run, via `LinearDecayCallback`; SB3's own schedules restart with every learn() call). Values used in the grid:
   - `ent_coef` 0.002, decaying linearly to 0 by 8M steps;
   - learning rate 3e-4, decaying linearly to 0 by 8M steps;
   - `n_epochs` 5, `target_kl` 0.02;
   - 2,048 steps per update (`--n-steps 256` × 8 envs), batch size 256, `--normalize-reward`;
   - network 64-64 (input is now ~390 values).
3. ~~Multi-year prices~~ — done: `--prices data-external/prices_2023.csv,data-external/prices_2025.csv`. Leave-one-year-out: three runs per configuration, each trained on two years and evaluated (`--evaluate-savings --prices <third year>`) on the third. Episode starts are drawn over the whole concatenation; at the seam, prices and clock jump to the next file, like the existing wrap from December back to January. The price term of the reward is scaled by the 1st/99th percentiles of the loaded prices, so training pools them over both years and evaluation uses the test year's; costs and savings are unaffected.
4. ~~Validation / automatic pruning~~ — done (`src/validation.py`, `train.py --val-every`). For the runs: `--val-every 5` (500K steps) with `--val-hourly-jobs data-internal/allusers-main-30.log` in every stage (A–C train on the generator) and `--val-prices` = the two training years. Defaults: 4 windows × 2 episodes (≈ 4 months) at ×1.0 and ×1.5, about 65 s per validation (≈ 2% of training time). Validation episodes run without the loss barrier, otherwise a policy that loses jobs ends episodes early and sees different arrivals. Feasible = completions ≥ 99.5% of the baseline's and lost ≤ baseline's at both scales; score = mean savings vs baseline_off. `validation_best.json` names the best checkpoint; `cluster/run_v2.sh` evaluates it (and the last one) on the held-out year at the end. Still open: drop a configuration whose seeds all score below a threshold — done by hand when comparing the grid results. Validation history spans stages, since all stages write to the same models directory: filter by step to get a stage's best.
5. ~~Evaluation fixes~~ — done: `--eval-months` defaults to 13 (also in `train_iter.py` and the analyze scripts); "Annual Savings Rate" scaled by 365 / evaluated days (was 12 / months, i.e. 336 days counted as a year); `eval_summary.json` with the totals; `--eval-tag`; `analyze_eval_seeds.py` (several evaluation seeds in parallel, mean ± std). Still open, outside the repo: align `analyzer.py` (the colleague's heuristic evaluator; it and `best*.py` were moved out by the user) to the same episode count and dispatcher before comparing with RL; the heuristics were written for the v1 action (magnitude 1–335, refill) and need adapting to the v2 action.
6. ~~Cluster setup~~ — done (`cluster/`). Slurm cluster running Apptainer containers; AMD nodes with 256 cores; GPU partition (AMD) currently down, so CPU only.
   - `cluster/powersched-cpu.def`: Python 3.12 slim + CPU PyTorch, versions pinned to the dev venv (torch 2.10.0, SB3 2.7.1, …); 370 MB, builds in under a minute; the repo is bind-mounted, so code changes need no rebuild. `PYTHONNOUSERSITE=1`, because Apptainer mounts `$HOME` and `~/.local` packages otherwise shadow the image's. The existing `powersched.def` (bakes in the GitHub master zip, ROCm PyTorch, unpinned) is left as it is.
   - `cluster/run_v2.sh <paper|de|e> <seed> <test-year>`: one run (curriculum, validation, final evaluation of best + last checkpoint on the held-out year), resumable; `SMOKE=1` for a short version. Grid arm (c) uses the new `--job-arrival-scale-range 0.5,2.0` (training only; per-episode uniform draw).
   - `cluster/v2_grid.slurm`: array 0–26 = curriculum × seed (1–3) × test year; 9 cores (8 envs + main process; 27 tasks = 243 cores fit one 256-core node), 8 GB, 24 h per task; resubmit an index to resume. Partition/account still to fill in for the cluster.
   - Still open: (b) of step 4 — compare seeds/configurations after the grid (a summary over `<session>/eval/*.json`).
7. ~~Smoke run~~ — done 2026-11-08: `SMOKE=1` paper curriculum, seed 1, test 2024, through `v2_grid.slurm` inside the image; all stages, validation and final evaluation work. Main trace with 8 envs: 184–193 steps/s (8M ≈ 12 h), 2.8 GB per task. From stage C on (0.3–0.6M steps) the policy saves 12–17% vs baseline_off at validation, but by delaying jobs (average wait 6–28 h vs 0, up to 26K jobs queued at window ends), so completion is 97.7–98.8% of the baseline and every such checkpoint is infeasible under the 99.5% rule. The rule rejects the right thing; if long-trained policies still sit just below it, decide what delay is acceptable (e.g. average wait ≤ N h and no overdue jobs) and re-rank from `validation.jsonl`, which keeps wait and queue figures for that.
8. **Grid running** since 2026-11-08 13:47–13:50 on Virgo (GSI), partition `long`, in `/lustre/rz/orybalch/powersched/sessions/v2/`: Slurm jobs 21211317 (task 0) and 21211368 (tasks 1–26), code `94fed6c`. Stage A ran at 181 s per 100K steps there (local: 149 s), so ~15–16 h per run; expected done ~2026-11-09 06:00. Next: collect `validation.jsonl` and `eval/*.json`, compare curricula and seeds per held-out year.
9. **Feasibility by wait budget** (2026-11-08 evening, after 6 h of grid). Validation's completion rule (≥ 99.5% of the baseline's completions) turned out to be a hidden wait limit: jobs still queued at the 4 window ends count as not completed (Little's law: ~550 jobs/h × 15 h ≈ 8K per window end ≈ 2–3%), so every checkpoint that saved money failed it although it lost no jobs. Now: feasible = no more lost jobs than the baseline; best checkpoint kept per wait budget (average wait at the worst scale ≤ 6/12/24 h/any), since acceptable waiting depends on the jobs. `run_v2.sh` evaluates the best per budget, recomputed from `validation.jsonl`, so the running grid picks it up at its end (if the cluster checkout is updated before then). `analyze_validation.py` compares runs. First look (validation up to ~3M steps, training-year prices): `de` +18.8 ± 3.7% within 12 h (9/9 runs), +21.3 ± 1.7% unlimited; `e` +21.2 ± 4.6% within 24 h, waits longer (15–35 h) for about the same savings; `paper` collapsed to keep-all-on in stages A–C (entropy −0.05 to −0.6), −38% so far; nothing saves money under 6 h. Savings plateau near 20% from ~9 h of wait on, so `e`'s extra waiting buys nothing. Lower waits need a higher job-age weight: a job for the weights round.
10. **To check when the grid finishes** (notes from the evening of 2026-11-08; the cluster checkout was updated to `3243dd9` at ~21:00 while the jobs ran, so stages started after that use the new validation).
    - **Held-out year is the real test.** The ~20% above is validation on the training years. Look at `<session>/eval/*.json`: savings vs baseline_off on the held-out year, oracle capture, completion and wait over 13 continuous months (one end of timeline only, so deferral past the end is negligible there). Compare the best checkpoint per wait budget with the last one.
    - **Seed stability** (the main v2 goal): at ~3M steps all 9 `de` and all 9 `e` runs were in the same +15–25% band. Confirm on the held-out year (spread over seeds and over test years).
    - **`de` vs `e`:** same savings, but `e` waits 15–35 h vs 9–15 h for `de`. Check whether that holds at 8M and on the held-out year at ×1.0 (where `e`'s randomised arrival scale should help most).
    - **Why `paper` trails** (hypotheses, untested): (1) its validation during A–C measures transfer to a workload and prices it has not seen, so compare only from stage D on; `paper_s1_test2023`, the first in E, reached +14.9% at 8.7 h. (2) The learning-rate and entropy decay run over all 8M steps, so `paper` meets the main trace at ~60% of both, with a policy already near-deterministic from the easy stages (entropy −0.05 to −0.6 vs about −1 for `de`/`e`), and only 5M steps left. (3) `VecNormalize`'s return statistics average over all steps, so after 3M synthetic steps they adapt slowly to the main trace; check by comparing `*_vecnormalize.pkl` just before and after a stage change (e.g. `paper_s3_test2023` 2.9M vs 3.1M). If `paper` still trails at the end: drop A–C, or reset normalisation and schedules per stage.
    - **`paper` stage B is slow** on the cluster (23–37 steps/s vs ~170 locally); cause unknown (queue size in stage B? node contention?). Check `stageB.log` and the queue metrics in TensorBoard.
    - **Short waits:** no checkpoint saves money with an average wait under 6 h (job-age weight 0.2). For short-wait results the weights round needs higher job-age weights. A negative value in a budget column of `analyze_validation.py` means a feasible checkpoint exists but costs more than the baseline (effectively "none").
    - **`e_s3_test2023`** had a weak value function (explained variance 0.10 vs 0.5–0.9 elsewhere) at ~4M; see whether it recovered.
    - **Training health at ~1M steps** was fine: approx_kl 0.008–0.018 (target 0.02), clip_fraction 0.08–0.14, explained variance rising to 0.6–0.9, no lost jobs after the first ~100K steps.
    - **Timing:** main-trace runs at 74–133 steps/s on the cluster (local: 184–193); the slowest may exceed the 24 h limit. Follow-up array 21216729 (`--dependency=afterany` on 21211317 and 21211368) reruns all 27 indices: finished ones exit quickly, timed-out ones resume, and missing per-budget evaluations are added.
    - **Commands:** on the submitter node, `apptainer exec --no-home --bind "$PWD" --pwd "$PWD" --env MPLCONFIGDIR=/tmp/mpl-$USER powersched-cpu.sif python analyze_validation.py --sessions sessions/v2`; or rsync events/validation/curriculum.log/`eval/***` via sadpc017 into `sessions/v2_cluster/` (ssh -t for the key passphrase) and run it locally.

## Training grid (first round)

- Weights: e4p4s2 (eff 0.4, price 0.4, idle 0, age 0.2, drop 0) only; expand to other weight combinations if v2 works.
- Curriculum, as a grid dimension:
  - (a) A–E as in the paper (A: flat 150 jobs/h, B: flat 1,000, C: flat 600 + 5% small bursts, D: main trace ×1.5, all on logic prices; E: main trace ×1.5 on real prices);
  - (b) D+E only;
  - (c) E only, with the arrival scale randomised per episode between 0.5 and 2.0.
- Seeds: 3 per cell. Prices: leave-one-year-out.
- Loss barrier (`--flush-after-drop-streak 3`) on; drop penalty off (unchanged from the paper).

## Evaluation protocol

- Argmax actions (`deterministic=True`), as in the paper; also report sampled actions if they differ much.
- Held-out price year; 26 episodes; evaluation seeds 100, 200, 301; mean ± spread.
- Savings vs idle-free from total cost, plus completion, average wait, jobs lost.
- Masking probe (`analyze_obs_probe.py`) on the final models: aggregates, queue summaries, time features, prices flattened / shifted by 12 hours.

## Open decisions

- Priority (user): a good result matters more than the paper deadline; asking for a deadline extension is an option.

- Whether to attempt the v2 results within the paper revision, or report seed sensitivity as a limitation (deadline extension possible).
- Whether to replace the 335-value node vector by a summary (deferred; not in v2).
- Drop penalty vs loss barrier (deferred; barrier kept).
- Structured queue encoder (DeepSets): no longer needed for the flat queue, which is gone; reconsider only if summaries prove too coarse.

## Notes from the work so far

- Parallel-env scaling: tried starting all training envs at midnight so their peak-arrival hours coincide; ~13% faster, not adopted (it would fix the logic-price phase). User preference: no micro-optimisation for now.
- Throughput on the dev workstation (i7-12700K, 8 P-cores + 4 E-cores): 8 independent single-env processes each run ~80 steps/s, so the machine scales fine across processes.
- The PPO step (`--n-epochs`, `--target-kl`, `--decay-steps`) is the last step prepared; check `git log` to see whether it was committed.

## Workflow

- Changes go to `v2` in small steps; the user commits (commit messages are prepared with each step). Things that are not v2-specific go to master.
- Old v1 runs and evaluations live in `sessions/netarch*/` and `data-internal/paper/revision_eval/` (not in git).
