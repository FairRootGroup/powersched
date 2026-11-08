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
| (pending) | `--prices` takes several comma-separated CSVs, concatenated in order with their own timestamps (`load_price_csv`); passes unchanged through `train_iter.py` and the analyze scripts. |

Already on master before v2: numpy backlog speed-up, `--normalize-reward`, `--ent-coef` kept on resume, `analyze_obs_probe.py`.

## Still to do

1. ~~Parallel environments~~ — done. Scaling is sub-linear because a lockstep batch waits for its slowest env and step time is very uneven (median 6 ms, p90 35 ms, depending on arrivals): 1 env 65 steps/s, 4 → 107, 8 → 157, 16 → 231. Use `--n-envs` only to fill cores when there are fewer runs than cores; e.g. 27 runs × 8 envs on a 256-core node ≈ 14 h for 8M steps.
2. **PPO settings for v2.** Options added: `--n-epochs`, `--target-kl`, `--decay-steps` (linear decay of learning rate and `ent_coef` to 0 over the whole run, via `LinearDecayCallback`; SB3's own schedules restart with every learn() call). Values for the v2 runs:
   - `ent_coef` 0.001–0.003, decaying to 0 over training;
   - learning rate 3e-4 decaying linearly to 0;
   - `n_epochs` 5, `target_kl` 0.02;
   - 2,048 steps per update, batch size 256, `--normalize-reward`;
   - network 64-64 (input is now ~390 values).
3. ~~Multi-year prices~~ — done: `--prices data-external/prices_2023.csv,data-external/prices_2025.csv`. Leave-one-year-out: three runs per configuration, each trained on two years and evaluated (`--evaluate-savings --prices <third year>`) on the third. Episode starts are drawn over the whole concatenation; at the seam, prices and clock jump to the next file, like the existing wrap from December back to January. The price term of the reward is scaled by the 1st/99th percentiles of the loaded prices, so training pools them over both years and evaluation uses the test year's; costs and savings are unaffected.
4. **Validation callback / automatic pruning.** Every 500K steps (or at stage ends), evaluate deterministically (argmax) on validation data: about 2 months at arrival scales 1.0 and 1.5. Score = savings vs idle-free, subject to completion ≥ 99.5% and no lost jobs. Keep each run's best checkpoint; drop a configuration if all its seeds score below a threshold. Report all seeds of kept configurations.
5. **Evaluation fixes.** `--eval-months 13` (26 episodes, 364 days; 12 gives only 336 days). Report means over several evaluation seeds. Align `analyzer.py` (the colleague's heuristic evaluator; it and `best*.py` were moved out of the repo by the user) to the same episode count and dispatcher before comparing with RL. Note: the heuristics were written for the v1 action (magnitude 1–335, refill) and need adapting to the v2 action.
6. **Cluster setup.** Slurm cluster running Apptainer containers; AMD nodes with 256 cores; GPU partition (AMD) currently down, so CPU only. Apptainer image (`powersched.def` exists; CPU-only PyTorch), Slurm array job script, one run per array task (or one node running several runs via `train_iter.py --parallel` × `--n-envs`).
7. **Smoke run** of a few hours before the grid.

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
