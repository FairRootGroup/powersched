# PowerSched v2: training setup plan

Branch: `v2`. Status as of 12 Nov 2026: setup done, two training rounds done (curricula, reward weights).

## Why v2

Findings from the paper revision (details in `data-internal/paper/revision_findings.md`, not in git):

- **Training is seed-sensitive.** Retraining the paper recipe with other seeds gave one collapsed run, one stuck run and two runs saving −2.3% and +1.0% on 2024 prices, against 6.4% for the published e4p4s2. The PPO settings (64-step rollouts, 10 passes over each, no entropy bonus, unscaled rewards) let policies collapse into "switch everything off" or "keep everything on".
- **The policies barely use the 10,000-value flattened queue.** A masking probe showed two of three published models ignore it, and the third depends on slot order. All rely on 5 aggregate statistics.
- **The 1–335 node-count action breaks argmax evaluation** when trained with an entropy bonus: the count stays nearly uniform, and its most likely value is arbitrary.
- **Savings halve on held-out prices** (2023 → 2024), although 2024 offers more headroom. Training used a single price year.
- **Workload and prices ran on different clocks** in training, offset by a random, changing number of hours.

## Setup (done)

| commit | change |
|---|---|
| `00548ec` | One `JobQueue` for all waiting jobs (up to 52,500, oldest first) for agent and baselines; backlog and refill action removed; queue no longer observed. Vectorised dispatch, identical decisions to the old loops (`test/test_job_queue.py`). `easy` (EASY backfilling) is the default dispatcher. |
| `74f67e6` | Action magnitude indexes `NODE_CHANGE_STEPS` (1, 2, 4, … 256 nodes): 27 actions instead of 1,005. |
| `befe1de` | Observation adds queue summaries (counts and core-hours per age bin, counts per node-request bin, oldest age, overdue count) and time features (sin/cos hour of day, weekday). Arrivals and time features follow the price clock (local timestamps incl. DST). |
| `b55eda5` | Observation scaled by fixed transforms to roughly [-1, 1] (no running statistics to save). |
| `4de22d5` | `--n-envs` parallel environments (`SubprocVecEnv`). 1 env 65 steps/s, 8 → 157, 16 → 231 (sub-linear: a batch waits for its slowest env). |
| `2d5944a` | `--n-epochs`, `--target-kl`, `--decay-steps` (linear decay of learning rate and `ent_coef` over the whole run, across stages and resumes). |
| `90748a1` | `--prices` takes several comma-separated CSVs, concatenated with their own timestamps. The price term of the reward is scaled by the 1st/99th percentiles of the loaded prices. |
| `3673ee1` | Checkpoint validation (`--val-every`, `src/validation.py`). |
| `5cf2ca1` | `--eval-months 13` by default (364 days), annual rate over the evaluated days, `eval_summary.json`, `--eval-tag`, `analyze_eval_seeds.py` (mean ± std over evaluation seeds). |
| later | `--job-arrival-scale-range`; `cluster/` (Apptainer image, `run_v2.sh`, Slurm arrays); `analyze_validation.py`, `analyze_grid.py`. |

Already on master before v2: numpy backlog speed-up, `--normalize-reward`, `--ent-coef` kept on resume, `analyze_obs_probe.py`.

**PPO settings** (in `cluster/run_v2.sh`): 2,048 steps per update (256 × 8 envs), batch 256, 5 epochs, target KL 0.02, `ent_coef` 0.002 and learning rate 3e-4 both decaying linearly to 0 over 8M steps, `--normalize-reward`, network 64-64 (385 inputs), loss barrier `--flush-after-drop-streak 3`, drop penalty off.

**Validation and selection** (`src/validation.py`): every 500K steps, argmax actions on 4 windows × 2 episodes of the two training years, at ×1.0 and ×1.5, fixed arrivals, no loss barrier. Feasible = no more lost jobs than the baseline at both scales. Per wait budget (average wait at the worse scale ≤ 6 / 12 / 24 h / none) the feasible checkpoint with the highest positive savings is the pick. Validation never affects training: every run trains 8M steps and keeps all checkpoints.

**Evaluation:** held-out price year, 364 days continuous, evaluation seeds 100 / 200 / 301, argmax, ×1.5 and ×1.0; savings vs the idle-free baseline from total cost, plus completion, wait, lost jobs, oracle capture.

**Running:** `cluster/run_v2.sh <paper|de|e> <seed> <test-year> [weights]` (one run: curriculum, validation, evaluation of the picks and the last checkpoint; resumable; `SMOKE=1` short); `cluster/v2_grid.slurm` (round 1, 27 runs), `cluster/v2_weights.slurm` (round 2, 36 runs); 9 cores, 8 GB, 24 h per task on Virgo `long`, plus a follow-up array (`--dependency=afterany:<jobid>`) for time-outs. Summaries: `analyze_grid.py --sessions sessions/v2_cluster`, `analyze_validation.py --sessions sessions/v2_cluster`. Fetching results from the cluster: rsync with include filters (validation, `curriculum.log`, `eval/***`, events, or `8000000.zip` only) into `sessions/v2_cluster/`.

## Results: round 1, curricula (2026-11-10)

27 runs: curricula `paper` (A–E as in the paper), `de` (D 1M steps on logic prices, E 7M on the two training years), `e` (E only, arrival scale drawn per episode from 0.5–2.0) × seeds 1–3 × held-out year 2023/2024/2025; weights e4p4s2; main trace ×1.5 in D and E. Pick without a wait budget; mean ± std over 9 runs.

| curriculum | ×1.5: savings | wait | ×1.0: savings | wait | lost jobs |
|---|---|---|---|---|---|
| **de** | **+23.1 ± 1.5%** (+21.0 … +25.9) | 12.3 h | **+24.4 ± 2.1%** | 8.5 h | 0 |
| e | +21.3 ± 2.3% | 22.1 h | +24.1 ± 2.7% | 15.5 h | 0 |
| paper | +19.6 ± 3.4% | 19.7 h | +21.6 ± 3.3% | 13.2 h | 221K at ×1.5 |

- **`de` is the configuration to build on:** all 9 runs within +21 … +26% at ×1.5, never a lost job, shortest waits; its last checkpoint is as good as the validated pick.
- **`e`** saves about as much but waits twice as long; its last checkpoints vary (down to −5%).
- **`paper`** catches up only in stage E and varies more; the synthetic stages A–C make the policy nearly deterministic before it meets the real workload.
- **v1 comparison:** the published e4p4s2 gave 6.4% on held-out 2024. Not like for like: v2 also changed dispatcher, clock, evaluation length and action space, each relative to the baseline in its own environment.
- **Oracle capture** (contiguous): 20–31%; for comparing runs, not an absolute grade (the oracle knows all future arrivals).

## Results: round 2, reward weights (2026-11-12)

Curriculum `de`; the paper's e4p4s2 (round 1), e3p5s2, e2p2s6 and two new combinations, e3p3s4 (between e4p4s2 and e2p2s6 on job age) and e2p6s2 (price-heavy) × 3 seeds × 3 held-out years. Pick without a wait budget (decided: no time rule; lost jobs are reported). Ordered by wait:

| weights | ×1.5: savings | wait | ×1.0: savings | wait | last checkpoint ×1.5 | lost jobs (×1.5 / ×1.0) |
|---|---|---|---|---|---|---|
| e2p2s6 | +19.0 ± 1.3% | 3.0 h | +17.8 ± 1.6% | 2.7 h | +18.5 ± 1.8% | 0 |
| e3p3s4 | +23.8 ± 1.6% | 5.7 h | +23.6 ± 1.6% | 4.8 h | +21.9 ± 1.4% | 0 |
| e2p6s2 | +26.9 ± 3.1% | 9.1 h | +25.9 ± 3.0% | 8.0 h | +24.6 ± 2.4% | 324K / 166K (1 run) |
| e3p5s2 | +25.2 ± 1.7% | 11.4 h | +26.0 ± 2.1% | 8.2 h | +24.6 ± 1.4% | 0 |
| e4p4s2 | +23.1 ± 1.5% | 12.3 h | +24.4 ± 2.1% | 8.5 h | +23.1 ± 2.0% | 0 |

- **A savings-vs-wait frontier:** e2p2s6 (3 h) → e3p3s4 (5–6 h) → e3p5s2 / e2p6s2 (8–11 h). e4p4s2, the paper's headline weights, saves less than both price-heavy combinations.
- **The lossy pick:** e2p6s2 includes `de_e2p6s2_s2_test2025` @ 1.0M, the end of stage D (logic prices only). It built a backlog in validation (26,675 jobs overdue at the ×1.5 window ends) but lost no job within the 4-week windows, so it passed; over the held-out year it lost 324K jobs. Without that run e2p6s2 gives +26.0 ± 2.0% at 7.5 h. A 12 h budget would have avoided every lossy pick of both rounds (not adopted as a rule).
- **Short waits are possible now:** best within a 6 h validation wait, ×1.5: e3p3s4 +23.8%, e2p6s2 +23.5%, e2p2s6 +19.0% (all 9 runs); e3p5s2 4 of 9 runs, e4p4s2 none.
- **Paper's questions:** e2p2s6 again waits least (3 h vs 0.5–1 h in the paper); e4p4s2 no longer saves most; e3p5s2, inconsistent in the paper, is as stable as the others.
- **Seed stability holds:** ±1.3–2.1 points over 9 runs (e2p6s2 ±3.1 with the lossy run).
- **No in-sample advantage:** each run's last checkpoint also evaluated on its two training years (×1.0, `eval/m8000000_x1.0_in<year>.json`): seen minus held-out savings −0.4 … +0.5 points per combination. 2024 is the best year for all 45 models, trained on it or not, so the spread over 9 runs is mostly the price year; within a year the seeds agree to about ±1 point.
- **Validation predicts held-out** within about 2 points, same ordering up to near-ties.
- **Load sweep** (`de_s1_test2023` @ 8M, 2023, ×0.25 … ×2.0): +16 … +24%, no lost jobs at any scale; at ×1.0 about 8 points come from lower energy (tighter packing), the rest from a 16% lower average price paid. The main trace loads the cluster lightly (cores in use 5% at ×1.0, 8% at ×1.5).

## Open

1. **Selection: last-stage candidates only** (planned after the group meeting). Pick only among checkpoints of the last curriculum stage (`de`: after 1.0M, `paper`: after 4.0M); earlier ones never trained on the target setting. Both stage-D picks of the two rounds lost jobs (`de_e2p6s2_s2_test2025` @ 1.0M, `paper_s2_test2025` @ 4.0M, 213K). With the filter the e2p6s2 run's pick becomes 4.0M (already evaluated: +24.0 / +25.2%, 0 lost); no retraining. Implementation (drafted, reverted before the meeting): `after_step` in `best_per_budget` / `record_result`, `train.py --val-after-step`, forwarded by `train_iter.py`, set per curriculum in `cluster/run_v2.sh` (training and final pick), `last_stage_start(session)` from `curriculum.log` for `analyze_grid.py` / `analyze_validation.py`, a test in `test_validation.py`.
2. **Validation starts from a favourable state** (known gap): each window starts with all nodes on and an empty queue, and jobs are lost only after 2 weeks or with a full queue, so a policy with a growing backlog can pass. Ideas, not adopted:
   - overdue limit at the window ends, relative to the baseline (needs the baseline's overdue count recorded); an absolute limit would rule out overloaded scenarios;
   - warm-up: 1–2 unscored episodes per window (validation +50–100%).
   Until then: check held-out lost jobs (`analyze_grid.py` lists them) before trusting a pick.
3. **Training length:** validation plateaus from ~2.5M for e4p4s2, but higher job-age weights keep improving to 8M (e2p2s6 +11.6% at 2M → +17.5% at 8M), and the spread between runs shrinks with training. Confound: with `--decay-steps 8000000` learning rate and entropy are still high at 2M. Keep 8M for the paper; for exploratory rounds test 4M with `--decay-steps 4000000` once.
4. **Blind 4th price year** (2026 via the Energy-Charts API, or 2022 as a stress test), evaluated only after freezing all choices: the 2023–2025 held-out years informed choices and are not blind for the final claim. The workload is not held out either (one trace).
5. **Rule-based baselines** (reviewers): `analyzer.py` (the colleague's heuristics, outside the repo) needs the same episode count and dispatcher, and adapting to the v2 action.
6. **Masking probe on the v2 models** (`analyze_obs_probe.py`): queue summaries, time features, prices flattened / shifted by 12 h.
7. **Price-heavy runs are slower** (e2p6s2 ~58, e3p5s2 ~66 steps/s vs ~100 for the others; queues are not larger); cause not checked.

## Open decisions

- Priority (user): a good result matters more than the paper deadline; asking for a deadline extension is an option.
- Which weights to present as the main result (frontier: e3p3s4 for short waits, e2p6s2 / e3p5s2 for most savings).
- Whether to replace the 335-value node vector by a summary (deferred).
- Drop penalty vs loss barrier (deferred; barrier kept).

## Notes

Internal working notes (deck, reviews, figures, workstation measurements) are kept outside the repository in `data-internal/paper/v2_notes.md`.

## Workflow

- Changes go to `v2` in small steps; the user commits (commit messages are prepared with each step). Things that are not v2-specific go to master.
- Old v1 runs and evaluations live in `sessions/netarch*/` and `data-internal/paper/revision_eval/` (not in git).
