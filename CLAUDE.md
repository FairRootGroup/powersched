# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PowerSched is a reinforcement learning project that uses PPO (Proximal Policy Optimization) to optimize compute cluster scheduling based on electricity prices and job efficiency. The system trains an RL agent to make decisions about when to turn nodes on/off and how to schedule jobs in a high-performance computing environment.

## Project Structure

```
powersched/
├── src/                    # Core source code
│   ├── environment.py      # Main RL environment
│   ├── config.py           # Configuration constants
│   ├── job_management.py   # Job queue and scheduling
│   ├── node_management.py  # Node control logic
│   ├── reward_calculation.py # Reward computation
│   ├── metrics_tracker.py  # Performance metrics
│   ├── workload_generator.py # Job generation (legacy)
│   ├── workloadgen.py      # Synthetic workload generator (no historical logs)
│   ├── workloadgen_cli.py  # Shared CLI helpers for workload generator
│   ├── baseline.py         # Baseline comparisons
│   ├── prices.py               # Electricity price modeling
│   ├── sampler_*.py        # Job samplers (historical log replay)
│   ├── callbacks.py        # Training callbacks
│   ├── validation.py       # Checkpoint validation and best-checkpoint selection (--val-every)
│   ├── weights.py          # Reward weights
│   ├── plot_config.py      # Plot configuration
│   ├── plotter.py          # Visualization (plot_dashboard, plot_cumulative_savings, plot_episode_summary)
│   ├── analysis_naming.py  # Standardized directory/model name construction from weight configs
│   ├── analysis_reporting.py # Savings reporting and validation utilities
│   ├── arrival_scale.py    # Job arrival scaling validation
│   └── evaluation_summary.py # Episode summary formatting and occupancy calculations
├── test/                   # Test files (all start with test_)
│   ├── run_all.py          # Run all tests
│   ├── test_checkenv.py    # Environment validation
│   ├── test_env.py         # Quick environment test
│   ├── test_sanity_env.py  # Environment sanity checks (invariants, determinism)
│   ├── test_sanity_workloadgen.py      # Workload generator sanity/property tests
│   ├── test_determinism_workloadgen.py # Workload generator determinism verification
│   ├── test_inspect_workloadgen.py     # Distribution inspection and plotting tool
│   ├── test_job_completion_metrics.py  # Job completion tracking tests
│   ├── test_plotter.py     # Plotter function tests
│   ├── test_sampler_*.py   # Sampler tests
│   ├── test_output/        # Output files from test runs (plots, etc.)
│   └── test_*.py           # Other unit tests
├── .github/workflows/      # CI/CD
│   └── tests.yml           # GitHub Actions test workflow
├── train.py                # Main training script
├── train_iter.py           # Sequential training
├── cluster/                # Apptainer image (powersched-cpu.def), v2 run script (run_v2.sh), Slurm arrays (v2_grid.slurm, v2_weights.slurm)
├── analyze_arrivalscale_occupancy.py  # Analysis: arrival scale effects on occupancy
├── analyze_lambda_occupancy.py        # Analysis: lambda parameter effects on occupancy
├── analyze_seed_occupancy.py          # Analysis: seed-based training run comparison
├── analyze_obs_probe.py               # Analysis: which observation parts a trained policy uses (masking probe)
├── analyze_eval_seeds.py              # Analysis: evaluate one model over several evaluation seeds (mean ± std)
├── analyze_validation.py              # Analysis: best validated checkpoint per wait budget across a grid of runs
├── analyze_grid.py                    # Analysis: held-out-year results of a grid per config, wait budget and scale
├── data/                   # Sample data
│   └── workload_statistics/ # Workload log analysis scripts and aggregate stats
├── data-internal/          # Full Slurm logs
└── sessions/               # Training outputs
```

## Core Components

- **Environment** (`src/environment.py`): Gymnasium-compatible RL environment simulating a compute cluster with 335 nodes, job queues, and electricity pricing
- **Training** (`train.py`): Main training script using stable-baselines3 PPO with tensorboard logging and model checkpointing
- **Pricing** (`src/prices.py`): Electricity price modeling and data handling
- **Samplers**: Job duration (`src/sampler_duration.py`), job characteristics (`src/sampler_jobs.py`), and hourly statistical sampler (`src/sampler_hourly.py`) sampling from real data
- **Workload Generator** (`src/workloadgen.py`, `src/workloadgen_cli.py`): Synthetic job generator that produces configurable, deterministic job streams without relying on historical logs. Supports flat/poisson/uniform arrival modes with optional burst injectors.
- **Plotting** (`src/plotter.py`): Visualization of training progress, rewards, and cluster state (`plot_dashboard`, `plot_cumulative_savings`, `plot_episode_summary`)
- **Analysis utilities** (`src/analysis_naming.py`, `src/analysis_reporting.py`, `src/evaluation_summary.py`): Session naming, savings reporting, and episode summary formatting
- **Callbacks** (`src/callbacks.py`): Custom callbacks for training monitoring and logging
- **Weights** (`src/weights.py`): Reward weight configuration and management

## Development Commands

**Important:** Always activate the virtual environment before running any commands:
```bash
source venv/bin/activate
```

**Setup Environment:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Environment Check:**
```bash
python -m test.test_checkenv
```

**Quick Test Run:**
```bash
python -m test.test_env
```

**Main Training:**
```bash
python ./train.py
```

**Training with Visualization:**
```bash
python ./train.py --render human
```

**Evaluate Trained Model (No Training):**
```bash
python ./train.py --evaluate-savings --session my_experiment
```
`--eval-months` counts 4-week months of 2 episodes; the default 13 is 364 days, a full price year (12 covers only 336 days). Totals are also written to `eval_summary.json` in the evaluation directory; `--eval-tag TAG` appends TAG to that directory's name so evaluations on other test prices or seeds are kept apart.

**Evaluate over several evaluation seeds** (different job arrivals; mean ± std):
```bash
python analyze_eval_seeds.py --seeds 1,2,3,4,5 --parallel 5 --tag prices2024 --out eval_2024.json -- \
    --session my_experiment --prices data-external/prices_2024.csv --hourly-jobs data-internal/allusers-main-30.log --model 8000000
```
Everything after `--` goes to `train.py` unchanged.
This runs the trained model for the specified number of months and generates:
- Episode-by-episode cost and job completion statistics
- Cumulative savings plot comparing agent vs two baselines
- Comprehensive job processing metrics (completion rates, wait times, queue sizes)
- Annual savings projections

**Sequential Training with Different Weights:**
```bash
python ./train_iter.py
```

**Run All Tests:**
```bash
python test/run_all.py
```

**Run Individual Tests:**
```bash
# Environment tests
python -m test.test_checkenv
python -m test.test_env

# Environment sanity tests (three modes)
python -m test.test_sanity_env --steps 200                                    # Quick invariants
python -m test.test_sanity_env --check-gym --check-determinism --steps 300    # Full checks
python -m test.test_sanity_env --prices data-external/prices_2023.csv --hourly-jobs data/allusers-gpu-30.log --steps 300  # With external data
python -m test.test_sanity_env --dispatcher easy --check-determinism --steps 300  # Under the EASY dispatcher

# Dispatcher tests (EASY backfilling)
python -m test.test_dispatcher_easy

# Job queue and vectorized dispatch (equivalence with the previous per-job dispatcher loops)
python -m test.test_job_queue

# Observation: queue summaries and time features match the simulation
python -m test.test_observation

# Checkpoint validation: same arrivals for every policy, feasibility, best-checkpoint record
python -m test.test_validation

# Workload generator tests
python -m test.test_sanity_workloadgen
python -m test.test_determinism_workloadgen

# Workload generator inspection (distribution plots, determinism self-check)
python -m test.test_inspect_workloadgen --workload-gen poisson --wg-poisson-lambdas4 200,10,6,24 --hours 336 --plot
python -m test.test_inspect_workloadgen --workload-gen poisson --wg-poisson-lambdas4 200,10,6,24 --hours 336 --plot --wg-burst-small-prob 0.2 --wg-burst-heavy-prob 0.02

# Price tests
python -m test.test_price_history
python -m test.test_prices_cycling

# Sampler tests
python -m test.test_sampler_duration --print-stats --test-samples 10
python -m test.test_sampler_hourly --file-path data/allusers-gpu-30.log --test-day
python -m test.test_sampler_hourly_aggregated --file-path data/allusers-gpu-30.log
python -m test.test_sampler_jobs --file-path data/allusers-gpu-30.log
python -m test.test_sampler_jobs_aggregated --file-path data/allusers-gpu-30.log
```

**v2 training grid (cluster):**
```bash
apptainer build powersched-cpu.sif cluster/powersched-cpu.def   # CPU-only deps; the repo is bind-mounted
mkdir -p sessions/v2/slurm
sbatch --partition=<partition> cluster/v2_grid.slurm           # 27 runs: curricula paper/de/e x seeds 1-3 x held-out year
cluster/run_v2.sh paper 1 2024                                 # one run locally (venv); SMOKE=1 for a short version
sbatch --partition=<partition> cluster/v2_weights.slurm        # weights round: de x 4 weight combinations x seeds 1-3 x held-out year (36 runs)
cluster/run_v2.sh de 1 2024 e2p2s6                             # one run with other weights (e<eff>p<price>s<age> in tenths; default e4p4s2)
```
`cluster/run_v2.sh` trains the curriculum (stages are cumulative `--iter-limit`s, 8M steps), validates every 500K steps on the two training years, then evaluates the best validated checkpoint per wait budget (6/12/24 h/any) and the last one on the held-out year (`analyze_eval_seeds.py`, seeds 100/200/301, x1.5 and x1.0) into `<session>/eval/`. Rerunning resumes. Summarise a finished grid with `python analyze_grid.py --sessions sessions/v2` (held-out savings, wait, completion and lost jobs per config, wait budget and arrival scale; `grid_eval.csv`).

**GitHub Actions:**
Tests run automatically on push/PR to master/main via `.github/workflows/tests.yml`.

## Key Training Parameters

The system uses weighted reward components:
- `--efficiency-weight` (default 0.7): Weight for job processing efficiency
- `--price-weight` (default 0.2): Weight for electricity price optimization
- `--idle-weight` (default 0.1): Penalty weight for idle nodes
- `--job-age-weight` (default 0.0): Penalty weight for job waiting time
- `--drop-weight` (default 0.0): Penalty weight for lost jobs (age expiry or queue-full rejection)

**Workload generator options** (pass `--workload-gen` to enable; replaces historical log samplers):
- `--workload-gen`: Arrival mode — `flat`, `poisson`, or `uniform` (default: disabled)
- `--wg-poisson-lambda` (default 200.0): Poisson lambda for job arrivals
- `--wg-poisson-lambdas4`: Four-value override: `arrivals,duration,nodes,cores`
- `--wg-max-jobs-hour` (default 1500): Hard cap on jobs per hour
- `--wg-flat-jobs-hour` (default 200): Target jobs/hour for flat mode
- `--wg-flat-jitter` (default 0): +/- jitter for flat arrivals
- `--wg-flat-targets4`: Four-value flat targets: `arrivals,duration,nodes,cores`
- `--wg-flat-jitters4`: Four-value flat jitters: `arrivals,duration,nodes,cores`
- `--wg-uniform-ranges4`: Four-value uniform ranges: `a_min:a_max,d_min:d_max,n_min:n_max,c_min:c_max`
- `--wg-burst-small-prob` (default 0.0): Per-hour probability of a small-job burst (additive on top of base arrivals)
- `--wg-burst-heavy-prob` (default 0.0): Per-hour probability of a heavy-job burst (additive on top of base arrivals)

Additional training options:
- `--ent-coef` (default 0.0): Entropy coefficient for PPO loss calculation
- `--iter-limit`: Maximum number of training iterations (1 iteration = 100K steps)
- `--session`: Session ID for organizing training runs
- `--render`: Visualization mode ("human" or "none")
- `--seed`: Random seed for reproducibility (seeds environment, numpy, torch, and PPO)
- `--seed-sweep`: Isolate outputs under a seed-specific session subdirectory
- `--device`: Training device ("auto", "cuda", or "cpu")
- `--net-arch`: Hidden layer sizes for policy/value networks (e.g., "64,64" or "256,128")
- `--n-epochs` (default 10): PPO passes over each rollout per update
- `--target-kl` (default off): Stop an update's remaining passes once the approximate KL to the previous policy exceeds this (e.g. 0.02)
- `--decay-steps` (default 0 = constant): Decay learning rate and `--ent-coef` linearly to 0 by this total timestep count (`LinearDecayCallback`); uses the model's total step count, so the decay runs once across iterations, curriculum stages and resumes
- `--n-envs` (default 1): Environments per run, each in its own process (`SubprocVecEnv`); `--n-steps` is per environment. Each env is seeded `seed + i`. Speeds up one run, at lower efficiency per core (measured on the main trace: 1 env 65 steps/s, 8 envs 157, 16 envs 231), so prefer `train_iter.py --parallel` across runs when there are more runs than cores
- `--normalize-reward`: Train on rewards scaled by a running return std (SB3 `VecNormalize`, observations untouched); statistics are saved as `<step>_vecnormalize.pkl` next to each checkpoint and reloaded on resume. Evaluation is unaffected
- `--val-every K` (default 0 = off): Validate the checkpoint every K iterations and after the last one (`--iter-limit`), see `src/validation.py`. Deterministic (argmax) runs on fixed windows: `--val-windows` (4) windows spread evenly over the validation prices, `--val-window-episodes` (2) episodes each, at every `--val-arrival-scales` (1.0,1.5), arrivals seeded by `--val-seed`, always full-length episodes (no loss barrier), so all checkpoints see the same jobs and prices. Workload `--val-hourly-jobs` (default `--hourly-jobs`; set it in stages that train on `--workload-gen`), prices `--val-prices` (default `--prices`; never the held-out test year). Feasible if, at every scale, no more jobs are lost than by the baseline (`--val-min-completion` can add a completion requirement, off by default: jobs still queued at a window's end count as not completed, so it acts as a hidden wait limit). Since acceptable waiting depends on the jobs, the feasible checkpoint with the highest positive savings is kept per wait budget (average wait at the worst scale ≤ 6 h, 12 h, 24 h, any): `validation_best.json` (`by_wait_budget`; top-level `step` = no limit), with the history in `validation.jsonl`, both next to the models; `val/*` in TensorBoard. About 65–90 s per validation. `train_iter.py` forwards all `--val-*` options. Compare runs with `python analyze_validation.py --sessions <dir>` (best per budget per run and per curriculum, `validation_best.csv`, savings-vs-wait plot `validation_frontier.png`)
- `--model`: Load a specific model checkpoint by timestep number
- `--job-arrival-scale` (default 1.0): Scale factor for sampled arrivals per step
- `--job-arrival-scale-range LOW,HIGH`: Training only: each episode draws its arrival scale uniformly from the range (e.g. `0.5,2.0`) instead of `--job-arrival-scale`; evaluation and validation keep fixed scales. Not with `--workload-gen`
- `--jobs-exact-replay`: Replay raw jobs in timeline order without aggregation
- `--jobs-exact-replay-aggregate`: Aggregate jobs per time-bin before enqueueing
- `--dispatcher` (default `easy`): Job dispatcher shared by agent and baselines. `easy` goes oldest job first with EASY backfilling (reservations are planned on online nodes only); `firstfit` starts every queued job that fits now, oldest first
- `--reservation-depth` (default 1): With `--dispatcher easy`, how many of the oldest blocked jobs get a reservation (1 = EASY, larger approaches conservative backfilling)
- `--plot-dashboard`: Generate combined dashboard plot after evaluation
- `--dashboard-hours` (default 336): Hours to show in dashboard plot

## Data Files

- `data/`: Contains job duration samples and price data
- `data/workload_statistics/`: Aggregate workload statistics and analysis scripts (`analyze_workload_logs.py`, `workload_logs.txt`) used to calibrate workload generator parameters
- `data-internal/`: Contains complete Slurm logs with job characteristics (nodes, cores, duration)
- `sessions/`: Training session outputs (logs, models, plots)
- `test/test_output/`: Output files from test runs (e.g., distribution plots from `test_inspect_workloadgen.py`)
- Models are saved as `.zip` files every 100K steps during training

## Samplers

The project includes three job samplers (all in `src/`), each with different use cases:

### 1. Duration Sampler (`src/sampler_duration.py`)
Simple sampler that only samples job durations from duration-only logs. Used for basic testing.

### 2. Jobs Sampler (`src/sampler_jobs.py`)
Deterministic replay sampler that replays historical job batches in sequence:
- Parses Slurm logs and bins jobs by time period (default: hourly)
- Replays jobs in chronological order, preserving exact historical patterns
- Includes **aggregation support**: groups similar jobs by (nodes, cores, duration) to reduce job count
- Converts sub-hour jobs to hourly equivalents by adjusting resource requirements to preserve core-hours
- Use `sample_aggregated()` for pre-aggregated jobs, `sample_hourly()` for full hourly conversion

### 3. Hourly Sampler (`src/sampler_hourly.py`)
Statistical sampler that builds hour-of-day distributions (24 distributions, one per hour):
- Captures daily patterns (busy hours vs quiet hours)
- Properly handles zero-job hours in the distribution
- Samples job count, duration, nodes, and cores-per-node independently
- Generates randomized but statistically realistic job patterns

**Aggregation support** (used by default in the environment):
- `precalculate_hourly_templates()`: Pre-computes aggregated job templates per hour
- Sub-hour jobs are binned by resource profile (nodes, cores) and converted to equivalent 1-hour jobs
- Hourly+ jobs are kept individually with rounded duration
- `sample_aggregated()`: Samples from templates with proportional scaling based on sampled job count
- Preserves resource profiles while reducing the number of discrete job objects

## Workload Generator (`src/workloadgen.py`)

A synthetic, deterministic job generator that produces configurable job streams without relying on historical Slurm logs. Key design goals:

- **Deterministic**: same `seed` + same `WorkloadGenConfig` → identical job stream
- **Controllable**: dial job rate, duration mix, node/cores mix, and stress modes
- **No log dependency**: works independently of `data/` files

### Arrival Modes
- `flat`: constant arrivals at `flat_jobs_per_hour` ± `flat_jitter`
- `poisson`: Poisson(λ) arrivals; each attribute (duration, nodes, cores) also sampled from Poisson with its own λ
- `uniform`: discrete-uniform in `[uniform_min_new_jobs_per_hour, max_new_jobs_per_hour]`

Job attributes (duration, nodes, cores) are sampled with the same mode as arrivals, using per-attribute lambdas/targets/jitters.

### Burst Injectors (additive on top of base arrivals)
- **Small burst** (`burst_small_prob`): each hour, with the given probability, inject many short low-resource jobs
- **Heavy burst** (`burst_heavy_prob`): each hour, with the given probability, inject a few long high-resource jobs

### CLI Integration (`src/workloadgen_cli.py`)
Provides `add_workloadgen_args()` and `build_workloadgen_config()` — shared by `train.py`, `test_sanity_env.py`, and `test_inspect_workloadgen.py` to avoid config duplication.

### Inspection Tool (`test/test_inspect_workloadgen.py`)
Interactive script that runs the generator for a configurable number of hours, prints distribution statistics, runs a determinism self-check, and optionally saves distribution plots to `test/test_output/`.

## Architecture Notes

- Uses stable-baselines3 PPO with custom ComputeClusterEnv
- Environment simulates 2-week episodes (336 hours) with hourly decisions
- State space includes node states, the 24h price forecast, pending job statistics (count, core-hours, avg duration, max nodes), waiting-job summaries (counts and core-hours per age bin, counts per node-request bin, oldest age and overdue count; bins in `QUEUE_AGE_BIN_EDGES` / `QUEUE_SIZE_BIN_EDGES`), and time features (sin/cos of hour of day and weekday); the job queue itself is not observed
- The observation is a scaled copy of the raw state (`ComputeClusterEnv._observation`): fixed transforms to roughly [-1, 1] (nodes: -1 off / 0 idle / remaining hours / 170; prices / 100; counts and core-hours log-scaled to [0, 1]), so it means the same in every curriculum stage and price year and needs no saved statistics. `env.state` stays in simulation units
- One simulation clock: job arrivals (hourly sampler) and the time features follow the hour of day of the current price hour. With `--prices`, that is the CSV's local timestamp (`prices.load_price_csv`), including DST and the real weekday. `--prices` takes several comma-separated CSVs (e.g. `data-external/prices_2023.csv,data-external/prices_2025.csv`), concatenated in order, each hour keeping its own timestamp
- Action space: `[action_type, magnitude]` - action_type 0 = switch nodes off, 1 = keep, 2 = switch on; magnitude indexes `NODE_CHANGE_STEPS` (1, 2, 4, ... 256 nodes)
- Rewards balance efficiency, cost savings, and resource utilization
- Cluster configuration: 335 nodes max, 96 cores per node, up to 16 nodes per job
- Job queue: one FIFO queue of all waiting jobs (`JobQueue`, max 52,500 jobs; arrivals beyond it are rejected), max 1500 new jobs per hour, max 170h runtime
- Power consumption: 150W idle, 450W used per node
- Baseline comparison: greedy scheduler that keeps all nodes on and processes jobs FIFO

## Evaluation Metrics

When using `--evaluate-savings`, the system outputs:

**Per Episode:**
- Total cost for the episode
- Savings vs baseline (with idle nodes) and baseline_off (no idle nodes)
- Job completion rate (completed/submitted)
- Average wait time per job
- Maximum queue size reached

**Cumulative Analysis:**
- Total savings over evaluation period
- Average monthly cost reduction percentage
- Projected annual savings rate
- Job processing comparison: agent vs baseline completion rates, wait times, and queue sizes

## Training Session Management

Sessions are organized under `sessions/` directory with subdirectories for:
- `logs/`: Tensorboard training logs
- `models/`: Model checkpoints saved every 100K steps
- `plots/`: Training visualization plots and cumulative savings analysis

Use `--session` parameter to create named training runs for organization and comparison.

The cumulative savings plot (generated during `--evaluate-savings`) is saved to the session's plots directory and shows:
- Agent costs vs baseline costs over time
- Two baseline comparisons: with idle nodes (baseline) and without idle nodes (baseline_off)
- Visual representation of cost reduction achieved by the trained agent
