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
├── analyze_arrivalscale_occupancy.py  # Analysis: arrival scale effects on occupancy
├── analyze_lambda_occupancy.py        # Analysis: lambda parameter effects on occupancy
├── analyze_seed_occupancy.py          # Analysis: seed-based training run comparison
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
python ./train.py --evaluate-savings --eval-months 12 --session my_experiment
```
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
python -m test.test_sanity_env --prices data/prices_2023.csv --hourly-jobs data/allusers-gpu-30.log --steps 300  # With external data

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
- `--model`: Load a specific model checkpoint by timestep number
- `--job-arrival-scale` (default 1.0): Scale factor for sampled arrivals per step
- `--jobs-exact-replay`: Replay raw jobs in timeline order without aggregation
- `--jobs-exact-replay-aggregate`: Aggregate jobs per time-bin before enqueueing
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
- State space includes node counts, job queue, electricity prices, pending job statistics (count, core-hours, avg duration, max nodes), and backlog size
- Action space: `[action_type, magnitude, do_refill]` - controls nodes online/offline and whether to refill the job queue from the backlog
- Rewards balance efficiency, cost savings, and resource utilization
- Cluster configuration: 335 nodes max, 96 cores per node, up to 16 nodes per job
- Job queue: max 2500 jobs, max 1500 new jobs per hour, max 170h runtime; overflow goes to backlog (max 50000)
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
