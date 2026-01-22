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
│   ├── workload_generator.py # Job generation
│   ├── baseline.py         # Baseline comparisons
│   ├── prices.py           # Price modeling
│   ├── prices_deterministic.py # Deterministic pricing
│   ├── sampler_*.py        # Job samplers
│   ├── callbacks.py        # Training callbacks
│   ├── weights.py          # Reward weights
│   ├── plot_config.py      # Plot configuration
│   └── plot.py             # Visualization
├── test/                   # Test files (all start with test_)
│   ├── run_all.py          # Run all tests
│   ├── test_checkenv.py    # Environment validation
│   ├── test_env.py         # Quick environment test
│   ├── test_sanity_env.py  # Environment sanity checks (invariants, determinism)
│   ├── test_sampler_*.py   # Sampler tests
│   └── test_*.py           # Other unit tests
├── .github/workflows/      # CI/CD
│   └── tests.yml           # GitHub Actions test workflow
├── train.py                # Main training script
├── train_iter.py           # Sequential training
├── data/                   # Sample data
├── data-internal/          # Full Slurm logs
└── sessions/               # Training outputs
```

## Core Components

- **Environment** (`src/environment.py`): Gymnasium-compatible RL environment simulating a compute cluster with 335 nodes, job queues, and electricity pricing
- **Training** (`train.py`): Main training script using stable-baselines3 PPO with tensorboard logging and model checkpointing
- **Pricing** (`src/prices.py`, `src/prices_deterministic.py`): Electricity price modeling and data handling
- **Samplers**: Job duration (`src/sampler_duration.py`), job characteristics (`src/sampler_jobs.py`), and hourly statistical sampler (`src/sampler_hourly.py`) sampling from real data
- **Plotting** (`src/plot.py`): Visualization of training progress, rewards, and cluster state
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

# Price tests
python -m test.test_price_history
python -m test.test_prices_cycling

# Sampler tests
python -m test.test_sampler_duration --print-stats --test-samples 10
python -m test.test_sampler_hourly --file-path data/allusers-gpu-30.log --test-day
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

Additional training options:
- `--ent-coef` (default 0.0): Entropy coefficient for PPO loss calculation
- `--iter-limit`: Maximum number of training iterations (1 iteration = 100K steps)
- `--session`: Session ID for organizing training runs
- `--render`: Visualization mode ("human" or "none")

## Data Files

- `data/`: Contains job duration samples and price data
- `data-internal/`: Contains complete Slurm logs with job characteristics (nodes, cores, duration)
- `sessions/`: Training session outputs (logs, models, plots)
- Models are saved as `.zip` files every 100K steps during training

## Samplers

The project includes three job samplers (all in `src/`):

1. **Duration Sampler** (`src/sampler_duration.py`): Samples job durations from simple duration logs
2. **Jobs Sampler** (`src/sampler_jobs.py`): Pattern-based replay of historical job batches with full characteristics
3. **Hourly Sampler** (`src/sampler_hourly.py`): Statistical sampler that builds hour-of-day distributions from Slurm logs
   - Captures daily patterns (busy vs quiet hours)
   - Properly handles zero-job hours
   - Samples job count, duration, nodes, and cores-per-node independently
   - Generates randomized but realistic job patterns

## Architecture Notes

- Uses stable-baselines3 PPO with custom ComputeClusterEnv
- Environment simulates 2-week episodes (336 hours) with hourly decisions
- State space includes node counts, job queue, electricity prices, and time
- Action space controls the number of nodes to bring online/offline
- Rewards balance efficiency, cost savings, and resource utilization
- Cluster configuration: 335 nodes max, 96 cores per node, up to 16 nodes per job
- Job queue: max 1000 jobs, max 1500 new jobs per hour, max 170h runtime
- Power consumption: 150W idle, 450W used per node

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