# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PowerSched is a reinforcement learning project that uses PPO (Proximal Policy Optimization) to optimize compute cluster scheduling based on electricity prices and job efficiency. The system trains an RL agent to make decisions about when to turn nodes on/off and how to schedule jobs in a high-performance computing environment.

## Core Components

- **Environment** (`environment.py`): Gymnasium-compatible RL environment simulating a compute cluster with 335 nodes, job queues, and electricity pricing
- **Training** (`train.py`): Main training script using stable-baselines3 PPO with tensorboard logging and model checkpointing
- **Pricing** (`prices.py`): Electricity price modeling and data handling
- **Samplers**: Job duration (`sampler_duration.py`) and job characteristics (`sampler_jobs.py`) sampling from real data
- **Plotting** (`plot.py`): Visualization of training progress, rewards, and cluster state
- **Callbacks** (`callbacks.py`): Custom callbacks for training monitoring and logging
- **Weights** (`weights.py`): Reward weight configuration and management

## Development Commands

**Setup Environment:**
```bash
pip install -r requirements.txt
```

**Environment Check:**
```bash
python ./checkenv.py
```

**Quick Test Run:**
```bash
python ./testenv.py
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

**Sequential Training with Different Weights:**
```bash
python ./train_iter.py
```

**Run Tests:**
```bash
python test_sampler_duration.py --print-stats --plot
python test_sampler_jobs.py --print-stats --plot
python test_aggregated_jobs.py
```

**Generate Cumulative Savings Plot:**
```bash
python plot_savings.py [session_name] --months 12 --show
```

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
- `sessions/`: Training session outputs (logs, models, plots)
- Models are saved as `.zip` files every 100K steps during training

## Architecture Notes

- Uses stable-baselines3 PPO with custom ComputeClusterEnv
- Environment simulates 2-week episodes (336 hours) with hourly decisions
- State space includes node counts, job queue, electricity prices, and time
- Action space controls the number of nodes to bring online/offline
- Rewards balance efficiency, cost savings, and resource utilization
- Cluster configuration: 335 nodes max, 96 cores per node, up to 16 nodes per job
- Job queue: max 1000 jobs, max 1500 new jobs per hour, max 170h runtime
- Power consumption: 150W idle, 450W used per node

## Training Session Management

Sessions are organized under `sessions/` directory with subdirectories for:
- `logs/`: Tensorboard training logs
- `models/`: Model checkpoints saved every 100K steps
- `plots/`: Training visualization plots

Use `--session` parameter to create named training runs for organization and comparison.