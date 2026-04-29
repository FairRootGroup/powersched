# Oracle Benchmarks for PowerSched

## Motivation

PowerSched trains a reinforcement learning agent to reduce electricity costs by shifting
compute workload towards cheaper price periods. A key open question is:

> **How much savings are achievable in principle?**

Without an upper bound, we can observe that the agent saves X euros versus the greedy
baseline, but we cannot say whether that is 10% or 90% of what is theoretically possible.
An *oracle* — a scheduler with perfect foresight that solves for the optimal schedule
offline — provides that upper bound.

The oracle does not need to run in real time. It receives the same job stream and price
data that the environment sees, accumulates them over the episode, and solves for the
minimum-cost schedule at episode end.

---

## What the Agent Is Up Against

The agent operates under hard real-world constraints:

- Jobs arrive online; the agent cannot know future arrivals
- A job, once started, runs continuously until it finishes (no preemption)
- Each job has fixed resource requirements (nodes, cores) that must be met simultaneously
- Jobs expire if they wait longer than `MAX_AGE` (336 hours)
- The cluster has a hard capacity cap (335 nodes, 96 cores/node)

The core lever available to the agent is **workload shifting**: delay jobs that arrive
during expensive price periods until prices drop, then burst through the backlog.

---

## Oracle Hierarchy

Four levels of oracle are possible, ordered from most optimistic (lowest cost, easiest
to compute) to most realistic (tightest bound, hardest to compute):

| Oracle | Arrival times | Deadlines | Job continuity | Node/core requirements | Complexity |
|---|:---:|:---:|:---:|:---:|---|
| **Liquid** (implemented) | ✗ | ✗ | ✗ | ✗ | Trivial — sort + greedy fill |
| **LP relaxation** | ✓ | ✓ | ✗ | ✓ | Easy — standard LP solver |
| **Job-contiguous greedy** | ✓ | ✓ | ✓ | ✓ | Fast heuristic |
| **ILP (exact optimal)** | ✓ | ✓ | ✓ | ✓ | NP-hard, intractable at scale |

---

## Oracle 1 — Liquid Lower Bound (Implemented)

### Idea

Treat all arriving work as a divisible fluid. Compute total core-hours demanded by the
episode's workload, then redistribute that fluid freely across all 336 hours, filling the
cheapest hours first up to cluster capacity.

### Algorithm

1. Accumulate `W = Σ (duration × nodes × cores_per_node)` over all arriving jobs.
2. Sort the 336 hourly prices cheapest-first.
3. Fill each hour up to `335 × 96 = 32,160` core-hours until `W` is exhausted.
4. Cost = `Σ_t  (core_hours_t / 96) × 0.45 kW × price_t`

### What it ignores

- Jobs cannot be split across hours; a 100-hour job on 4 nodes is **not** equivalent to
  400 node-hours of fluid.
- Work cannot be done before jobs arrive.
- Jobs have deadlines.

### When it is useful

The liquid oracle is a fast sanity check. If the agent's cost is far above the liquid
oracle, there is meaningful room to improve. If the agent's cost is close, the agent is
near-optimal (or the oracle bound is loose for this workload).

It is particularly loose for workloads with many long-running jobs, where splitting is
physically impossible. It is fairly tight for workloads with many short jobs.

---

## Oracle 2 — Job-Contiguous Greedy

### Idea

Honor all real job constraints (arrival times, deadlines, continuous execution, fixed
resource requirements) but give the scheduler **perfect foresight**: it sees all future
prices and arrivals before making any decision. It assigns each job to the cheapest
contiguous time window in which it fits.

### Algorithm

**Setup** (run once at episode end):
- `capacity[t]` = cores committed at hour `t`, initialized to 0 for all 336 hours.
- Full price series `price[0..335]`.
- Full job list with `(arrival, duration, nodes, cores)` for every job.

**For each job** (sorted by decreasing scheduling difficulty — see below):

1. Determine the feasibility window: start time `t` must satisfy
   `arrival ≤ t ≤ arrival + MAX_AGE − duration`
2. For each candidate start `t` in that window, check:
   `max(capacity[t : t+duration]) + nodes × cores ≤ 335 × 96`
3. Among all feasible candidates, pick the cheapest:
   `t* = argmin Σ_{h=t}^{t+duration−1} price[h]`
4. Commit: `capacity[t* : t*+duration] += nodes × cores`
5. Accumulate cost.

Both the feasibility check and the cost sum are numpy slice operations, accelerated to
O(1) per candidate with prefix sums.

**Sort order** (matters for greedy quality):

The recommended order is **least-slack first**: sort by
`(feasibility window size) / duration` ascending, so jobs with the fewest placement
options get first pick. This is analogous to Earliest Deadline First scheduling and
produces tighter, more realistic assignments than arrival-time order.

### Complexity

`O(N_jobs × MAX_AGE)` per episode with prefix-sum acceleration. For a typical episode
(~50,000 jobs, 336-hour window) this is approximately 17 million simple operations —
fast enough to run at episode end with negligible overhead.

### What it still ignores vs. the real agent

- **Perfect foresight**: the oracle knows all future prices and arrivals; the agent does
  not.
- **No scheduling overhead**: the oracle makes one globally optimal assignment; the agent
  takes one action per hour under uncertainty.

These are the only remaining freedoms the oracle has over the agent. Everything else
(arrival times, deadlines, continuity, resource requirements) is respected.

---

## How the Benchmarks Fit Together

With both oracles implemented, a single evaluation run produces:

```
baseline_cost          greedy FIFO, all nodes always on
baseline_cost_off      greedy FIFO, idle nodes turned off
agent_cost             trained RL agent
oracle_cost            liquid oracle (optimistic lower bound)
oracle_contiguous_cost job-contiguous greedy oracle (tight lower bound)
```

This gives three meaningful gaps:

| Gap | Meaning |
|---|---|
| `baseline_off − agent` | Savings the agent actually achieved |
| `baseline_off − oracle_jcg` | Maximum workload-shifting savings available in principle |
| `agent − oracle_jcg` | How far the agent is from the realistic optimum |
| `oracle_jcg − oracle_liq` | Cost of honoring job-continuity constraints |

The key metric for evaluating agent quality becomes:

```
Agent Capture Rate = (baseline_off − agent) / (baseline_off − oracle_jcg)
```

A capture rate of 80% means the agent recovers 80% of the theoretically achievable
workload-shifting savings, with the remaining 20% left on the table.

---

## Implementation Status

| Component | Status |
|---|---|
| Liquid oracle (`src/oracle.py`) | ✅ Implemented |
| Wired into simulation (runs alongside every episode) | ✅ Implemented |
| `--oracle` flag in `train.py` and `train_iter.py` | ✅ Implemented |
| Oracle cost reported in evaluation output | ✅ Implemented |
| Job-contiguous greedy oracle | ✅ Implemented |
