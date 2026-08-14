# Evaluation results

CSV outputs from evaluating trained policies against baselines, used to produce
the figures/tables in `data-internal/paper/draft.tex`. Each row is one
(model, arrival-rate sweep point) evaluation.

## Files

- **`eval_main.csv`** - main-trace job arrivals under real 2023 day-ahead
  electricity prices, swept across arrival-rate `scale` 0.25–5.0. Backs the
  "real prices" overview/decomposition figures.
- **`eval_main_logic_price.csv`** - same main-trace arrivals and `scale`
  sweep, but under the deterministic two-level "logic price" signal (12h
  expensive / 12h cheap, 10–250 €/MWh) instead of real prices. Backs
  `fig_1_overview_main_logic_price` / `fig_2_decomp_main_logic_price`.
- **`eval_poisson.csv`** - synthetic Poisson-arrival workload under real
  prices, swept across `lambda` (250–3000 jobs/hour). Out-of-curriculum
  generalization check, backs the Poisson decomposition figure.

## Columns

- `model` - policy identifier (e.g. `e3p5s2`, `e4p4s2`, `e2p2s6`).
- `arrivals` - realized average jobs/hour for that run.
- `scale` / `lambda` - sweep parameter (arrival-rate scale factor, or Poisson
  arrival rate).
- `cost_agent`, `cost_baseline`, `cost_baseline_off` - total € cost for the
  agent, the always-on baseline, and the idle-free baseline.
- `savings_vs_baseline_pct`, `savings_vs_baseline_off_pct` - agent savings
  relative to each baseline, as a percentage.
- `savings_abs_vs_baseline_eur`, `savings_abs_vs_baseline_off_eur` - same
  savings, in absolute €.
- `cost_per_1k_agent/baseline/baseline_off` - € cost per 1,000 jobs
  processed; `cost_per_1k_delta` is the agent-vs-baseline_off difference.
- `completion_pct` - share of submitted jobs completed.
- `mean_wait_h` - mean job wait time in hours.
- `dropped_jobs`, `dropped_per_eur` - jobs lost to age expiry/queue-full
  rejection, and per € saved.
- `power_agent/baseline/baseline_off_mwh` - total energy consumption (MWh);
  `delta_power_mwh` is the agent-vs-baseline_off difference.
- `eur_per_mwh_agent/baseline/baseline_off` - effective electricity price
  paid; `delta_eur_per_mwh` is the agent-vs-baseline_off difference.
- `oracle_liquid_eur`, `oracle_contiguous_eur` - cost under perfect-foresight
  oracle baselines (unconstrained vs. contiguous scheduling).
  `oracle_capture_pct` - share of the idle-free-to-oracle savings gap the
  agent recovers.
- `savings_model_eur` - total modeled savings vs. baseline_off, decomposed
  into:
  - `power_savings_pct` / power-volume effect - savings from using less
    energy overall (implied by `delta_power_mwh`).
  - `price_savings_eur`, `price_savings_pct` - savings from shifting energy
    use toward cheaper price periods ("price-timing" effect), as opposed to
    just consuming less energy.

The energy-volume vs. price-timing split is the paper's central diagnostic
for distinguishing genuine price-aware scheduling from savings that merely
come from avoiding idle-node overhead.
