[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21888087.svg)](https://doi.org/10.5281/zenodo.21888087)

#### powersched

`python ./checkenv.py` - Check the environment.

`python ./testenv.py` - Short training run.

`python ./train.py` - Infinite training run with tensorboard logs and intermediate models save.

`python ./train_iter.py` - Sequentially launch `./train.py` with different weights.

`./train.py` accepts `--render` argument with "human" or "none" ("none" is the default). "none" trains silently, while "human" runs intentionally slower, adds some debug output and graph output after each episode.

### Curriculum Training

The current training setup is intentionally curriculum-based. The target behavior is not merely "use fewer nodes" or "wait longer"; it is the more specific policy:

- execute little work during expensive hours,
- defer safely while cheap hours are still ahead,
- then clear backlog aggressively during cheap hours,
- while keeping overdue backlog and job loss near zero.

This is now encoded directly in the environment and reward design:

- the agent sees a 24h price forecast window,
- cheap-hour execution is rewarded and expensive-hour execution is penalized,
- cheap hours penalize under-service when backlog exists,
- overdue backlog after the 24h grace period becomes intrinsically bad,
- and end-of-episode pending and overdue metrics make "saving money by not serving work" visible.

The practical reason for using a curriculum instead of only training longer is that the full problem has several easy but wrong local optima:

- serve immediately and ignore price timing,
- trickle a small amount of work continuously,
- or over-defer until backlog becomes unstable.

Those behaviors can produce tolerable short-horizon rewards, so simply running PPO for more steps does not guarantee discovery of the desired defer-then-clear policy. The curriculum reduces variance and improves credit assignment by first teaching the core phase behavior under deterministic logic prices and only then adding load, burstiness, realistic arrivals, price noise, and finally real prices.

Current intended sequence:

1. Stage A: flat arrivals + logic prices.
2. Stage B: high-load flat arrivals + logic prices.
3. Stage C: bursty arrivals + logic prices.
4. Stage D: main arrivals + logic prices.
5. Stage E: main arrivals + noisy logic prices.
6. Stage F: main arrivals + real prices.

In short: more steps on the full problem mostly improve whatever basin the optimizer already occupies; the curriculum is meant to make the correct basin discoverable first.

For a more formal write-up, see [curriculum_argument.md](curriculum_argument.md) and [curriculum_setup.md](curriculum_setup.md).

### Multi-run evaluation analysis

Use the CSV meta-analysis plotter to aggregate existing evaluation runs:

```bash
python meta-eval/plot_eval_oracle.py --multi-run
```

By default this reads `analysis/v2_e2p2s6`, `analysis/v2_e3p5s2`, and
`analysis/v2_e4p4s2`, and writes figures and tables under
`analysis/multi_run_meta_analysis/`. To select other model directories or an
output directory:

```bash
python meta-eval/plot_eval_oracle.py --multi-run \
  --input-dirs analysis/v2_e2p2s6 analysis/v2_e3p5s2 analysis/v2_e4p4s2 \
  --output-dir analysis/multi_run_meta_analysis
```

Each model directory must contain `s<train>_pr<year>_seed<eval>_jobs<workload>/eval_main.csv`
files covering 2023, 2024, and 2025, with three training seeds and three
evaluation seeds per trained model. Seed identifiers are parsed from directory
names. The workload suffix does not distinguish runs: duplicate
model/year/training-seed/evaluation-seed combinations are rejected. Rows align
by model and workload `scale`; stochastic `arrivals` values are averaged.

Aggregation first averages evaluation seeds, then training seeds, then years.
Each level retains its own sample standard deviation (`ddof=1`). Figures use
the resulting grand mean and, for complete metrics, the sample SD of all 27
original measurements.
The error bars describe observed spread, not a standard error or confidence
interval. Ratios and percentages are averaged from their per-run values.

The existing six figure layouts are saved as PNG and PDF with the suffix
`multi_run` (overridable with `--name-suffix`). Full-precision CSV intermediate
tables and separate, narrow LaTeX tables for each metric are written to
`multi_run_tables/` inside the output directory. Source evaluation CSVs are
never overwritten.

The CSV exports are `raw_measurements.csv`, `evaluation_seed.csv`,
`training_seed.csv`, `year.csv`, and `grand_total.csv`. Aggregate columns use
`<metric>_mean` and `<metric>_std_eval`, `_std_train`, `_std_year`, or
`_std_total`, plus `<metric>_n_valid` for the number of available inputs at
that level. LaTeX files use `<level>_<metric>.tex` and contain model, scale,
year, and training-seed identifiers where applicable. Signed contribution
plots derive signs per run before aggregation and also export these derived
metrics.

Validation rejects missing files, duplicate runs or rows, incomplete seed and
scale grids, conflicting model names, inconsistent schemas, non-numeric values,
and infinities. Blank metric values are skipped independently at each level;
other metrics from the same run still contribute. All 27 run rows are still
required. An entirely blank metric group remains unavailable (`NaN` in memory,
blank in CSV, `--` in LaTeX), and an SD needs at least two valid inputs.

When a metric has blanks, the evaluation-seed, training-seed, and year means
retain equal weighting over the available inputs at each level. The final
spread is calculated from the available raw values around that hierarchical
grand mean, using `n_valid - 1` as the denominator. With uneven missingness,
this center can differ from the direct mean of the available raw values.
The tables report valid counts, so skipped values are visible. For example,
the two blank `dropped_per_eur` values in the supplied dataset leave 26 valid
measurements for each affected model/scale, while other metrics retain 27.

Omit `--multi-run` to retain the existing single-CSV behavior:

```bash
python meta-eval/plot_eval_oracle.py path/to/eval_main.csv
```
