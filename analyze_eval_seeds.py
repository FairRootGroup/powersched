#!/usr/bin/env python3
"""
Evaluate one model over several evaluation seeds (different job arrivals) and report mean ± std.

Each seed runs `train.py --evaluate-savings --seed S --eval-tag <tag>_seedS <train args>` in its own
process; the per-seed totals come from the eval_summary.json that train.py writes. Everything after
`--` is passed to train.py unchanged (session, weights, model, prices, workload, dispatcher, ...).

Example (a held-out test year):
  python analyze_eval_seeds.py --seeds 1,2,3,4,5 --parallel 5 --tag prices2024 --out eval_2024.json -- \\
      --session v2_run --efficiency-weight 0.4 --price-weight 0.4 --idle-weight 0 --job-age-weight 0.2 \\
      --prices data-external/prices_2024.csv --hourly-jobs data-internal/allusers-main-30.log \\
      --job-arrival-scale 1.5 --flush-after-drop-streak 3 --model 8000000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Reported as mean ± std over seeds (when present in the summaries).
METRICS = [
    "savings_off_pct", "savings_pct", "prop_savings_off_pct", "prop_savings_pct",
    "avg_monthly_savings_pct_off", "completion_pct", "baseline_completion_pct",
    "jobs_lost", "baseline_jobs_lost", "avg_wait_h", "baseline_avg_wait_h",
    "agent_mean_price", "baseline_off_mean_price", "capture_contiguous_pct", "capture_liquid_pct",
]
SUMMARY_LINE = "Summary written to "


def run_seed(seed: int, tag: str, train_args: list[str], log_dir: str | None) -> dict:
    eval_tag = f"{tag}_seed{seed}" if tag else f"seed{seed}"
    command = [sys.executable, "train.py", "--evaluate-savings", "--seed", str(seed),
               "--eval-tag", eval_tag, *train_args]
    env = dict(os.environ, MPLBACKEND="Agg")
    done = subprocess.run(command, capture_output=True, text=True, env=env)
    output = done.stdout + done.stderr
    if log_dir:
        with open(os.path.join(log_dir, f"eval_{eval_tag}.log"), "w") as f:
            f.write(output)
    paths = [line[len(SUMMARY_LINE):].strip() for line in output.splitlines() if line.startswith(SUMMARY_LINE)]
    if done.returncode != 0 or not paths:
        tail = "\n".join(output.splitlines()[-30:])
        raise RuntimeError(f"seed {seed}: train.py failed (exit {done.returncode}) or wrote no summary:\n{tail}")
    with open(paths[-1]) as f:
        return json.load(f)


def main() -> None:
    if "--" not in sys.argv:
        sys.exit("usage: analyze_eval_seeds.py [options] -- <train.py arguments>  (see --help)")
    split = sys.argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="1,2,3,4,5", help="Comma-separated evaluation seeds (default: 1,2,3,4,5).")
    parser.add_argument("--parallel", type=int, default=1, help="Seeds evaluated at the same time (default: 1).")
    parser.add_argument("--tag", default="", help="Prefix of each seed's --eval-tag, e.g. the test year (default: none).")
    parser.add_argument("--out", default="", help="Write per-seed summaries and mean/std to this JSON file.")
    parser.add_argument("--log-dir", default="", help="Save each seed's train.py output here.")
    args = parser.parse_args(sys.argv[1:split])
    train_args = sys.argv[split + 1:]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        runs = list(pool.map(lambda s: run_seed(s, args.tag, train_args, args.log_dir or None), seeds))

    stats = {}
    for key in METRICS:
        values = [r[key] for r in runs if r.get(key) is not None]
        if values:
            stats[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                          "min": float(np.min(values)), "max": float(np.max(values))}

    print(f"\nModel step {runs[0]['model_step']}, prices {runs[0]['prices']}, {runs[0]['episodes']} episodes, seeds {seeds}")
    print(f"{'metric':<30} {'mean':>10} {'std':>8} {'min':>10} {'max':>10}   per seed")
    for key, s in stats.items():
        per_seed = ", ".join(f"{r[key]:.2f}" for r in runs if r.get(key) is not None)
        print(f"{key:<30} {s['mean']:>10.2f} {s['std']:>8.2f} {s['min']:>10.2f} {s['max']:>10.2f}   {per_seed}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"seeds": seeds, "train_args": train_args, "stats": stats, "runs": runs}, f, indent=2)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
