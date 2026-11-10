#!/usr/bin/env python3
"""
Summarise the held-out-year evaluations of a training grid (as written by cluster/run_v2.sh).

Each run is a session <config>_s<seed>_test<year> with models/*/validation.jsonl (validation on
the training years) and eval/m<step>_x<scale>_test<year>.json (analyze_eval_seeds.py output:
one year, 364 days, on the held-out price year, averaged over evaluation seeds). <config> is the
curriculum (paper, de, e), optionally with a weights tag (e.g. de_e2p2s6); runs are grouped by it.

For every run and wait budget, the checkpoint is the one validation picks (src.validation:
best feasible checkpoint with average wait <= budget at the worst scale), plus "last" (8M steps).
The script then reports, per config, budget and arrival scale, the held-out savings vs
baseline_off (mean ± std over runs), wait, completion and lost jobs; savings per held-out year;
and every evaluated checkpoint that lost jobs.

Example:
  python analyze_grid.py --sessions sessions/v2_cluster
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

import numpy as np

from src.validation import WAIT_BUDGETS, best_per_budget, budget_label, load_history

RUN_NAME = re.compile(r"^(?P<config>.+)_s(?P<seed>\d+)_test(?P<year>\d{4})$")
EVAL_NAME = re.compile(r"^m(?P<step>\d+)_x(?P<scale>[\d.]+)_test\d{4}\.json$")


def load_runs(sessions: str, last_step: int) -> list[dict]:
    runs = []
    for session in sorted(glob.glob(os.path.join(sessions, "*", ""))):
        name = os.path.basename(session.rstrip(os.sep))
        m = RUN_NAME.match(name)
        models = glob.glob(os.path.join(session, "models", "*", ""))
        if not m or not models:
            continue
        evals = {}
        for path in glob.glob(os.path.join(session, "eval", "*.json")):
            em = EVAL_NAME.match(os.path.basename(path))
            if em:
                with open(path) as f:
                    stats = json.load(f)["stats"]
                evals[(int(em["step"]), em["scale"])] = {k: v["mean"] for k, v in stats.items()}
        best = best_per_budget(load_history(models[0]))
        picks = {label: (e["step"] if e else None) for label, e in best.items()}
        picks["last"] = last_step
        runs.append(dict(name=name, **m.groupdict(), picks=picks, evals=evals))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", default="sessions/v2", help="Directory holding the run sessions (default: sessions/v2).")
    parser.add_argument("--last-step", type=int, default=8_000_000, help="Step of the last checkpoint (default: 8000000).")
    parser.add_argument("--out", default="", help="CSV with one row per run, budget and scale (default: <sessions>/grid_eval.csv).")
    args = parser.parse_args()

    runs = load_runs(args.sessions, args.last_step)
    if not runs:
        raise SystemExit(f"no <config>_s<seed>_test<year> sessions with models under {args.sessions}")
    labels = [budget_label(b) for b in WAIT_BUDGETS] + ["last"]
    scales = sorted({s for r in runs for (_, s) in r["evals"]}, key=float, reverse=True)
    configs = sorted({r["config"] for r in runs})

    rows, missing = [], []
    for r in runs:
        for label in labels:
            step = r["picks"].get(label)
            if step is None:
                continue
            for scale in scales:
                e = r["evals"].get((step, scale))
                if e is None:
                    missing.append(f"{r['name']} {label} {step} x{scale}")
                    continue
                rows.append(dict(run=r["name"], config=r["config"], seed=r["seed"], test_year=r["year"], budget=label,
                                 step=step, scale=scale, **{k: e.get(k) for k in (
                                     "savings_off_pct", "avg_wait_h", "completion_pct", "baseline_completion_pct",
                                     "jobs_lost", "capture_contiguous_pct")}))

    def sel(config, label, scale, key, year=None):
        return [x[key] for x in rows if x["config"] == config and x["budget"] == label and x["scale"] == scale
                and (year is None or x["test_year"] == year) and x.get(key) is not None]

    print(f"{len(runs)} runs, {len(rows)} evaluated picks" + (f", {len(missing)} picks not evaluated" if missing else ""))
    for scale in scales:
        print(f"\nHeld-out year, arrival scale x{scale}: savings vs baseline_off % (mean ± std over runs, n, range), "
              "average wait h, completion %, lost jobs (sum)")
        print(f"{'config':14s} {'budget':6s} {'savings':>17s} {'range':>15s} {'wait':>6s} {'compl':>7s} {'lost':>9s}")
        for config in configs:
            for label in labels:
                v = sel(config, label, scale, "savings_off_pct")
                if not v:
                    print(f"{config:14s} {label:6s} {'—':>17s}")
                    continue
                w, c, lost = (sel(config, label, scale, k) for k in ("avg_wait_h", "completion_pct", "jobs_lost"))
                print(f"{config:14s} {label:6s} {np.mean(v):+6.1f} ± {np.std(v):4.1f} ({len(v):2d}) "
                      f"{min(v):+6.1f} … {max(v):+6.1f} {np.mean(w):6.1f} {np.mean(c):7.2f} {int(sum(lost)):9d}")

    years = sorted({r["year"] for r in runs})
    for scale in scales:
        print(f"\nSavings % per held-out year, x{scale} (mean over seeds)")
        print(f"{'config':14s} {'budget':6s} " + " ".join(f"{y:>8s}" for y in years))
        for config in configs:
            for label in ("12h", "any", "last"):
                cells = [sel(config, label, scale, "savings_off_pct", y) for y in years]
                print(f"{config:14s} {label:6s} " + " ".join(f"{np.mean(c):+8.1f}" if c else f"{'—':>8s}" for c in cells))

    lossy = sorted({(x["run"], x["step"], x["scale"], x["jobs_lost"], x["avg_wait_h"], x["savings_off_pct"])
                    for x in rows if (x.get("jobs_lost") or 0) > 0})
    if lossy:
        print("\nEvaluated checkpoints that lost jobs on the held-out year:")
        for run, step, scale, lost, wait, sav in lossy:
            print(f"  {run:22s} {step / 1e6:4.1f}M x{scale}: lost {lost:,.0f}, wait {wait:.1f} h, savings {sav:+.1f}%")
    if missing:
        print("\nNot evaluated: " + "; ".join(missing))

    out = args.out or os.path.join(args.sessions, "grid_eval.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWritten {out}")


if __name__ == "__main__":
    main()
