#!/usr/bin/env python3
"""
Summarise the observation ablation (cluster/eval_probe.sh): per run (mean over evaluation seeds),
per weight combination (mean over its runs) and, as in the paper's appendix table, the range over
the weight combinations per variant.

  python analyze_probe.py --sessions sessions/v2 [--scale 1.0] [--out probe_runs.csv]

Savings, energy and effective price are relative to the idle-free baseline; "lossy runs" counts
runs whose mean over seeds loses jobs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import pandas as pd

VARIANTS = ["full", "prices_roll12", "prices_const", "time_const", "aggregates_zero", "queue_summary_zero"]
ORDER = ["e2p2s6", "e3p3s4", "e4p4s2", "e3p5s2", "e2p6s2"]


def load(sessions: str, scale: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(os.path.join(sessions, "de*", "eval", "probe", f"m*_x{scale}_*_s*.json")):
        session = f.split(os.sep)[-4]
        m = re.match(r"de(?:_(e\dp\ds\d))?_s(\d+)_test(\d{4})$", session)
        v = re.match(rf"m(\d+)_x{re.escape(scale)}_(.+)_s(\d+)\.json$", os.path.basename(f))
        if not m or not v:
            continue
        r = json.load(open(f))
        rows.append(dict(
            weights=m[1] or "e4p4s2", run=session, step=int(v[1]), variant=v[2], seed=int(v[3]),
            savings=r["savings_pct_vs_idle_free_total"], wait=r["avg_wait_h"], lost=r["jobs_lost"],
            energy=100 * r["agent_energy_mwh"] / r["baseline_off_energy_mwh"],
            price=100 * (r["agent_cost"] / r["agent_energy_mwh"]) / (r["baseline_off_cost"] / r["baseline_off_energy_mwh"]),
        ))
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sessions", default="sessions/v2")
    p.add_argument("--scale", default="1.0")
    p.add_argument("--out", default="", help="Write the per-run table to this CSV.")
    a = p.parse_args()

    d = load(a.sessions, a.scale)
    if d.empty:
        raise SystemExit(f"no probe results at x{a.scale} under {a.sessions}")
    runs = d.groupby(["weights", "run", "step", "variant"]).mean(numeric_only=True).reset_index().drop(columns="seed")
    full = runs[runs.variant == "full"].set_index("run")["savings"]
    runs["delta_savings"] = runs.savings - runs.run.map(full)
    print(f"{len(d)} evaluations, {runs.run.nunique()} runs, scale {a.scale}")

    pd.set_option("display.width", 200)
    variants = [v for v in VARIANTS if v in set(runs.variant)]
    weights = [w for w in ORDER if w in set(runs.weights)]
    for col in ["savings", "wait", "energy", "price"]:
        t = runs.pivot_table(index="weights", columns="variant", values=col, aggfunc="mean").loc[weights, variants]
        print(f"\n{col} (mean over runs)\n{t.round(2).to_string()}")

    print("\nRange over weight combinations (appendix table)")
    per_weights = runs.groupby(["variant", "weights"]).mean(numeric_only=True)
    for v in variants:
        x = per_weights.loc[v]
        lossy = int((runs[runs.variant == v].lost > 0).sum())
        n = int((runs.variant == v).sum())
        cells = "  ".join(f"{c} {x[c].min():.1f}..{x[c].max():.1f}" for c in ["savings", "wait", "energy", "price"])
        dsav = runs[runs.variant == v].delta_savings
        print(f"  {v:<20} {cells}  lossy runs {lossy}/{n}  change per run {dsav.min():+.1f}..{dsav.max():+.1f}")

    if a.out:
        runs.to_csv(a.out, index=False)
        print(f"\nPer-run table written to {a.out}")


if __name__ == "__main__":
    main()
