#!/usr/bin/env python3
"""
Summarise rule-based controller evaluations (cluster/eval_heuristics.sh): per setting and arrival
scale, the mean ± standard deviation over the price years (each year is the mean over its
evaluation seeds), as in the paper's table, with energy and effective price relative to the
idle-free baseline.

  python analyze_heuristics.py [--dir sessions/heuristics] [--tag test] [--out heuristics.csv]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="sessions/heuristics")
    p.add_argument("--tag", default="test", help="Result tag of cluster/eval_heuristics.sh (default: test).")
    p.add_argument("--out", default="", help="Write the per-year table to this CSV.")
    a = p.parse_args()

    rows = []
    for f in glob.glob(os.path.join(a.dir, "test*", "eval", f"*_{a.tag}.json")):
        year = re.search(r"test(\d{4})", f)[1]
        m = re.match(rf"(.+)_x([\d.]+)_{re.escape(a.tag)}\.json$", os.path.basename(f))
        for r in json.load(open(f))["runs"]:
            rows.append(dict(
                setting=m[1], scale=float(m[2]), year=year, seed=r["seed"],
                savings=r["savings_off_pct"], wait=r["avg_wait_h"],
                lost=r["jobs_lost"] - r["baseline_jobs_lost"],
                completion=100 * r["jobs_completed"] / r["baseline_jobs_completed"],
                energy=100 * r["agent_energy_mwh"] / r["baseline_off_energy_mwh"],
                price=100 * r["agent_mean_price"] / r["baseline_off_mean_price"],
            ))
    if not rows:
        raise SystemExit(f"no results with tag {a.tag!r} under {a.dir}")
    d = pd.DataFrame(rows)
    years = d.groupby(["setting", "scale", "year"]).mean(numeric_only=True).reset_index().drop(columns="seed")
    g = years.groupby(["setting", "scale"])
    out = g[["savings", "wait", "energy", "price", "completion", "lost"]].mean()
    out.insert(1, "savings_sd", g.savings.std())
    out.insert(3, "wait_sd", g.wait.std())
    out["years"] = g.size()
    out["seeds"] = d.groupby(["setting", "scale"]).seed.nunique()

    pd.set_option("display.width", 200)
    for scale in sorted(d.scale.unique()):
        print(f"\n=== arrival scale {scale} ===")
        print(out.xs(scale, level="scale").sort_values("wait").round(2).to_string())
    if a.out:
        years.to_csv(a.out, index=False)
        print(f"\nPer-year table written to {a.out}")


if __name__ == "__main__":
    main()
