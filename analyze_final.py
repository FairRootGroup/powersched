#!/usr/bin/env python3
"""
Numbers of the paper's results from the final-checkpoint evaluations (cluster/eval_final.sh: final
checkpoint, 8M steps, evaluation seeds 101/201/301, result tag prefix f101_), one block per section.

  python analyze_final.py [--sessions sessions/v2] [--heuristics sessions/heuristics] [--section NAME]

A run's value is the mean over its evaluation seeds; means and sample standard deviations (n - 1) are
over the runs of a weight combination. Savings, energy and effective price are relative to the
idle-free baseline unless stated otherwise. "Lossy" runs lose at least one job in at least one seed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

ORDER = ["e2p2s6", "e3p3s4", "e4p4s2", "e3p5s2", "e2p6s2"]
STEP = 8000000
SESSION_RE = re.compile(r"^(de|e|paper)(?:_(e\dp\ds\d))?_s(\d)_test(\d{4})$")


def run_row(path: str) -> dict:
    """Mean over evaluation seeds of one evaluation file, plus derived ratios."""
    runs = json.load(open(path))["runs"]
    mean = lambda key: float(np.mean([r[key] for r in runs]))
    row = {k: mean(k) for k in ["savings_off_pct", "savings_pct", "avg_wait_h", "jobs_lost", "agent_cost",
                                 "baseline_off_cost", "agent_energy_mwh", "baseline_off_energy_mwh",
                                 "agent_neg_price_energy_mwh", "agent_neg_price_cost",
                                 "baseline_off_neg_price_energy_mwh", "baseline_off_neg_price_cost",
                                 "negative_price_hours", "jobs_submitted"]}
    row["lossy"] = any(r["jobs_lost"] > 0 for r in runs)
    row["lost_total"] = sum(r["jobs_lost"] for r in runs)
    row["completion_vs_base"] = mean_ratio(runs, "jobs_completed", "baseline_jobs_completed")
    row["energy_ratio"] = mean_ratio(runs, "agent_energy_mwh", "baseline_off_energy_mwh")
    row["price_ratio"] = float(np.mean([(r["agent_cost"] / r["agent_energy_mwh"]) /
                                        (r["baseline_off_cost"] / r["baseline_off_energy_mwh"]) * 100 for r in runs]))
    if all(r.get("oracle_contiguous_cost") for r in runs):
        row["headroom"] = mean_ratio(runs, "oracle_contiguous_cost", "baseline_off_cost", complement=True)
        row["capture"] = float(np.mean([r["capture_contiguous_pct"] for r in runs]))
    row["seed_sd"] = float(np.std([r["savings_off_pct"] for r in runs], ddof=1))
    row["jobs_per_hour"] = row["jobs_submitted"] / 8736
    return row


def mean_ratio(runs, num, den, complement=False):
    values = [100 * r[num] / r[den] for r in runs]
    return float(np.mean([100 - v for v in values] if complement else values))


def load(sessions: str, tag: str, scales=("1.0", "1.5"), curricula=("de",)) -> pd.DataFrame:
    rows = []
    for sd in sorted(glob.glob(os.path.join(sessions, "*"))):
        m = SESSION_RE.match(os.path.basename(sd))
        if not m or m[1] not in curricula:
            continue
        for scale in scales:
            f = os.path.join(sd, "eval", f"m{STEP}_x{scale}_{tag}.json")
            if os.path.exists(f):
                rows.append(dict(curriculum=m[1], weights=m[2] or "e4p4s2", seed=int(m[3]), heldout=m[4],
                                 scale=scale, **run_row(f)))
    return pd.DataFrame(rows)


def fmt(x, sd=None, digits=1):
    return f"{x:.{digits}f}" if sd is None else f"{x:.{digits}f} ± {sd:.{digits}f}"


def per_weights(d: pd.DataFrame, cols, by=("weights", "scale")):
    g = d.groupby(list(by))
    out = g[cols].mean()
    for c in cols:
        out[c + "_sd"] = g[c].std()
    out["lossy_runs"] = g["lossy"].sum()
    out["runs"] = g.size()
    return out


def section_main(a):
    d = load(a.sessions, "f101_heldout")
    print(f"== Main results: held-out years ({len(d)} run-scales; lossy {int(d.lossy.sum())})")
    t = per_weights(d, ["savings_off_pct", "avg_wait_h"])
    for w in ORDER:
        cells = [f"x{s}: {fmt(t.loc[(w, s), 'savings_off_pct'], t.loc[(w, s), 'savings_off_pct_sd'])} % / "
                 f"{fmt(t.loc[(w, s), 'avg_wait_h'], t.loc[(w, s), 'avg_wait_h_sd'])} h" for s in ("1.0", "1.5")]
        print(f"  {w}: " + " | ".join(cells))
    for s in ("1.0", "1.5"):
        x = d[d.scale == s]
        m = x.groupby("weights")
        print(f"  x{s}: completion vs idle-free {x.completion_vs_base.min():.1f}..{x.completion_vs_base.max():.1f} %, "
              f"effective price {m.price_ratio.mean().min():.0f}..{m.price_ratio.mean().max():.0f} % "
              f"(i.e. {100 - m.price_ratio.mean().max():.0f}..{100 - m.price_ratio.mean().min():.0f} % below), "
              f"energy {m.energy_ratio.mean().min():.0f}..{m.energy_ratio.mean().max():.0f} %, "
              f"vs always-on {m.savings_pct.mean().min():.1f}..{m.savings_pct.mean().max():.1f} %, "
              f"jobs/h {x.jobs_per_hour.mean():.0f}")


def section_crossyear(a):
    held = load(a.sessions, "f101_heldout", scales=("1.0",))
    print("== Cross-year check (x1.0): savings on a year by runs trained on it minus by runs holding it out")
    diffs, best_year = {}, []
    for w in ORDER:
        dw = []
        for y in ("2023", "2024", "2025"):
            seen = load(a.sessions, f"f101_in{y}", scales=("1.0",))
            seen = seen[seen.weights == w]
            dw.append(seen.savings_off_pct.mean() - held[(held.weights == w) & (held.heldout == y)].savings_off_pct.mean())
        diffs[w] = np.mean(dw)
        print(f"  {w}: {diffs[w]:+.1f} (per year {', '.join(f'{x:+.1f}' for x in dw)})")
    # Which year gives each run its highest savings (training years and held-out year, x1.0)?
    per_run = {}
    for y in ("2023", "2024", "2025"):
        for _, r in load(a.sessions, f"f101_in{y}", scales=("1.0",)).iterrows():
            per_run.setdefault((r.weights, r.seed, r.heldout), {})[y] = r.savings_off_pct
    for _, r in held.iterrows():
        per_run.setdefault((r.weights, r.seed, r.heldout), {})[r.heldout] = r.savings_off_pct
    tops = [max(v, key=v.get) for v in per_run.values() if len(v) == 3]
    print(f"  best year per run: {pd.Series(tops).value_counts().to_dict()} of {len(tops)} runs")


def section_2022(a):
    held = load(a.sessions, "f101_heldout")
    real = load(a.sessions, "f101_new2022")
    scaled = load(a.sessions, "f101_scaled2022", scales=("1.5",))
    print("== 2022 (main trace)")
    for s in ("1.0", "1.5"):
        x = real[real.scale == s]
        print(f"  x{s}: lossy runs {int(x.lossy.sum())}/45, lost jobs {x.lost_total.sum():,.0f} (all seeds), "
              f"min savings {x.savings_off_pct.min():.1f} %, runs with savings <= 0: {int((x.savings_off_pct <= 0).sum())}")
    print("  table (x1.5): held out | 2022 without lossy runs (n) | lossy runs real | scaled")
    for w in ORDER:
        h = held[(held.weights == w) & (held.scale == "1.5")]
        r = real[(real.weights == w) & (real.scale == "1.5")]
        ok = r[~r.lossy]
        sc = scaled[scaled.weights == w]
        print(f"    {w}: {fmt(h.savings_off_pct.mean(), h.savings_off_pct.std())} | "
              f"{fmt(ok.savings_off_pct.mean(), ok.savings_off_pct.std())} ({len(ok)}), wait {ok.avg_wait_h.mean():.1f} h | "
              f"{int(r.lossy.sum())} | {int(sc.lossy.sum())}")
    r15 = real[real.scale == "1.5"]
    print(f"  rescaled x1.5: lossy {int(scaled.lossy.sum())}/45 ({scaled.lost_total.sum():,.0f} jobs, all seeds) "
          f"vs real {int(r15.lossy.sum())}/45 ({r15.lost_total.sum():,.0f})")
    both = r15.merge(scaled, on=["weights", "seed", "heldout"], suffixes=("_real", "_scaled"))
    ok = both[~both.lossy_real & ~both.lossy_scaled]
    print(f"  runs without losses at either level: {len(ok)}; savings {ok.savings_off_pct_real.mean():.1f} vs "
          f"{ok.savings_off_pct_scaled.mean():.1f} %, waits {ok.avg_wait_h_real.mean():.1f} vs {ok.avg_wait_h_scaled.mean():.1f} h")
    hok = r15[~r15.lossy]
    hy = held[held.scale == "1.5"]
    print(f"  headroom (x1.5): 2022 {real.headroom.mean():.0f} %, held-out years "
          f"{hy.groupby('heldout').headroom.mean().round(0).to_dict()}; capture 2022 (no-loss runs) {hok.capture.mean():.0f} %, "
          f"held-out years {hy.groupby('heldout').capture.mean().round(0).to_dict()}")


def section_trace(a):
    main = load(a.sessions, "f101_heldout")
    tr = load(a.sessions, "f101_trace2611", scales=("0.6", "0.9", "1.0", "1.5"))
    tr22 = load(a.sessions, "f101_trace2611_new2022", scales=("0.6", "0.9", "1.0", "1.5"))
    m22 = load(a.sessions, "f101_new2022")
    print("== Second trace (matched: main x1.0 ~ second x0.6, main x1.5 ~ second x0.9)")
    jph = lambda d, s: d[d.scale == s].jobs_per_hour.mean()
    print(f"  jobs/h: main {jph(main, '1.0'):.0f}/{jph(main, '1.5'):.0f}; second "
          + "/".join(f"{jph(tr, s):.0f}" for s in ("0.6", "0.9", "1.0", "1.5")))
    print(f"  lossy at matched rates: {int(tr[tr.scale.isin(['0.6', '0.9'])].lossy.sum())}")
    diffs, waits = [], []
    for w in ORDER:
        cells = []
        for sm, st in (("1.0", "0.6"), ("1.5", "0.9")):
            x = main[(main.weights == w) & (main.scale == sm)]
            y = tr[(tr.weights == w) & (tr.scale == st)]
            cells.append(f"{fmt(x.savings_off_pct.mean(), x.savings_off_pct.std())} {x.avg_wait_h.mean():.1f} h -> "
                         f"{fmt(y.savings_off_pct.mean(), y.savings_off_pct.std())} {y.avg_wait_h.mean():.1f} h")
            diffs.append(y.savings_off_pct.mean() - x.savings_off_pct.mean())
            waits.append(100 * (y.avg_wait_h.mean() / x.avg_wait_h.mean() - 1))
        print(f"  {w}: " + " | ".join(cells))
    print(f"  savings difference {min(diffs):+.1f}..{max(diffs):+.1f} points, waits {min(waits):+.0f}..{max(waits):+.0f} %")
    for s in ("1.0", "1.5"):
        x = tr[tr.scale == s]
        g = x.groupby("weights")
        print(f"  second x{s}: lossy {g.lossy.sum().loc[ORDER].astype(int).to_dict()}, "
              f"savings {g.savings_off_pct.mean().loc[ORDER].round(1).to_dict()}, wait {g.avg_wait_h.mean().loc[ORDER].round(1).to_dict()}")
    for s in ("0.6", "0.9", "1.0", "1.5"):
        x = tr22[tr22.scale == s]
        print(f"  second + 2022 x{s}: lossy {int(x.lossy.sum())}/45 {x.groupby('weights').lossy.sum().loc[ORDER].astype(int).to_dict()}")
    for s in ("1.0", "1.5"):
        print(f"  main + 2022 x{s}: lossy {int(m22[m22.scale == s].lossy.sum())}/45")


def section_curricula(a):
    d = load(a.sessions, "f101_heldout", curricula=("de", "e", "paper"))
    d = d[d.weights == "e4p4s2"]
    print("== Curricula (e4p4s2, 9 runs each)")
    for c in ("de", "e", "paper"):
        for s in ("1.0", "1.5"):
            x = d[(d.curriculum == c) & (d.scale == s)]
            print(f"  {c} x{s}: {fmt(x.savings_off_pct.mean(), x.savings_off_pct.std())} %, "
                  f"wait {fmt(x.avg_wait_h.mean(), x.avg_wait_h.std())} h, savings range {x.savings_off_pct.min():.1f}..{x.savings_off_pct.max():.1f}, "
                  f"lossy {int(x.lossy.sum())}, lost {x.lost_total.sum():,.0f}")


def section_variability(a):
    d = load(a.sessions, "f101_heldout", scales=("1.5",))
    print("== Variability (held-out years, x1.5)")
    for w in ORDER:
        x = d[d.weights == w]
        pooled = np.sqrt(x.groupby("heldout").savings_off_pct.var().mean())
        pooled_wait = np.sqrt(x.groupby("heldout").avg_wait_h.var().mean())
        years = x.groupby("heldout").savings_off_pct.mean()
        print(f"  {w}: training seed (pooled SD within year) {pooled:.1f}, price year (SD of year means) {years.std():.1f}, "
              f"evaluation seed (mean SD within run) {x.seed_sd.mean():.1f}; best year {years.idxmax()}; "
              f"savings {x.savings_off_pct.min():.1f}..{x.savings_off_pct.max():.1f}, wait {x.avg_wait_h.min():.1f}..{x.avg_wait_h.max():.1f} h, "
              f"wait pooled SD {pooled_wait:.1f}")


def section_exposure(a):
    d = load(a.sessions, "f101_heldout", scales=("1.5",))
    d["neg_share"] = 100 * d.agent_neg_price_energy_mwh / d.agent_energy_mwh
    d["neg_share_base"] = 100 * d.baseline_off_neg_price_energy_mwh / d.baseline_off_energy_mwh
    d["neg_points"] = 100 * (d.baseline_off_neg_price_cost - d.agent_neg_price_cost) / d.baseline_off_cost
    d["price_pos"] = (d.agent_cost - d.agent_neg_price_cost) / (d.agent_energy_mwh - d.agent_neg_price_energy_mwh)
    d["price_pos_base"] = ((d.baseline_off_cost - d.baseline_off_neg_price_cost) /
                           (d.baseline_off_energy_mwh - d.baseline_off_neg_price_energy_mwh))
    d["price_pos_ratio"] = 100 * d.price_pos / d.price_pos_base
    print("== Negative-price exposure (held-out years, x1.5; R3.2)")
    cols = ["savings_off_pct", "agent_cost", "baseline_off_cost", "agent_energy_mwh", "baseline_off_energy_mwh",
            "neg_share", "neg_share_base", "neg_points", "price_pos_ratio"]
    print(d.groupby("weights")[cols].mean().loc[ORDER].round(2).to_string())


def section_heuristics(a):
    rl = load(a.sessions, "f101_heldout")
    print("== RL reference for the rule-based comparison")
    for s in ("1.0", "1.5"):
        g = rl[rl.scale == s].groupby("weights")
        print(f"  x{s}: energy {g.energy_ratio.mean().min():.0f}..{g.energy_ratio.mean().max():.0f} %, "
              f"price {g.price_ratio.mean().min():.0f}..{g.price_ratio.mean().max():.0f} %; "
              + ", ".join(f"{w} {g.savings_off_pct.mean()[w]:.1f} %/{g.avg_wait_h.mean()[w]:.1f} h" for w in ORDER))
    print("  rule settings: python analyze_heuristics.py --tag f101")


def section_probe(a):
    print(f"== Observation ablation: python analyze_probe.py --sessions {a.sessions} --step {STEP} --seeds 101,201,301")


SECTIONS = {"main": section_main, "crossyear": section_crossyear, "2022": section_2022, "trace": section_trace,
            "curricula": section_curricula, "variability": section_variability, "exposure": section_exposure,
            "heuristics": section_heuristics, "probe": section_probe}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sessions", default="sessions/v2")
    p.add_argument("--section", choices=sorted(SECTIONS), action="append", help="Only these sections (default: all).")
    a = p.parse_args()
    for name in a.section or SECTIONS:
        SECTIONS[name](a)
        print()


if __name__ == "__main__":
    main()
