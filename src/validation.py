"""Validation of training checkpoints: a fixed, deterministic evaluation used to pick the best ones.

Every checkpoint is run (argmax actions) on the same windows of the same validation data: for each
arrival scale one env, and for each window a hard reset with a fixed seed and price start index, so
arrivals and prices are identical for all checkpoints and only the agent's decisions differ.

A checkpoint is feasible if, at every arrival scale, it loses no more jobs than the baseline
(optionally also: completes at least `min_completion_ratio` of the baseline's jobs; off by default,
because jobs still queued at a window's end count as not completed, which turns the completion
ratio into a hidden wait limit). Its score is the savings vs baseline_off (the baseline without idle
nodes), in percent, averaged over the scales.

How long jobs may wait depends on the jobs, so there is no single best checkpoint: for each wait
budget (average wait in hours at the worst scale; WAIT_BUDGETS) the best feasible checkpoint within
it is recorded. Feasibility and budgets are recomputed from the raw figures in the history, so a
history written under another rule is re-ranked as it is.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.config import EPISODE_HOURS

HISTORY_FILE = "validation.jsonl"
BEST_FILE = "validation_best.json"
WAIT_BUDGETS: tuple[float | None, ...] = (6.0, 12.0, 24.0, None)  # None = no limit


@dataclass
class ValidationConfig:
    arrival_scales: tuple[float, ...] = (1.0, 1.5)
    windows: int = 4                 # windows spread evenly over the validation prices
    episodes_per_window: int = 2     # 2-week episodes run back to back within a window
    seed: int = 12345                # window w uses seed + w
    min_completion_ratio: float = 0.0  # 0 = off


def window_starts(n_prices: int | None, windows: int) -> list[int]:
    """Price start index of each window: evenly spread, centred in its share of the data."""
    if n_prices is None:  # synthetic prices repeat daily; any start works
        return [w * EPISODE_HOURS for w in range(windows)]
    return [int((w + 0.5) * n_prices / windows) for w in range(windows)]


def budget_label(budget: float | None) -> str:
    return "any" if budget is None else f"{budget:g}h"


def scale_feasible(s: dict, min_completion_ratio: float = 0.0) -> bool:
    return bool(s["jobs_lost"] <= s["baseline_jobs_lost"] and s["completion_ratio"] >= min_completion_ratio)


def entry_feasible(entry: dict, min_completion_ratio: float = 0.0) -> bool:
    return all(scale_feasible(s, min_completion_ratio) for s in entry["scales"].values())


def entry_wait(entry: dict) -> float:
    """Average wait at the worst scale (inf if the entry predates wait tracking)."""
    return max(s.get("avg_wait_h", math.inf) for s in entry["scales"].values())


def best_per_budget(history: list[dict], budgets=WAIT_BUDGETS, min_completion_ratio: float = 0.0) -> dict:
    """{budget label: the feasible entry with the highest savings and wait within the budget, or None}."""
    best = {}
    for budget in budgets:
        ok = [h for h in history if entry_feasible(h, min_completion_ratio)
              and (budget is None or entry_wait(h) <= budget)]
        best[budget_label(budget)] = max(ok, key=lambda h: h["savings_off_pct"]) if ok else None
    return best


def run_validation(model, make_env: Callable[[float], object], config: ValidationConfig,
                   envs: dict | None = None) -> dict:
    """
    Run the deterministic validation for `model`. `make_env(arrival_scale)` builds an evaluation env;
    pass a dict as `envs` to keep the envs between calls (they are reused with hard resets).
    """
    if envs is None:
        envs = {}
    per_scale = {}
    for scale in config.arrival_scales:
        env = envs.get(scale)
        if env is None:
            env = envs[scale] = make_env(scale)
        n_prices = None if env.prices.external_prices is None else len(env.prices.external_prices)
        totals = dict(agent_cost=0.0, baseline_cost_off=0.0, jobs_submitted=0, jobs_completed=0,
                      baseline_jobs_completed=0, jobs_launched=0, jobs_lost=0, baseline_jobs_lost=0,
                      pending_jobs_end=0, overdue_jobs_end=0, episodes=0)
        wait_hours = 0.0  # summed over launched jobs
        for w, start in enumerate(window_starts(n_prices, config.windows)):
            obs, _ = env.reset(seed=config.seed + w, options={"hard_reset": True, "price_start_index": start})
            for episode in range(config.episodes_per_window):
                if episode > 0:
                    obs, _ = env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                ep = env.metrics.episode_costs[-1]
                totals["agent_cost"] += float(ep["agent_cost"])
                totals["baseline_cost_off"] += float(ep["baseline_cost_off"])
                totals["jobs_submitted"] += int(ep["jobs_submitted"])
                totals["jobs_completed"] += int(ep["jobs_completed"])
                totals["baseline_jobs_completed"] += int(ep["baseline_jobs_completed"])
                launched = int(ep.get("jobs_launched", ep["jobs_completed"]))
                totals["jobs_launched"] += launched
                wait_hours += float(ep["avg_wait_time"]) * launched
                totals["jobs_lost"] += int(ep.get("jobs_lost_total", ep.get("jobs_dropped", 0)))
                totals["baseline_jobs_lost"] += int(ep.get("baseline_jobs_lost_total", ep.get("baseline_jobs_dropped", 0)))
                totals["episodes"] += 1
            # Queue left at the end of each window (the last episode's end); overdue = waiting > 24 h.
            totals["pending_jobs_end"] += int(ep.get("pending_jobs_end", 0))
            totals["overdue_jobs_end"] += int(ep.get("overdue_jobs_end", 0))
        totals["avg_wait_h"] = wait_hours / totals["jobs_launched"] if totals["jobs_launched"] > 0 else 0.0
        base_off = totals["baseline_cost_off"]
        totals["savings_off_pct"] = 100.0 * (base_off - totals["agent_cost"]) / base_off if base_off > 0 else 0.0
        done_base = totals["baseline_jobs_completed"]
        totals["completion_ratio"] = totals["jobs_completed"] / done_base if done_base > 0 else 1.0
        totals["feasible"] = scale_feasible(totals, config.min_completion_ratio)
        per_scale[f"{scale:g}"] = totals

    feasible = all(s["feasible"] for s in per_scale.values())
    mean_savings = float(np.mean([s["savings_off_pct"] for s in per_scale.values()]))
    return {"scales": per_scale, "feasible": feasible, "savings_off_pct": mean_savings,
            "score": mean_savings if feasible else None}


def load_history(models_dir: str) -> list[dict]:
    path = os.path.join(models_dir, HISTORY_FILE)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def record_result(models_dir: str, step: int, result: dict, min_completion_ratio: float = 0.0) -> dict:
    """
    Append the result for checkpoint `step` to the history and rewrite the best-checkpoint file.
    A step validated again (e.g. after a resume) replaces its earlier entry.

    validation_best.json holds {"by_wait_budget": {label: entry or null}} plus, at the top level,
    the best entry without a wait limit (its "step" field is what older run scripts read).
    Returns the by-budget dict.
    """
    entry = {"step": int(step), **result}
    history = [h for h in load_history(models_dir) if h["step"] != entry["step"]] + [entry]
    history.sort(key=lambda h: h["step"])
    with open(os.path.join(models_dir, HISTORY_FILE), "w") as f:
        for h in history:
            f.write(json.dumps(h) + "\n")
    best = best_per_budget(history, min_completion_ratio=min_completion_ratio)
    best_path = os.path.join(models_dir, BEST_FILE)
    if best["any"] is not None:
        with open(best_path, "w") as f:
            json.dump({**best["any"], "by_wait_budget": best}, f, indent=2)
    elif os.path.exists(best_path):
        os.remove(best_path)
    return best


def format_result(step: int, result: dict) -> str:
    parts = [f"Validation at {step:,} steps:"]
    for scale, s in result["scales"].items():
        parts.append(f"x{scale}: savings_off {s['savings_off_pct']:+.2f}%, wait {s['avg_wait_h']:.1f}h, "
                     f"lost {s['jobs_lost']} (baseline {s['baseline_jobs_lost']}), completion {100 * s['completion_ratio']:.2f}% of baseline, "
                     f"queued at window ends {s['pending_jobs_end']} (overdue {s['overdue_jobs_end']}){'' if s['feasible'] else ' INFEASIBLE'}")
    parts.append(f"score {result['score']:+.2f}" if result["score"] is not None else "score n/a (infeasible: loses jobs)")
    return "\n  ".join(parts)


def format_best(best: dict) -> str:
    return ", ".join(f"{label}: " + ("none" if e is None else f"{e['step']:,} ({e['savings_off_pct']:+.1f}%, wait {entry_wait(e):.1f}h)")
                     for label, e in best.items())
