"""Validation of training checkpoints: a fixed, deterministic evaluation used to pick the best one.

Every checkpoint is run (argmax actions) on the same windows of the same validation data: for each
arrival scale one env, and for each window a hard reset with a fixed seed and price start index, so
arrivals and prices are identical for all checkpoints and only the agent's decisions differ.

A checkpoint is feasible if, at every arrival scale, it completes at least `min_completion_ratio`
of the jobs the baseline completes and loses no more jobs than the baseline. Its score is the
savings vs baseline_off (the baseline without idle nodes), in percent, averaged over the scales.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.config import EPISODE_HOURS

HISTORY_FILE = "validation.jsonl"
BEST_FILE = "validation_best.json"


@dataclass
class ValidationConfig:
    arrival_scales: tuple[float, ...] = (1.0, 1.5)
    windows: int = 4                 # windows spread evenly over the validation prices
    episodes_per_window: int = 2     # 2-week episodes run back to back within a window
    seed: int = 12345                # window w uses seed + w
    min_completion_ratio: float = 0.995


def window_starts(n_prices: int | None, windows: int) -> list[int]:
    """Price start index of each window: evenly spread, centred in its share of the data."""
    if n_prices is None:  # synthetic prices repeat daily; any start works
        return [w * EPISODE_HOURS for w in range(windows)]
    return [int((w + 0.5) * n_prices / windows) for w in range(windows)]


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
                      baseline_jobs_completed=0, jobs_lost=0, baseline_jobs_lost=0, episodes=0)
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
                totals["jobs_lost"] += int(ep.get("jobs_lost_total", ep.get("jobs_dropped", 0)))
                totals["baseline_jobs_lost"] += int(ep.get("baseline_jobs_lost_total", ep.get("baseline_jobs_dropped", 0)))
                totals["episodes"] += 1
        base_off = totals["baseline_cost_off"]
        totals["savings_off_pct"] = 100.0 * (base_off - totals["agent_cost"]) / base_off if base_off > 0 else 0.0
        done_base = totals["baseline_jobs_completed"]
        totals["completion_ratio"] = totals["jobs_completed"] / done_base if done_base > 0 else 1.0
        totals["feasible"] = bool(totals["completion_ratio"] >= config.min_completion_ratio
                                  and totals["jobs_lost"] <= totals["baseline_jobs_lost"])
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


def record_result(models_dir: str, step: int, result: dict) -> dict | None:
    """
    Append the result for checkpoint `step` to the history and rewrite the best-checkpoint file.
    A step validated again (e.g. after a resume) replaces its earlier entry. Returns the best entry.
    """
    entry = {"step": int(step), **result}
    history = [h for h in load_history(models_dir) if h["step"] != entry["step"]] + [entry]
    history.sort(key=lambda h: h["step"])
    with open(os.path.join(models_dir, HISTORY_FILE), "w") as f:
        for h in history:
            f.write(json.dumps(h) + "\n")
    scored = [h for h in history if h.get("score") is not None]
    best = max(scored, key=lambda h: h["score"]) if scored else None
    best_path = os.path.join(models_dir, BEST_FILE)
    if best is not None:
        with open(best_path, "w") as f:
            json.dump(best, f, indent=2)
    elif os.path.exists(best_path):
        os.remove(best_path)
    return best


def format_result(step: int, result: dict) -> str:
    parts = [f"Validation at {step:,} steps:"]
    for scale, s in result["scales"].items():
        parts.append(f"x{scale}: savings_off {s['savings_off_pct']:+.2f}%, completion {100 * s['completion_ratio']:.2f}% of baseline, "
                     f"lost {s['jobs_lost']} (baseline {s['baseline_jobs_lost']}){'' if s['feasible'] else ' INFEASIBLE'}")
    parts.append(f"score {result['score']:+.2f}" if result["score"] is not None else "score n/a (infeasible)")
    return "\n  ".join(parts)
