#!/usr/bin/env python3
"""
Sweep workload Poisson lambda values and analyze:
1) lambda -> agent occupancy (nodes)
2) occupancy -> proportional savings (%)
3) occupancy -> proportional savings_off (%)
4) lambda -> completion rate
5) occupancy -> proportional effective savings (%)
6) occupancy -> proportional effective savings_off (%)
7) occupancy -> average wait delta (agent - baseline)
8) occupancy -> (baseline_off - agent) cost_per_1000_completed_jobs / baseline_off
9) occupancy -> (baseline_off - agent) proportional power / baseline_off
10) lambda -> baseline and baseline_off occupancies
11) lambda -> mean jobs/hour (with std)
12) lambda -> dropped-jobs delta (agent - baseline)

For each lambda, this script runs train.py in evaluation mode for one year
(12 months = 24 episodes), parses per-episode metrics from stdout, computes
mean/std, and fits 3rd-order polynomial trend lines.

FAST DEBUG MODE: python analyze_lambda_occupancy.py --eval-months 1 --lambdas 1200,2000,3500 --no-plot-dashboard
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.analysis_naming import build_analysis_dir_name
from src.analysis_reporting import compute_savings_totals


EPISODE_RE = re.compile(
    r"Episode\s+(?P<episode>\d+):.*?"
    r"Savings=€(?P<savings>-?[\d,]+(?:\.\d+)?)\/€(?P<savings_off>-?[\d,]+(?:\.\d+)?),.*?"
    r"Power=(?P<agent_power>-?[\d.]+)\/(?P<baseline_power>-?[\d.]+)\/(?P<baseline_off_power>-?[\d.]+)\s*MWh.*?"
    r"CostPer1kCompleted=(?P<agent_cost_1k>-?[\d,]+(?:\.\d+)?|n/a)\/"
    r"(?P<baseline_cost_1k>-?[\d,]+(?:\.\d+)?|n/a)\/"
    r"(?P<baseline_off_cost_1k>-?[\d,]+(?:\.\d+)?|n/a)\s*€/1k.*?"
    r"Jobs=[\d,]+\/[\d,]+\s+\((?P<completion_rate>-?[\d.]+)%\),\s*"
    r"AvgWait=(?P<avg_wait>-?[\d.]+)h,.*?"
    r"(?:Dropped|Lost)=(?P<agent_dropped>-?[\d,]+),.*?"
    r"Agent Occupancy \(Nodes\)=\s*(?P<occupancy>-?[\d.]+)%,\s*"
    r"Baseline Occupancy \(Nodes\)=\s*(?P<baseline_occupancy>-?[\d.]+)%"
    r"(?:.*?"
    r"PropPower=(?P<agent_prop_power>-?[\d.]+)\/(?P<baseline_prop_power>-?[\d.]+)\/(?P<baseline_off_prop_power>-?[\d.]+)\s*MWh.*?"
    r"PropCost=€(?P<agent_prop_cost>-?[\d,]+(?:\.\d+)?)\/€(?P<baseline_prop_cost>-?[\d,]+(?:\.\d+)?)\/"
    r"€(?P<baseline_off_prop_cost>-?[\d,]+(?:\.\d+)?).*?"
    r"PropSavings=€(?P<prop_savings>-?[\d,]+(?:\.\d+)?)\/€(?P<prop_savings_off>-?[\d,]+(?:\.\d+)?))?",
    re.MULTILINE,
)

WAIT_SUMMARY_RE = re.compile(
    r"=== JOB PROCESSING METRICS ===.*?"
    r"Agent:.*?Average Wait Time:\s*(?P<agent_wait>-?[\d.]+)\s*hours.*?"
    r"Baseline:.*?Average Wait Time:\s*(?P<baseline_wait>-?[\d.]+)\s*hours",
    re.DOTALL,
)

ARRIVALS_SUMMARY_RE = re.compile(
    r"Job Arrivals/Hour \(mean\s*(?:±|\+/-)\s*std\):\s*(?P<mean>-?[\d.]+)\s*(?:±|\+/-)\s*(?P<std>-?[\d.]+)"
)

DROPPED_AGENT_SUMMARY_RE = re.compile(
    r"Total (?:Dropped|Lost) Jobs \(Agent\):\s*(?P<agent>[\d,]+)"
)

DROPPED_BASELINE_SUMMARY_RE = re.compile(
    r"Total (?:Dropped|Lost) Jobs \(Baseline\):\s*(?P<baseline>[\d,]+)"
)


@dataclass
class LambdaRunStats:
    lambda_value: int
    episodes: int
    occupancy_mean: float
    occupancy_std: float
    baseline_occupancy_mean: float
    baseline_occupancy_std: float
    baseline_off_occupancy_mean: float
    baseline_off_occupancy_std: float
    arrivals_per_hour_mean: float
    arrivals_per_hour_std: float
    dropped_jobs_agent_total: float
    dropped_jobs_baseline_total: float
    dropped_jobs_delta_total: float
    savings_mean: float
    savings_std: float
    savings_off_mean: float
    savings_off_std: float
    prop_savings_mean: float
    prop_savings_std: float
    prop_savings_off_mean: float
    prop_savings_off_std: float
    prop_savings_pct_mean: float
    prop_savings_pct_std: float
    prop_savings_pct_off_mean: float
    prop_savings_pct_off_std: float
    completion_rate_mean: float
    completion_rate_std: float
    agent_avg_wait_hours: float
    baseline_avg_wait_hours: float
    wait_delta_hours: float
    effective_savings_mean: float
    effective_savings_std: float
    effective_savings_off_mean: float
    effective_savings_off_std: float
    prop_effective_savings_mean: float
    prop_effective_savings_std: float
    prop_effective_savings_off_mean: float
    prop_effective_savings_off_std: float
    prop_effective_savings_pct_mean: float
    prop_effective_savings_pct_std: float
    prop_effective_savings_pct_off_mean: float
    prop_effective_savings_pct_off_std: float
    cost_per_1k_delta_pct_baseline_mean: float
    cost_per_1k_delta_pct_baseline_std: float
    cost_per_1k_delta_pct_baseline_off_mean: float
    cost_per_1k_delta_pct_baseline_off_std: float
    power_delta_pct_baseline_off_mean: float
    power_delta_pct_baseline_off_std: float
    prop_power_delta_pct_baseline_off_mean: float
    prop_power_delta_pct_baseline_off_std: float
    evaluation_savings: float
    annualized_savings: float
    evaluation_savings_off: float
    annualized_savings_off: float
    prop_evaluation_savings: float
    prop_annualized_savings: float
    prop_evaluation_savings_off: float
    prop_annualized_savings_off: float
    command: list[str]
    command_str: str
    occupancy_samples: list[float] = field(default_factory=list)
    baseline_occupancy_samples: list[float] = field(default_factory=list)
    baseline_off_occupancy_samples: list[float] = field(default_factory=list)
    dropped_jobs_agent_samples: list[float] = field(default_factory=list)
    savings_samples: list[float] = field(default_factory=list)
    savings_off_samples: list[float] = field(default_factory=list)
    prop_savings_samples: list[float] = field(default_factory=list)
    prop_savings_off_samples: list[float] = field(default_factory=list)
    completion_rate_samples: list[float] = field(default_factory=list)
    effective_savings_samples: list[float] = field(default_factory=list)
    effective_savings_off_samples: list[float] = field(default_factory=list)
    prop_effective_savings_samples: list[float] = field(default_factory=list)
    prop_effective_savings_off_samples: list[float] = field(default_factory=list)
    cost_per_1k_delta_pct_baseline_samples: list[float] = field(default_factory=list)
    cost_per_1k_delta_pct_baseline_off_samples: list[float] = field(default_factory=list)
    power_delta_pct_baseline_off_samples: list[float] = field(default_factory=list)
    prop_power_delta_pct_baseline_off_samples: list[float] = field(default_factory=list)


def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def _to_float_or_nan(raw: str | None) -> float:
    if raw is None:
        return float("nan")
    val = raw.strip().lower()
    if val in {"n/a", "nan"}:
        return float("nan")
    return _to_float(raw)


def parse_episode_metrics(
    stdout: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    occupancy = []
    baseline_occupancy = []
    agent_dropped = []
    savings = []
    savings_off = []
    completion_rate = []
    avg_wait = []
    agent_cost_1k = []
    baseline_cost_1k = []
    baseline_off_cost_1k = []
    agent_power = []
    baseline_off_power = []
    prop_savings = []
    prop_savings_off = []
    agent_prop_power = []
    baseline_prop_cost = []
    baseline_off_prop_cost = []
    baseline_off_prop_power = []

    for match in EPISODE_RE.finditer(stdout):
        flat_savings = _to_float(match.group("savings"))
        flat_savings_off = _to_float(match.group("savings_off"))
        flat_agent_power = _to_float(match.group("agent_power"))
        flat_baseline_off_power = _to_float(match.group("baseline_off_power"))
        parsed_prop_savings = _to_float_or_nan(match.group("prop_savings"))
        parsed_prop_savings_off = _to_float_or_nan(match.group("prop_savings_off"))
        parsed_agent_prop_power = _to_float_or_nan(match.group("agent_prop_power"))
        parsed_baseline_prop_cost = _to_float_or_nan(match.group("baseline_prop_cost"))
        parsed_baseline_off_prop_cost = _to_float_or_nan(match.group("baseline_off_prop_cost"))
        parsed_baseline_off_prop_power = _to_float_or_nan(match.group("baseline_off_prop_power"))
        if not (
            np.isfinite(parsed_prop_savings)
            and np.isfinite(parsed_prop_savings_off)
            and np.isfinite(parsed_agent_prop_power)
            and np.isfinite(parsed_baseline_prop_cost)
            and np.isfinite(parsed_baseline_off_prop_cost)
            and np.isfinite(parsed_baseline_off_prop_power)
        ):
            raise RuntimeError(
                f"Episode {match.group('episode')} summary is missing PropPower/PropCost/PropSavings metrics. "
                "Update train.py output before running occupancy analyses."
            )
        occupancy.append(_to_float(match.group("occupancy")))
        baseline_occupancy.append(_to_float(match.group("baseline_occupancy")))
        agent_dropped.append(_to_float(match.group("agent_dropped")))
        savings.append(flat_savings)
        savings_off.append(flat_savings_off)
        completion_rate.append(_to_float(match.group("completion_rate")))
        avg_wait.append(_to_float(match.group("avg_wait")))
        agent_cost_1k.append(_to_float_or_nan(match.group("agent_cost_1k")))
        baseline_cost_1k.append(_to_float_or_nan(match.group("baseline_cost_1k")))
        baseline_off_cost_1k.append(_to_float_or_nan(match.group("baseline_off_cost_1k")))
        agent_power.append(flat_agent_power)
        baseline_off_power.append(flat_baseline_off_power)
        prop_savings.append(parsed_prop_savings)
        prop_savings_off.append(parsed_prop_savings_off)
        agent_prop_power.append(parsed_agent_prop_power)
        baseline_prop_cost.append(parsed_baseline_prop_cost)
        baseline_off_prop_cost.append(parsed_baseline_off_prop_cost)
        baseline_off_prop_power.append(parsed_baseline_off_prop_power)

    if not occupancy:
        raise RuntimeError(
            "Could not parse episode metrics from train.py output. "
            "Expected lines like 'Episode X: ... Savings=€.../€..., Power=..., CostPer1kCompleted=..., "
            "Agent Occupancy (Nodes)=...%, PropPower=..., PropSavings=€.../€...'."
        )

    return (
        np.asarray(occupancy, dtype=float),
        np.asarray(baseline_occupancy, dtype=float),
        np.asarray(agent_dropped, dtype=float),
        np.asarray(savings, dtype=float),
        np.asarray(savings_off, dtype=float),
        np.asarray(completion_rate, dtype=float),
        np.asarray(avg_wait, dtype=float),
        np.asarray(agent_cost_1k, dtype=float),
        np.asarray(baseline_cost_1k, dtype=float),
        np.asarray(baseline_off_cost_1k, dtype=float),
        np.asarray(agent_power, dtype=float),
        np.asarray(baseline_off_power, dtype=float),
        np.asarray(prop_savings, dtype=float),
        np.asarray(prop_savings_off, dtype=float),
        np.asarray(agent_prop_power, dtype=float),
        np.asarray(baseline_prop_cost, dtype=float),
        np.asarray(baseline_off_prop_cost, dtype=float),
        np.asarray(baseline_off_prop_power, dtype=float),
    )


def parse_wait_summary(stdout: str) -> tuple[float | None, float | None]:
    match = WAIT_SUMMARY_RE.search(stdout)
    if not match:
        return None, None
    return _to_float(match.group("agent_wait")), _to_float(match.group("baseline_wait"))


def parse_arrivals_summary(stdout: str) -> tuple[float | None, float | None]:
    match = ARRIVALS_SUMMARY_RE.search(stdout)
    if not match:
        return None, None
    return _to_float(match.group("mean")), _to_float(match.group("std"))


def parse_dropped_totals_summary(stdout: str) -> tuple[float | None, float | None]:
    agent_match = DROPPED_AGENT_SUMMARY_RE.search(stdout)
    baseline_match = DROPPED_BASELINE_SUMMARY_RE.search(stdout)
    agent_total = _to_float(agent_match.group("agent")) if agent_match else None
    baseline_total = _to_float(baseline_match.group("baseline")) if baseline_match else None
    return agent_total, baseline_total


def safe_divide(numer: np.ndarray, denom: float) -> np.ndarray:
    if abs(denom) < 1e-12:
        return np.full_like(numer, np.nan, dtype=float)
    return numer / denom


def safe_divide_arrays(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    numer_arr = np.asarray(numer, dtype=float)
    denom_arr = np.asarray(denom, dtype=float)
    out = np.full_like(numer_arr, np.nan, dtype=float)
    finite = np.isfinite(numer_arr) & np.isfinite(denom_arr)
    valid = finite & (np.abs(denom_arr) >= 1e-12)
    out[valid] = numer_arr[valid] / denom_arr[valid]
    return out


def finite_mean_std(values: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan"), float("nan")
    vals = values[finite]
    return float(np.mean(vals)), float(np.std(vals))


def make_run_stats(
    lambda_value: int,
    eval_months: int,
    command: list[str],
    occupancy: np.ndarray,
    baseline_occupancy: np.ndarray,
    agent_dropped: np.ndarray,
    savings: np.ndarray,
    savings_off: np.ndarray,
    completion_rate: np.ndarray,
    agent_avg_wait_hours: float,
    baseline_avg_wait_hours: float,
    agent_cost_1k: np.ndarray,
    baseline_cost_1k: np.ndarray,
    baseline_off_cost_1k: np.ndarray,
    agent_power: np.ndarray,
    baseline_off_power: np.ndarray,
    prop_savings: np.ndarray,
    prop_savings_off: np.ndarray,
    agent_prop_power: np.ndarray,
    baseline_prop_cost: np.ndarray,
    baseline_off_prop_cost: np.ndarray,
    baseline_off_prop_power: np.ndarray,
    arrivals_per_hour_mean: float,
    arrivals_per_hour_std: float,
    dropped_jobs_agent_total: float,
    dropped_jobs_baseline_total: float,
) -> LambdaRunStats:
    wait_delta_hours = agent_avg_wait_hours - baseline_avg_wait_hours
    effective_savings = safe_divide(savings * (completion_rate/100)**2, wait_delta_hours+1)
    effective_savings_off = safe_divide(savings_off * (completion_rate/100)**2, wait_delta_hours+1)
    prop_savings_pct = safe_divide_arrays(prop_savings * 100.0, baseline_prop_cost)
    prop_savings_pct_off = safe_divide_arrays(prop_savings_off * 100.0, baseline_off_prop_cost)
    prop_effective_savings = safe_divide(prop_savings * (completion_rate / 100) ** 2, wait_delta_hours + 1)
    prop_effective_savings_off = safe_divide(prop_savings_off * (completion_rate / 100) ** 2, wait_delta_hours + 1)
    prop_effective_savings_pct = safe_divide(prop_savings_pct * (completion_rate / 100) ** 2, wait_delta_hours + 1)
    prop_effective_savings_pct_off = safe_divide(prop_savings_pct_off * (completion_rate / 100) ** 2, wait_delta_hours + 1)
    effective_savings_mean, effective_savings_std = finite_mean_std(effective_savings)
    effective_savings_off_mean, effective_savings_off_std = finite_mean_std(effective_savings_off)
    prop_savings_pct_mean, prop_savings_pct_std = finite_mean_std(prop_savings_pct)
    prop_savings_pct_off_mean, prop_savings_pct_off_std = finite_mean_std(prop_savings_pct_off)
    prop_effective_savings_mean, prop_effective_savings_std = finite_mean_std(prop_effective_savings)
    prop_effective_savings_off_mean, prop_effective_savings_off_std = finite_mean_std(prop_effective_savings_off)
    prop_effective_savings_pct_mean, prop_effective_savings_pct_std = finite_mean_std(prop_effective_savings_pct)
    prop_effective_savings_pct_off_mean, prop_effective_savings_pct_off_std = finite_mean_std(prop_effective_savings_pct_off)
    cost_per_1k_delta_pct_baseline = safe_divide_arrays((baseline_cost_1k - agent_cost_1k) * 100.0, baseline_cost_1k)
    cost_per_1k_delta_pct_baseline_off = safe_divide_arrays((baseline_off_cost_1k - agent_cost_1k) * 100.0, baseline_off_cost_1k)
    power_delta_pct_baseline_off = safe_divide_arrays((baseline_off_power - agent_power) * 100.0, baseline_off_power)
    prop_power_delta_pct_baseline_off = safe_divide_arrays(
        (baseline_off_prop_power - agent_prop_power) * 100.0,
        baseline_off_prop_power,
    )
    cost_per_1k_delta_pct_baseline_mean, cost_per_1k_delta_pct_baseline_std = finite_mean_std(cost_per_1k_delta_pct_baseline)
    cost_per_1k_delta_pct_baseline_off_mean, cost_per_1k_delta_pct_baseline_off_std = finite_mean_std(cost_per_1k_delta_pct_baseline_off)
    power_delta_pct_baseline_off_mean, power_delta_pct_baseline_off_std = finite_mean_std(power_delta_pct_baseline_off)
    prop_power_delta_pct_baseline_off_mean, prop_power_delta_pct_baseline_off_std = finite_mean_std(
        prop_power_delta_pct_baseline_off
    )
    baseline_off_occupancy = baseline_occupancy.copy()
    dropped_jobs_delta_total = dropped_jobs_agent_total - dropped_jobs_baseline_total
    evaluation_savings, annualized_savings = compute_savings_totals(savings, eval_months)
    evaluation_savings_off, annualized_savings_off = compute_savings_totals(savings_off, eval_months)
    prop_evaluation_savings, prop_annualized_savings = compute_savings_totals(prop_savings, eval_months)
    prop_evaluation_savings_off, prop_annualized_savings_off = compute_savings_totals(prop_savings_off, eval_months)
    return LambdaRunStats(
        lambda_value=lambda_value,
        episodes=int(occupancy.size),
        occupancy_mean=float(np.mean(occupancy)),
        occupancy_std=float(np.std(occupancy)),
        baseline_occupancy_mean=float(np.mean(baseline_occupancy)),
        baseline_occupancy_std=float(np.std(baseline_occupancy)),
        baseline_off_occupancy_mean=float(np.mean(baseline_off_occupancy)),
        baseline_off_occupancy_std=float(np.std(baseline_off_occupancy)),
        arrivals_per_hour_mean=float(arrivals_per_hour_mean),
        arrivals_per_hour_std=float(arrivals_per_hour_std),
        dropped_jobs_agent_total=float(dropped_jobs_agent_total),
        dropped_jobs_baseline_total=float(dropped_jobs_baseline_total),
        dropped_jobs_delta_total=float(dropped_jobs_delta_total),
        savings_mean=float(np.mean(savings)),
        savings_std=float(np.std(savings)),
        savings_off_mean=float(np.mean(savings_off)),
        savings_off_std=float(np.std(savings_off)),
        prop_savings_mean=float(np.mean(prop_savings)),
        prop_savings_std=float(np.std(prop_savings)),
        prop_savings_off_mean=float(np.mean(prop_savings_off)),
        prop_savings_off_std=float(np.std(prop_savings_off)),
        prop_savings_pct_mean=prop_savings_pct_mean,
        prop_savings_pct_std=prop_savings_pct_std,
        prop_savings_pct_off_mean=prop_savings_pct_off_mean,
        prop_savings_pct_off_std=prop_savings_pct_off_std,
        completion_rate_mean=float(np.mean(completion_rate)),
        completion_rate_std=float(np.std(completion_rate)),
        agent_avg_wait_hours=float(agent_avg_wait_hours),
        baseline_avg_wait_hours=float(baseline_avg_wait_hours),
        wait_delta_hours=float(wait_delta_hours),
        effective_savings_mean=effective_savings_mean,
        effective_savings_std=effective_savings_std,
        effective_savings_off_mean=effective_savings_off_mean,
        effective_savings_off_std=effective_savings_off_std,
        prop_effective_savings_mean=prop_effective_savings_mean,
        prop_effective_savings_std=prop_effective_savings_std,
        prop_effective_savings_off_mean=prop_effective_savings_off_mean,
        prop_effective_savings_off_std=prop_effective_savings_off_std,
        prop_effective_savings_pct_mean=prop_effective_savings_pct_mean,
        prop_effective_savings_pct_std=prop_effective_savings_pct_std,
        prop_effective_savings_pct_off_mean=prop_effective_savings_pct_off_mean,
        prop_effective_savings_pct_off_std=prop_effective_savings_pct_off_std,
        cost_per_1k_delta_pct_baseline_mean=cost_per_1k_delta_pct_baseline_mean,
        cost_per_1k_delta_pct_baseline_std=cost_per_1k_delta_pct_baseline_std,
        cost_per_1k_delta_pct_baseline_off_mean=cost_per_1k_delta_pct_baseline_off_mean,
        cost_per_1k_delta_pct_baseline_off_std=cost_per_1k_delta_pct_baseline_off_std,
        power_delta_pct_baseline_off_mean=power_delta_pct_baseline_off_mean,
        power_delta_pct_baseline_off_std=power_delta_pct_baseline_off_std,
        prop_power_delta_pct_baseline_off_mean=prop_power_delta_pct_baseline_off_mean,
        prop_power_delta_pct_baseline_off_std=prop_power_delta_pct_baseline_off_std,
        evaluation_savings=evaluation_savings,
        annualized_savings=annualized_savings,
        evaluation_savings_off=evaluation_savings_off,
        annualized_savings_off=annualized_savings_off,
        prop_evaluation_savings=prop_evaluation_savings,
        prop_annualized_savings=prop_annualized_savings,
        prop_evaluation_savings_off=prop_evaluation_savings_off,
        prop_annualized_savings_off=prop_annualized_savings_off,
        command=command,
        command_str=shlex.join(command),
        occupancy_samples=occupancy.tolist(),
        baseline_occupancy_samples=baseline_occupancy.tolist(),
        baseline_off_occupancy_samples=baseline_off_occupancy.tolist(),
        dropped_jobs_agent_samples=agent_dropped.tolist(),
        savings_samples=savings.tolist(),
        savings_off_samples=savings_off.tolist(),
        prop_savings_samples=prop_savings.tolist(),
        prop_savings_off_samples=prop_savings_off.tolist(),
        completion_rate_samples=completion_rate.tolist(),
        effective_savings_samples=effective_savings.tolist(),
        effective_savings_off_samples=effective_savings_off.tolist(),
        prop_effective_savings_samples=prop_effective_savings.tolist(),
        prop_effective_savings_off_samples=prop_effective_savings_off.tolist(),
        cost_per_1k_delta_pct_baseline_samples=cost_per_1k_delta_pct_baseline.tolist(),
        cost_per_1k_delta_pct_baseline_off_samples=cost_per_1k_delta_pct_baseline_off.tolist(),
        power_delta_pct_baseline_off_samples=power_delta_pct_baseline_off.tolist(),
        prop_power_delta_pct_baseline_off_samples=prop_power_delta_pct_baseline_off.tolist(),
    )


def polyfit_curve(x: np.ndarray, y: np.ndarray, max_degree: int = 3) -> tuple[np.ndarray | None, int]:
    finite = np.isfinite(x) & np.isfinite(y)
    xf = x[finite]
    yf = y[finite]
    if xf.size < 2:
        return None, 0
    degree = min(max_degree, xf.size - 1)
    coeffs = np.polyfit(xf, yf, degree)
    return coeffs, degree


def unique_ints_sorted(values: list[int]) -> list[int]:
    return sorted({int(v) for v in values})


def geometric_int_space(low: int, high: int, n: int) -> list[int]:
    if n <= 1 or low == high:
        return [int(low)]
    vals = np.geomspace(low, high, n)
    return unique_ints_sorted([int(round(v)) for v in vals])


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def build_train_command(args: argparse.Namespace, lambda_value: int) -> list[str]:
    cmd = [
        sys.executable,
        "./train.py",
        "--prices",
        args.prices,
        "--session",
        args.session,
        "--efficiency-weight",
        str(args.efficiency_weight),
        "--price-weight",
        str(args.price_weight),
        "--idle-weight",
        str(args.idle_weight),
        "--job-age-weight",
        str(args.job_age_weight),
        "--drop-weight",
        str(args.drop_weight),
        "--evaluate-savings",
        "--eval-months",
        str(args.eval_months),
        "--model",
        str(args.model),
        "--workload-gen",
        "poisson",
        "--wg-poisson-lambdas4",
        f"{lambda_value},{args.wg_duration_lambda},{args.wg_nodes_lambda},{args.wg_cores_lambda}",
        "--wg-max-jobs-hour",
        str(args.wg_max_jobs_hour),
        "--wg-burst-small-prob",
        str(args.wg_burst_small_prob),
        "--wg-burst-heavy-prob",
        str(args.wg_burst_heavy_prob),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.plot_dashboard:
        cmd.append("--plot-dashboard")
    if args.oracle:
        cmd.append("--oracle")
    if args.dashboard_hours is not None:
        cmd.extend(["--dashboard-hours", str(args.dashboard_hours)])
    return cmd


def run_lambda_eval(args: argparse.Namespace, project_root: Path, lambda_value: int) -> tuple[LambdaRunStats, str]:
    command = build_train_command(args, lambda_value)
    print(f"[run] lambda={lambda_value}: {shlex.join(command)}")
    completed = subprocess.run(
        command,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    if args.echo_train_output:
        print(combined_output)
    if completed.returncode != 0:
        raise RuntimeError(
            f"train.py failed for lambda={lambda_value} with code {completed.returncode}.\n"
            f"Last output lines:\n{os_tail(combined_output, lines=40)}"
        )

    (
        occupancy,
        baseline_occupancy,
        agent_dropped,
        savings,
        savings_off,
        completion_rate,
        avg_wait,
        agent_cost_1k,
        baseline_cost_1k,
        baseline_off_cost_1k,
        agent_power,
        baseline_off_power,
        prop_savings,
        prop_savings_off,
        agent_prop_power,
        baseline_prop_cost,
        baseline_off_prop_cost,
        baseline_off_prop_power,
    ) = parse_episode_metrics(combined_output)
    agent_wait_summary, baseline_wait_summary = parse_wait_summary(combined_output)
    if agent_wait_summary is None or baseline_wait_summary is None:
        print(f"[warn] lambda={lambda_value}: could not parse run-level wait summary; effective savings may be NaN.")
        agent_avg_wait_hours = float(np.mean(avg_wait))
        baseline_avg_wait_hours = float(np.mean(avg_wait))
    else:
        agent_avg_wait_hours = float(agent_wait_summary)
        baseline_avg_wait_hours = float(baseline_wait_summary)
    arrivals_per_hour_mean, arrivals_per_hour_std = parse_arrivals_summary(combined_output)
    if arrivals_per_hour_mean is None or arrivals_per_hour_std is None:
        print(f"[warn] lambda={lambda_value}: could not parse run-level arrivals/hour summary; values set to NaN.")
        arrivals_per_hour_mean = float("nan")
        arrivals_per_hour_std = float("nan")
    dropped_jobs_agent_total, dropped_jobs_baseline_total = parse_dropped_totals_summary(combined_output)
    if dropped_jobs_agent_total is None:
        dropped_jobs_agent_total = float(np.sum(agent_dropped))
        print(f"[warn] lambda={lambda_value}: could not parse run-level agent dropped total; using sum of episode Dropped= values.")
    if dropped_jobs_baseline_total is None:
        dropped_jobs_baseline_total = 0.0
        print(f"[warn] lambda={lambda_value}: could not parse run-level baseline dropped total; defaulting to 0.")
    stats = make_run_stats(
        lambda_value,
        args.eval_months,
        command,
        occupancy,
        baseline_occupancy,
        agent_dropped,
        savings,
        savings_off,
        completion_rate,
        agent_avg_wait_hours,
        baseline_avg_wait_hours,
        agent_cost_1k,
        baseline_cost_1k,
        baseline_off_cost_1k,
        agent_power,
        baseline_off_power,
        prop_savings,
        prop_savings_off,
        agent_prop_power,
        baseline_prop_cost,
        baseline_off_prop_cost,
        baseline_off_prop_power,
        arrivals_per_hour_mean,
        arrivals_per_hour_std,
        dropped_jobs_agent_total,
        dropped_jobs_baseline_total,
    )
    print(
        f"[ok ] lambda={lambda_value}: "
        f"occupancy={stats.occupancy_mean:.2f}%±{stats.occupancy_std:.2f}, "
        f"baseline_occ={stats.baseline_occupancy_mean:.2f}%±{stats.baseline_occupancy_std:.2f}, "
        f"arrivals/h={stats.arrivals_per_hour_mean:.2f}±{stats.arrivals_per_hour_std:.2f}, "
        f"dropped_delta={stats.dropped_jobs_delta_total:.0f}, "
        f"completion={stats.completion_rate_mean:.2f}%±{stats.completion_rate_std:.2f}, "
        f"prop_savings={stats.prop_savings_mean:.0f}±{stats.prop_savings_std:.0f}, "
        f"prop_savings_off={stats.prop_savings_off_mean:.0f}±{stats.prop_savings_off_std:.0f}, "
        f"prop_eval_savings={stats.prop_evaluation_savings:.0f}/{stats.prop_evaluation_savings_off:.0f}, "
        f"prop_annualized_savings={stats.prop_annualized_savings:.0f}/{stats.prop_annualized_savings_off:.0f}, "
        f"wait_delta={stats.wait_delta_hours:.3f}h"
    )
    return stats, combined_output


def os_tail(text: str, lines: int = 20) -> str:
    parts = text.rstrip().splitlines()
    if not parts:
        return ""
    return "\n".join(parts[-lines:])


def select_lambda_schedule(
    args: argparse.Namespace,
    evaluate: Callable[[int], LambdaRunStats],
) -> list[int]:
    """
    Adaptive lambda selection when --lambdas is not provided:
    1) Start at reference lambda (clamped to min/max).
    2) Probe downward (divide by bracket_factor) until occupancy reaches
       target_occupancy_min (+ tolerance) or bracket_steps is exhausted.
    3) Probe upward (multiply by bracket_factor) until occupancy reaches
       target_occupancy_max (- tolerance) or bracket_steps is exhausted.
    4) Fill remaining points with log spacing between discovered low/high.
    5) If log spacing collapses due to integer rounding, insert candidates
       into the largest gaps until num_points is reached or no new values
       can be inserted.
    """
    if args.lambdas:
        explicit = unique_ints_sorted(
            [clamp_int(v, args.min_lambda, args.max_lambda) for v in parse_int_list(args.lambdas)]
        )
        for lam in explicit:
            evaluate(lam)
        return explicit

    ref = clamp_int(args.reference_lambda, args.min_lambda, args.max_lambda)
    evaluate(ref)

    # Probe low side.
    low = ref
    for _ in range(args.bracket_steps):
        occ = evaluate(low).occupancy_mean
        if occ <= args.target_occupancy_min + args.occupancy_tolerance:
            break
        nxt = clamp_int(int(round(low / args.bracket_factor)), args.min_lambda, args.max_lambda)
        if nxt == low:
            break
        low = nxt
        evaluate(low)

    # Probe high side.
    high = ref
    for _ in range(args.bracket_steps):
        occ = evaluate(high).occupancy_mean
        if occ >= args.target_occupancy_max - args.occupancy_tolerance:
            break
        nxt = clamp_int(int(round(high * args.bracket_factor)), args.min_lambda, args.max_lambda)
        if nxt == high:
            break
        high = nxt
        evaluate(high)

    # Fill remaining points log-spaced across discovered range.
    discovered = unique_ints_sorted(list(evaluate.cache_keys()))
    low_bound = min(discovered)
    high_bound = max(discovered)
    target_points = max(args.num_points, len(discovered))

    for lam in geometric_int_space(low_bound, high_bound, target_points):
        evaluate(lam)

    # If geometric spacing collapsed to fewer unique integers, fill largest gaps.
    while len(evaluate.cache_keys()) < target_points:
        ordered = unique_ints_sorted(list(evaluate.cache_keys()))
        if len(ordered) < 2:
            break
        gaps = sorted(
            ((ordered[i + 1] - ordered[i], ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)),
            reverse=True,
        )
        inserted = False
        for _, a, b in gaps:
            cand = int(round(math.sqrt(a * b)))
            if cand in (a, b):
                cand = (a + b) // 2
            cand = clamp_int(cand, args.min_lambda, args.max_lambda)
            if cand not in evaluate.cache_keys():
                evaluate(cand)
                inserted = True
                break
        if not inserted:
            break

    return unique_ints_sorted(list(evaluate.cache_keys()))


def write_summary_csv(path: Path, stats_by_lambda: list[LambdaRunStats]) -> None:
    fieldnames = [
        "lambda",
        "episodes",
        "occupancy_mean_pct",
        "occupancy_std_pct",
        "baseline_occupancy_mean_pct",
        "baseline_occupancy_std_pct",
        "baseline_off_occupancy_mean_pct",
        "baseline_off_occupancy_std_pct",
        "arrivals_per_hour_mean",
        "arrivals_per_hour_std",
        "dropped_jobs_agent_total",
        "dropped_jobs_baseline_total",
        "dropped_jobs_delta_total",
        "completion_rate_mean_pct",
        "completion_rate_std_pct",
        "agent_avg_wait_hours",
        "baseline_avg_wait_hours",
        "wait_delta_hours",
        "savings_mean_eur",
        "savings_std_eur",
        "savings_off_mean_eur",
        "savings_off_std_eur",
        "prop_savings_mean_eur",
        "prop_savings_std_eur",
        "prop_savings_off_mean_eur",
        "prop_savings_off_std_eur",
        "prop_savings_pct_mean",
        "prop_savings_pct_std",
        "prop_savings_pct_off_mean",
        "prop_savings_pct_off_std",
        "effective_savings_mean",
        "effective_savings_std",
        "effective_savings_off_mean",
        "effective_savings_off_std",
        "prop_effective_savings_mean",
        "prop_effective_savings_std",
        "prop_effective_savings_off_mean",
        "prop_effective_savings_off_std",
        "prop_effective_savings_pct_mean",
        "prop_effective_savings_pct_std",
        "prop_effective_savings_pct_off_mean",
        "prop_effective_savings_pct_off_std",
        "cost_per_1k_delta_pct_baseline_mean",
        "cost_per_1k_delta_pct_baseline_std",
        "cost_per_1k_delta_pct_baseline_off_mean",
        "cost_per_1k_delta_pct_baseline_off_std",
        "power_delta_pct_baseline_off_mean",
        "power_delta_pct_baseline_off_std",
        "prop_power_delta_pct_baseline_off_mean",
        "prop_power_delta_pct_baseline_off_std",
        "evaluation_savings_eur",
        "annualized_savings_eur",
        "evaluation_savings_off_eur",
        "annualized_savings_off_eur",
        "prop_evaluation_savings_eur",
        "prop_annualized_savings_eur",
        "prop_evaluation_savings_off_eur",
        "prop_annualized_savings_off_eur",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in sorted(stats_by_lambda, key=lambda x: x.lambda_value):
            writer.writerow(
                {
                    "lambda": s.lambda_value,
                    "episodes": s.episodes,
                    "occupancy_mean_pct": f"{s.occupancy_mean:.6f}",
                    "occupancy_std_pct": f"{s.occupancy_std:.6f}",
                    "baseline_occupancy_mean_pct": f"{s.baseline_occupancy_mean:.6f}",
                    "baseline_occupancy_std_pct": f"{s.baseline_occupancy_std:.6f}",
                    "baseline_off_occupancy_mean_pct": f"{s.baseline_off_occupancy_mean:.6f}",
                    "baseline_off_occupancy_std_pct": f"{s.baseline_off_occupancy_std:.6f}",
                    "arrivals_per_hour_mean": f"{s.arrivals_per_hour_mean:.6f}",
                    "arrivals_per_hour_std": f"{s.arrivals_per_hour_std:.6f}",
                    "dropped_jobs_agent_total": f"{s.dropped_jobs_agent_total:.6f}",
                    "dropped_jobs_baseline_total": f"{s.dropped_jobs_baseline_total:.6f}",
                    "dropped_jobs_delta_total": f"{s.dropped_jobs_delta_total:.6f}",
                    "completion_rate_mean_pct": f"{s.completion_rate_mean:.6f}",
                    "completion_rate_std_pct": f"{s.completion_rate_std:.6f}",
                    "agent_avg_wait_hours": f"{s.agent_avg_wait_hours:.6f}",
                    "baseline_avg_wait_hours": f"{s.baseline_avg_wait_hours:.6f}",
                    "wait_delta_hours": f"{s.wait_delta_hours:.6f}",
                    "savings_mean_eur": f"{s.savings_mean:.6f}",
                    "savings_std_eur": f"{s.savings_std:.6f}",
                    "savings_off_mean_eur": f"{s.savings_off_mean:.6f}",
                    "savings_off_std_eur": f"{s.savings_off_std:.6f}",
                    "prop_savings_mean_eur": f"{s.prop_savings_mean:.6f}",
                    "prop_savings_std_eur": f"{s.prop_savings_std:.6f}",
                    "prop_savings_off_mean_eur": f"{s.prop_savings_off_mean:.6f}",
                    "prop_savings_off_std_eur": f"{s.prop_savings_off_std:.6f}",
                    "prop_savings_pct_mean": f"{s.prop_savings_pct_mean:.6f}",
                    "prop_savings_pct_std": f"{s.prop_savings_pct_std:.6f}",
                    "prop_savings_pct_off_mean": f"{s.prop_savings_pct_off_mean:.6f}",
                    "prop_savings_pct_off_std": f"{s.prop_savings_pct_off_std:.6f}",
                    "effective_savings_mean": f"{s.effective_savings_mean:.6f}",
                    "effective_savings_std": f"{s.effective_savings_std:.6f}",
                    "effective_savings_off_mean": f"{s.effective_savings_off_mean:.6f}",
                    "effective_savings_off_std": f"{s.effective_savings_off_std:.6f}",
                    "prop_effective_savings_mean": f"{s.prop_effective_savings_mean:.6f}",
                    "prop_effective_savings_std": f"{s.prop_effective_savings_std:.6f}",
                    "prop_effective_savings_off_mean": f"{s.prop_effective_savings_off_mean:.6f}",
                    "prop_effective_savings_off_std": f"{s.prop_effective_savings_off_std:.6f}",
                    "prop_effective_savings_pct_mean": f"{s.prop_effective_savings_pct_mean:.6f}",
                    "prop_effective_savings_pct_std": f"{s.prop_effective_savings_pct_std:.6f}",
                    "prop_effective_savings_pct_off_mean": f"{s.prop_effective_savings_pct_off_mean:.6f}",
                    "prop_effective_savings_pct_off_std": f"{s.prop_effective_savings_pct_off_std:.6f}",
                    "cost_per_1k_delta_pct_baseline_mean": f"{s.cost_per_1k_delta_pct_baseline_mean:.6f}",
                    "cost_per_1k_delta_pct_baseline_std": f"{s.cost_per_1k_delta_pct_baseline_std:.6f}",
                    "cost_per_1k_delta_pct_baseline_off_mean": f"{s.cost_per_1k_delta_pct_baseline_off_mean:.6f}",
                    "cost_per_1k_delta_pct_baseline_off_std": f"{s.cost_per_1k_delta_pct_baseline_off_std:.6f}",
                    "power_delta_pct_baseline_off_mean": f"{s.power_delta_pct_baseline_off_mean:.6f}",
                    "power_delta_pct_baseline_off_std": f"{s.power_delta_pct_baseline_off_std:.6f}",
                    "prop_power_delta_pct_baseline_off_mean": f"{s.prop_power_delta_pct_baseline_off_mean:.6f}",
                    "prop_power_delta_pct_baseline_off_std": f"{s.prop_power_delta_pct_baseline_off_std:.6f}",
                    "evaluation_savings_eur": f"{s.evaluation_savings:.6f}",
                    "annualized_savings_eur": f"{s.annualized_savings:.6f}",
                    "evaluation_savings_off_eur": f"{s.evaluation_savings_off:.6f}",
                    "annualized_savings_off_eur": f"{s.annualized_savings_off:.6f}",
                    "prop_evaluation_savings_eur": f"{s.prop_evaluation_savings:.6f}",
                    "prop_annualized_savings_eur": f"{s.prop_annualized_savings:.6f}",
                    "prop_evaluation_savings_off_eur": f"{s.prop_evaluation_savings_off:.6f}",
                    "prop_annualized_savings_off_eur": f"{s.prop_annualized_savings_off:.6f}",
                }
            )


def make_plot(
    path: Path,
    stats_by_lambda: list[LambdaRunStats],
    fit: bool = False,
    individual_dir: Path | None = None,
) -> None:
    ordered = sorted(stats_by_lambda, key=lambda x: x.lambda_value)
    if not ordered:
        return

    def _minmax_error_array(sample_lists: list[list[float]], means: np.ndarray) -> np.ndarray:
        errors = []
        for mean, samples in zip(means, sample_lists):
            vals = np.asarray(samples, dtype=float)
            finite = vals[np.isfinite(vals)]
            if finite.size == 0 or not np.isfinite(mean):
                errors.append((np.nan, np.nan))
                continue
            errors.append(
                (
                    max(float(mean - np.min(finite)), 0.0),
                    max(float(np.max(finite) - mean), 0.0),
                )
            )
        return np.asarray(errors, dtype=float).T

    lambdas = np.array([s.lambda_value for s in ordered], dtype=float)
    occ_mean = np.array([s.occupancy_mean for s in ordered], dtype=float)
    occ_std = np.array([s.occupancy_std for s in ordered], dtype=float)
    occ_minmax = _minmax_error_array([s.occupancy_samples for s in ordered], occ_mean)
    baseline_occ_mean = np.array([s.baseline_occupancy_mean for s in ordered], dtype=float)
    baseline_occ_std = np.array([s.baseline_occupancy_std for s in ordered], dtype=float)
    baseline_occ_minmax = _minmax_error_array([s.baseline_occupancy_samples for s in ordered], baseline_occ_mean)
    baseline_off_occ_mean = np.array([s.baseline_off_occupancy_mean for s in ordered], dtype=float)
    baseline_off_occ_std = np.array([s.baseline_off_occupancy_std for s in ordered], dtype=float)
    baseline_off_occ_minmax = _minmax_error_array(
        [s.baseline_off_occupancy_samples for s in ordered],
        baseline_off_occ_mean,
    )
    arrivals_per_hour_mean = np.array([s.arrivals_per_hour_mean for s in ordered], dtype=float)
    arrivals_per_hour_std = np.array([s.arrivals_per_hour_std for s in ordered], dtype=float)
    dropped_jobs_delta_total = np.array([s.dropped_jobs_delta_total for s in ordered], dtype=float)
    sav_mean = np.array([s.savings_mean for s in ordered], dtype=float)
    sav_std = np.array([s.savings_std for s in ordered], dtype=float)
    sav_off_mean = np.array([s.savings_off_mean for s in ordered], dtype=float)
    sav_off_std = np.array([s.savings_off_std for s in ordered], dtype=float)
    prop_sav_pct_mean = np.array([s.prop_savings_pct_mean for s in ordered], dtype=float)
    prop_sav_pct_std = np.array([s.prop_savings_pct_std for s in ordered], dtype=float)
    prop_sav_pct_off_mean = np.array([s.prop_savings_pct_off_mean for s in ordered], dtype=float)
    prop_sav_pct_off_std = np.array([s.prop_savings_pct_off_std for s in ordered], dtype=float)
    completion_mean = np.array([s.completion_rate_mean for s in ordered], dtype=float)
    completion_std = np.array([s.completion_rate_std for s in ordered], dtype=float)
    eff_sav_mean = np.array([s.effective_savings_mean for s in ordered], dtype=float)
    eff_sav_std = np.array([s.effective_savings_std for s in ordered], dtype=float)
    eff_sav_off_mean = np.array([s.effective_savings_off_mean for s in ordered], dtype=float)
    eff_sav_off_std = np.array([s.effective_savings_off_std for s in ordered], dtype=float)
    prop_eff_sav_pct_mean = np.array([s.prop_effective_savings_pct_mean for s in ordered], dtype=float)
    prop_eff_sav_pct_std = np.array([s.prop_effective_savings_pct_std for s in ordered], dtype=float)
    prop_eff_sav_pct_off_mean = np.array([s.prop_effective_savings_pct_off_mean for s in ordered], dtype=float)
    prop_eff_sav_pct_off_std = np.array([s.prop_effective_savings_pct_off_std for s in ordered], dtype=float)
    wait_delta_hours = np.array([s.wait_delta_hours for s in ordered], dtype=float)
    cost_per_1k_delta_base_off_mean = np.array([s.cost_per_1k_delta_pct_baseline_off_mean for s in ordered], dtype=float)
    cost_per_1k_delta_base_off_std = np.array([s.cost_per_1k_delta_pct_baseline_off_std for s in ordered], dtype=float)
    power_delta_base_off_mean = np.array([s.power_delta_pct_baseline_off_mean for s in ordered], dtype=float)
    power_delta_base_off_std = np.array([s.power_delta_pct_baseline_off_std for s in ordered], dtype=float)
    prop_power_delta_base_off_mean = np.array([s.prop_power_delta_pct_baseline_off_mean for s in ordered], dtype=float)
    prop_power_delta_base_off_std = np.array([s.prop_power_delta_pct_baseline_off_std for s in ordered], dtype=float)

    lam_min = float(np.min(lambdas))
    lam_max = float(np.max(lambdas))
    if lam_max <= lam_min:
        lam_max = lam_min + 1.0
    norm = matplotlib.colors.Normalize(vmin=lam_min, vmax=lam_max)
    cmap = plt.get_cmap("turbo")
    point_colors = cmap(norm(lambdas))
    colorbar_label = "Poisson lambda (point color)"

    def _error_at(arr: np.ndarray | None, idx: int) -> float | np.ndarray | None:
        if arr is None:
            return None
        if arr.ndim == 1:
            v = float(arr[idx])
            return v if np.isfinite(v) else None
        if arr.ndim == 2:
            lower = float(arr[0, idx])
            upper = float(arr[1, idx])
            if not (np.isfinite(lower) and np.isfinite(upper)):
                return None
            return np.asarray([[lower], [upper]], dtype=float)
        raise ValueError("Expected 1D or 2D error array.")

    def _draw_point(
        ax: plt.Axes,
        x: float,
        y: float,
        color: np.ndarray,
        marker: str = "o",
        xerr_std: float | np.ndarray | None = None,
        yerr_std: float | np.ndarray | None = None,
        xerr_range: float | np.ndarray | None = None,
        yerr_range: float | np.ndarray | None = None,
    ) -> None:
        if xerr_range is not None or yerr_range is not None:
            ax.errorbar(
                x,
                y,
                xerr=xerr_range,
                yerr=yerr_range,
                fmt="none",
                capsize=4,
                ecolor=color,
                elinewidth=0.9,
                alpha=0.35,
                zorder=1,
            )
        ax.errorbar(
            x,
            y,
            xerr=xerr_std,
            yerr=yerr_std,
            fmt=marker,
            markersize=6,
            capsize=2.5,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            alpha=0.95,
            zorder=2,
        )

    def _maybe_plot_fit(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
        coeffs = None
        deg = 0
        if fit:
            coeffs, deg = polyfit_curve(x, y, max_degree=3)
        if coeffs is None:
            return
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            return
        x_fit = np.linspace(float(np.min(x[finite])), float(np.max(x[finite])), 250)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), color="black", lw=2, label=f"poly deg {deg}")
        ax.legend()

    def plot_colored_points(
        ax: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        xerr: np.ndarray | None = None,
        yerr: np.ndarray | None = None,
        xerr_range: np.ndarray | None = None,
        yerr_range: np.ndarray | None = None,
    ) -> None:
        for i, (xi, yi, c) in enumerate(zip(x, y, point_colors)):
            if not (np.isfinite(xi) and np.isfinite(yi)):
                continue
            _draw_point(
                ax,
                float(xi),
                float(yi),
                c,
                xerr_std=_error_at(xerr, i),
                yerr_std=_error_at(yerr, i),
                xerr_range=_error_at(xerr_range, i),
                yerr_range=_error_at(yerr_range, i),
            )

    def draw_baseline_occupancy_pair(ax: plt.Axes) -> None:
        for i, (xv, c) in enumerate(zip(lambdas, point_colors)):
            y_base = float(baseline_occ_mean[i])
            y_base_off = float(baseline_off_occ_mean[i])
            if np.isfinite(xv) and np.isfinite(y_base):
                _draw_point(
                    ax,
                    xv,
                    y_base,
                    c,
                    marker="o",
                    yerr_std=_error_at(baseline_occ_std, i),
                    yerr_range=_error_at(baseline_occ_minmax, i),
                )
            if np.isfinite(xv) and np.isfinite(y_base_off):
                _draw_point(
                    ax,
                    xv,
                    y_base_off,
                    c,
                    marker="^",
                    yerr_std=_error_at(baseline_off_occ_std, i),
                    yerr_range=_error_at(baseline_off_occ_minmax, i),
                )
        ax.scatter([], [], marker="o", color="black", label="Baseline")
        ax.scatter([], [], marker="^", color="black", label="Baseline_off")
        ax.legend()

    panel_specs: list[tuple[str, Callable[[plt.Axes], None]]] = []

    def _panel(
        slug: str,
        title: str,
        xlabel: str,
        ylabel: str,
        draw_body: Callable[[plt.Axes], None],
    ) -> None:
        def _draw(ax: plt.Axes) -> None:
            draw_body(ax)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
        panel_specs.append((slug, _draw))

    _panel(
        "01_lambda_vs_agent_occupancy",
        "Lambda vs Occupancy/Episode",
        "Poisson lambda (arrivals)",
        "Agent Occupancy (Nodes, %) / Episode",
        lambda ax: (
            plot_colored_points(ax, lambdas, occ_mean, yerr=occ_std, yerr_range=occ_minmax),
            _maybe_plot_fit(ax, lambdas, occ_mean),
        ),
    )
    _panel(
        "02_occupancy_vs_prop_savings",
        "Occupancy/Episode vs Proportional Savings (%)",
        "Agent Occupancy (Nodes, %) / Episode",
        "Prop Savings vs Baseline (%)",
        lambda ax: (
            plot_colored_points(ax, occ_mean, prop_sav_pct_mean, xerr=occ_std, yerr=prop_sav_pct_std),
            _maybe_plot_fit(ax, occ_mean, prop_sav_pct_mean),
        ),
    )
    _panel(
        "03_occupancy_vs_prop_savings_off",
        "Occupancy vs Proportional Savings_off (%)",
        "Agent Occupancy (Nodes, %) / Episode",
        "Prop Savings vs Baseline_off (%)",
        lambda ax: (
            plot_colored_points(ax, occ_mean, prop_sav_pct_off_mean, xerr=occ_std, yerr=prop_sav_pct_off_std),
            _maybe_plot_fit(ax, occ_mean, prop_sav_pct_off_mean),
        ),
    )
    _panel(
        "04_lambda_vs_completion_rate",
        "Lambda vs Agent Completion Rate",
        "Poisson lambda (arrivals)",
        "Completion Rate (%)",
        lambda ax: (plot_colored_points(ax, lambdas, completion_mean, yerr=completion_std), _maybe_plot_fit(ax, lambdas, completion_mean)),
    )
    _panel(
        "05_occupancy_vs_prop_effective_savings",
        "Occupancy vs Proportional Effective Savings (%)",
        "Agent Occupancy (Nodes, %) / Episode",
        "Prop Effective Savings vs Baseline (% adjusted)",
        lambda ax: (
            plot_colored_points(ax, occ_mean, prop_eff_sav_pct_mean, xerr=occ_std, yerr=prop_eff_sav_pct_std),
            _maybe_plot_fit(ax, occ_mean, prop_eff_sav_pct_mean),
        ),
    )
    _panel(
        "06_occupancy_vs_prop_effective_savings_off",
        "Occupancy vs Proportional Effective Savings_off (%)",
        "Agent Occupancy (Nodes, %) / Episode",
        "Prop Effective Savings vs Baseline_off (% adjusted)",
        lambda ax: (
            plot_colored_points(ax, occ_mean, prop_eff_sav_pct_off_mean, xerr=occ_std, yerr=prop_eff_sav_pct_off_std),
            _maybe_plot_fit(ax, occ_mean, prop_eff_sav_pct_off_mean),
        ),
    )
    _panel(
        "07_occupancy_vs_average_wait_delta",
        "Occupancy vs Average Wait Delta",
        "Agent Occupancy (Nodes, %) / Episode",
        "Average Wait Delta (Agent - Baseline, hours)",
        lambda ax: (
            plot_colored_points(ax, occ_mean, wait_delta_hours, xerr=occ_std),
            _maybe_plot_fit(ax, occ_mean, wait_delta_hours),
        ),
    )
    _panel(
        "08_occupancy_vs_cost_per_1k_delta_baseline_off",
        "Occupancy vs Cost/1k Delta vs Baseline_off",
        "Agent Occupancy (Nodes, %) / Episode",
        "(Baseline_off - Agent) / Baseline_off  [%]",
        lambda ax: (
            plot_colored_points(
                ax,
                occ_mean,
                cost_per_1k_delta_base_off_mean,
                xerr=occ_std,
                yerr=cost_per_1k_delta_base_off_std,
            ),
            _maybe_plot_fit(ax, occ_mean, cost_per_1k_delta_base_off_mean),
        ),
    )
    _panel(
        "09_occupancy_vs_prop_power_delta_baseline_off",
        "Occupancy vs Prop Power Delta vs Baseline_off",
        "Agent Occupancy (Nodes, %) / Episode",
        "Prop Power Delta vs Baseline_off (%)",
        lambda ax: (
            plot_colored_points(
                ax,
                occ_mean,
                prop_power_delta_base_off_mean,
                xerr=occ_std,
                yerr=prop_power_delta_base_off_std,
            ),
            _maybe_plot_fit(ax, occ_mean, prop_power_delta_base_off_mean),
        ),
    )
    _panel(
        "10_lambda_vs_baseline_occupancies",
        "Lambda vs Baseline Occupancies",
        "Poisson lambda (arrivals)",
        "Baseline Occupancy (Nodes, %) / Episode",
        draw_baseline_occupancy_pair,
    )
    _panel(
        "11_lambda_vs_jobs_per_hour",
        "Lambda vs Job Arrivals/Hour",
        "Poisson lambda (arrivals)",
        "Job Arrivals/Hour (mean ± std)",
        lambda ax: (plot_colored_points(ax, lambdas, arrivals_per_hour_mean, yerr=arrivals_per_hour_std), _maybe_plot_fit(ax, lambdas, arrivals_per_hour_mean)),
    )
    _panel(
        "12_lambda_vs_dropped_jobs_delta",
        "Lambda vs Dropped Jobs Delta",
        "Poisson lambda (arrivals)",
        "Dropped Jobs Delta (Agent - Baseline)",
        lambda ax: (plot_colored_points(ax, lambdas, dropped_jobs_delta_total), _maybe_plot_fit(ax, lambdas, dropped_jobs_delta_total)),
    )

    fig, axes = plt.subplots(4, 3, figsize=(22, 22), constrained_layout=True)
    for ax, (_, draw_fn) in zip(axes.ravel(), panel_specs):
        draw_fn(ax)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.02)
    cbar.set_label(colorbar_label)
    fig.savefig(path, dpi=220)
    plt.close(fig)

    if individual_dir is not None:
        individual_dir.mkdir(parents=True, exist_ok=True)
        for slug, draw_fn in panel_specs:
            panel_path = individual_dir / f"{slug}.png"
            fig_i, ax_i = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
            draw_fn(ax_i)
            sm_i = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm_i.set_array([])
            cbar_i = fig_i.colorbar(sm_i, ax=ax_i, pad=0.02)
            cbar_i.set_label(colorbar_label)
            fig_i.savefig(panel_path, dpi=220)
            plt.close(fig_i)


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep Poisson lambdas and fit occupancy/savings trend lines."
    )

    # Core train.py params (default values mirror the command from the request).
    parser.add_argument("--prices", default="./data/prices_2023.csv")
    parser.add_argument("--session", default="")
    parser.add_argument("--efficiency-weight", type=float, default=0.6)
    parser.add_argument("--price-weight", type=float, default=0.1)
    parser.add_argument("--idle-weight", type=float, default=0.1)
    parser.add_argument("--job-age-weight", type=float, default=0.2)
    parser.add_argument("--drop-weight", type=float, default=0.0)
    parser.add_argument("--eval-months", type=int, default=12)
    parser.add_argument("--model", type=int, default=1000000)

    parser.add_argument("--plot-dashboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dashboard-hours", type=int, default=None)

    parser.add_argument("--wg-duration-lambda", type=float, default=0.3)
    parser.add_argument("--wg-nodes-lambda", type=float, default=1.0)
    parser.add_argument("--wg-cores-lambda", type=float, default=3.0)
    parser.add_argument("--wg-max-jobs-hour", type=int, default=70000)
    parser.add_argument("--wg-burst-small-prob", type=float, default=0.1329)
    parser.add_argument("--wg-burst-heavy-prob", type=float, default=0.0043)

    # Lambda sweep controls.
    parser.add_argument("--lambdas", type=str, default="", help="Explicit comma-separated lambda list. If set, adaptive selection is disabled.")
    parser.add_argument("--reference-lambda", type=int, default=2000)
    parser.add_argument("--min-lambda", type=int, default=100)
    parser.add_argument("--max-lambda", type=int, default=12000)
    parser.add_argument("--num-points", type=int, default=7, help="Target number of lambda points for adaptive sweep.")
    parser.add_argument("--bracket-steps", type=int, default=2, help="Low/high probing steps from reference lambda.")
    parser.add_argument("--bracket-factor", type=float, default=2.0, help="Factor for high/low probing.")
    parser.add_argument("--target-occupancy-min", type=float, default=20.0)
    parser.add_argument("--target-occupancy-max", type=float, default=100.0)
    parser.add_argument("--occupancy-tolerance", type=float, default=2.0)

    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--save-logs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--echo-train-output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fit", action="store_true", default=False, help="Enable polynomial fitting of datasets")
    parser.add_argument("--oracle", action="store_true", default=False, help="Enable oracle for evaluation")
    parser.add_argument("--seed", type=int, default=None, help="Forwarded to train.py (sampling mode only).")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.num_points < 2:
        parser.error("--num-points must be >= 2")
    if args.eval_months <= 0:
        parser.error("--eval-months must be > 0")
    if args.min_lambda <= 0:
        parser.error("--min-lambda must be > 0")
    if args.max_lambda < args.min_lambda:
        parser.error("--max-lambda must be >= --min-lambda")
    if args.bracket_factor <= 1.0:
        parser.error("--bracket-factor must be > 1")

    project_root = Path(__file__).resolve().parent
    train_py = project_root / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"Could not find train.py at: {train_py}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir_name = build_analysis_dir_name(
            prefix="lambda_occupancy_sweep",
            timestamp=timestamp,
            model=args.model,
            efficiency_weight=args.efficiency_weight,
            price_weight=args.price_weight,
            idle_weight=args.idle_weight,
            job_age_weight=args.job_age_weight,
        )
        out_dir = project_root / "analysis" / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    if args.save_logs:
        logs_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[int, LambdaRunStats] = {}

    def evaluate(lam: int) -> LambdaRunStats:
        lam = clamp_int(lam, args.min_lambda, args.max_lambda)
        if lam in cache:
            return cache[lam]
        stats, raw_output = run_lambda_eval(args, project_root, lam)
        cache[lam] = stats
        if args.save_logs:
            log_path = logs_dir / f"lambda_{lam}.log"
            log_path.write_text(raw_output)
        return stats

    # Attach cache reader for select_lambda_schedule().
    evaluate.cache_keys = cache.keys  # type: ignore[attr-defined]

    selected_lambdas = select_lambda_schedule(args, evaluate)
    stats_ordered = [cache[l] for l in sorted(selected_lambdas)]

    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"
    plot_path = out_dir / "trendlines.png"
    individual_plots_dir = out_dir / "plots_individual"

    write_summary_csv(csv_path, stats_ordered)
    with json_path.open("w") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(),
                "selected_lambdas": selected_lambdas,
                "args": vars(args),
                "results": [asdict(s) for s in stats_ordered],
            },
            f,
            indent=2,
        )
    make_plot(plot_path, stats_ordered, fit=args.fit, individual_dir=individual_plots_dir)

    print("\nSweep complete.")
    print(f"  Lambdas: {selected_lambdas}")
    print(f"  Evaluation months: {args.eval_months}")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  Plot: {plot_path}")
    print(f"  Individual Plots: {individual_plots_dir}")


if __name__ == "__main__":
    main()
