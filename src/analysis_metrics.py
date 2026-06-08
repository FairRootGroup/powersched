"""Shared parsing, computation, and plotting helpers for sweep analysis scripts."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from src.analysis_reporting import compute_savings_totals


# ── Regexes ───────────────────────────────────────────────────────────────────

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


# ── Float helpers ─────────────────────────────────────────────────────────────

def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def _to_float_or_nan(raw: str | None) -> float:
    if raw is None:
        return float("nan")
    val = raw.strip().lower()
    if val in {"n/a", "nan"}:
        return float("nan")
    return _to_float(raw)


# ── Parsing ───────────────────────────────────────────────────────────────────

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


# ── Math helpers ──────────────────────────────────────────────────────────────

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


def polyfit_curve(x: np.ndarray, y: np.ndarray, max_degree: int = 3) -> tuple[np.ndarray | None, int]:
    finite = np.isfinite(x) & np.isfinite(y)
    xf = x[finite]
    yf = y[finite]
    if xf.size < 2:
        return None, 0
    degree = min(max_degree, xf.size - 1)
    coeffs = np.polyfit(xf, yf, degree)
    return coeffs, degree


# ── RunStats ──────────────────────────────────────────────────────────────────

@dataclass
class RunStats:
    sweep_key: float
    replay_mode: str
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


def make_run_stats(
    sweep_key: float,
    replay_mode: str,
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
) -> RunStats:
    wait_delta_hours = agent_avg_wait_hours - baseline_avg_wait_hours
    effective_savings = safe_divide(savings * (completion_rate / 100) ** 2, wait_delta_hours + 1)
    effective_savings_off = safe_divide(savings_off * (completion_rate / 100) ** 2, wait_delta_hours + 1)
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
    prop_power_delta_pct_baseline_off_mean, prop_power_delta_pct_baseline_off_std = finite_mean_std(prop_power_delta_pct_baseline_off)
    baseline_off_occupancy = baseline_occupancy.copy()
    dropped_jobs_delta_total = dropped_jobs_agent_total - dropped_jobs_baseline_total
    evaluation_savings, annualized_savings = compute_savings_totals(savings, eval_months)
    evaluation_savings_off, annualized_savings_off = compute_savings_totals(savings_off, eval_months)
    prop_evaluation_savings, prop_annualized_savings = compute_savings_totals(prop_savings, eval_months)
    prop_evaluation_savings_off, prop_annualized_savings_off = compute_savings_totals(prop_savings_off, eval_months)

    return RunStats(
        sweep_key=float(sweep_key),
        replay_mode=replay_mode,
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


# ── Plot helpers ──────────────────────────────────────────────────────────────

def os_tail(text: str, lines: int = 20) -> str:
    parts = text.rstrip().splitlines()
    if not parts:
        return ""
    return "\n".join(parts[-lines:])


def minmax_error_array(sample_lists: list[list[float]], means: np.ndarray) -> np.ndarray:
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


def error_at(arr: np.ndarray | None, idx: int) -> float | np.ndarray | None:
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


def draw_point(
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


def maybe_plot_fit(ax: plt.Axes, x: np.ndarray, y: np.ndarray, fit: bool) -> None:
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


# ── CSV helpers ───────────────────────────────────────────────────────────────

CSV_COMMON_FIELDNAMES: list[str] = [
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


def csv_common_row(s: RunStats) -> dict[str, object]:
    return {
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


# ── Misc utilities ────────────────────────────────────────────────────────────

def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def unique_ints_sorted(values: list[int]) -> list[int]:
    return sorted({int(v) for v in values})
