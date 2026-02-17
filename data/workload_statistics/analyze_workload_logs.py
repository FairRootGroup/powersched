#!/usr/bin/env python3
"""Summarize workload logs per file and estimate burst probabilities.

For each input file, compute:
- arrivals per hour: mean, stddev
- duration (hours): mean, stddev
- nodes: mean, stddev
- cores: mean, stddev
- Pearson correlations:
  - duration vs nodes
  - duration vs cores
- burst probability suggestions for:
  - wg-burst-small-prob
  - wg-burst-heavy-prob

The script supports:
- whitespace-delimited Slurm-like logs (as in data-internal/allusers-*.log)
- standard CSV files with matching column names
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from statistics import fmean, stdev
from typing import Dict, Iterable, List, Sequence, Tuple


SUBMIT_CANDIDATES = ("Submit", "submit", "SUBMIT", "submission_time", "timestamp")
DURATION_CANDIDATES = ("ElapsedRaw", "elapsed_raw", "ELAPSEDRAW", "duration_seconds", "duration")
NODES_CANDIDATES = ("NNodes", "nnodes", "NNODES", "nodes")
CORES_CANDIDATES = ("NCPUS", "ncpus", "NCPUs", "cores", "cores_per_node")


def pick_column(columns: Sequence[str], candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find any of {candidates} in columns: {columns}")


def parse_submit_hour(value: str) -> datetime:
    dt = datetime.fromisoformat(value.strip())
    return dt.replace(minute=0, second=0, microsecond=0)


def parse_duration_hours(value: str) -> float:
    raw = value.strip()
    # Common case in these logs: ElapsedRaw is integer seconds.
    try:
        seconds = float(raw)
        return seconds / 3600.0
    except ValueError:
        pass

    # Fallback: parse HH:MM:SS or D-HH:MM:SS
    day_part = 0
    time_part = raw
    if "-" in raw:
        maybe_day, maybe_time = raw.split("-", 1)
        if maybe_day.isdigit():
            day_part = int(maybe_day)
            time_part = maybe_time
    parts = time_part.split(":")
    if len(parts) == 3:
        hh, mm, ss = parts
    elif len(parts) == 2:
        hh, mm = parts
        ss = "0"
    else:
        raise ValueError(f"Cannot parse duration: {raw!r}")
    total_seconds = (day_part * 24 + int(hh)) * 3600 + int(mm) * 60 + int(ss)
    return total_seconds / 3600.0


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mx = fmean(x)
    my = fmean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return fmean(values), stdev(values)


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    q = min(max(float(q), 0.0), 1.0)
    s = sorted(float(v) for v in values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def parse_min_max(raw: str) -> Tuple[int, int]:
    parts = [p.strip() for p in str(raw).split(":")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected min:max")
    try:
        lo = int(parts[0])
        hi = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid min:max pair '{raw}'") from exc
    if lo > hi:
        raise argparse.ArgumentTypeError(f"min > max in '{raw}'")
    return lo, hi


def hourly_axis(arrivals_by_hour: Counter, include_zero_hours: bool) -> List[datetime]:
    if not arrivals_by_hour:
        return []
    if not include_zero_hours:
        return sorted(arrivals_by_hour.keys())

    start = min(arrivals_by_hour)
    end = max(arrivals_by_hour)
    current = start
    values: List[datetime] = []
    while current <= end:
        values.append(current)
        current += timedelta(hours=1)
    return values


def iter_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        first_nonempty = None
        for line in fh:
            if line.strip():
                first_nonempty = line
                break
        if first_nonempty is None:
            return

        if "," in first_nonempty:
            fh.seek(0)
            reader = csv.DictReader(fh)
            for row in reader:
                if not row:
                    continue
                yield row
            return

        # Whitespace-delimited format with a dashed separator on line 2.
        columns = first_nonempty.split()
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if set(stripped) <= {"-"}:
                continue
            parts = line.split()
            if len(parts) < len(columns):
                continue
            row = {columns[i]: parts[i] for i in range(len(columns))}
            yield row


def summarize_file(
    path: Path,
    include_zero_hours: bool,
    small_duration_max: float,
    small_nodes_max: float,
    small_cores_max: float,
    heavy_duration_min: float,
    heavy_nodes_min: float,
    heavy_cores_min: float,
    assumed_small_jobs_min: int,
    assumed_small_jobs_max: int,
    assumed_heavy_jobs_min: int,
    assumed_heavy_jobs_max: int,
    baseline_quantile: float,
) -> Dict[str, float]:
    arrivals_by_hour: Counter = Counter()
    small_jobs_by_hour: Counter = Counter()
    heavy_jobs_by_hour: Counter = Counter()
    durations_h: List[float] = []
    nodes: List[float] = []
    cores: List[float] = []
    skipped = 0

    rows = iter_rows(path)
    rows = iter(rows)
    try:
        first = next(rows)
    except StopIteration:
        return {
            "jobs": 0,
            "skipped_rows": 0,
            "arrival_mean": float("nan"),
            "arrival_std": float("nan"),
            "duration_mean_h": float("nan"),
            "duration_std_h": float("nan"),
            "nodes_mean": float("nan"),
            "nodes_std": float("nan"),
            "cores_mean": float("nan"),
            "cores_std": float("nan"),
            "corr_duration_nodes": float("nan"),
            "corr_duration_cores": float("nan"),
            "hours_observed": 0,
            "small_baseline_jobs_per_hour": float("nan"),
            "heavy_baseline_jobs_per_hour": float("nan"),
            "small_event_prob": float("nan"),
            "heavy_event_prob": float("nan"),
            "small_volume_prob": float("nan"),
            "heavy_volume_prob": float("nan"),
            "suggested_wg_burst_small_prob": float("nan"),
            "suggested_wg_burst_heavy_prob": float("nan"),
        }

    # Determine column mapping from the first row, then process all rows (including first).
    keys = list(first.keys())
    submit_col = pick_column(keys, SUBMIT_CANDIDATES)
    duration_col = pick_column(keys, DURATION_CANDIDATES)
    nodes_col = pick_column(keys, NODES_CANDIDATES)
    cores_col = pick_column(keys, CORES_CANDIDATES)

    def consume(row: Dict[str, str]) -> None:
        nonlocal skipped
        try:
            hour = parse_submit_hour(row[submit_col])
            duration = parse_duration_hours(row[duration_col])
            node_count = float(row[nodes_col])
            core_count = float(row[cores_col])
        except Exception:
            skipped += 1
            return
        arrivals_by_hour[hour] += 1
        if duration <= small_duration_max and node_count <= small_nodes_max and core_count <= small_cores_max:
            small_jobs_by_hour[hour] += 1
        if duration >= heavy_duration_min and node_count >= heavy_nodes_min and core_count >= heavy_cores_min:
            heavy_jobs_by_hour[hour] += 1
        durations_h.append(duration)
        nodes.append(node_count)
        cores.append(core_count)

    consume(first)
    for row in rows:
        consume(row)

    axis = hourly_axis(arrivals_by_hour, include_zero_hours)
    arrival_series = [float(arrivals_by_hour.get(h, 0)) for h in axis]
    small_series = [float(small_jobs_by_hour.get(h, 0)) for h in axis]
    heavy_series = [float(heavy_jobs_by_hour.get(h, 0)) for h in axis]

    arrival_mean, arrival_std = mean_std(arrival_series)
    duration_mean_h, duration_std_h = mean_std(durations_h)
    nodes_mean, nodes_std = mean_std(nodes)
    cores_mean, cores_std = mean_std(cores)

    n_hours = len(axis)
    active_idx = [i for i, a in enumerate(arrival_series) if a > 0.0]
    if active_idx:
        active_small = [small_series[i] for i in active_idx]
        active_heavy = [heavy_series[i] for i in active_idx]
    else:
        active_small = small_series
        active_heavy = heavy_series

    # Learn baseline from active hours to avoid zero-heavy timelines collapsing
    # burst baselines to 0.
    small_baseline = quantile(active_small, baseline_quantile)
    heavy_baseline = quantile(active_heavy, baseline_quantile)

    small_event_threshold = small_baseline + float(assumed_small_jobs_min)
    heavy_event_threshold = heavy_baseline + float(assumed_heavy_jobs_min)

    if n_hours > 0:
        small_event_prob = sum(1 for v in small_series if v >= small_event_threshold and v > small_baseline) / n_hours
        heavy_event_prob = sum(1 for v in heavy_series if v >= heavy_event_threshold and v > heavy_baseline) / n_hours
    else:
        small_event_prob = float("nan")
        heavy_event_prob = float("nan")

    small_expected_jobs_per_burst = (float(assumed_small_jobs_min) + float(assumed_small_jobs_max)) / 2.0
    heavy_expected_jobs_per_burst = (float(assumed_heavy_jobs_min) + float(assumed_heavy_jobs_max)) / 2.0

    small_excess = sum(max(v - small_baseline, 0.0) for v in small_series)
    heavy_excess = sum(max(v - heavy_baseline, 0.0) for v in heavy_series)

    if n_hours > 0 and small_expected_jobs_per_burst > 0.0:
        small_volume_prob = min(max(small_excess / (n_hours * small_expected_jobs_per_burst), 0.0), 1.0)
    else:
        small_volume_prob = float("nan")
    if n_hours > 0 and heavy_expected_jobs_per_burst > 0.0:
        heavy_volume_prob = min(max(heavy_excess / (n_hours * heavy_expected_jobs_per_burst), 0.0), 1.0)
    else:
        heavy_volume_prob = float("nan")

    # wg-burst-*-prob are event probabilities, so event-rate estimates are the
    # primary recommendation. Volume estimates are kept for diagnostics.
    if math.isfinite(small_event_prob):
        suggested_small_prob = min(max(small_event_prob, 0.0), 1.0)
    elif math.isfinite(small_volume_prob):
        suggested_small_prob = min(max(small_volume_prob, 0.0), 1.0)
    else:
        suggested_small_prob = float("nan")

    if math.isfinite(heavy_event_prob):
        suggested_heavy_prob = min(max(heavy_event_prob, 0.0), 1.0)
    elif math.isfinite(heavy_volume_prob):
        suggested_heavy_prob = min(max(heavy_volume_prob, 0.0), 1.0)
    else:
        suggested_heavy_prob = float("nan")

    return {
        "jobs": len(durations_h),
        "skipped_rows": skipped,
        "arrival_mean": arrival_mean,
        "arrival_std": arrival_std,
        "duration_mean_h": duration_mean_h,
        "duration_std_h": duration_std_h,
        "nodes_mean": nodes_mean,
        "nodes_std": nodes_std,
        "cores_mean": cores_mean,
        "cores_std": cores_std,
        "corr_duration_nodes": pearson_corr(durations_h, nodes),
        "corr_duration_cores": pearson_corr(durations_h, cores),
        "hours_observed": n_hours,
        "small_baseline_jobs_per_hour": small_baseline,
        "heavy_baseline_jobs_per_hour": heavy_baseline,
        "small_event_prob": small_event_prob,
        "heavy_event_prob": heavy_event_prob,
        "small_volume_prob": small_volume_prob,
        "heavy_volume_prob": heavy_volume_prob,
        "suggested_wg_burst_small_prob": suggested_small_prob,
        "suggested_wg_burst_heavy_prob": suggested_heavy_prob,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize workload logs by file.")
    parser.add_argument(
        "files",
        nargs="*",
        help="Input files (CSV or whitespace logs). Default: data-internal/allusers-*.log",
    )
    parser.add_argument(
        "--include-zero-hours",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include zero-arrival hours between first and last submission time (default: true).",
    )
    parser.add_argument("--small-duration-max", type=float, default=8.0, help="Small-job duration threshold (<=).")
    parser.add_argument("--small-nodes-max", type=float, default=2.0, help="Small-job nodes threshold (<=).")
    parser.add_argument("--small-cores-max", type=float, default=16.0, help="Small-job cores threshold (<=).")
    parser.add_argument("--heavy-duration-min", type=float, default=72.0, help="Heavy-job duration threshold (>=).")
    parser.add_argument("--heavy-nodes-min", type=float, default=4.0, help="Heavy-job nodes threshold (>=).")
    parser.add_argument("--heavy-cores-min", type=float, default=32.0, help="Heavy-job cores threshold (>=).")
    parser.add_argument(
        "--assumed-burst-small-jobs",
        type=parse_min_max,
        default=(50, 250),
        help="Assumed small burst size range as min:max (default: 50:250).",
    )
    parser.add_argument(
        "--assumed-burst-heavy-jobs",
        type=parse_min_max,
        default=(1, 12),
        help="Assumed heavy burst size range as min:max (default: 1:12).",
    )
    parser.add_argument(
        "--baseline-quantile",
        type=float,
        default=0.5,
        help="Quantile used as non-burst baseline for hourly small/heavy counts (default: 0.5).",
    )
    args = parser.parse_args()

    if args.files:
        files = [Path(p) for p in args.files]
    else:
        files = sorted(Path("data-internal").glob("allusers-*.log"))

    if not files:
        raise SystemExit("No input files found.")

    for path in files:
        stats = summarize_file(
            path,
            include_zero_hours=args.include_zero_hours,
            small_duration_max=float(args.small_duration_max),
            small_nodes_max=float(args.small_nodes_max),
            small_cores_max=float(args.small_cores_max),
            heavy_duration_min=float(args.heavy_duration_min),
            heavy_nodes_min=float(args.heavy_nodes_min),
            heavy_cores_min=float(args.heavy_cores_min),
            assumed_small_jobs_min=int(args.assumed_burst_small_jobs[0]),
            assumed_small_jobs_max=int(args.assumed_burst_small_jobs[1]),
            assumed_heavy_jobs_min=int(args.assumed_burst_heavy_jobs[0]),
            assumed_heavy_jobs_max=int(args.assumed_burst_heavy_jobs[1]),
            baseline_quantile=float(args.baseline_quantile),
        )
        print(f"\n=== {path.name} ===")
        print(f"jobs={stats['jobs']} skipped_rows={stats['skipped_rows']}")
        print(f"hours_observed={stats['hours_observed']}")
        print(
            "arrivals_per_hour: "
            f"mean={stats['arrival_mean']:.4f} std={stats['arrival_std']:.4f}"
        )
        print(
            "duration_hours: "
            f"mean={stats['duration_mean_h']:.4f} std={stats['duration_std_h']:.4f}"
        )
        print(f"nodes: mean={stats['nodes_mean']:.4f} std={stats['nodes_std']:.4f}")
        print(f"cores: mean={stats['cores_mean']:.4f} std={stats['cores_std']:.4f}")
        print(
            "corr(duration, nodes)="
            f"{stats['corr_duration_nodes']:.6f} "
            "corr(duration, cores)="
            f"{stats['corr_duration_cores']:.6f}"
        )
        print(
            "burst_baseline_jobs_per_hour: "
            f"small={stats['small_baseline_jobs_per_hour']:.4f} "
            f"heavy={stats['heavy_baseline_jobs_per_hour']:.4f}"
        )
        print(
            "burst_prob_estimates: "
            f"small_event={stats['small_event_prob']:.4f} "
            f"small_volume={stats['small_volume_prob']:.4f} "
            f"heavy_event={stats['heavy_event_prob']:.4f} "
            f"heavy_volume={stats['heavy_volume_prob']:.4f}"
        )
        print(
            "suggested_flags: "
            f"--wg-burst-small-prob {stats['suggested_wg_burst_small_prob']:.4f} "
            f"--wg-burst-heavy-prob {stats['suggested_wg_burst_heavy_prob']:.4f}"
        )


if __name__ == "__main__":
    main()
