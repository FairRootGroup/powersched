"""Shared CLI helpers for workload generator configuration.

Provides:
- Argument parsers for quad-parameter CLI flags (floats, ints, ranges)
- add_workloadgen_args(): register workload-gen argparse flags on a parser
- build_workloadgen_config(): construct WorkloadGenConfig from parsed args
"""

from __future__ import annotations

import argparse

from src.config import (
    MAX_JOB_DURATION,
    MIN_NODES_PER_JOB, MAX_NODES_PER_JOB,
    MIN_CORES_PER_JOB,
    CORES_PER_NODE,
)
from src.workloadgen import WorkloadGenConfig


def parse_quad_floats(raw: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in str(raw).split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Expected 4 comma-separated floats: arrivals,duration,nodes,cores"
        )
    try:
        a, b, c, d = (float(p) for p in parts)
        return (a, b, c, d)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float in '{raw}'") from exc


def parse_quad_ints(raw: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in str(raw).split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Expected 4 comma-separated ints: arrivals,duration,nodes,cores"
        )
    try:
        a, b, c, d = (int(p) for p in parts)
        return (a, b, c, d)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid int in '{raw}'") from exc


def parse_quad_ranges(raw: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    parts = [p.strip() for p in str(raw).split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Expected 4 comma-separated ranges: a_min:a_max,d_min:d_max,n_min:n_max,c_min:c_max"
        )
    ranges = []
    for part in parts:
        bounds = [b.strip() for b in part.split(":")]
        if len(bounds) != 2:
            raise argparse.ArgumentTypeError(f"Invalid range '{part}', expected min:max")
        try:
            low = int(bounds[0])
            high = int(bounds[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid int in range '{part}'") from exc
        if low > high:
            raise argparse.ArgumentTypeError(f"Range min > max in '{part}'")
        ranges.append((low, high))
    (a, b, c, d) = ranges
    return (a, b, c, d)


def add_workloadgen_args(parser: argparse.ArgumentParser) -> None:
    """Register workload-generator CLI flags on *parser*."""
    parser.add_argument(
        "--workload-gen", type=str, default="",
        choices=["", "flat", "poisson", "uniform"],
        help="Enable workload generator (default: disabled).",
    )
    parser.add_argument("--wg-poisson-lambda", type=float, default=200.0, help="Poisson lambda for arrivals (used when --wg-poisson-lambdas4 is not set).")
    parser.add_argument("--wg-poisson-lambdas4", type=parse_quad_floats, default=None, help="arrivals,duration,nodes,cores")
    parser.add_argument("--wg-max-jobs-hour", type=int, default=1500, help="Cap jobs/hour for the workload generator.")
    parser.add_argument("--wg-flat-jobs-hour", type=int, default=200, help="Flat target for arrivals (used when --wg-flat-targets4 is not set).")
    parser.add_argument("--wg-flat-jitter", type=int, default=0, help="Flat jitter for arrivals (used when --wg-flat-jitters4 is not set).")
    parser.add_argument("--wg-flat-targets4", type=parse_quad_ints, default=None, help="arrivals,duration,nodes,cores")
    parser.add_argument("--wg-flat-jitters4", type=parse_quad_ints, default=None, help="arrivals,duration,nodes,cores")
    parser.add_argument("--wg-uniform-ranges4", type=parse_quad_ranges, default=None, help="a_min:a_max,d_min:d_max,n_min:n_max,c_min:c_max")
    parser.add_argument("--wg-burst-small-prob", type=float, default=0.0, help="Probability of additive small-job burst per hour.")
    parser.add_argument("--wg-burst-heavy-prob", type=float, default=0.0, help="Probability of additive heavy-job burst per hour.")


def build_workloadgen_config(
    args: argparse.Namespace,
    min_duration: int = 1,
    max_duration: int = MAX_JOB_DURATION,
    min_nodes: int = MIN_NODES_PER_JOB,
    max_nodes: int = MAX_NODES_PER_JOB,
    min_cores: int = MIN_CORES_PER_JOB,
    max_cores: int = CORES_PER_NODE,
) -> WorkloadGenConfig | None:
    """Build a WorkloadGenConfig from parsed argparse *args*.

    Returns None if the workload generator is not enabled (--workload-gen is empty).
    """
    arrivals = getattr(args, "workload_gen", "")
    if not arrivals:
        return None

    uniform_min_jobs = 0
    max_jobs_hour = args.wg_max_jobs_hour

    if args.wg_uniform_ranges4 is not None:
        (
            (uniform_min_jobs, max_jobs_hour),
            (min_duration, max_duration),
            (min_nodes, max_nodes),
            (min_cores, max_cores),
        ) = args.wg_uniform_ranges4

    duration_mid = (min_duration + max_duration) // 2
    nodes_mid = (min_nodes + max_nodes) // 2
    cores_mid = (min_cores + max_cores) // 2

    if args.wg_poisson_lambdas4 is not None:
        poisson_lambda_arrivals, poisson_lambda_duration, poisson_lambda_nodes, poisson_lambda_cores = args.wg_poisson_lambdas4
    else:
        poisson_lambda_arrivals = args.wg_poisson_lambda
        poisson_lambda_duration = float(duration_mid)
        poisson_lambda_nodes = float(nodes_mid)
        poisson_lambda_cores = float(cores_mid)

    if args.wg_flat_targets4 is not None:
        flat_jobs_per_hour, flat_duration_target, flat_nodes_target, flat_cores_target = args.wg_flat_targets4
    else:
        flat_jobs_per_hour = args.wg_flat_jobs_hour
        flat_duration_target = duration_mid
        flat_nodes_target = nodes_mid
        flat_cores_target = cores_mid

    if args.wg_flat_jitters4 is not None:
        flat_jitter_arrivals, flat_duration_jitter, flat_nodes_jitter, flat_cores_jitter = args.wg_flat_jitters4
    else:
        flat_jitter_arrivals = args.wg_flat_jitter
        flat_duration_jitter = 0
        flat_nodes_jitter = 0
        flat_cores_jitter = 0

    return WorkloadGenConfig(
        arrivals=arrivals,
        uniform_min_new_jobs_per_hour=uniform_min_jobs,
        max_new_jobs_per_hour=max_jobs_hour,
        poisson_lambda=poisson_lambda_arrivals,
        poisson_lambda_duration=poisson_lambda_duration,
        poisson_lambda_nodes=poisson_lambda_nodes,
        poisson_lambda_cores=poisson_lambda_cores,
        flat_jobs_per_hour=flat_jobs_per_hour,
        flat_jitter=flat_jitter_arrivals,
        flat_duration_target=flat_duration_target,
        flat_nodes_target=flat_nodes_target,
        flat_cores_target=flat_cores_target,
        flat_duration_jitter=flat_duration_jitter,
        flat_nodes_jitter=flat_nodes_jitter,
        flat_cores_jitter=flat_cores_jitter,
        burst_small_prob=args.wg_burst_small_prob,
        burst_heavy_prob=args.wg_burst_heavy_prob,
        min_duration=min_duration,
        max_duration=max_duration,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        min_cores=min_cores,
        max_cores=max_cores,
    )
