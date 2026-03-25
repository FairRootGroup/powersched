"""Workload generation logic for the PowerSched environment."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
import numpy as np
from src.arrival_scale import validate_job_arrival_scale
from src.config import (
    MAX_NEW_JOBS_PER_HOUR, MAX_JOB_DURATION, MIN_NODES_PER_JOB,
    MAX_NODES_PER_JOB, MIN_CORES_PER_JOB, CORES_PER_NODE
)

if TYPE_CHECKING:
    from src.sampler_jobs import DurationSampler as JobsSampler
    from src.sampler_hourly import HourlySampler
    from src.sampler_duration import DurationSampler as DurationsSampler
    from src.workloadgen import WorkloadGenerator


def generate_jobs(
    current_hour: int,
    external_jobs: str | None,
    external_hourly_jobs: str | None,
    external_durations: str | None,
    workload_gen: WorkloadGenerator | None,
    jobs_sampler: JobsSampler | None,
    hourly_sampler: HourlySampler,
    durations_sampler: DurationsSampler,
    np_random: np.random.Generator,
    job_arrival_scale: float = 1.0,
    jobs_exact_replay: bool = False,
) -> tuple[int, list[int], list[int], list[int]]:
    """
    Generate new jobs for the current hour using configured workload source.

    Args:
        current_hour: Current simulation hour (0-indexed)
        external_jobs: Path to external jobs file (or None)
        external_hourly_jobs: Path to external hourly jobs file (or None)
        external_durations: Path to external durations file (or None)
        workload_gen: Workload generator object (or None)
        jobs_sampler: Jobs sampler object
        hourly_sampler: Hourly sampler object
        durations_sampler: Durations sampler object
        np_random: NumPy random generator
        job_arrival_scale: Multiplier for sampled arrivals per step.
            - 1.0: unchanged
            - >1.0: upsample jobs
            - 0.0..1.0: downsample jobs
        jobs_exact_replay: If True, replay raw jobs in log order for --jobs mode.
    Returns:
        Tuple of (new_jobs_count, new_jobs_durations, new_jobs_nodes, new_jobs_cores)
    """
    job_arrival_scale = validate_job_arrival_scale(job_arrival_scale)
    new_jobs_durations = []
    new_jobs_nodes = []
    new_jobs_cores = []
    new_jobs_count = 0
    arrival_scale_applied_in_source = False

    if external_jobs and not workload_gen:
        if jobs_exact_replay:
            # Replay jobs exactly as they appear in the parsed timeline (one bin per step).
            sampled = jobs_sampler.sample(1, wrap=True)
            raw_jobs = next(iter(sampled.values()), [])
            for job in raw_jobs:
                duration_hours = max(1, int(math.ceil(int(job['duration_minutes']) / 60)))
                nnodes = min(max(int(job['nnodes']), MIN_NODES_PER_JOB), MAX_NODES_PER_JOB)
                cores_per_node = min(max(int(job['cores_per_node']), MIN_CORES_PER_JOB), CORES_PER_NODE)
                new_jobs_count += 1
                new_jobs_durations.append(duration_hours)
                new_jobs_nodes.append(nnodes)
                new_jobs_cores.append(cores_per_node)
        else:
            # Use pre-aggregated hourly templates for pattern-based replay.
            sampled = jobs_sampler.sample_one_hourly(wrap=True)
            jobs = sampled.get("hourly_jobs", [])
            if len(jobs) > 0:
                for job in jobs:
                    instances = max(1, int(job.get('instances', 1)))
                    new_jobs_count += instances
                    new_jobs_durations.extend([job['duration_hours']] * instances)
                    new_jobs_nodes.extend([job['nnodes']] * instances)
                    new_jobs_cores.extend([job['cores_per_node']] * instances)

    elif external_hourly_jobs:
        # Use hourly sampler for statistical sampling with aggregated jobs
        hour_of_day = (current_hour - 1) % 24

        jobs = hourly_sampler.sample_aggregated(
            hour_of_day,
            rng=np_random,
            arrival_scale=job_arrival_scale,
        )
        arrival_scale_applied_in_source = True

        if len(jobs) > 0:
            for job in jobs:
                new_jobs_count += 1
                new_jobs_durations.append(job['duration_hours'])
                new_jobs_nodes.append(job['nodes'])
                new_jobs_cores.append(job['cores_per_node'])

    else:
        # Use Workload Generator for Randomizer
        if workload_gen is not None:
            jobs = workload_gen.sample(current_hour - 1, np_random)
            new_jobs_count = len(jobs)
            if new_jobs_count > 0:
                for j in jobs:
                    new_jobs_durations.append(j.duration)
                    new_jobs_nodes.append(j.nodes)
                    new_jobs_cores.append(j.cores_per_node)
        # Legacy Randomizer
        else:
            new_jobs_count = np_random.integers(0, MAX_NEW_JOBS_PER_HOUR + 1)
            if external_durations:
                new_jobs_durations = durations_sampler.sample(new_jobs_count).tolist()
            else:
                new_jobs_durations = np_random.integers(1, MAX_JOB_DURATION + 1, size=new_jobs_count).tolist()
            # Generate random node and core requirements
            for _ in range(new_jobs_count):
                new_jobs_nodes.append(np_random.integers(MIN_NODES_PER_JOB, MAX_NODES_PER_JOB + 1))
                new_jobs_cores.append(np_random.integers(MIN_CORES_PER_JOB, CORES_PER_NODE + 1))

    # Global arrival scaling for sources that do not apply it internally.
    if (not arrival_scale_applied_in_source) and new_jobs_count > 0 and job_arrival_scale != 1.0:
        if job_arrival_scale <= 0.0:
            return 0, [], [], []

        whole = int(np.floor(job_arrival_scale))
        frac = float(job_arrival_scale - whole)

        scaled_durations: list[int] = []
        scaled_nodes: list[int] = []
        scaled_cores: list[int] = []

        if whole > 0:
            scaled_durations.extend(new_jobs_durations * whole)
            scaled_nodes.extend(new_jobs_nodes * whole)
            scaled_cores.extend(new_jobs_cores * whole)

        if frac > 0.0:
            for d, n, c in zip(new_jobs_durations, new_jobs_nodes, new_jobs_cores):
                if np_random.random() < frac:
                    scaled_durations.append(d)
                    scaled_nodes.append(n)
                    scaled_cores.append(c)

        new_jobs_durations = scaled_durations
        new_jobs_nodes = scaled_nodes
        new_jobs_cores = scaled_cores
        new_jobs_count = len(new_jobs_durations)

    return new_jobs_count, new_jobs_durations, new_jobs_nodes, new_jobs_cores
