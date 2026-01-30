"""Workload generation logic for the PowerSched environment."""

import numpy as np
from src.config import (
    MAX_NEW_JOBS_PER_HOUR, MAX_JOB_DURATION, MIN_NODES_PER_JOB,
    MAX_NODES_PER_JOB, MIN_CORES_PER_JOB, CORES_PER_NODE
)


def generate_jobs(current_hour, external_jobs, external_hourly_jobs,
                 external_durations, workload_gen, jobs_sampler, hourly_sampler,
                 durations_sampler, np_random):
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

    Returns:
        Tuple of (new_jobs_count, new_jobs_durations, new_jobs_nodes, new_jobs_cores)
    """
    new_jobs_durations = []
    new_jobs_nodes = []
    new_jobs_cores = []
    new_jobs_count = 0

    if external_jobs and not workload_gen:
        # Use jobs sampler for pattern-based replay
        jobs = jobs_sampler.sample_one_hourly(wrap=True)["hourly_jobs"]
        if len(jobs) > 0:
            for job in jobs:
                new_jobs_count += 1
                new_jobs_durations.append(job['duration_hours'])
                new_jobs_nodes.append(job['nnodes'])
                new_jobs_cores.append(job['cores_per_node'])

    elif external_hourly_jobs:
        # Use hourly sampler for statistical sampling with aggregated jobs
        hour_of_day = (current_hour - 1) % 24

        jobs = hourly_sampler.sample_aggregated(hour_of_day, rng=np_random)

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
                new_jobs_durations = durations_sampler.sample(new_jobs_count)
            else:
                new_jobs_durations = np_random.integers(1, MAX_JOB_DURATION + 1, size=new_jobs_count)
            # Generate random node and core requirements
            for _ in range(new_jobs_count):
                new_jobs_nodes.append(np_random.integers(MIN_NODES_PER_JOB, MAX_NODES_PER_JOB + 1))
                new_jobs_cores.append(np_random.integers(MIN_CORES_PER_JOB, CORES_PER_NODE + 1))

    return new_jobs_count, new_jobs_durations, new_jobs_nodes, new_jobs_cores
