"""Job queue management and scheduling logic for the PowerSched environment."""

from collections import deque
from typing import Any

import numpy as np
from src.config import (
    MAX_NODES, CORES_PER_NODE, MAX_BACKLOG_SIZE, MAX_JOB_AGE
)
from src.metrics_tracker import MetricsTracker


def age_backlog_queue(backlog_queue: deque, _metrics: MetricsTracker, _is_baseline: bool = False) -> int:
    """
    Age jobs waiting in the backlog queue.

    Returns the number of jobs dropped for exceeding ``MAX_JOB_AGE``.
    Callers are responsible for updating any drop counters exactly once.
    """
    if not backlog_queue:
        return 0

    dropped = 0
    kept = []
    for job in backlog_queue:
        job[1] += 1
        if job[1] > MAX_JOB_AGE:
            dropped += 1
        else:
            kept.append(job)

    backlog_queue.clear()
    backlog_queue.extend(kept)
    return dropped


def age_job_queue(job_queue_2d: np.ndarray, next_empty_slot: int) -> tuple[int, int]:
    """
    Age jobs already waiting in the main queue once for the current step.

    Returns the updated ``next_empty_slot`` together with the number of jobs
    dropped for exceeding ``MAX_JOB_AGE``.
    """
    dropped = 0

    for job_idx, job in enumerate(job_queue_2d):
        job_duration = job[0]
        if job_duration <= 0:
            continue

        job_queue_2d[job_idx][1] += 1
        if job_queue_2d[job_idx][1] > MAX_JOB_AGE:
            job_queue_2d[job_idx] = [0, 0, 0, 0]
            if job_idx < next_empty_slot:
                next_empty_slot = job_idx
            dropped += 1

    return next_empty_slot, dropped


def fill_queue_from_backlog(job_queue_2d: np.ndarray, backlog_queue: deque, next_empty_slot: int) -> tuple[int, int]:
    """
    Move jobs from backlog queue into the real queue (FIFO) until full.
    """
    if not backlog_queue:
        return next_empty_slot, 0

    moved = 0
    while backlog_queue and next_empty_slot < len(job_queue_2d):
        job_queue_2d[next_empty_slot] = backlog_queue.popleft()
        moved += 1

        next_empty_slot += 1
        while next_empty_slot < len(job_queue_2d) and job_queue_2d[next_empty_slot][0] != 0:
            next_empty_slot += 1

    return next_empty_slot, moved


def validate_next_empty(job_queue_2d: np.ndarray, next_empty: int) -> None:
    """Validator for debugging queue consistency."""
    n = len(job_queue_2d)
    if next_empty < n:
        assert job_queue_2d[next_empty][0] == 0, "next_empty_slot not empty"
    # everything before must be non-empty
    if next_empty > 0:
        assert np.all(job_queue_2d[:next_empty, 0] != 0), "hole before next_empty_slot"


def process_ongoing_jobs(
        nodes: np.ndarray,
        cores_available: np.ndarray,
        running_jobs: dict[int, dict[str, Any]],
        metrics: MetricsTracker | None = None,
        is_baseline: bool = False,
) -> list[int]:
    """
    Process ongoing jobs: decrement their duration, complete finished jobs,
    release resources, and optionally record completion metrics.

    Args:
        nodes: Array of node states
        cores_available: Array of available cores per node
        running_jobs: Dictionary of currently running jobs
        metrics: Optional metrics tracker to update when jobs finish
        is_baseline: Whether finished jobs belong to the baseline simulation

    Returns:
        List of completed job IDs
    """
    completed_jobs = []
    completed_wait_time = 0

    for job_id, job_data in running_jobs.items():
        job_data['duration'] -= 1

        # Check if job is completed
        if job_data['duration'] <= 0:
            completed_jobs.append(job_id)
            completed_wait_time += int(job_data.get('wait_time', 0))
            # Release resources
            for node_idx, cores_used in job_data['allocation']:
                cores_available[node_idx] += cores_used

    # Remove completed jobs
    for job_id in completed_jobs:
        del running_jobs[job_id]

    if metrics is not None and completed_jobs:
        completed_count = len(completed_jobs)
        if is_baseline:
            metrics.baseline_jobs_completed += completed_count
            metrics.baseline_total_job_wait_time += completed_wait_time
            metrics.episode_baseline_jobs_completed += completed_count
            metrics.episode_baseline_total_job_wait_time += completed_wait_time
        else:
            metrics.jobs_completed += completed_count
            metrics.total_job_wait_time += completed_wait_time
            metrics.episode_jobs_completed += completed_count
            metrics.episode_total_job_wait_time += completed_wait_time

    # Update node times based on remaining jobs
    # Reset all nodes first
    for i in range(MAX_NODES):
        if nodes[i] > 0:  # Don't touch turned-off nodes
            nodes[i] = 0

    # Set node times based on jobs
    for job_id, job_data in running_jobs.items():
        remaining_time = job_data['duration']
        for node_idx, _ in job_data['allocation']:
            nodes[node_idx] = max(nodes[node_idx], remaining_time)

    return completed_jobs


def add_new_jobs(
        job_queue_2d: np.ndarray,
        new_jobs_count: int,
        new_jobs_durations: list[int],
        new_jobs_nodes: list[int],
        new_jobs_cores: list[int],
        next_empty_slot: int,
        backlog_queue: deque | None = None,
) -> tuple[list[Any], int, int]:
    """
    Add new jobs to the queue.

    Args:
        job_queue_2d: 2D job queue array (MAX_QUEUE_SIZE, 4)
        new_jobs_count: Number of new jobs to add
        new_jobs_durations: List of job durations
        new_jobs_nodes: List of nodes required per job
        new_jobs_cores: List of cores per node required per job
        next_empty_slot: Index of next empty slot in queue
        backlog_queue: Optional deque for overflow jobs

    Returns:
        Tuple of (list of added jobs, updated next_empty_slot, num_dropped)
    """
    new_jobs = []
    num_dropped = 0
    for i in range(new_jobs_count):
        # Check if we have space in the queue
        if next_empty_slot >= len(job_queue_2d):
            if backlog_queue is None:
                break  # Queue is full
            if len(backlog_queue) >= MAX_BACKLOG_SIZE:
                num_dropped += 1
                continue  # Backlog full, drop incoming job
            job_entry = [
                new_jobs_durations[i],
                0,  # Age starts at 0
                new_jobs_nodes[i],  # Number of nodes required
                new_jobs_cores[i],  # Cores per node required
            ]
            backlog_queue.append(job_entry)
            new_jobs.append(job_entry)
            continue

        # Add job to the known empty slot
        job_queue_2d[next_empty_slot] = [
            new_jobs_durations[i],
            0,  # Age starts at 0
            new_jobs_nodes[i],  # Number of nodes required
            new_jobs_cores[i]   # Cores per node required
        ]
        new_jobs.append(job_queue_2d[next_empty_slot])

        # Find next empty slot
        next_empty_slot += 1
        while next_empty_slot < len(job_queue_2d) and job_queue_2d[next_empty_slot][0] != 0:
            next_empty_slot += 1

    return new_jobs, next_empty_slot, num_dropped


def assign_jobs_to_available_nodes(
        job_queue_2d: np.ndarray,
        nodes: np.ndarray,
        cores_available: np.ndarray,
        running_jobs: dict[int, dict[str, Any]],
        next_empty_slot: int,
        next_job_id: int,
        metrics: MetricsTracker,
        is_baseline: bool = False,
) -> tuple[int, int, int, int]:
    """
    Assign jobs from queue to available nodes.

    Args:
        job_queue_2d: 2D job queue array (MAX_QUEUE_SIZE, 4)
        nodes: Array of node states
        cores_available: Array of available cores per node
        running_jobs: Dictionary of currently running jobs
        next_empty_slot: Index of next empty slot in queue
        next_job_id: Next available job ID
        metrics: MetricsTracker object to update with completion/wait-time metrics
        is_baseline: Whether this is baseline simulation

    Returns:
        Tuple of (num_processed_jobs, updated next_empty_slot, num_dropped, updated next_job_id).
        Callers are responsible for recording ``num_dropped`` exactly once.
    """
    num_processed_jobs = 0
    num_dropped = 0

    for job_idx, job in enumerate(job_queue_2d):
        job_duration, job_age, job_nodes, job_cores_per_node = job

        if job_duration <= 0:
            continue

        # Candidates: node is on and has enough free cores
        mask = (nodes >= 0) & (cores_available >= job_cores_per_node)
        candidate_nodes = np.where(mask)[0]

        if len(candidate_nodes) >= job_nodes:
            # Assign job to first job_nodes candidates
            job_allocation = []
            for i in range(job_nodes):
                node_idx = candidate_nodes[i]
                cores_available[node_idx] -= job_cores_per_node
                nodes[node_idx] = max(nodes[node_idx], job_duration)
                job_allocation.append((node_idx, job_cores_per_node))

            running_jobs[next_job_id] = {
                "duration": job_duration,
                "allocation": job_allocation,
                "wait_time": int(job_age),
            }
            # Record scheduling delay when the job starts.
            # Completion metrics are tracked separately when the job actually finishes.
            if is_baseline:
                metrics.baseline_jobs_launched += 1
                metrics.baseline_total_job_wait_time_launch += int(job_age)
                metrics.episode_baseline_jobs_launched += 1
                metrics.episode_baseline_total_job_wait_time_launch += int(job_age)
            else:
                metrics.jobs_launched += 1
                metrics.total_job_wait_time_launch += int(job_age)
                metrics.episode_jobs_launched += 1
                metrics.episode_total_job_wait_time_launch += int(job_age)
            next_job_id += 1

            # Clear job from queue
            job_queue_2d[job_idx] = [0, 0, 0, 0]

            # Update next_empty_slot if we cleared a slot before it
            if job_idx < next_empty_slot:
                next_empty_slot = job_idx

            num_processed_jobs += 1
            continue

    return num_processed_jobs, next_empty_slot, num_dropped, next_job_id
