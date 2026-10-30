"""Job queue management and scheduling logic for the PowerSched environment."""

from collections import deque
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from src.config import (
    MAX_NODES, CORES_PER_NODE, MAX_BACKLOG_SIZE, MAX_JOB_AGE, MAX_JOB_DURATION
)
from src.metrics_tracker import MetricsTracker

# firstfit: scan queue slots in order and start every job that fits now.
# easy:     oldest job first, with EASY backfilling (reservations for the oldest blocked jobs).
DISPATCHERS: tuple[str, ...] = ("firstfit", "easy")


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


def _start_job(
        job_queue_2d: np.ndarray,
        job_idx: int,
        chosen_nodes: np.ndarray,
        nodes: np.ndarray,
        cores_available: np.ndarray,
        running_jobs: dict[int, dict[str, Any]],
        job_id: int,
        metrics: MetricsTracker,
        is_baseline: bool,
) -> None:
    """Start the queued job at ``job_idx`` on ``chosen_nodes`` and clear its queue slot."""
    job_duration, job_age, _job_nodes, job_cores_per_node = job_queue_2d[job_idx]

    job_allocation = []
    for node_idx in chosen_nodes:
        cores_available[node_idx] -= job_cores_per_node
        nodes[node_idx] = max(nodes[node_idx], job_duration)
        job_allocation.append((node_idx, job_cores_per_node))

    running_jobs[job_id] = {
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

    # Clear job from queue
    job_queue_2d[job_idx] = [0, 0, 0, 0]


def _free_core_timeline(
        nodes: np.ndarray,
        running_jobs: dict[int, dict[str, Any]],
        horizon: int,
) -> np.ndarray:
    """
    Free cores per (hour from now, node) over ``horizon`` hours, from the running jobs.

    A running job with ``duration`` d holds its cores for hours [0, d). Offline nodes have no free cores.
    """
    change = np.zeros((horizon + 1, len(nodes)), dtype=np.int32)
    change[0] = np.where(nodes >= 0, CORES_PER_NODE, 0)
    for job_data in running_jobs.values():
        release = int(job_data['duration'])
        for node_idx, cores_used in job_data['allocation']:
            change[0, node_idx] -= cores_used
            change[release, node_idx] += cores_used
    return np.cumsum(change[:-1], axis=0)


def _earliest_start(
        timeline: np.ndarray,
        job_duration: int,
        job_nodes: int,
        job_cores_per_node: int,
        monotone: bool,
) -> int | None:
    """
    Earliest hour at which ``job_nodes`` nodes each keep ``job_cores_per_node`` free cores
    for ``job_duration`` hours, or None if that never happens within the timeline.
    """
    if monotone:
        # Without reservations free cores never drop, so a window's minimum is its first hour.
        window_min = timeline[:len(timeline) - job_duration + 1]
    else:
        window_min = sliding_window_view(timeline, job_duration, axis=0).min(axis=-1)
    fits = np.count_nonzero(window_min >= job_cores_per_node, axis=1) >= job_nodes
    starts = np.flatnonzero(fits)
    return int(starts[0]) if len(starts) else None


def _assign_jobs_easy(
        job_queue_2d: np.ndarray,
        nodes: np.ndarray,
        cores_available: np.ndarray,
        running_jobs: dict[int, dict[str, Any]],
        next_empty_slot: int,
        next_job_id: int,
        metrics: MetricsTracker,
        is_baseline: bool,
        reservation_depth: int,
) -> tuple[int, int, int, int]:
    """
    EASY backfilling over the visible queue, oldest job first.

    A job starts if it fits now without delaying a reservation made for an older job. The first
    ``reservation_depth`` jobs that cannot start get a reservation at their earliest possible start
    (1 = EASY; larger values approach conservative backfilling). Reservations are planned on the online
    nodes only: a job that cannot fit on them at any time gets no reservation and does not hold back
    younger jobs, since providing enough online capacity is the capacity controller's job.
    Reservations are recomputed on every call, i.e. at every scheduling pass.
    """
    num_processed_jobs = 0

    waiting = np.flatnonzero(job_queue_2d[:, 0] > 0)
    # Oldest first; jobs of equal age keep their slot order.
    order = waiting[np.argsort(-job_queue_2d[waiting, 1], kind="stable")]

    # Built on the first reservation; from then on it also tracks jobs started in this pass.
    timeline = None
    reservation_starts: list[int] = []

    for job_idx in order:
        job_duration, _job_age, job_nodes, job_cores_per_node = (int(v) for v in job_queue_2d[job_idx])

        if reservation_starts:
            # Free cores only drop where a reservation begins, so checking those hours
            # (and now) gives each node's minimum over the job's runtime.
            checkpoints = [0] + [t for t in reservation_starts if t < job_duration]
            free_cores = timeline[checkpoints].min(axis=0)
        else:
            free_cores = cores_available

        candidate_nodes = np.flatnonzero((nodes >= 0) & (free_cores >= job_cores_per_node))
        if len(candidate_nodes) >= job_nodes:
            chosen_nodes = candidate_nodes[:job_nodes]
            _start_job(job_queue_2d, job_idx, chosen_nodes, nodes, cores_available,
                       running_jobs, next_job_id, metrics, is_baseline)
            if timeline is not None:
                timeline[:job_duration, chosen_nodes] -= job_cores_per_node
            next_job_id += 1
            num_processed_jobs += 1
            if job_idx < next_empty_slot:
                next_empty_slot = job_idx
            continue

        if len(reservation_starts) >= reservation_depth:
            continue

        if timeline is None:
            # Each reservation starts once running jobs and earlier reservations end at the latest.
            max_running = max((int(j['duration']) for j in running_jobs.values()), default=0)
            timeline = _free_core_timeline(nodes, running_jobs,
                                           max_running + (reservation_depth + 1) * MAX_JOB_DURATION)

        start = _earliest_start(timeline, job_duration, job_nodes, job_cores_per_node,
                                monotone=not reservation_starts)
        if start is None:
            continue  # never fits on the online nodes: no reservation

        window_free = timeline[start:start + job_duration].min(axis=0)
        reserved_nodes = np.flatnonzero(window_free >= job_cores_per_node)[:job_nodes]
        timeline[start:start + job_duration, reserved_nodes] -= job_cores_per_node
        reservation_starts.append(start)

    return num_processed_jobs, next_empty_slot, 0, next_job_id


def assign_jobs_to_available_nodes(
        job_queue_2d: np.ndarray,
        nodes: np.ndarray,
        cores_available: np.ndarray,
        running_jobs: dict[int, dict[str, Any]],
        next_empty_slot: int,
        next_job_id: int,
        metrics: MetricsTracker,
        is_baseline: bool = False,
        dispatcher: str = "firstfit",
        reservation_depth: int = 1,
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
        dispatcher: One of DISPATCHERS
        reservation_depth: Number of reservations for the "easy" dispatcher (1 = EASY)

    Returns:
        Tuple of (num_processed_jobs, updated next_empty_slot, num_dropped, updated next_job_id).
        Callers are responsible for recording ``num_dropped`` exactly once.
    """
    if dispatcher == "easy":
        return _assign_jobs_easy(job_queue_2d, nodes, cores_available, running_jobs, next_empty_slot,
                                 next_job_id, metrics, is_baseline, reservation_depth)
    if dispatcher != "firstfit":
        raise ValueError(f"Unknown dispatcher {dispatcher!r}, expected one of {DISPATCHERS}")

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
            _start_job(job_queue_2d, job_idx, candidate_nodes[:job_nodes], nodes, cores_available,
                       running_jobs, next_job_id, metrics, is_baseline)
            next_job_id += 1

            # Update next_empty_slot if we cleared a slot before it
            if job_idx < next_empty_slot:
                next_empty_slot = job_idx

            num_processed_jobs += 1
            continue

    return num_processed_jobs, next_empty_slot, num_dropped, next_job_id
