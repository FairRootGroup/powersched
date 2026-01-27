"""Job queue management and scheduling logic for the PowerSched environment."""

import numpy as np
from src.config import (
    MAX_NODES, MAX_JOB_AGE, CORES_PER_NODE
)


def age_backlog_queue(backlog_queue, _metrics, _is_baseline=False):
    """
    Age jobs waiting in the backlog queue.
    NOTE: dropping based on MAX_JOB_AGE is temporarily disabled via an `if False` hotfix,
    so jobs are always kept even if job[1] > MAX_JOB_AGE.
    TODO: re-enable drops by removing the `if False` guard and using `job[1] > MAX_JOB_AGE`;
    metrics updates are already in the disabled branch.
    """
    if not backlog_queue:
        return 0

    dropped = 0
    kept = []
    for job in backlog_queue:
        job[1] += 1
        # TEMP HOTFIX: disable age-based dropping (keep logic for later).
        # if False and job[1] > MAX_JOB_AGE:
        #     dropped += 1
        #     if _is_baseline:
        #         _metrics.baseline_jobs_dropped += 1
        #         _metrics.baseline_dropped_this_episode += 1
        #         _metrics.episode_baseline_jobs_dropped += 1
        #     else:
        #         _metrics.jobs_dropped += 1
        #         _metrics.dropped_this_episode += 1
        #         _metrics.episode_jobs_dropped += 1
        # else:
        kept.append(job)

    backlog_queue.clear()
    backlog_queue.extend(kept)
    return dropped


def fill_queue_from_backlog(job_queue_2d, backlog_queue, next_empty_slot):
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


def validate_next_empty(job_queue_2d, next_empty):
    """Validator for debugging queue consistency."""
    n = len(job_queue_2d)
    if next_empty < n:
        assert job_queue_2d[next_empty][0] == 0, "next_empty_slot not empty"
    # everything before must be non-empty
    if next_empty > 0:
        assert np.all(job_queue_2d[:next_empty, 0] != 0), "hole before next_empty_slot"


def process_ongoing_jobs(nodes, cores_available, running_jobs):
    """
    Process ongoing jobs: decrement their duration, complete finished jobs,
    and release resources.

    Args:
        nodes: Array of node states
        cores_available: Array of available cores per node
        running_jobs: Dictionary of currently running jobs

    Returns:
        List of completed job IDs
    """
    completed_jobs = []

    for job_id, job_data in running_jobs.items():
        job_data['duration'] -= 1

        # Check if job is completed
        if job_data['duration'] <= 0:
            completed_jobs.append(job_id)
            # Release resources
            for node_idx, cores_used in job_data['allocation']:
                cores_available[node_idx] += cores_used

    # Remove completed jobs
    for job_id in completed_jobs:
        del running_jobs[job_id]

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


def add_new_jobs(job_queue_2d, new_jobs_count, new_jobs_durations, new_jobs_nodes,
                 new_jobs_cores, next_empty_slot, backlog_queue=None):
    """
    Add new jobs to the queue.

    Args:
        job_queue_2d: 2D job queue array (MAX_QUEUE_SIZE, 4)
        new_jobs_count: Number of new jobs to add
        new_jobs_durations: List of job durations
        new_jobs_nodes: List of nodes required per job
        new_jobs_cores: List of cores per node required per job
        next_empty_slot: Index of next empty slot in queue

    Returns:
        Tuple of (list of added jobs (real queue + backlog queue), updated next_empty_slot)
    """
    new_jobs = []
    for i in range(new_jobs_count):
        # Check if we have space in the queue
        if next_empty_slot >= len(job_queue_2d):
            if backlog_queue is None:
                break  # Queue is full
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

    return new_jobs, next_empty_slot


def assign_jobs_to_available_nodes(job_queue_2d, nodes, cores_available, running_jobs,
                                   next_empty_slot, next_job_id, metrics, is_baseline=False):
    """
    Assign jobs from queue to available nodes.

    Args:
        job_queue_2d: 2D job queue array (MAX_QUEUE_SIZE, 4)
        nodes: Array of node states
        cores_available: Array of available cores per node
        running_jobs: Dictionary of currently running jobs
        next_empty_slot: Index of next empty slot in queue
        next_job_id: Next available job ID
        metrics: MetricsTracker object to update with job completion metrics
        is_baseline: Whether this is baseline simulation

    Returns:
        Tuple of (num_processed_jobs, updated next_empty_slot, num_dropped, updated next_job_id)
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
            }
            next_job_id += 1

            # Clear job from queue
            job_queue_2d[job_idx] = [0, 0, 0, 0]

            # Update next_empty_slot if we cleared a slot before it
            if job_idx < next_empty_slot:
                next_empty_slot = job_idx

            # Track job completion and wait time
            if is_baseline:
                metrics.baseline_jobs_completed += 1
                metrics.baseline_total_job_wait_time += job_age
                metrics.episode_baseline_jobs_completed += 1
                metrics.episode_baseline_total_job_wait_time += job_age
            else:
                metrics.jobs_completed += 1
                metrics.total_job_wait_time += job_age
                metrics.episode_jobs_completed += 1
                metrics.episode_total_job_wait_time += job_age

            num_processed_jobs += 1
            continue

        # Not enough resources -> job waits and ages (or gets dropped)
        new_age = job_age + 1

        if new_age > MAX_JOB_AGE:
            # Clear job from queue
            job_queue_2d[job_idx] = [0, 0, 0, 0]

            # Update next_empty_slot if we cleared a slot before it
            if job_idx < next_empty_slot:
                next_empty_slot = job_idx
            num_dropped += 1

            if is_baseline:
                metrics.baseline_jobs_dropped += 1
                metrics.episode_baseline_jobs_dropped += 1
            else:
                metrics.jobs_dropped += 1
                metrics.episode_jobs_dropped += 1
        else:
            job_queue_2d[job_idx][1] = new_age

    return num_processed_jobs, next_empty_slot, num_dropped, next_job_id
