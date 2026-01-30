"""Baseline comparison simulation logic for the PowerSched environment."""

import numpy as np
from src.job_management import (
    process_ongoing_jobs,
    add_new_jobs,
    assign_jobs_to_available_nodes,
    fill_queue_from_backlog,
    age_backlog_queue,
)
from src.reward_calculation import power_cost


def baseline_step(baseline_state, baseline_cores_available, baseline_running_jobs,
                 current_price, new_jobs_count, new_jobs_durations, new_jobs_nodes,
                 new_jobs_cores, baseline_next_empty_slot, next_job_id, metrics, env_print,
                 baseline_backlog_queue):
    """
    Run one step of the baseline simulation for comparison.

    The baseline keeps all nodes online all the time and schedules jobs greedily.

    Args:
        baseline_state: Baseline state dictionary with 'nodes' and 'job_queue'
        baseline_cores_available: Array of available cores per node (baseline)
        baseline_running_jobs: Dictionary of baseline running jobs
        current_price: Current electricity price
        new_jobs_count: Number of new jobs arriving this step
        new_jobs_durations: List of job durations
        new_jobs_nodes: List of nodes required per job
        new_jobs_cores: List of cores per node required per job
        baseline_next_empty_slot: Index of next empty slot in baseline queue
        next_job_id: Next available job ID
        metrics: MetricsTracker object to update with baseline job metrics
        env_print: Print function for logging

    Returns:
        Tuple of (baseline_cost, baseline_cost_off, updated baseline_next_empty_slot, updated next_job_id)
    """
    job_queue_2d = baseline_state['job_queue'].reshape(-1, 4)

    process_ongoing_jobs(baseline_state['nodes'], baseline_cores_available, baseline_running_jobs)

    # Age helper queue and fill real queue before new arrivals
    age_backlog_queue(baseline_backlog_queue, metrics, _is_baseline=True)
    baseline_next_empty_slot, _ = fill_queue_from_backlog(
        job_queue_2d, baseline_backlog_queue, baseline_next_empty_slot
    )

    _new_baseline_jobs, baseline_next_empty_slot = add_new_jobs(
        job_queue_2d, new_jobs_count, new_jobs_durations,
        new_jobs_nodes, new_jobs_cores, baseline_next_empty_slot, baseline_backlog_queue
    )
    metrics.baseline_jobs_submitted += new_jobs_count
    metrics.episode_baseline_jobs_submitted += new_jobs_count

    _, baseline_next_empty_slot, _, next_job_id = assign_jobs_to_available_nodes(
        job_queue_2d, baseline_state['nodes'], baseline_cores_available,
        baseline_running_jobs, baseline_next_empty_slot, next_job_id, metrics, is_baseline=True
    )

    num_used_nodes = np.sum(baseline_state['nodes'] > 0)
    num_on_nodes = np.sum(baseline_state['nodes'] > -1)
    num_idle_nodes = num_on_nodes - num_used_nodes
    num_unprocessed_jobs = np.sum(job_queue_2d[:, 0] > 0)

    # Track baseline max queue size (queue only, without backlog)
    if num_unprocessed_jobs > metrics.baseline_max_queue_size_reached:
        metrics.baseline_max_queue_size_reached = num_unprocessed_jobs
    if num_unprocessed_jobs > metrics.episode_baseline_max_queue_size_reached:
        metrics.episode_baseline_max_queue_size_reached = num_unprocessed_jobs

    # Track baseline max backlog size
    backlog_size = len(baseline_backlog_queue)
    if backlog_size > metrics.baseline_max_backlog_size_reached:
        metrics.baseline_max_backlog_size_reached = backlog_size
    if backlog_size > metrics.episode_baseline_max_backlog_size_reached:
        metrics.episode_baseline_max_backlog_size_reached = backlog_size

    baseline_state['job_queue'] = job_queue_2d.flatten()

    baseline_cost = power_cost(num_used_nodes, num_idle_nodes, current_price)
    env_print(f"    > baseline_cost: €{baseline_cost:.4f} | used nodes: {num_used_nodes}, idle nodes: {num_idle_nodes}")
    baseline_cost_off = power_cost(num_used_nodes, 0, current_price)
    env_print(f"    > baseline_cost_off: €{baseline_cost_off:.4f} | used nodes: {num_used_nodes}, idle nodes: 0")

    return baseline_cost, baseline_cost_off, baseline_next_empty_slot, next_job_id
