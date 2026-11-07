"""Baseline comparison simulation logic for the PowerSched environment."""

from typing import Any, Callable

import numpy as np
from src.job_management import JobQueue, process_ongoing_jobs, dispatch_jobs
from src.metrics_tracker import MetricsTracker
from src.reward_calculation import power_consumption_mwh
from src.config import CORES_PER_NODE, MAX_JOB_AGE


def baseline_step(
        baseline_state: dict[str, np.ndarray],
        baseline_cores_available: np.ndarray,
        baseline_running_jobs: dict[int, dict[str, Any]],
        current_price: float,
        new_jobs_durations: list[int],
        new_jobs_nodes: list[int],
        new_jobs_cores: list[int],
        next_job_id: int,
        metrics: MetricsTracker,
        env_print: Callable[..., None],
        baseline_queue: JobQueue,
        dispatcher: str = "firstfit",
        reservation_depth: int = 1,
) -> tuple[float, float, float, float, int, int, int]:
    """
    Run one step of the baseline simulation for comparison.

    The baseline keeps all nodes online all the time and schedules jobs greedily.

    Args:
        baseline_state: Baseline state dictionary with 'nodes'
        baseline_cores_available: Array of available cores per node (baseline)
        baseline_running_jobs: Dictionary of baseline running jobs
        current_price: Current electricity price
        new_jobs_durations: List of job durations
        new_jobs_nodes: List of nodes required per job
        new_jobs_cores: List of cores per node required per job
        next_job_id: Next available job ID
        metrics: MetricsTracker object to update with baseline job metrics
        env_print: Print function for logging
        baseline_queue: The baseline's waiting jobs
        dispatcher: Job dispatcher, same as the agent's (see job_management.DISPATCHERS)
        reservation_depth: Number of reservations for the "easy" dispatcher

    Returns:
        Tuple of (
            baseline_cost, baseline_cost_off,
            baseline_power_mwh, baseline_power_off_mwh,
            updated next_job_id,
            baseline_num_used_nodes, baseline_num_used_cores
        )
    """
    process_ongoing_jobs(
        baseline_state['nodes'],
        baseline_cores_available,
        baseline_running_jobs,
        metrics,
        is_baseline=True,
    )

    # Age jobs already waiting before admitting this step's new arrivals.
    aged_dropped = baseline_queue.age(MAX_JOB_AGE)

    _added, rejected_queue_full = baseline_queue.add(new_jobs_durations, new_jobs_nodes, new_jobs_cores)
    metrics.baseline_jobs_submitted += len(new_jobs_durations)
    metrics.episode_baseline_jobs_submitted += len(new_jobs_durations)
    if rejected_queue_full > 0:
        metrics.baseline_jobs_rejected_queue_full += rejected_queue_full
        metrics.episode_baseline_jobs_rejected_queue_full += rejected_queue_full

    _num_launched, next_job_id = dispatch_jobs(
        baseline_queue, baseline_state['nodes'], baseline_cores_available,
        baseline_running_jobs, next_job_id, metrics, is_baseline=True,
        dispatcher=dispatcher, reservation_depth=reservation_depth
    )

    total_dropped = aged_dropped + rejected_queue_full
    if total_dropped > 0:
        metrics.baseline_jobs_dropped += total_dropped
        metrics.episode_baseline_jobs_dropped += total_dropped

    num_used_nodes = np.sum(baseline_state['nodes'] > 0)
    num_on_nodes = np.sum(baseline_state['nodes'] > -1)
    num_idle_nodes = num_on_nodes - num_used_nodes
    num_unprocessed_jobs = len(baseline_queue)

    # Track baseline max queue size
    if num_unprocessed_jobs > metrics.baseline_max_queue_size_reached:
        metrics.baseline_max_queue_size_reached = num_unprocessed_jobs
    if num_unprocessed_jobs > metrics.episode_baseline_max_queue_size_reached:
        metrics.episode_baseline_max_queue_size_reached = num_unprocessed_jobs

    num_used_cores = int(num_on_nodes * CORES_PER_NODE - np.sum(baseline_cores_available))
    baseline_power_mwh = power_consumption_mwh(num_on_nodes, num_used_cores)
    baseline_power_off_mwh = power_consumption_mwh(num_used_nodes, num_used_cores)
    baseline_cost = baseline_power_mwh * current_price
    baseline_cost_off = baseline_power_off_mwh * current_price
    env_print(f"    > baseline_cost: €{baseline_cost:.4f} | used nodes: {num_used_nodes}, idle nodes: {num_idle_nodes}")
    env_print(f"    > baseline_cost_off: €{baseline_cost_off:.4f} | used nodes: {num_used_nodes}, idle nodes: 0")

    return (
        baseline_cost,
        baseline_cost_off,
        baseline_power_mwh,
        baseline_power_off_mwh,
        next_job_id,
        num_used_nodes,
        num_used_cores,
    )
