import numpy as np

from src.config import CORES_PER_NODE, MAX_NODES
from src.job_management import assign_jobs_to_available_nodes, process_ongoing_jobs
from src.metrics_tracker import MetricsTracker


def _empty_cluster():
    nodes = np.zeros(MAX_NODES, dtype=np.int32)
    cores_available = np.full(MAX_NODES, CORES_PER_NODE, dtype=np.int32)
    running_jobs: dict[int, dict[str, object]] = {}
    return nodes, cores_available, running_jobs


def test_agent_completed_jobs_increment_only_when_job_finishes():
    metrics = MetricsTracker()
    nodes, cores_available, running_jobs = _empty_cluster()
    job_queue_2d = np.array([[1, 3, 1, 4]], dtype=np.int32)

    num_launched, next_empty_slot, num_dropped, next_job_id = assign_jobs_to_available_nodes(
        job_queue_2d,
        nodes,
        cores_available,
        running_jobs,
        next_empty_slot=0,
        next_job_id=0,
        metrics=metrics,
        is_baseline=False,
    )

    assert num_launched == 1
    assert num_dropped == 0
    assert next_empty_slot == 0
    assert next_job_id == 1
    assert metrics.jobs_completed == 0
    assert metrics.episode_jobs_completed == 0
    assert metrics.total_job_wait_time == 0
    assert metrics.episode_total_job_wait_time == 0
    assert running_jobs[0]["wait_time"] == 3

    completed_jobs = process_ongoing_jobs(
        nodes,
        cores_available,
        running_jobs,
        metrics,
        is_baseline=False,
    )

    assert completed_jobs == [0]
    assert not running_jobs
    assert metrics.jobs_completed == 1
    assert metrics.episode_jobs_completed == 1
    assert metrics.total_job_wait_time == 3
    assert metrics.episode_total_job_wait_time == 3


def test_baseline_completed_jobs_increment_only_when_job_finishes():
    metrics = MetricsTracker()
    nodes, cores_available, running_jobs = _empty_cluster()
    job_queue_2d = np.array([[1, 2, 1, 8]], dtype=np.int32)

    num_launched, next_empty_slot, num_dropped, next_job_id = assign_jobs_to_available_nodes(
        job_queue_2d,
        nodes,
        cores_available,
        running_jobs,
        next_empty_slot=0,
        next_job_id=0,
        metrics=metrics,
        is_baseline=True,
    )

    assert num_launched == 1
    assert num_dropped == 0
    assert next_empty_slot == 0
    assert next_job_id == 1
    assert metrics.baseline_jobs_completed == 0
    assert metrics.episode_baseline_jobs_completed == 0
    assert metrics.baseline_total_job_wait_time == 0
    assert metrics.episode_baseline_total_job_wait_time == 0
    assert running_jobs[0]["wait_time"] == 2

    completed_jobs = process_ongoing_jobs(
        nodes,
        cores_available,
        running_jobs,
        metrics,
        is_baseline=True,
    )

    assert completed_jobs == [0]
    assert not running_jobs
    assert metrics.baseline_jobs_completed == 1
    assert metrics.episode_baseline_jobs_completed == 1
    assert metrics.baseline_total_job_wait_time == 2
    assert metrics.episode_baseline_total_job_wait_time == 2
