"""Tests for JobQueue and for dispatch_jobs against the previous per-slot dispatcher loops.

Run with: python -m test.test_job_queue
"""

import copy

import numpy as np

from src.config import CORES_PER_NODE, MAX_JOB_DURATION, MAX_NODES
from src.job_management import (
    JobQueue, _earliest_start, _free_core_timeline, dispatch_jobs, process_ongoing_jobs
)
from src.metrics_tracker import MetricsTracker


# --- Reference: the dispatchers as they were before the queue and backlog were merged ---

def _reference_start(queue_2d, job_idx, chosen_nodes, nodes, cores_available, running_jobs, job_id):
    duration, age, _n, cores = (int(v) for v in queue_2d[job_idx])
    allocation = []
    for node_idx in chosen_nodes:
        cores_available[node_idx] -= cores
        nodes[node_idx] = max(nodes[node_idx], duration)
        allocation.append((node_idx, cores))
    running_jobs[job_id] = {"duration": duration, "allocation": allocation, "wait_time": age}
    queue_2d[job_idx] = 0


def _reference_firstfit(queue_2d, nodes, cores_available, running_jobs, next_job_id):
    for job_idx in range(len(queue_2d)):
        duration, _age, job_nodes, cores = queue_2d[job_idx]
        if duration <= 0:
            continue
        candidates = np.flatnonzero((nodes >= 0) & (cores_available >= cores))
        if len(candidates) >= job_nodes:
            _reference_start(queue_2d, job_idx, candidates[:job_nodes], nodes, cores_available,
                             running_jobs, next_job_id)
            next_job_id += 1
    return next_job_id


def _reference_easy(queue_2d, nodes, cores_available, running_jobs, next_job_id, reservation_depth):
    waiting = np.flatnonzero(queue_2d[:, 0] > 0)
    order = waiting[np.argsort(-queue_2d[waiting, 1], kind="stable")]
    timeline = None
    reservation_starts = []
    for job_idx in order:
        duration, _age, job_nodes, cores = (int(v) for v in queue_2d[job_idx])
        if reservation_starts:
            checkpoints = [0] + [t for t in reservation_starts if t < duration]
            free = timeline[checkpoints].min(axis=0)
        else:
            free = cores_available
        candidates = np.flatnonzero((nodes >= 0) & (free >= cores))
        if len(candidates) >= job_nodes:
            chosen = candidates[:job_nodes]
            _reference_start(queue_2d, job_idx, chosen, nodes, cores_available, running_jobs, next_job_id)
            if timeline is not None:
                timeline[:duration, chosen] -= cores
            next_job_id += 1
            continue
        if len(reservation_starts) >= reservation_depth:
            continue
        if timeline is None:
            max_running = max((int(j['duration']) for j in running_jobs.values()), default=0)
            timeline = _free_core_timeline(nodes, running_jobs,
                                           max_running + (reservation_depth + 1) * MAX_JOB_DURATION)
        start = _earliest_start(timeline, duration, job_nodes, cores, monotone=not reservation_starts)
        if start is None:
            continue
        window_free = timeline[start:start + duration].min(axis=0)
        reserved = np.flatnonzero(window_free >= cores)[:job_nodes]
        timeline[start:start + duration, reserved] -= cores
        reservation_starts.append(start)
    return next_job_id


# --- Random scenarios ---

def _random_jobs(rng, count, max_nodes=16):
    return (rng.integers(1, 40, count), rng.integers(1, max_nodes + 1, count),
            rng.choice([1, 2, 8, 24, 48, 96], count))


def _random_cluster(rng):
    """A cluster with some nodes off and some jobs already running."""
    nodes = np.zeros(MAX_NODES, dtype=np.int32)
    nodes[rng.choice(MAX_NODES, size=int(rng.integers(0, MAX_NODES // 2)), replace=False)] = -1
    cores_available = np.full(MAX_NODES, CORES_PER_NODE, dtype=np.int32)
    running_jobs = {}
    warmup = JobQueue()
    warmup.add(*_random_jobs(rng, int(rng.integers(0, 400))))
    _, next_job_id = dispatch_jobs(warmup, nodes, cores_available, running_jobs, 0, MetricsTracker())
    for _ in range(int(rng.integers(0, 20))):  # let some jobs progress or finish
        process_ongoing_jobs(nodes, cores_available, running_jobs)
    return nodes, cores_available, running_jobs, next_job_id


def _random_queue(rng, count):
    """Queue rows in arrival order: ages never increase along the queue."""
    durations, job_nodes, cores = _random_jobs(rng, count)
    ages = np.sort(rng.integers(0, 300, count))[::-1]
    return np.column_stack([durations, ages, job_nodes, cores]).astype(np.int32)


def _compare(dispatcher, reservation_depth, trials, seed):
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        nodes, cores_available, running_jobs, next_job_id = _random_cluster(rng)
        rows = _random_queue(rng, int(rng.integers(0, 3000)))

        ref = (nodes.copy(), cores_available.copy(), copy.deepcopy(running_jobs))
        ref_queue = np.zeros((len(rows) + 5, 4), dtype=np.int32)
        ref_queue[:len(rows)] = rows
        if dispatcher == "easy":
            ref_next = _reference_easy(ref_queue, *ref, next_job_id, reservation_depth)
        else:
            ref_next = _reference_firstfit(ref_queue, *ref, next_job_id)

        queue = JobQueue()
        queue._buf[:len(rows)] = rows
        queue._n = len(rows)
        started, new_next = dispatch_jobs(queue, nodes, cores_available, running_jobs, next_job_id,
                                          MetricsTracker(), dispatcher=dispatcher,
                                          reservation_depth=reservation_depth)

        assert new_next == ref_next and started == ref_next - next_job_id
        assert np.array_equal(nodes, ref[0]) and np.array_equal(cores_available, ref[1])
        assert running_jobs == ref[2]
        assert np.array_equal(queue.rows(), ref_queue[ref_queue[:, 0] > 0])


def test_firstfit_matches_reference():
    _compare("firstfit", 1, trials=60, seed=1)


def test_easy_matches_reference():
    for depth in (1, 2, 3):
        _compare("easy", depth, trials=40, seed=10 + depth)


def test_queue_add_rejects_when_full():
    queue = JobQueue(capacity=5)
    assert queue.add([1, 2, 3], [1, 1, 1], [1, 1, 1]) == (3, 0)
    assert queue.add([4, 5, 6], [1, 1, 1], [1, 1, 1]) == (2, 1)
    assert [int(d) for d in queue.rows()[:, 0]] == [1, 2, 3, 4, 5]
    assert all(int(a) == 0 for a in queue.rows()[:, 1])


def test_queue_age_drops_oldest_and_keeps_order():
    queue = JobQueue()
    for duration in range(1, 6):
        queue.age(10)
        queue.add([duration], [1], [1])
    # Ages are now 4, 3, 2, 1, 0 for durations 1..5; two more steps expire only duration 1 (age 6).
    assert queue.age(5) == 0
    assert queue.age(5) == 1
    assert [int(d) for d in queue.rows()[:, 0]] == [2, 3, 4, 5]
    assert [int(a) for a in queue.rows()[:, 1]] == [5, 4, 3, 2]


def test_queue_remove_and_clear():
    queue = JobQueue()
    queue.add([1, 2, 3, 4], [1, 1, 1, 1], [1, 1, 1, 1])
    queue.remove([0, 2])
    assert [int(d) for d in queue.rows()[:, 0]] == [2, 4]
    queue.clear()
    assert not queue and len(queue.rows()) == 0


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASSED: {name}")
    print(f"All {len(tests)} job queue tests passed.")


if __name__ == "__main__":
    main()
