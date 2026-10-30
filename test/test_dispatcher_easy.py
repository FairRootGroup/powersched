"""Tests for the EASY backfilling dispatcher (job_management, dispatcher="easy").

Run with: python -m test.test_dispatcher_easy
"""

import numpy as np

from src.config import CORES_PER_NODE, MAX_JOB_DURATION, MAX_NODES
from src.job_management import (
    assign_jobs_to_available_nodes, _earliest_start, _free_core_timeline,
)
from src.metrics_tracker import MetricsTracker


def _cluster(online: int):
    """Cluster with the first ``online`` nodes powered and idle, the rest off."""
    nodes = np.full(MAX_NODES, -1, dtype=np.int32)
    nodes[:online] = 0
    cores_available = np.zeros(MAX_NODES, dtype=np.int32)
    cores_available[:online] = CORES_PER_NODE
    return nodes, cores_available, {}


def _add_running(running_jobs, nodes, cores_available, node_ids, cores_used, remaining):
    job_id = len(running_jobs) + 1000
    running_jobs[job_id] = {
        "duration": remaining,
        "allocation": [(n, cores_used) for n in node_ids],
        "wait_time": 0,
    }
    for n in node_ids:
        cores_available[n] -= cores_used
        nodes[n] = max(nodes[n], remaining)


def _dispatch(queue, nodes, cores_available, running_jobs, dispatcher="easy", reservation_depth=1):
    job_queue_2d = np.array(queue, dtype=np.int32)
    num_launched, _, num_dropped, _ = assign_jobs_to_available_nodes(
        job_queue_2d, nodes, cores_available, running_jobs,
        next_empty_slot=len(job_queue_2d), next_job_id=0, metrics=MetricsTracker(),
        dispatcher=dispatcher, reservation_depth=reservation_depth,
    )
    assert num_dropped == 0
    started = [i for i in range(len(job_queue_2d)) if job_queue_2d[i, 0] == 0]
    return started, num_launched


# Queue rows are [duration, age, nodes, cores_per_node].

def test_short_job_backfills_before_reservation():
    nodes, cores, running = _cluster(4)
    _add_running(running, nodes, cores, [0, 1], CORES_PER_NODE, remaining=5)
    # Head needs all 4 nodes -> reserved at hour 5. The 3 h job finishes before that.
    started, _ = _dispatch([[2, 10, 4, 96], [3, 5, 2, 96]], nodes, cores, running)
    assert started == [1]
    assert [n for n, _ in running[0]["allocation"]] == [2, 3]


def test_backfill_that_would_delay_head_is_held():
    queue = [[2, 10, 4, 96], [6, 5, 2, 96]]
    nodes, cores, running = _cluster(4)
    _add_running(running, nodes, cores, [0, 1], CORES_PER_NODE, remaining=5)
    started, _ = _dispatch(queue, nodes, cores, running)
    assert started == []

    # firstfit has no reservations and starts it.
    nodes, cores, running = _cluster(4)
    _add_running(running, nodes, cores, [0, 1], CORES_PER_NODE, remaining=5)
    started, _ = _dispatch(queue, nodes, cores, running, dispatcher="firstfit")
    assert started == [1]


def test_backfill_on_nodes_outside_reservation():
    nodes, cores, running = _cluster(4)
    _add_running(running, nodes, cores, [0, 1], CORES_PER_NODE, remaining=5)
    # Head needs 3 nodes -> reserved on nodes 0-2 at hour 5; node 3 is free for a long job.
    started, _ = _dispatch([[2, 10, 3, 96], [10, 5, 1, 96]], nodes, cores, running)
    assert started == [1]
    assert [n for n, _ in running[0]["allocation"]] == [3]


def test_oldest_job_goes_first_regardless_of_slot():
    queue = [[1, 0, 1, 96], [1, 7, 1, 96]]
    nodes, cores, running = _cluster(1)
    started, _ = _dispatch(queue, nodes, cores, running)
    assert started == [1]

    nodes, cores, running = _cluster(1)
    started, _ = _dispatch(queue, nodes, cores, running, dispatcher="firstfit")
    assert started == [0]


def test_head_that_never_fits_online_nodes_gets_no_reservation():
    nodes, cores, running = _cluster(2)
    # Head needs 4 nodes but only 2 are online: no reservation, so the long job may start.
    started, _ = _dispatch([[1, 10, 4, 96], [50, 1, 2, 96]], nodes, cores, running)
    assert started == [1]


def test_reservation_respects_shared_cores():
    nodes, cores, running = _cluster(1)
    _add_running(running, nodes, cores, [0], 48, remaining=4)
    # Head needs the whole node -> reserved at hour 4. Of two half-node jobs, only the one
    # that finishes by then may use the free 48 cores.
    started, _ = _dispatch([[2, 10, 1, 96], [5, 5, 1, 48], [3, 4, 1, 48]], nodes, cores, running)
    assert started == [2]


def test_jobs_start_without_reservations_when_everything_fits():
    nodes, cores, running = _cluster(3)
    started, launched = _dispatch([[2, 10, 2, 96], [1, 5, 1, 96]], nodes, cores, running)
    assert started == [0, 1]
    assert launched == 2


def test_deeper_reservations_protect_more_jobs():
    def setup():
        nodes, cores, running = _cluster(4)
        _add_running(running, nodes, cores, [0, 1], CORES_PER_NODE, remaining=5)
        _add_running(running, nodes, cores, [2], CORES_PER_NODE, remaining=8)
        return nodes, cores, running

    # A (4 nodes) is reserved at hour 8. B (3 nodes, 3 h) could run in hours 5-8 on nodes 0, 1, 3.
    # C (1 node, 6 h) fits on node 3 now, but would block B's slot.
    queue = [[2, 10, 4, 96], [3, 9, 3, 96], [6, 1, 1, 96]]
    started, _ = _dispatch(queue, *setup(), reservation_depth=1)
    assert started == [2]
    started, _ = _dispatch(queue, *setup(), reservation_depth=2)
    assert started == []


def test_timeline_starts_at_cores_available():
    rng = np.random.default_rng(0)
    nodes, cores, running = _cluster(20)
    for _ in range(30):
        n = int(rng.integers(0, 20))
        used = int(rng.integers(1, 49))
        if cores[n] >= used:
            _add_running(running, nodes, cores, [n], used, remaining=int(rng.integers(1, 50)))
    timeline = _free_core_timeline(nodes, running, horizon=60)
    assert np.array_equal(timeline[0], cores)
    assert np.all(timeline[-1][:20] == CORES_PER_NODE)


def test_easy_never_delays_the_oldest_blocked_job():
    """Randomized check of the EASY guarantee: backfilling never pushes back the head job's earliest start."""
    rng = np.random.default_rng(42)
    horizon = 2 * MAX_JOB_DURATION + 1
    checked = 0
    for _ in range(400):
        online = int(rng.integers(2, 9))
        nodes, cores, running = _cluster(online)
        for _ in range(int(rng.integers(0, 3 * online))):
            n = int(rng.integers(0, online))
            used = int(rng.choice([24, 48, 96]))
            if cores[n] >= used:
                _add_running(running, nodes, cores, [n], used, remaining=int(rng.integers(1, 30)))

        queue = np.column_stack([
            rng.integers(1, 30, size=8),            # duration
            rng.permutation(8) * 3,                 # distinct ages
            rng.integers(1, online + 1, size=8),    # nodes
            rng.choice([24, 48, 96], size=8),       # cores per node
        ]).astype(np.int32)
        head = int(np.argmax(queue[:, 1]))
        h_dur, _, h_nodes, h_cores = (int(v) for v in queue[head])

        before = _earliest_start(_free_core_timeline(nodes, running, horizon), h_dur, h_nodes, h_cores, monotone=True)
        if before is None or before == 0:
            continue  # head starts right away, nothing to protect

        started, _ = _dispatch(queue, nodes, cores, running)
        assert head not in started
        assert np.all(cores >= 0)
        after = _earliest_start(_free_core_timeline(nodes, running, horizon), h_dur, h_nodes, h_cores, monotone=True)
        assert after is not None and after <= before, (before, after)
        checked += 1
    assert checked > 50, checked


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASSED: {name}")
    print(f"All {len(tests)} dispatcher tests passed.")


if __name__ == "__main__":
    main()
