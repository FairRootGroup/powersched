"""Job queue management and scheduling logic for the PowerSched environment."""

from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from src.config import (
    MAX_NODES, CORES_PER_NODE, MAX_QUEUE_SIZE, MAX_JOB_DURATION
)
from src.metrics_tracker import MetricsTracker

# firstfit: go through the queue in arrival order and start every job that fits now.
# easy:     oldest job first, with EASY backfilling (reservations for the oldest blocked jobs).
DISPATCHERS: tuple[str, ...] = ("firstfit", "easy")

# How many queued jobs the dispatchers test for a fit in one vectorized batch.
_FIT_CHUNK = 512


class JobQueue:
    """
    All waiting jobs as [duration, age, nodes, cores_per_node] rows, in arrival order.

    Every job ages by one hour per step, so arrival order is also oldest-first order, and the
    jobs that exceed the maximum age are always a prefix of the queue.
    """

    def __init__(self, capacity: int = MAX_QUEUE_SIZE) -> None:
        self.capacity = capacity
        self._buf = np.zeros((capacity, 4), dtype=np.int32)
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def __bool__(self) -> bool:
        return self._n > 0

    def rows(self) -> np.ndarray:
        """View of the waiting jobs, oldest first."""
        return self._buf[:self._n]

    def add(self, durations, nodes, cores) -> tuple[int, int]:
        """Append new jobs (age 0) while there is room. Returns (added, rejected because the queue is full)."""
        count = len(durations)
        added = min(count, self.capacity - self._n)
        if added:
            new = self._buf[self._n:self._n + added]
            new[:, 0] = durations[:added]
            new[:, 1] = 0
            new[:, 2] = nodes[:added]
            new[:, 3] = cores[:added]
            self._n += added
        return added, count - added

    def age(self, max_age: int) -> int:
        """Add one hour to every job's age and drop those older than ``max_age``. Returns the drop count."""
        rows = self.rows()
        rows[:, 1] += 1
        # Ages never increase along the queue, so the expired jobs are the leading ones.
        dropped = int(np.searchsorted(-rows[:, 1], -max_age, side="left"))
        if dropped:
            self._remove_prefix(dropped)
        return dropped

    def remove(self, indices) -> None:
        """Remove the jobs at ``indices`` (positions in ``rows()``), keeping the order of the rest."""
        if len(indices) == 0:
            return
        keep = np.ones(self._n, dtype=bool)
        keep[indices] = False
        kept = self.rows()[keep]
        self._n = len(kept)
        self._buf[:self._n] = kept

    def clear(self) -> None:
        self._n = 0

    def _remove_prefix(self, count: int) -> None:
        remaining = self._n - count
        self._buf[:remaining] = self._buf[count:self._n]
        self._n = remaining


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


def _start_job(
        job: np.ndarray,
        chosen_nodes: np.ndarray,
        nodes: np.ndarray,
        cores_available: np.ndarray,
        running_jobs: dict[int, dict[str, Any]],
        job_id: int,
        metrics: MetricsTracker,
        is_baseline: bool,
) -> None:
    """Start ``job`` (a queue row) on ``chosen_nodes``. The caller removes it from the queue."""
    job_duration, job_age, _job_nodes, job_cores_per_node = (int(v) for v in job)

    job_allocation = []
    for node_idx in chosen_nodes:
        cores_available[node_idx] -= job_cores_per_node
        nodes[node_idx] = max(nodes[node_idx], job_duration)
        job_allocation.append((node_idx, job_cores_per_node))

    running_jobs[job_id] = {
        "duration": job_duration,
        "allocation": job_allocation,
        "wait_time": job_age,
    }
    # Record scheduling delay when the job starts.
    # Completion metrics are tracked separately when the job actually finishes.
    if is_baseline:
        metrics.baseline_jobs_launched += 1
        metrics.baseline_total_job_wait_time_launch += job_age
        metrics.episode_baseline_jobs_launched += 1
        metrics.episode_baseline_total_job_wait_time_launch += job_age
    else:
        metrics.jobs_launched += 1
        metrics.total_job_wait_time_launch += job_age
        metrics.episode_jobs_launched += 1
        metrics.episode_total_job_wait_time_launch += job_age


def _nodes_with_free_cores(free_cores: np.ndarray) -> np.ndarray:
    """
    ``counts[c]`` = number of nodes with at least ``c`` free cores, for c in 0..CORES_PER_NODE + 1.

    ``free_cores`` is -1 for offline nodes. The last entry is always 0, so job core requests
    above CORES_PER_NODE can be clipped to it and never fit.
    """
    per_value = np.bincount(free_cores[free_cores >= 0], minlength=CORES_PER_NODE + 1)
    counts = np.zeros(CORES_PER_NODE + 2, dtype=np.int64)
    counts[:-1] = np.cumsum(per_value[::-1])[::-1]
    return counts


class _Capacity:
    """
    Free cores a waiting job may use, per class of job durations, with fit counts kept up to date.

    Without reservations there is one class: the free cores now. With reservations, free cores
    only drop where a reservation begins, so for a job whose runtime spans the first k reservation
    starts, the usable free cores are the minimum over now and those k hours (class k).
    """

    def __init__(self, online: np.ndarray, cores_available: np.ndarray,
                 timeline: np.ndarray | None, reservation_starts: list[int]) -> None:
        self.online = online
        self.cores_available = cores_available
        self.timeline = timeline
        self.starts = sorted(reservation_starts)
        self.free = np.empty((len(self.starts) + 1, len(online)), dtype=np.int64)
        self.free[0] = np.where(online, cores_available, -1)
        for k, t in enumerate(self.starts, start=1):
            self.free[k] = np.where(online, np.minimum(self.free[k - 1], timeline[t]), -1)
        self.counts = np.stack([_nodes_with_free_cores(f) for f in self.free])

    def job_classes(self, durations: np.ndarray) -> np.ndarray:
        if not self.starts:
            return np.zeros(len(durations), dtype=np.intp)
        return np.searchsorted(self.starts, durations, side="left")

    def take(self, chosen_nodes: np.ndarray) -> None:
        """Update free cores and counts on ``chosen_nodes`` after a job started there."""
        for k in range(len(self.free)):
            old = self.free[k, chosen_nodes]
            if k == 0:
                new = self.cores_available[chosen_nodes].astype(np.int64)
            else:
                new = np.minimum(self.free[k - 1, chosen_nodes], self.timeline[self.starts[k - 1], chosen_nodes])
            self.free[k, chosen_nodes] = new
            counts = self.counts[k]
            for o, n in zip(old.tolist(), new.tolist()):
                if n < o:
                    counts[n + 1:o + 1] -= 1


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


def dispatch_jobs(
        queue: JobQueue,
        nodes: np.ndarray,
        cores_available: np.ndarray,
        running_jobs: dict[int, dict[str, Any]],
        next_job_id: int,
        metrics: MetricsTracker,
        is_baseline: bool = False,
        dispatcher: str = "firstfit",
        reservation_depth: int = 1,
) -> tuple[int, int]:
    """
    Start waiting jobs on the online nodes and remove them from ``queue``.

    firstfit goes through the queue in arrival order and starts every job that fits now.

    easy goes oldest job first (the same order) with EASY backfilling. A job starts if it fits
    now without delaying a reservation made for an older job. The first ``reservation_depth``
    jobs that cannot start get a reservation at their earliest possible start (1 = EASY; larger
    values approach conservative backfilling). Reservations are planned on the online nodes only:
    a job that cannot fit on them at any time gets no reservation and does not hold back younger
    jobs, since providing enough online capacity is the capacity controller's job. Reservations
    are recomputed on every call, i.e. at every scheduling pass.

    In both, a starting job takes the lowest-numbered online nodes with enough free cores.
    Fits are tested for a whole batch of queued jobs at once and the fit counts are updated
    incrementally as jobs start, so the per-job Python work is only done for jobs that start
    (or get a reservation), not for every waiting job.

    Returns:
        Tuple of (number of jobs started, updated next_job_id).
    """
    if dispatcher not in DISPATCHERS:
        raise ValueError(f"Unknown dispatcher {dispatcher!r}, expected one of {DISPATCHERS}")
    rows = queue.rows()
    if len(rows) == 0:
        return 0, next_job_id

    durations = rows[:, 0]
    job_nodes = rows[:, 2]
    # Requests above CORES_PER_NODE index the always-zero last count, so they never fit.
    job_cores = np.minimum(rows[:, 3], CORES_PER_NODE + 1)
    online = nodes >= 0
    num_online = int(np.count_nonzero(online))
    reservations_left = reservation_depth if dispatcher == "easy" else 0
    # Built on the first reservation; from then on it also tracks jobs started in this pass.
    timeline = None
    reservation_starts: list[int] = []
    capacity = _Capacity(online, cores_available, timeline, reservation_starts)

    started: list[int] = []
    pos = 0
    while pos < len(rows):
        end = min(len(rows), pos + _FIT_CHUNK)
        job_class = capacity.job_classes(durations[pos:end])
        fits = capacity.counts[job_class, job_cores[pos:end]] >= job_nodes[pos:end]
        if reservations_left:
            # A job that does not fit now gets a reservation unless it can never fit on the
            # online nodes (each online node eventually frees all its cores).
            candidates = np.flatnonzero(fits | (job_nodes[pos:end] <= num_online))
        else:
            candidates = np.flatnonzero(fits)
        if len(candidates) == 0:
            if capacity.counts[0, 1] == 0 and not reservations_left:
                break  # no free cores on any online node: nothing else can start
            pos = end
            continue

        # Free cores only drop while jobs start, so a job that did not fit above cannot fit later
        # in this batch; the candidates are re-checked one by one as capacity is used up.
        next_pos = end
        for h in candidates.tolist():
            i = pos + h
            duration, cores, needed = int(durations[i]), int(job_cores[i]), int(job_nodes[i])
            k = int(job_class[h])
            if capacity.counts[k, cores] >= needed:
                chosen_nodes = np.flatnonzero(capacity.free[k] >= cores)[:needed]
                _start_job(rows[i], chosen_nodes, nodes, cores_available, running_jobs,
                           next_job_id, metrics, is_baseline)
                if timeline is not None:
                    timeline[:duration, chosen_nodes] -= cores
                capacity.take(chosen_nodes)
                next_job_id += 1
                started.append(i)
            elif reservations_left and needed <= num_online:
                if timeline is None:
                    # Each reservation starts once running jobs and earlier reservations end at the latest.
                    max_running = max((int(j['duration']) for j in running_jobs.values()), default=0)
                    timeline = _free_core_timeline(nodes, running_jobs,
                                                   max_running + (reservation_depth + 1) * MAX_JOB_DURATION)
                start = _earliest_start(timeline, duration, needed, cores, monotone=not reservation_starts)
                if start is not None:
                    window_free = timeline[start:start + duration].min(axis=0)
                    reserved_nodes = np.flatnonzero(window_free >= cores)[:needed]
                    timeline[start:start + duration, reserved_nodes] -= cores
                    reservation_starts.append(start)
                    reservations_left -= 1
                    # The reservation changes what every later job may use: rebuild and re-test.
                    capacity = _Capacity(online, cores_available, timeline, reservation_starts)
                    next_pos = i + 1
                    break
        pos = next_pos

    queue.remove(started)
    return len(started), next_job_id
