"""Tests for the numpy-backed Backlog against the previous deque-of-lists behaviour.

Run with: python -m test.test_backlog
"""

from collections import deque

import numpy as np

from src.config import MAX_JOB_AGE
from src.job_management import Backlog, age_backlog_queue, fill_queue_from_backlog


def _reference_age(backlog: deque) -> int:
    dropped = 0
    kept = []
    for job in backlog:
        job[1] += 1
        if job[1] > MAX_JOB_AGE:
            dropped += 1
        else:
            kept.append(job)
    backlog.clear()
    backlog.extend(kept)
    return dropped


def _reference_fill(job_queue_2d: np.ndarray, backlog: deque, next_empty_slot: int) -> tuple[int, int]:
    moved = 0
    while backlog and next_empty_slot < len(job_queue_2d):
        job_queue_2d[next_empty_slot] = backlog.popleft()
        moved += 1
        next_empty_slot += 1
        while next_empty_slot < len(job_queue_2d) and job_queue_2d[next_empty_slot][0] != 0:
            next_empty_slot += 1
    return next_empty_slot, moved


def _rows(backlog) -> list[list[int]]:
    return [[int(v) for v in job] for job in backlog]


def test_matches_deque_under_random_operations():
    rng = np.random.default_rng(0)
    capacity = 64  # small, so appends hit the buffer end and compaction runs
    for _ in range(200):
        new, ref = Backlog(capacity), deque()
        queue_new = np.zeros((40, 4), dtype=np.int32)
        queue_ref = queue_new.copy()
        slot_new = slot_ref = 0
        for _ in range(60):
            op = rng.integers(0, 4)
            if op == 0:
                for _ in range(int(rng.integers(0, 30))):
                    if len(ref) >= capacity:
                        break
                    job = [int(rng.integers(1, 170)), int(rng.integers(0, MAX_JOB_AGE + 1)),
                           int(rng.integers(1, 17)), int(rng.integers(1, 97))]
                    new.append(job)
                    ref.append(list(job))
            elif op == 1:
                assert age_backlog_queue(new, None) == _reference_age(ref)
            elif op == 2:
                assert fill_queue_from_backlog(queue_new, new, slot_new) == (r := _reference_fill(queue_ref, ref, slot_ref))
                slot_new = slot_ref = r[0]
            else:
                # Free some queue slots, keeping next_empty_slot at the first empty one.
                for idx in rng.choice(len(queue_ref), size=int(rng.integers(0, 10)), replace=False):
                    queue_new[idx] = queue_ref[idx] = 0
                empty = np.flatnonzero(queue_ref[:, 0] == 0)
                slot_new = slot_ref = int(empty[0]) if len(empty) else len(queue_ref)
            assert _rows(new) == _rows(ref)
            assert len(new) == len(ref) and bool(new) == bool(ref)
            assert np.array_equal(queue_new, queue_ref)


def test_age_drops_only_expired_jobs_and_keeps_order():
    backlog = Backlog(8)
    for age in (MAX_JOB_AGE, 3, MAX_JOB_AGE - 1, MAX_JOB_AGE):
        backlog.append([5, age, 1, 1])
    assert backlog.age(MAX_JOB_AGE) == 2
    assert [int(j[1]) for j in backlog] == [4, MAX_JOB_AGE]


def test_clear_and_pop_front():
    backlog = Backlog(4)
    for i in range(3):
        backlog.append([i + 1, 0, 1, 1])
    assert [int(r[0]) for r in backlog.pop_front(2)] == [1, 2]
    assert len(backlog) == 1
    backlog.clear()
    assert not backlog and len(backlog.rows()) == 0


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASSED: {name}")
    print(f"All {len(tests)} backlog tests passed.")


if __name__ == "__main__":
    main()
