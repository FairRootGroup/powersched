#!/usr/bin/env python3
"""Tests for --jobs replay modes: raw exact replay vs aggregated template replay.

Synthetic log has two hourly bins with known jobs:
  Bin 1 (00:xx):
    - 2x (2 nodes, 16 NCPUS=8/node, 30 min)  sub-hour, identical → merge in template mode
    - 1x (1 node,   8 NCPUS=8/node, 120 min) 2-hour job
  Bin 2 (01:xx):
    - 1x (4 nodes, 32 NCPUS=8/node, 60 min)  1-hour job
    - 1x (1 node,  96 NCPUS=96/node, 30 min) sub-hour job

Raw replay:   bin 1 → 3 jobs, bin 2 → 2 jobs; cores preserved as-parsed.
Template replay: bin 1 → 2 job instances (two sub-hour jobs collapsed into one
                 core-hour-equivalent entry), bin 2 → 2 job instances.
"""

import math
import sys
import tempfile
import textwrap

import numpy as np

from src.sampler_jobs import DurationSampler
from src.workload_generator import generate_jobs
from src.config import CORES_PER_NODE, MAX_NODES_PER_JOB

LOG_CONTENT = textwrap.dedent("""\
    JobID      User  Partition              Submit               Start                 End ElapsedRaw      NCPUS   NNodes\x20
    ------------------------------ --------- ---------- ------------------- ------------------- ------------------- ---------- ---------- --------\x20
          1    user01        gpu 2024-01-01T00:01:00 2024-01-01T00:01:00 2024-01-01T00:31:00       1800         16        2
          2    user01        gpu 2024-01-01T00:02:00 2024-01-01T00:02:00 2024-01-01T00:32:00       1800         16        2
          3    user01        gpu 2024-01-01T00:10:00 2024-01-01T00:10:00 2024-01-01T02:10:00       7200          8        1
          4    user01        gpu 2024-01-01T01:05:00 2024-01-01T01:05:00 2024-01-01T02:05:00       3600         32        4
          5    user01        gpu 2024-01-01T01:10:00 2024-01-01T01:10:00 2024-01-01T01:40:00       1800         96        1
""")

# Expected raw jobs after clamping/rounding applied by generate_jobs.
# duration = max(1, ceil(elapsed_seconds / 3600))
# cores_per_node = ncpus // nnodes, then clamped to [1, CORES_PER_NODE]
EXPECTED_RAW_BIN1 = [
    {"duration_hours": 1, "nnodes": 2, "cores_per_node": 8},
    {"duration_hours": 1, "nnodes": 2, "cores_per_node": 8},
    {"duration_hours": 2, "nnodes": 1, "cores_per_node": 8},
]
EXPECTED_RAW_BIN2 = [
    {"duration_hours": 1, "nnodes": 4, "cores_per_node": 8},
    {"duration_hours": 1, "nnodes": 1, "cores_per_node": 96},
]


def _make_sampler(log_path: str, template: bool) -> DurationSampler:
    s = DurationSampler()
    s.parse_jobs(log_path, 60)
    if template:
        s.precalculate_hourly_jobs(CORES_PER_NODE, MAX_NODES_PER_JOB)
    return s


def _call_generate(sampler, hour, exact_replay):
    rng = np.random.default_rng(0)
    return generate_jobs(
        current_hour=hour,
        external_jobs="synthetic",  # non-empty triggers --jobs path
        external_hourly_jobs=None,
        external_durations=None,
        workload_gen=None,
        jobs_sampler=sampler,
        hourly_sampler=None,
        durations_sampler=None,
        np_random=rng,
        jobs_exact_replay=exact_replay,
    )


def test_raw_replay_job_counts():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(LOG_CONTENT)
        path = f.name

    sampler = _make_sampler(path, template=False)

    count1, _, _, _ = _call_generate(sampler, hour=1, exact_replay=True)
    assert count1 == 3, f"bin 1 raw: expected 3 jobs, got {count1}"

    count2, _, _, _ = _call_generate(sampler, hour=2, exact_replay=True)
    assert count2 == 2, f"bin 2 raw: expected 2 jobs, got {count2}"

    print(f"PASS: raw replay job counts ({count1}, {count2})")


def test_raw_replay_cores_populated():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(LOG_CONTENT)
        path = f.name

    sampler = _make_sampler(path, template=False)

    for hour in (1, 2):
        _, _, _, cores = _call_generate(sampler, hour=hour, exact_replay=True)
        assert all(c > 0 for c in cores), \
            f"bin {hour} raw: cores_per_node contains zeros: {cores}"

    print("PASS: raw replay cores populated")


def test_raw_replay_job_attributes():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(LOG_CONTENT)
        path = f.name

    sampler = _make_sampler(path, template=False)

    _, dur1, nodes1, cores1 = _call_generate(sampler, hour=1, exact_replay=True)
    actual1 = sorted(zip(dur1, nodes1, cores1))
    expected1 = sorted(
        (j["duration_hours"], j["nnodes"], j["cores_per_node"]) for j in EXPECTED_RAW_BIN1
    )
    assert actual1 == expected1, f"bin 1 raw jobs mismatch:\n  got:      {actual1}\n  expected: {expected1}"

    _, dur2, nodes2, cores2 = _call_generate(sampler, hour=2, exact_replay=True)
    actual2 = sorted(zip(dur2, nodes2, cores2))
    expected2 = sorted(
        (j["duration_hours"], j["nnodes"], j["cores_per_node"]) for j in EXPECTED_RAW_BIN2
    )
    assert actual2 == expected2, f"bin 2 raw jobs mismatch:\n  got:      {actual2}\n  expected: {expected2}"

    print("PASS: raw replay job attributes match expected")


def test_template_replay_cores_populated():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(LOG_CONTENT)
        path = f.name

    sampler = _make_sampler(path, template=True)

    for hour in (1, 2):
        count, durations, nodes, cores = _call_generate(sampler, hour=hour, exact_replay=False)
        assert count > 0, f"bin {hour} template: expected jobs, got 0"
        assert all(c > 0 for c in cores), \
            f"bin {hour} template: cores_per_node contains zeros: {cores}"

    print("PASS: template replay cores populated")


def test_template_replay_bin1_collapses_subhour_pair():
    """Two identical sub-hour jobs in bin 1 should collapse to one template entry."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(LOG_CONTENT)
        path = f.name

    sampler = _make_sampler(path, template=True)

    count, durations, nodes, cores = _call_generate(sampler, hour=1, exact_replay=False)

    # Raw has 3 jobs; template collapses the two identical sub-hour jobs into one
    # core-hour-equivalent entry + keeps the 2h job = 2 job instances.
    assert count == 2, (
        f"bin 1 template: expected 2 job instances after sub-hour collapse, got {count}\n"
        f"  durations={durations}, nodes={nodes}, cores={cores}"
    )
    print(f"PASS: template replay bin 1 collapses sub-hour pair → {count} instances")


def test_template_replay_preserves_core_hours_bin1():
    """Template mode should preserve approximate total core-hours from the raw log."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(LOG_CONTENT)
        path = f.name

    # Raw core-hours for bin 1:
    #   job1: 2 nodes * 8 cores/node * (30/60)h = 8 core-hours
    #   job2: same = 8 core-hours
    #   job3: 1 node * 8 cores/node * (120/60)h = 16 core-hours
    # Total = 32 core-hours
    raw_core_hours_bin1 = 8 + 8 + 16

    sampler = _make_sampler(path, template=True)
    count, durations, nodes, cores = _call_generate(sampler, hour=1, exact_replay=False)

    template_core_hours = sum(
        n * c * d for d, n, c in zip(durations, nodes, cores)
    )
    # Allow small rounding due to ceil operations, but should be within 10%
    assert abs(template_core_hours - raw_core_hours_bin1) / raw_core_hours_bin1 < 0.1, (
        f"bin 1 template core-hours {template_core_hours:.1f} deviates "
        f"more than 10% from raw {raw_core_hours_bin1}"
    )
    print(f"PASS: template replay bin 1 core-hours {template_core_hours:.1f} ≈ raw {raw_core_hours_bin1}")


def main():
    tests = [
        test_raw_replay_job_counts,
        test_raw_replay_cores_populated,
        test_raw_replay_job_attributes,
        test_template_replay_cores_populated,
        test_template_replay_bin1_collapses_subhour_pair,
        test_template_replay_preserves_core_hours_bin1,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed.append(t.__name__)
        except Exception as e:
            print(f"ERROR: {t.__name__}: {e}")
            failed.append(t.__name__)

    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
