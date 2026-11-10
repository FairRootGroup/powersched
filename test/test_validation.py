"""Checks the checkpoint validation (src/validation.py): fixed arrivals, feasibility, best-checkpoint record.

Run with: python -m test.test_validation
"""

import contextlib
import io
import json
import os
import tempfile

import numpy as np

from src.environment import ComputeClusterEnv, Weights
from src.plot_config import PlotConfig
from src.validation import ValidationConfig, run_validation, record_result, load_history, window_starts, best_per_budget, BEST_FILE
from src.workloadgen import WorkloadGenConfig, WorkloadGenerator

CONFIG = ValidationConfig(arrival_scales=(1.0,), windows=2, episodes_per_window=1, seed=5)


class FixedPolicy:
    """Stands in for a PPO model: always the same action."""
    def __init__(self, action):
        self.action = np.array(action)

    def predict(self, obs, deterministic=True):
        return self.action, None


def _make_env(arrival_scale):
    workload = WorkloadGenerator(WorkloadGenConfig(arrivals="poisson", poisson_lambda=200.0,
                                                   poisson_lambda_duration=6.0, poisson_lambda_nodes=2.0,
                                                   poisson_lambda_cores=24.0))
    with contextlib.redirect_stdout(io.StringIO()):
        env = ComputeClusterEnv(weights=Weights(0.4, 0.4, 0.0, 0.2, 0.0), session="test_validation",
                                render_mode="none", external_prices=None, external_durations=None,
                                external_jobs=None, external_hourly_jobs=None, plot_config=PlotConfig(),
                                steps_per_iteration=10**9, evaluation_mode=True, workload_gen=workload,
                                job_arrival_scale=arrival_scale, output_dir="test/test_output")
    return env


def _run(policy, envs):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_validation(policy, _make_env, CONFIG, envs)


def test_same_arrivals_for_every_policy_and_feasibility():
    envs = {}
    keep_on = _run(FixedPolicy([2, 8]), envs)     # switch on up to 256 nodes per hour: all on
    all_off = _run(FixedPolicy([0, 8]), envs)     # switch everything off: the queue fills, jobs are lost
    keep_on_again = _run(FixedPolicy([2, 8]), envs)
    a, b = keep_on["scales"]["1"], all_off["scales"]["1"]
    assert a["jobs_submitted"] == b["jobs_submitted"] > 0
    assert a["baseline_cost_off"] == b["baseline_cost_off"]
    assert a["baseline_jobs_completed"] == b["baseline_jobs_completed"]
    assert a["episodes"] == b["episodes"] == CONFIG.windows * CONFIG.episodes_per_window
    assert keep_on == keep_on_again  # reusing the envs changes nothing
    assert keep_on["feasible"] and keep_on["score"] is not None
    assert not all_off["feasible"] and all_off["score"] is None


def _result(savings, wait, lost=0, overdue=0, submitted=100000):
    """A validation result with one scale; lost > 0 loses more jobs than the baseline."""
    scale = dict(savings_off_pct=savings, avg_wait_h=wait, jobs_lost=lost, baseline_jobs_lost=0, completion_ratio=0.97,
                 overdue_jobs_end=overdue, jobs_submitted=submitted)
    return {"scales": {"1": scale}, "feasible": lost == 0, "savings_off_pct": savings, "score": savings if lost == 0 else None}


def test_record_result_keeps_best_per_wait_budget():
    with tempfile.TemporaryDirectory() as d:
        best = record_result(d, 500000, _result(30.0, 2.0, lost=5))   # loses jobs: never best
        assert all(e is None for e in best.values())
        assert not os.path.exists(os.path.join(d, BEST_FILE))
        record_result(d, 1000000, _result(5.0, 4.0))
        record_result(d, 1500000, _result(15.0, 10.0))
        best = record_result(d, 2000000, _result(25.0, 20.0))
        assert {k: (e["step"] if e else None) for k, e in best.items()} == \
            {"6h": 1000000, "12h": 1500000, "24h": 2000000, "any": 2000000}
        # A re-validated step replaces its entry: 2M now waits 30 h, so 1.5M is best within 24 h.
        best = record_result(d, 2000000, _result(25.0, 30.0))
        assert best["24h"]["step"] == 1500000 and best["any"]["step"] == 2000000
        assert [h["step"] for h in load_history(d)] == [500000, 1000000, 1500000, 2000000]
        with open(os.path.join(d, BEST_FILE)) as f:
            saved = json.load(f)
        assert saved["step"] == 2000000 and saved["by_wait_budget"]["6h"]["step"] == 1000000
        # The completion ratio is ignored unless asked for (all entries above have 0.97).
        assert best_per_budget(load_history(d), min_completion_ratio=0.99)["any"] is None


def test_negative_savings_are_never_picked():
    history = [
        {"step": 1, **_result(-20.0, 1.0)},                # costs more than the baseline
        {"step": 2, **_result(10.0, 9.0, overdue=2000)},   # overdue jobs at window ends do not make it infeasible
    ]
    best = best_per_budget(history)
    assert best["6h"] is None                            # only the negative-savings checkpoint waits < 6 h
    assert best["12h"]["step"] == 2 and best["any"]["step"] == 2


def test_window_starts_spread_evenly():
    assert window_starts(17520, 4) == [2190, 6570, 10950, 15330]
    assert window_starts(None, 2) == [0, 336]


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASSED: {name}")
    print(f"All {len(tests)} validation tests passed.")


if __name__ == "__main__":
    main()
