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
from src.validation import ValidationConfig, run_validation, record_result, load_history, window_starts, BEST_FILE
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
    all_off = _run(FixedPolicy([0, 8]), envs)     # switch everything off
    keep_on_again = _run(FixedPolicy([2, 8]), envs)
    a, b = keep_on["scales"]["1"], all_off["scales"]["1"]
    assert a["jobs_submitted"] == b["jobs_submitted"] > 0
    assert a["baseline_cost_off"] == b["baseline_cost_off"]
    assert a["baseline_jobs_completed"] == b["baseline_jobs_completed"]
    assert a["episodes"] == b["episodes"] == CONFIG.windows * CONFIG.episodes_per_window
    assert keep_on == keep_on_again  # reusing the envs changes nothing
    assert keep_on["feasible"] and keep_on["score"] is not None
    assert not all_off["feasible"] and all_off["score"] is None


def test_record_result_keeps_best_feasible():
    def result(score):
        return {"scales": {}, "feasible": score is not None, "savings_off_pct": score if score is not None else 9.0, "score": score}
    with tempfile.TemporaryDirectory() as d:
        assert record_result(d, 500000, result(None)) is None
        assert not os.path.exists(os.path.join(d, BEST_FILE))
        assert record_result(d, 1000000, result(2.0))["step"] == 1000000
        assert record_result(d, 1500000, result(1.0))["step"] == 1000000
        # A re-validated step replaces its entry: 1M now scores 0.5, so 1.5M is best.
        assert record_result(d, 1000000, result(0.5))["step"] == 1500000
        assert [h["step"] for h in load_history(d)] == [500000, 1000000, 1500000]
        with open(os.path.join(d, BEST_FILE)) as f:
            assert json.load(f)["step"] == 1500000
        best = record_result(d, 2000000, result(3.0))
        assert best["step"] == 2000000 and best["score"] == 3.0


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
