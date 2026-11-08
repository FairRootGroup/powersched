"""Checks that the queue summaries and time features in the observation match the simulation.

Run with: python -m test.test_observation
"""

import numpy as np

from src.config import QUEUE_AGE_BIN_EDGES, QUEUE_SIZE_BIN_EDGES
from src.environment import ComputeClusterEnv, Weights
from src.plot_config import PlotConfig
from src.prices import load_price_csv
from src.workloadgen import WorkloadGenConfig, WorkloadGenerator


def _make_env(prices=None, price_times=None):
    workload = WorkloadGenerator(WorkloadGenConfig(arrivals="poisson", poisson_lambda=300.0,
                                                   poisson_lambda_duration=10.0, poisson_lambda_nodes=3.0,
                                                   poisson_lambda_cores=24.0))
    env = ComputeClusterEnv(weights=Weights(0.4, 0.4, 0.0, 0.2, 0.0), session="test_observation",
                            render_mode="none", external_prices=prices, external_durations=None,
                            external_jobs=None, external_hourly_jobs=None, plot_config=PlotConfig(),
                            steps_per_iteration=10**9, evaluation_mode=True, workload_gen=workload,
                            external_price_times=price_times, output_dir="test/test_output")
    return env


def _decode_time(features):
    hour = int(round(np.arctan2(features[0], features[1]) / (2 * np.pi) * 24)) % 24
    weekday = int(round(np.arctan2(features[2], features[3]) / (2 * np.pi) * 7)) % 7
    return hour, weekday


def test_queue_summaries_match_queue():
    env = _make_env()
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(3)
    for _ in range(200):
        # Mostly switch nodes off so the queue builds up and ages.
        obs, *_ = env.step(np.array([0 if rng.random() < 0.7 else 2, rng.integers(0, 9)]))
        rows = env.queue.rows()
        assert obs["queue_ages"].sum() == len(rows) == obs["pending_job_count"][0]
        assert obs["queue_sizes"].sum() == len(rows)
        core_hours = float((rows[:, 0].astype(np.float64) * rows[:, 2] * rows[:, 3]).sum())
        assert np.isclose(obs["queue_age_core_hours"].sum(), core_hours, rtol=1e-5)
        assert np.isclose(obs["pending_core_hours"][0], core_hours, rtol=1e-5)
        if len(rows):
            assert obs["queue_oldest"][0] == rows[0, 1]
            assert obs["queue_oldest"][1] == np.count_nonzero(rows[:, 1] > env.reward_calculator.DEFERRAL_GRACE_HOURS)
            expected_ages = np.bincount(np.searchsorted(QUEUE_AGE_BIN_EDGES, rows[:, 1], side="right"),
                                        minlength=len(QUEUE_AGE_BIN_EDGES) + 1)
            assert np.array_equal(obs["queue_ages"], expected_ages)
            expected_sizes = np.bincount(np.searchsorted(QUEUE_SIZE_BIN_EDGES, rows[:, 2], side="right"),
                                         minlength=len(QUEUE_SIZE_BIN_EDGES) + 1)
            assert np.array_equal(obs["queue_sizes"], expected_sizes)
    assert env.queue, "the test should leave jobs waiting"


def test_time_features_follow_price_timestamps():
    prices, times = load_price_csv("data-external/prices_2024.csv")
    env = _make_env(prices, times)
    obs, _ = env.reset(seed=1, options={"price_start_index": 0})
    for step in range(24 * 9):
        index = step % len(times)
        assert _decode_time(obs["time_features"]) == tuple(int(v) for v in times[index]), step
        assert env._clock() == tuple(int(v) for v in times[index])
        obs, *_ = env.step(np.array([1, 0]))


def test_time_features_without_timestamps():
    env = _make_env()
    obs, _ = env.reset(seed=1)
    start = env.prices.price_index - env.prices.PREDICTION_WINDOW
    for step in range(50):
        index = start + step
        assert _decode_time(obs["time_features"]) == (index % 24, (index // 24) % 7)
        obs, *_ = env.step(np.array([1, 0]))


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASSED: {name}")
    print(f"All {len(tests)} observation tests passed.")


if __name__ == "__main__":
    main()
