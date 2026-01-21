#!/usr/bin/env python3
"""Tests for the plotting module (plotter.py)."""

import os
import tempfile
import numpy as np

from src.plotter import plot_dashboard, plot_cumulative_savings, _compute_cumulative_savings
from src.plot import plot as plot_simple, plot_cumulative_savings as plot_cumulative_savings_simple
from src.weights import Weights
from src.metrics_tracker import MetricsTracker
from src.plot_config import PlotConfig


class MockEnv:
    """Mock environment with minimal attributes needed for plotting."""

    def __init__(self, num_hours=336):
        self.num_hours = num_hours
        self.session = "test_session"
        self.current_episode = 1
        self.current_step = 100
        self.weights = Weights(
            efficiency_weight=0.7,
            price_weight=0.2,
            idle_weight=0.1,
            job_age_weight=0.0,
            drop_weight=0.0,
        )

        # Metrics tracker with sample data
        self.metrics = MetricsTracker()
        self.metrics.jobs_submitted = 500
        self.metrics.jobs_completed = 450
        self.metrics.total_job_wait_time = 900
        self.metrics.max_queue_size_reached = 200
        self.metrics.baseline_jobs_submitted = 500
        self.metrics.baseline_jobs_completed = 400
        self.metrics.baseline_total_job_wait_time = 1200
        self.metrics.baseline_max_queue_size_reached = 300
        self.metrics.total_cost = 5000
        self.metrics.baseline_cost = 6000
        self.metrics.baseline_cost_off = 5500

        # Time series data
        self.metrics.price_stats = list(np.random.uniform(50, 150, num_hours))
        self.metrics.on_nodes = list(np.random.randint(100, 300, num_hours))
        self.metrics.used_nodes = list(np.random.randint(50, 200, num_hours))
        self.metrics.job_queue_sizes = list(np.random.randint(0, 500, num_hours))
        self.metrics.running_jobs_counts = list(np.random.randint(10, 100, num_hours))
        self.metrics.eff_rewards = list(np.random.uniform(0, 1, num_hours))
        self.metrics.price_rewards = list(np.random.uniform(0, 1, num_hours))
        self.metrics.idle_penalties = list(np.random.uniform(0, 0.5, num_hours))
        self.metrics.job_age_penalties = list(np.random.uniform(0, 0.3, num_hours))
        self.metrics.rewards = list(np.random.uniform(-1, 1, num_hours))

        # Baseline wait time (used directly in plotting)
        self.baseline_total_job_wait_time = 1200

        # Plot config
        self.plot_config = PlotConfig(
            plot_eff_reward=True,
            plot_price_reward=True,
            plot_idle_penalty=True,
            plot_job_age_penalty=True,
            plot_total_reward=True,
        )

        # Additional attributes for plot_simple (src/plot.py)
        self.next_plot_save = 0
        self.steps_per_iteration = 100


def make_episode_costs(n_episodes=12):
    """Create sample episode cost data for testing cumulative savings."""
    costs = []
    for i in range(n_episodes):
        agent = 5000 + np.random.uniform(-500, 500)
        baseline = 6000 + np.random.uniform(-300, 300)
        baseline_off = 5500 + np.random.uniform(-400, 400)
        costs.append({
            "agent_cost": agent,
            "baseline_cost": baseline,
            "baseline_cost_off": baseline_off,
        })
    return costs


class TestComputeCumulativeSavings:
    """Tests for _compute_cumulative_savings helper."""

    def test_empty_input(self):
        result = _compute_cumulative_savings([])
        assert result is None

    def test_none_input(self):
        result = _compute_cumulative_savings(None)
        assert result is None

    def test_single_episode(self):
        costs = [{"agent_cost": 100, "baseline_cost": 120, "baseline_cost_off": 110}]
        result = _compute_cumulative_savings(costs)
        assert result is not None
        assert len(result["cum_s"]) == 1
        assert result["cum_s"][0] == 20  # 120 - 100
        assert result["cum_s_off"][0] == 10  # 110 - 100

    def test_multiple_episodes(self):
        costs = make_episode_costs(6)
        result = _compute_cumulative_savings(costs)
        assert result is not None
        assert len(result["months"]) == 6
        assert len(result["cum_s"]) == 6
        assert len(result["cum_s_off"]) == 6
        # Cumulative savings should be monotonically calculated
        for i in range(1, len(result["cum_s"])):
            expected_diff = costs[i]["baseline_cost"] - costs[i]["agent_cost"]
            actual_diff = result["cum_s"][i] - result["cum_s"][i - 1]
            np.testing.assert_almost_equal(actual_diff, expected_diff)

    def test_months_calculation(self):
        costs = make_episode_costs(4)
        result = _compute_cumulative_savings(costs)
        # Each episode is 2 weeks, so months = (episode_num * 2) / 4.33
        expected_months = (np.arange(1, 5) * 2.0) / 4.33
        np.testing.assert_array_almost_equal(result["months"], expected_months)


class TestPlotDashboard:
    """Tests for plot_dashboard function."""

    def test_saves_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = MockEnv(num_hours=48)
            env.plots_dir = tmpdir

            plot_dashboard(env, num_hours=48, max_nodes=335, save=True, show=False)

            # Check that a file was created
            files = os.listdir(tmpdir)
            assert len(files) == 1
            assert files[0].endswith(".png")

    def test_with_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = MockEnv(num_hours=48)
            env.plots_dir = tmpdir

            plot_dashboard(env, num_hours=48, max_nodes=335, save=True, show=False, suffix="test")

            files = os.listdir(tmpdir)
            assert len(files) == 1
            assert "test" in files[0]

    def test_no_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = MockEnv(num_hours=48)
            env.plots_dir = tmpdir

            plot_dashboard(env, num_hours=48, max_nodes=335, save=False, show=False)

            files = os.listdir(tmpdir)
            assert len(files) == 0

    def test_skip_all_panels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = MockEnv(num_hours=48)
            env.plots_dir = tmpdir
            # Skip all panels
            env.plot_config = PlotConfig(
                skip_plot_price=True,
                skip_plot_online_nodes=True,
                skip_plot_used_nodes=True,
                skip_plot_job_queue=True,
            )

            # Should print message and not crash
            plot_dashboard(env, num_hours=48, max_nodes=335, save=True, show=False)

            # No file should be created when nothing to plot
            files = os.listdir(tmpdir)
            assert len(files) == 0

    def test_partial_panels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = MockEnv(num_hours=48)
            env.plots_dir = tmpdir
            # Only show price and nodes
            env.plot_config = PlotConfig(
                skip_plot_used_nodes=True,
                skip_plot_job_queue=True,
            )

            plot_dashboard(env, num_hours=48, max_nodes=335, save=True, show=False)

            files = os.listdir(tmpdir)
            assert len(files) == 1


class TestPlotCumulativeSavings:
    """Tests for plot_cumulative_savings function."""

    def test_saves_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = MockEnv()
            env.plots_dir = tmpdir
            costs = make_episode_costs(12)

            result = plot_cumulative_savings(
                env, costs, session_dir=tmpdir, save=True, show=False
            )

            assert result is not None
            files = os.listdir(tmpdir)
            assert len(files) == 1
            assert "cumulative_savings" in files[0]
            assert files[0].endswith(".png")

    def test_returns_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = MockEnv()
            costs = make_episode_costs(12)

            result = plot_cumulative_savings(
                env, costs, session_dir=tmpdir, save=False, show=False
            )

            assert result is not None
            assert "total_savings" in result
            assert "avg_monthly_savings_pct" in result
            assert "total_savings_off" in result
            assert "avg_monthly_savings_pct_off" in result

    def test_empty_costs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = MockEnv()

            result = plot_cumulative_savings(
                env, [], session_dir=tmpdir, save=True, show=False
            )

            assert result is None
            files = os.listdir(tmpdir)
            assert len(files) == 0

    def test_uses_env_plots_dir_when_no_session_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = MockEnv()
            env.plots_dir = tmpdir
            costs = make_episode_costs(4)

            plot_cumulative_savings(env, costs, session_dir=None, save=True, show=False)

            files = os.listdir(tmpdir)
            assert len(files) == 1

    def test_with_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = MockEnv()
            costs = make_episode_costs(4)

            plot_cumulative_savings(
                env, costs, session_dir=tmpdir, save=True, show=False, suffix="eval"
            )

            files = os.listdir(tmpdir)
            assert len(files) == 1
            assert "eval" in files[0]


class TestPlotSimple:
    """Tests for plot() function from src/plot.py."""

    def test_saves_file(self, output_dir):
        env = MockEnv(num_hours=48)
        env.plots_dir = output_dir + "/"  # plot_simple expects trailing slash

        plot_simple(env, num_hours=48, max_nodes=335, save=True, show=False, suffix=1)

        files = [f for f in os.listdir(output_dir) if f.startswith("e0.7") and "cumulative" not in f]
        assert len(files) >= 1
        assert files[-1].endswith(".png")

    def test_no_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = MockEnv(num_hours=48)
            env.plots_dir = tmpdir + "/"

            plot_simple(env, num_hours=48, max_nodes=335, save=False, show=False, suffix=1)

            files = os.listdir(tmpdir)
            assert len(files) == 0

    def test_with_skip_flags(self, output_dir):
        env = MockEnv(num_hours=48)
        env.plots_dir = output_dir + "/"
        env.plot_config = PlotConfig(
            skip_plot_price=True,
            skip_plot_online_nodes=True,
            skip_plot_used_nodes=True,
            skip_plot_job_queue=True,
        )

        plot_simple(env, num_hours=48, max_nodes=335, save=True, show=False, suffix=2)

        files = [f for f in os.listdir(output_dir) if f.startswith("e0.7") and "cumulative" not in f]
        assert len(files) >= 1  # Still saves even with nothing plotted


class TestPlotCumulativeSavingsSimple:
    """Tests for plot_cumulative_savings() from src/plot.py."""

    def test_saves_file(self, output_dir):
        env = MockEnv()
        costs = make_episode_costs(12)

        result = plot_cumulative_savings_simple(
            env, costs, session_dir=output_dir, months=6, save=True, show=False
        )

        assert result is not None
        files = [f for f in os.listdir(output_dir) if "cumulative_savings" in f]
        assert len(files) >= 1

    def test_returns_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = MockEnv()
            costs = make_episode_costs(12)

            result = plot_cumulative_savings_simple(
                env, costs, session_dir=tmpdir, save=False, show=False
            )

            assert result is not None
            assert "total_savings" in result
            assert "avg_monthly_savings_pct" in result

    def test_empty_costs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = MockEnv()

            result = plot_cumulative_savings_simple(
                env, [], session_dir=tmpdir, save=True, show=False
            )

            assert result is None

    def test_fewer_episodes_than_requested(self, output_dir):
        env = MockEnv()
        costs = make_episode_costs(4)  # Only 4 episodes, but request 12 months

        result = plot_cumulative_savings_simple(
            env, costs, session_dir=output_dir, months=12, save=True, show=False
        )

        # Should still work with available episodes
        assert result is not None


def get_output_dir():
    """Get persistent output directory for visual inspection of plots."""
    output_dir = os.path.join(os.path.dirname(__file__), "test_output", "plotter")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_visual_samples():
    """Generate sample plots for visual inspection (saved to test/test_output/plotter/)."""
    output_dir = get_output_dir()
    print(f"Generating visual samples in: {output_dir}")

    # Set seed for reproducible plots
    np.random.seed(42)

    # Generate dashboard plot (src/plotter.py)
    env = MockEnv(num_hours=168)  # One week
    env.plots_dir = output_dir
    plot_dashboard(env, num_hours=168, max_nodes=335, save=True, show=False, suffix="sample_dashboard")
    print("  - Dashboard plot (plotter.py) saved")

    # Generate cumulative savings plot (src/plotter.py)
    costs = make_episode_costs(12)
    plot_cumulative_savings(env, costs, session_dir=output_dir, save=True, show=False, suffix="sample_plotter")
    print("  - Cumulative savings plot (plotter.py) saved")

    # Generate simple plot (src/plot.py) - note: suffix must be an integer
    env2 = MockEnv(num_hours=168)
    env2.plots_dir = output_dir + "/"
    plot_simple(env2, num_hours=168, max_nodes=335, save=True, show=False, suffix=999)
    print("  - Simple plot (plot.py) saved")

    # Generate cumulative savings plot (src/plot.py)
    costs2 = make_episode_costs(12)
    plot_cumulative_savings_simple(env2, costs2, session_dir=output_dir, months=6, save=True, show=False)
    print("  - Cumulative savings plot (plot.py) saved")

    print(f"\nPlots saved to: {output_dir}")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")


def main():
    """Run tests without pytest (for quick manual testing)."""
    output_dir = get_output_dir()

    print("Testing _compute_cumulative_savings...")
    test_cs = TestComputeCumulativeSavings()
    test_cs.test_empty_input()
    test_cs.test_none_input()
    test_cs.test_single_episode()
    test_cs.test_multiple_episodes()
    test_cs.test_months_calculation()
    print("[OK] _compute_cumulative_savings tests passed")

    print("\nTesting plot_dashboard (src/plotter.py)...")
    test_pd = TestPlotDashboard()
    test_pd.test_saves_file()
    test_pd.test_with_suffix()
    test_pd.test_no_save()
    test_pd.test_skip_all_panels()
    test_pd.test_partial_panels()
    print("[OK] plot_dashboard tests passed")

    print("\nTesting plot_cumulative_savings (src/plotter.py)...")
    test_pcs = TestPlotCumulativeSavings()
    test_pcs.test_saves_file()
    test_pcs.test_returns_stats()
    test_pcs.test_empty_costs()
    test_pcs.test_uses_env_plots_dir_when_no_session_dir()
    test_pcs.test_with_suffix()
    print("[OK] plot_cumulative_savings tests passed")

    print("\nTesting plot (src/plot.py)...")
    test_ps = TestPlotSimple()
    test_ps.test_saves_file(output_dir)
    test_ps.test_no_save()
    test_ps.test_with_skip_flags(output_dir)
    print("[OK] plot tests passed")

    print("\nTesting plot_cumulative_savings (src/plot.py)...")
    test_pcss = TestPlotCumulativeSavingsSimple()
    test_pcss.test_saves_file(output_dir)
    test_pcss.test_returns_stats()
    test_pcss.test_empty_costs()
    test_pcss.test_fewer_episodes_than_requested(output_dir)
    print("[OK] plot_cumulative_savings (simple) tests passed")

    print("\n[OK] All plotter tests passed!")

    # Generate visual samples for inspection
    print("\n" + "=" * 60)
    generate_visual_samples()


if __name__ == "__main__":
    main()
