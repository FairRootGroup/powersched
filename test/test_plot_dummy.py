# test_plot_dummy.py
import os
import numpy as np
import matplotlib.pyplot as plt

# Import the new dashboard + cumulative savings plotters
from src.plotter import plot_dashboard, plot_cumulative_savings
from src.metrics_tracker import MetricsTracker
from src.plot_config import PlotConfig


class DummyWeights:
    def __init__(self, efficiency_weight=1.0, price_weight=1.0, idle_weight=1.0, job_age_weight=1.0):
        self.efficiency_weight = efficiency_weight
        self.price_weight = price_weight
        self.idle_weight = idle_weight
        self.job_age_weight = job_age_weight

    def __repr__(self):
        return (
            f"Weights(eff={self.efficiency_weight}, price={self.price_weight}, "
            f"idle={self.idle_weight}, age={self.job_age_weight})"
        )


class DummyEnv:
    """Minimal env-like object providing the attributes plot_dashboard() reads."""

    def __init__(self, num_hours=336, max_nodes=64, seed=123):
        rng = np.random.default_rng(seed)

        self.session = "dummy_session"
        self.current_episode = 1
        self.current_step = num_hours
        self.weights = DummyWeights(0.8, 1.2, 0.6, 0.9)

        # where to save
        self.plots_dir = "./_plots"
        os.makedirs(self.plots_dir, exist_ok=True)

        self.plot_config = PlotConfig(
            plot_eff_reward=True,
            plot_price_reward=True,
            plot_idle_penalty=True,
            plot_job_age_penalty=True,
            plot_total_reward=True,
        )

        self.metrics = MetricsTracker()

        # cost counters used in header
        self.metrics.episode_total_cost = float(rng.uniform(50_000, 90_000))
        self.metrics.episode_baseline_cost = float(self.metrics.episode_total_cost + rng.uniform(5_000, 20_000))
        self.metrics.episode_baseline_cost_off = float(self.metrics.episode_total_cost + rng.uniform(2_000, 10_000))

        # job counters used in header
        self.metrics.episode_jobs_submitted = int(rng.integers(5_000, 15_000))
        self.metrics.episode_jobs_completed = int(self.metrics.episode_jobs_submitted - rng.integers(0, 500))
        self.metrics.episode_total_job_wait_time = float(rng.uniform(10_000, 80_000))  # hours total (dummy)

        self.metrics.episode_baseline_jobs_submitted = self.metrics.episode_jobs_submitted
        self.metrics.episode_baseline_jobs_completed = int(self.metrics.episode_baseline_jobs_submitted - rng.integers(0, 900))
        self.metrics.episode_baseline_total_job_wait_time = float(rng.uniform(12_000, 120_000))

        self.metrics.episode_max_queue_size_reached = int(rng.integers(50, 1200))
        self.metrics.episode_baseline_max_queue_size_reached = int(rng.integers(50, 1500))

        # --- time series (length = num_hours) ---
        t = np.arange(num_hours)

        # price (€/MWh): daily-ish seasonality + noise
        self.metrics.episode_price_stats = (
            70 + 20 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 5, size=num_hours)
        ).clip(min=0)

        # online fluctuates slowly
        online = (
            0.7 * max_nodes
            + 0.2 * max_nodes * np.sin(2 * np.pi * t / 48)
            + rng.normal(0, 2, size=num_hours)
        )
        self.metrics.episode_on_nodes = np.clip(np.rint(online), 0, max_nodes).astype(int)

        # used <= online; used follows demand
        demand = (
            0.5 * max_nodes
            + 0.35 * max_nodes * np.sin(2 * np.pi * (t + 6) / 24)
            + rng.normal(0, 3, size=num_hours)
        )
        used = np.minimum(self.metrics.episode_on_nodes, np.clip(np.rint(demand), 0, max_nodes).astype(int))
        self.metrics.episode_used_nodes = used
        # approximate "running jobs" as used_nodes for the dummy
        self.metrics.episode_running_jobs_counts = self.metrics.episode_used_nodes.copy()


        # queue size: spikes + correlation with high demand/low capacity
        q = 200 + 300 * np.maximum(0, np.sin(2 * np.pi * (t - 4) / 72)) + 3 * (np.maximum(0, demand - online))
        q = q + rng.normal(0, 30, size=num_hours)
        self.metrics.episode_job_queue_sizes = np.clip(np.rint(q), 0, 2000).astype(int)

        # fake “running jobs” series for the overlay with the queue
        # heuristic: roughly proportional to used nodes, plus noise, but not exceeding queue size
        running = ((self.metrics.episode_used_nodes * rng.uniform(0, 10)) + rng.normal(0, 3, size=num_hours))
        running = np.clip(np.rint(running), 0, self.metrics.episode_job_queue_sizes).astype(int)
        self.metrics.episode_running_jobs_counts = running

        # reward components “scaled”
        # keep in plausible ranges: rewards ~ [0, 100], penalties ~ [-100, 0]
        on = np.maximum(self.metrics.episode_on_nodes, 1)
        self.metrics.episode_eff_rewards = np.clip(100.0 * (self.metrics.episode_used_nodes / on), 0.0, 100.0)
        self.metrics.episode_price_rewards = np.clip(rng.normal(10, 20, size=num_hours), -50, 50)
        self.metrics.episode_idle_penalties = -np.clip((self.metrics.episode_on_nodes - self.metrics.episode_used_nodes) * 2.0, 0, 100).astype(float)
        self.metrics.episode_job_age_penalties = -np.clip(self.metrics.episode_job_queue_sizes / 20.0, 0, 100).astype(float)

        # "Total reward" as a rough combination of the components
        self.metrics.episode_rewards = (
            0.5 * self.metrics.episode_eff_rewards
            + 0.3 * self.metrics.episode_price_rewards
            + 0.1 * self.metrics.episode_idle_penalties
            + 0.1 * self.metrics.episode_job_age_penalties
        )


def make_dummy_episode_costs(n_episodes=24, seed=123):
    rng = np.random.default_rng(seed)

    episode_costs = []
    agent_cost = 90_000.0
    for i in range(n_episodes):
        
        baseline = rng.normal(95_000, 3_000)
        baseline_off = baseline - rng.uniform(2_000, 7_000)

        # simulate learning: agent gets a bit cheaper over time
        agent_cost *= rng.uniform(0.992, 0.998)
        agent = agent_cost + rng.normal(0, 1_000)

        # dummy jobs & waits
        jobs_base = rng.integers(4000, 6000)
        jobs_agent = int(jobs_base * rng.uniform(0.1, 1.5))
        jobs_base = max(jobs_base, 1)

        avg_wait_agent = rng.uniform(5.0, 20.0)
        avg_wait_base = avg_wait_agent + rng.uniform(-3.0, 5.0)

        episode_costs.append({
            "agent_cost": float(agent),
            "baseline_cost": float(baseline),
            "baseline_cost_off": float(baseline_off),
            "jobs_completed": float(jobs_agent),
            "baseline_jobs_completed": float(jobs_base),
            "avg_wait_time": float(avg_wait_agent),
            "baseline_avg_wait_time": float(avg_wait_base),
        })

    return episode_costs



#def make_dummy_episode_costs(n_episodes=24, seed=123):
#    """
#    Produce a plausible episode_costs list:
#      - baseline_cost is highest
#      - baseline_cost_off is a bit lower
#      - agent_cost starts near baseline and improves slightly over episodes
#      - includes jobs_completed and baseline_jobs_completed so that
#        the jobs & combined panels in plot_cumulative_savings can be tested.
#    """
#    rng = np.random.default_rng(seed)
#
#    episode_costs = []
#    agent_cost = 90_000.0
#    for i in range(n_episodes):
#        baseline = rng.normal(95_000, 3_000)
#        baseline_off = baseline - rng.uniform(2_000, 7_000)
#
#        # simulate learning: agent gets a bit cheaper over time
#        agent_cost *= rng.uniform(0.992, 0.998)  # gradual improvement
#        agent = agent_cost + rng.normal(0, 1_000)
#
#        # jobs: baseline slightly worse or similar, agent slightly improving
#        base_jobs = rng.integers(4_000, 6_000)
#        # allow agent to be a bit better or worse, but centered near baseline
#        agent_jobs = int(base_jobs * rng.uniform(0.95, 1.05))
#
#        episode_costs.append({
#            "agent_cost": float(agent),
#            "baseline_cost": float(baseline),
#            "baseline_cost_off": float(baseline_off),
#            "jobs_completed": float(agent_jobs),
#            "baseline_jobs_completed": float(base_jobs),
#        })
#
#    return episode_costs


def main():
    num_hours = 24 * 14
    max_nodes = 64

    env = DummyEnv(num_hours=num_hours, max_nodes=max_nodes, seed=123)
    episode_costs = make_dummy_episode_costs(n_episodes=24, seed=999)

    # Per-hour dashboard
    plot_dashboard(
        env,
        num_hours=num_hours,
        max_nodes=max_nodes,
        episode_costs=episode_costs,  # accepted but not used in dashboard; fine for API compatibility
        save=True,
        show=True,
        suffix="dummy_dashboard",
    )

    # Cumulative savings + jobs on a separate, side-by-side canvas
    plot_cumulative_savings(
        env,
        episode_costs=episode_costs,
        session_dir=env.plots_dir,
        save=True,
        show=True,
        suffix="dummy_cumulative",
    )
    plt.show()

if __name__ == "__main__":
    main()
