"""Metrics tracking and episode recording for the PowerSched environment."""


class MetricsTracker:
    """Tracks metrics throughout training episodes."""

    def __init__(self):
        """Initialize all metric counters."""
        self.reset_episode_metrics()
        self.reset_state_metrics()

        # Cumulative metrics across all episodes
        self.episode_costs = []

    def reset_episode_metrics(self):
        """Reset metrics that persist across episodes."""
        # Job tracking metrics for agent (cumulative across episodes)
        self.jobs_dropped = 0
        self.jobs_rejected_queue_full = 0

        # Job tracking metrics for baseline (cumulative across episodes)
        self.baseline_jobs_dropped = 0
        self.baseline_jobs_rejected_queue_full = 0

    def reset_state_metrics(self):
        """Reset metrics at the start of each episode."""
        # Episode-level metrics
        self.current_hour = 0
        self.episode_reward = 0

        # Time series data for plotting
        self.on_nodes = []
        self.used_nodes = []
        self.job_queue_sizes = []
        self.price_stats = []

        self.eff_rewards = []
        self.price_rewards = []
        self.idle_penalties = []
        self.job_age_penalties = []

        # Cost tracking
        self.total_cost = 0
        self.baseline_cost = 0
        self.baseline_cost_off = 0

        # Agent job metrics
        self.jobs_submitted = 0
        self.jobs_completed = 0
        self.total_job_wait_time = 0
        self.max_queue_size_reached = 0

        # Baseline job metrics
        self.baseline_jobs_submitted = 0
        self.baseline_jobs_completed = 0
        self.baseline_total_job_wait_time = 0
        self.baseline_max_queue_size_reached = 0

        # Per-episode drop counters
        self.dropped_this_episode = 0
        self.baseline_dropped_this_episode = 0

    def record_episode_completion(self, current_episode):
        """
        Record episode costs and metrics for long-term analysis.

        Args:
            current_episode: Current episode number

        Returns:
            Dictionary with episode data
        """
        # Calculate average wait times
        avg_wait_time = self.total_job_wait_time / self.jobs_completed if self.jobs_completed > 0 else 0
        baseline_avg_wait_time = self.baseline_total_job_wait_time / self.baseline_jobs_completed if self.baseline_jobs_completed > 0 else 0

        # Calculate completion rates
        completion_rate = (self.jobs_completed / self.jobs_submitted * 100) if self.jobs_submitted > 0 else 0
        baseline_completion_rate = (self.baseline_jobs_completed / self.baseline_jobs_submitted * 100) if self.baseline_jobs_submitted > 0 else 0

        drop_rate = (self.jobs_dropped / self.jobs_submitted * 100) if self.jobs_submitted else 0.0
        baseline_drop_rate = (self.baseline_jobs_dropped / self.baseline_jobs_submitted * 100) if self.baseline_jobs_submitted else 0.0

        episode_data = {
            'episode': current_episode,
            'agent_cost': float(self.total_cost),
            'baseline_cost': float(self.baseline_cost),
            'baseline_cost_off': float(self.baseline_cost_off),
            'savings_vs_baseline': float(self.baseline_cost - self.total_cost),
            'savings_vs_baseline_off': float(self.baseline_cost_off - self.total_cost),
            'savings_pct_baseline': float(((self.baseline_cost - self.total_cost) / self.baseline_cost) * 100) if self.baseline_cost > 0 else 0,
            'savings_pct_baseline_off': float(((self.baseline_cost_off - self.total_cost) / self.baseline_cost_off) * 100) if self.baseline_cost_off > 0 else 0,
            'total_reward': float(self.episode_reward),
            # Agent job metrics
            'jobs_submitted': self.jobs_submitted,
            'jobs_completed': self.jobs_completed,
            'avg_wait_time': float(avg_wait_time),
            'completion_rate': float(completion_rate),
            'max_queue_size': self.max_queue_size_reached,
            # Baseline job metrics
            'baseline_jobs_submitted': self.baseline_jobs_submitted,
            'baseline_jobs_completed': self.baseline_jobs_completed,
            'baseline_avg_wait_time': float(baseline_avg_wait_time),
            'baseline_completion_rate': float(baseline_completion_rate),
            'baseline_max_queue_size': self.baseline_max_queue_size_reached,
            # Drop metrics
            "jobs_dropped": self.jobs_dropped,
            "drop_rate": float(drop_rate),
            "jobs_rejected_queue_full": self.jobs_rejected_queue_full,
            "baseline_jobs_dropped": self.baseline_jobs_dropped,
            "baseline_drop_rate": float(baseline_drop_rate),
            "baseline_jobs_rejected_queue_full": self.baseline_jobs_rejected_queue_full,
        }
        self.episode_costs.append(episode_data)
        return episode_data
