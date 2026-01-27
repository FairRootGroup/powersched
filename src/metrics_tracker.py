"""Metrics tracking and episode recording for the PowerSched environment."""


class MetricsTracker:
    """Tracks metrics throughout training episodes."""

    def __init__(self):
        """Initialize all metric counters."""
        self.reset_timeline_metrics()
        self.reset_episode_metrics()

        # Cumulative metrics across all episodes
        self.episode_costs = []

    def reset_timeline_metrics(self):
        """Reset metrics that persist across episodes (full reset)."""
        self.total_time_hours = 0
        self.current_running_jobs = 0

        # Cost tracking (cumulative across episodes)
        self.total_cost = 0
        self.baseline_cost = 0
        self.baseline_cost_off = 0

        # Agent job metrics (cumulative across episodes)
        self.jobs_submitted = 0
        self.jobs_completed = 0
        self.total_job_wait_time = 0
        self.max_queue_size_reached = 0
        self.jobs_dropped = 0
        self.jobs_rejected_queue_full = 0

        # Baseline job metrics (cumulative across episodes)
        self.baseline_jobs_submitted = 0
        self.baseline_jobs_completed = 0
        self.baseline_total_job_wait_time = 0
        self.baseline_max_queue_size_reached = 0
        self.baseline_jobs_dropped = 0
        self.baseline_jobs_rejected_queue_full = 0

        # Time series data for plotting (cumulative)
        self.on_nodes = []
        self.used_nodes = []
        self.job_queue_sizes = []
        self.price_stats = []

        self.eff_rewards = []
        self.price_rewards = []
        self.idle_penalties = []
        self.job_age_penalties = []
        self.rewards = []

    def reset_episode_metrics(self):
        """Reset metrics at the start of each episode."""
        self.current_hour = 0
        self.episode_reward = 0
        self.episode_total_cost = 0
        self.episode_baseline_cost = 0
        self.episode_baseline_cost_off = 0

        # Agent job metrics (episode)
        self.episode_jobs_submitted = 0
        self.episode_jobs_completed = 0
        self.episode_total_job_wait_time = 0
        self.episode_max_queue_size_reached = 0
        self.episode_jobs_dropped = 0
        self.episode_jobs_rejected_queue_full = 0

        # Baseline job metrics (episode)
        self.episode_baseline_jobs_submitted = 0
        self.episode_baseline_jobs_completed = 0
        self.episode_baseline_total_job_wait_time = 0
        self.episode_baseline_max_queue_size_reached = 0
        self.episode_baseline_jobs_dropped = 0
        self.episode_baseline_jobs_rejected_queue_full = 0

        # Time series data for plotting (episode)
        self.episode_on_nodes = []
        self.episode_used_nodes = []
        self.episode_job_queue_sizes = []
        self.episode_price_stats = []

        self.episode_eff_rewards = []
        self.episode_price_rewards = []
        self.episode_idle_penalties = []
        self.episode_job_age_penalties = []
        self.episode_rewards = []
        self.episode_running_jobs_counts = []

    def record_episode_completion(self, current_episode):
        """
        Record episode costs and metrics for long-term analysis.

        Args:
            current_episode: Current episode number

        Returns:
            Dictionary with episode data
        """
        # Calculate average wait times
        avg_wait_time = (
            self.episode_total_job_wait_time / self.episode_jobs_completed
            if self.episode_jobs_completed > 0
            else 0
        )
        baseline_avg_wait_time = (
            self.episode_baseline_total_job_wait_time / self.episode_baseline_jobs_completed
            if self.episode_baseline_jobs_completed > 0
            else 0
        )

        # Calculate completion rates
        completion_rate = (
            (self.episode_jobs_completed / self.episode_jobs_submitted * 100)
            if self.episode_jobs_submitted > 0
            else 0
        )
        baseline_completion_rate = (
            (self.episode_baseline_jobs_completed / self.episode_baseline_jobs_submitted * 100)
            if self.episode_baseline_jobs_submitted > 0
            else 0
        )

        drop_rate = (
            (self.episode_jobs_dropped / self.episode_jobs_submitted * 100)
            if self.episode_jobs_submitted
            else 0.0
        )
        baseline_drop_rate = (
            (self.episode_baseline_jobs_dropped / self.episode_baseline_jobs_submitted * 100)
            if self.episode_baseline_jobs_submitted
            else 0.0
        )

        episode_data = {
            'episode': current_episode,
            'agent_cost': float(self.episode_total_cost),
            'baseline_cost': float(self.episode_baseline_cost),
            'baseline_cost_off': float(self.episode_baseline_cost_off),
            'savings_vs_baseline': float(self.episode_baseline_cost - self.episode_total_cost),
            'savings_vs_baseline_off': float(self.episode_baseline_cost_off - self.episode_total_cost),
            'savings_pct_baseline': float(((self.episode_baseline_cost - self.episode_total_cost) / self.episode_baseline_cost) * 100) if self.episode_baseline_cost > 0 else 0,
            'savings_pct_baseline_off': float(((self.episode_baseline_cost_off - self.episode_total_cost) / self.episode_baseline_cost_off) * 100) if self.episode_baseline_cost_off > 0 else 0,
            'total_reward': float(self.episode_reward),
            # Agent job metrics
            'jobs_submitted': self.episode_jobs_submitted,
            'jobs_completed': self.episode_jobs_completed,
            'avg_wait_time': float(avg_wait_time),
            'completion_rate': float(completion_rate),
            'max_queue_size': self.episode_max_queue_size_reached,
            # Baseline job metrics
            'baseline_jobs_submitted': self.episode_baseline_jobs_submitted,
            'baseline_jobs_completed': self.episode_baseline_jobs_completed,
            'baseline_avg_wait_time': float(baseline_avg_wait_time),
            'baseline_completion_rate': float(baseline_completion_rate),
            'baseline_max_queue_size': self.episode_baseline_max_queue_size_reached,
            # Drop metrics
            "jobs_dropped": self.episode_jobs_dropped,
            "drop_rate": float(drop_rate),
            "jobs_rejected_queue_full": self.episode_jobs_rejected_queue_full,
            "baseline_jobs_dropped": self.episode_baseline_jobs_dropped,
            "baseline_drop_rate": float(baseline_drop_rate),
            "baseline_jobs_rejected_queue_full": self.episode_baseline_jobs_rejected_queue_full,
        }
        self.episode_costs.append(episode_data)
        return episode_data
