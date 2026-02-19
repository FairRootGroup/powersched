"""Metrics tracking and episode recording for the PowerSched environment."""


class MetricsTracker:
    """Tracks metrics throughout training episodes."""

    def __init__(self) -> None:
        """Initialize all metric counters."""
        self.reset_timeline_metrics()
        self.reset_episode_metrics()

        # Cumulative metrics across all episodes
        self.episode_costs: list[dict[str, float | int]] = []

    def reset_timeline_metrics(self) -> None:
        """Reset metrics that persist across episodes (full reset)."""
        self.total_time_hours: int = 0
        self.current_running_jobs: int = 0

        # Cost tracking (cumulative across episodes)
        self.total_cost: float = 0.0
        self.baseline_cost: float = 0.0
        self.baseline_cost_off: float = 0.0

        # Agent job metrics (cumulative across episodes)
        self.jobs_submitted: int = 0
        self.jobs_completed: int = 0
        self.total_job_wait_time: int = 0
        self.max_queue_size_reached: int = 0
        self.max_backlog_size_reached: int = 0
        self.jobs_dropped: int = 0
        self.jobs_rejected_queue_full: int = 0

        # Baseline job metrics (cumulative across episodes)
        self.baseline_jobs_submitted: int = 0
        self.baseline_jobs_completed: int = 0
        self.baseline_total_job_wait_time: int = 0
        self.baseline_max_queue_size_reached: int = 0
        self.baseline_max_backlog_size_reached: int = 0
        self.baseline_jobs_dropped: int = 0
        self.baseline_jobs_rejected_queue_full: int = 0

        # Time series data for plotting (cumulative)
        self.on_nodes: list[int] = []
        self.used_nodes: list[int] = []
        self.job_queue_sizes: list[int] = []
        self.price_stats: list[float] = []

        self.eff_rewards: list[float] = []
        self.price_rewards: list[float] = []
        self.idle_penalties: list[float] = []
        self.job_age_penalties: list[float] = []
        self.rewards: list[float] = []

    def reset_episode_metrics(self) -> None:
        """Reset metrics at the start of each episode."""
        self.current_hour: int = 0
        self.episode_reward: float = 0.0
        self.episode_total_cost: float = 0.0
        self.episode_baseline_cost: float = 0.0
        self.episode_baseline_cost_off: float = 0.0

        # Agent job metrics (episode)
        self.episode_jobs_submitted: int = 0
        self.episode_jobs_completed: int = 0
        self.episode_total_job_wait_time: int = 0
        self.episode_max_queue_size_reached: int = 0
        self.episode_max_backlog_size_reached: int = 0
        self.episode_jobs_dropped: int = 0
        self.episode_jobs_rejected_queue_full: int = 0

        # Baseline job metrics (episode)
        self.episode_baseline_jobs_submitted: int = 0
        self.episode_baseline_jobs_completed: int = 0
        self.episode_baseline_total_job_wait_time: int = 0
        self.episode_baseline_max_queue_size_reached: int = 0
        self.episode_baseline_max_backlog_size_reached: int = 0
        self.episode_baseline_jobs_dropped: int = 0
        self.episode_baseline_jobs_rejected_queue_full: int = 0

        # Time series data for plotting (episode)
        self.episode_on_nodes = []
        self.episode_used_nodes = []
        self.episode_used_cores = []
        self.episode_baseline_used_nodes = []
        self.episode_baseline_used_cores = []
        self.episode_job_queue_sizes = []
        self.episode_price_stats = []

        self.episode_eff_rewards: list[float] = []
        self.episode_price_rewards: list[float] = []
        self.episode_idle_penalties: list[float] = []
        self.episode_job_age_penalties: list[float] = []
        self.episode_rewards: list[float] = []
        self.episode_running_jobs_counts: list[int] = []

    def record_episode_completion(self, current_episode: int) -> dict[str, float | int]:
        """
        Record episode costs and metrics for long-term analysis.

        Args:
            current_episode: Current episode number

        Returns:
            Dictionary with episode data
        """
        # Calculate average wait times
        avg_wait_time: float = (
            self.episode_total_job_wait_time / self.episode_jobs_completed
            if self.episode_jobs_completed > 0
            else 0.0
        )
        baseline_avg_wait_time: float = (
            self.episode_baseline_total_job_wait_time / self.episode_baseline_jobs_completed
            if self.episode_baseline_jobs_completed > 0
            else 0.0
        )

        # Calculate completion rates
        completion_rate: float = (
            (self.episode_jobs_completed / self.episode_jobs_submitted * 100)
            if self.episode_jobs_submitted > 0
            else 0.0
        )
        baseline_completion_rate: float = (
            (self.episode_baseline_jobs_completed / self.episode_baseline_jobs_submitted * 100)
            if self.episode_baseline_jobs_submitted > 0
            else 0.0
        )

        drop_rate: float = (
            (self.episode_jobs_dropped / self.episode_jobs_submitted * 100)
            if self.episode_jobs_submitted
            else 0.0
        )
        baseline_drop_rate: float = (
            (self.episode_baseline_jobs_dropped / self.episode_baseline_jobs_submitted * 100)
            if self.episode_baseline_jobs_submitted
            else 0.0
        )

        episode_data: dict[str, float | int] = {
            'episode': current_episode,
            'agent_cost': self.episode_total_cost,
            'baseline_cost': self.episode_baseline_cost,
            'baseline_cost_off': self.episode_baseline_cost_off,
            'savings_vs_baseline': self.episode_baseline_cost - self.episode_total_cost,
            'savings_vs_baseline_off': self.episode_baseline_cost_off - self.episode_total_cost,
            'savings_pct_baseline': ((self.episode_baseline_cost - self.episode_total_cost) / self.episode_baseline_cost) * 100 if self.episode_baseline_cost > 0 else 0.0,
            'savings_pct_baseline_off': ((self.episode_baseline_cost_off - self.episode_total_cost) / self.episode_baseline_cost_off) * 100 if self.episode_baseline_cost_off > 0 else 0.0,
            'total_reward': self.episode_reward,
            # Agent job metrics
            'jobs_submitted': self.episode_jobs_submitted,
            'jobs_completed': self.episode_jobs_completed,
            'avg_wait_time': avg_wait_time,
            'completion_rate': completion_rate,
            'max_queue_size': self.episode_max_queue_size_reached,
            'max_backlog_size': self.episode_max_backlog_size_reached,
            # Baseline job metrics
            'baseline_jobs_submitted': self.episode_baseline_jobs_submitted,
            'baseline_jobs_completed': self.episode_baseline_jobs_completed,
            'baseline_avg_wait_time': baseline_avg_wait_time,
            'baseline_completion_rate': baseline_completion_rate,
            'baseline_max_queue_size': self.episode_baseline_max_queue_size_reached,
            'baseline_max_backlog_size': self.episode_baseline_max_backlog_size_reached,
            # Drop metrics
            "jobs_dropped": self.episode_jobs_dropped,
            "drop_rate": drop_rate,
            "jobs_rejected_queue_full": self.episode_jobs_rejected_queue_full,
            "baseline_jobs_dropped": self.episode_baseline_jobs_dropped,
            "baseline_drop_rate": baseline_drop_rate,
            "baseline_jobs_rejected_queue_full": self.episode_baseline_jobs_rejected_queue_full,
        }
        self.episode_costs.append(episode_data)
        return episode_data
