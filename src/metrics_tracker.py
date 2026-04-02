"""Metrics tracking and episode recording for the PowerSched environment."""

from src.config import COST_IDLE_MW, COST_USED_MW, CORES_PER_NODE, MAX_NODES


class MetricsTracker:
    """Tracks metrics throughout training episodes."""

    @staticmethod
    def _effective_mean_price(total_cost: float, total_power_mwh: float) -> float:
        """Effective mean price in €/MWh, weighted by consumed energy."""
        return (total_cost / total_power_mwh) if total_power_mwh > 0.0 else 0.0

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        """Safe division; returns NaN when denominator is not positive."""
        return (numerator / denominator) if denominator > 0.0 else float("nan")

    @staticmethod
    def _jobs_lost_total(dropped: int, flushed: int) -> int:
        """Total lost jobs, including regular drops and streak-triggered flushes."""
        return int(dropped) + int(flushed)

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
        self.oracle_cost: float = 0.0
        self.oracle_contiguous_cost: float = 0.0
        self.oracle_contiguous_unscheduled: int = 0
        self.oracle_contiguous_spillover: int = 0
        self.total_power_consumption_mwh: float = 0.0
        self.baseline_power_consumption_mwh: float = 0.0
        self.baseline_power_consumption_off_mwh: float = 0.0

        # Agent job metrics (cumulative across episodes)
        self.jobs_submitted: int = 0
        self.jobs_launched: int = 0
        self.jobs_completed: int = 0
        self.total_job_wait_time: int = 0
        self.total_job_wait_time_launch: int = 0
        self.max_queue_size_reached: int = 0
        self.max_backlog_size_reached: int = 0
        self.jobs_dropped: int = 0
        self.jobs_flushed: int = 0
        self.jobs_rejected_queue_full: int = 0

        # Baseline job metrics (cumulative across episodes)
        self.baseline_jobs_submitted: int = 0
        self.baseline_jobs_launched: int = 0
        self.baseline_jobs_completed: int = 0
        self.baseline_total_job_wait_time: int = 0
        self.baseline_total_job_wait_time_launch: int = 0
        self.baseline_max_queue_size_reached: int = 0
        self.baseline_max_backlog_size_reached: int = 0
        self.baseline_jobs_dropped: int = 0
        self.baseline_jobs_flushed: int = 0
        self.baseline_jobs_rejected_queue_full: int = 0

        # Time series data for plotting (cumulative)
        self.on_nodes: list[int] = []
        self.used_nodes: list[int] = []
        self.job_queue_sizes: list[int] = []
        self.price_stats: list[float] = []
        self.launched_jobs_counts: list[int] = []
        self.launched_cores: list[int] = []

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
        self.episode_oracle_cost: float = 0.0
        self.episode_oracle_contiguous_cost: float = 0.0
        self.episode_oracle_contiguous_unscheduled: int = 0
        self.episode_oracle_contiguous_spillover: int = 0
        self.episode_total_power_consumption_mwh: float = 0.0
        self.episode_baseline_power_consumption_mwh: float = 0.0
        self.episode_baseline_power_consumption_off_mwh: float = 0.0

        # Agent job metrics (episode)
        self.episode_jobs_submitted: int = 0
        self.episode_jobs_launched: int = 0
        self.episode_jobs_completed: int = 0
        self.episode_total_job_wait_time: int = 0
        self.episode_total_job_wait_time_launch: int = 0
        self.episode_max_queue_size_reached: int = 0
        self.episode_max_backlog_size_reached: int = 0
        self.episode_jobs_dropped: int = 0
        self.episode_jobs_flushed: int = 0
        self.episode_max_drop_streak: int = 0
        self.episode_jobs_rejected_queue_full: int = 0

        # Baseline job metrics (episode)
        self.episode_baseline_jobs_submitted: int = 0
        self.episode_baseline_jobs_launched: int = 0
        self.episode_baseline_jobs_completed: int = 0
        self.episode_baseline_total_job_wait_time: int = 0
        self.episode_baseline_total_job_wait_time_launch: int = 0
        self.episode_baseline_max_queue_size_reached: int = 0
        self.episode_baseline_max_backlog_size_reached: int = 0
        self.episode_baseline_jobs_dropped: int = 0
        self.episode_baseline_jobs_flushed: int = 0
        self.episode_baseline_jobs_rejected_queue_full: int = 0

        # End-of-episode pending-work snapshot.
        # These are updated every step so record_episode_completion() can report what
        # backlog remains when the episode terminates.
        self.episode_pending_jobs_end: int = 0
        self.episode_pending_core_demand_end: float = 0.0
        self.episode_pending_core_hours_end: float = 0.0
        self.episode_overdue_jobs_end: int = 0
        self.episode_overdue_age_core_hours_end: float = 0.0

        # Time series data for plotting (episode)
        self.episode_on_nodes: list[int] = []
        self.episode_used_nodes: list[int] = []
        self.episode_used_cores: list[int] = []
        self.episode_baseline_used_nodes: list[int] = []
        self.episode_baseline_used_cores: list[int] = []
        self.episode_job_queue_sizes: list[int] = []
        self.episode_price_stats: list[float] = []
        self.episode_launched_jobs_counts: list[int] = []
        self.episode_launched_cores: list[int] = []

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
        # Scheduling wait is measured when work launches, not when it eventually finishes.
        # That makes "defer into cheap hours" visible immediately instead of only after completion.
        avg_wait_time: float = (
            self.episode_total_job_wait_time_launch / self.episode_jobs_launched
            if self.episode_jobs_launched > 0
            else 0.0
        )
        baseline_avg_wait_time: float = (
            self.episode_baseline_total_job_wait_time_launch / self.episode_baseline_jobs_launched
            if self.episode_baseline_jobs_launched > 0
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
        jobs_lost_total = self._jobs_lost_total(self.episode_jobs_dropped, self.episode_jobs_flushed)
        flush_rate: float = (
            (self.episode_jobs_flushed / self.episode_jobs_submitted * 100)
            if self.episode_jobs_submitted
            else 0.0
        )
        loss_rate: float = (
            (jobs_lost_total / self.episode_jobs_submitted * 100)
            if self.episode_jobs_submitted
            else 0.0
        )
        baseline_drop_rate: float = (
            (self.episode_baseline_jobs_dropped / self.episode_baseline_jobs_submitted * 100)
            if self.episode_baseline_jobs_submitted
            else 0.0
        )
        baseline_jobs_lost_total = self._jobs_lost_total(
            self.episode_baseline_jobs_dropped,
            self.episode_baseline_jobs_flushed,
        )
        baseline_flush_rate: float = (
            (self.episode_baseline_jobs_flushed / self.episode_baseline_jobs_submitted * 100)
            if self.episode_baseline_jobs_submitted
            else 0.0
        )
        baseline_loss_rate: float = (
            (baseline_jobs_lost_total / self.episode_baseline_jobs_submitted * 100)
            if self.episode_baseline_jobs_submitted
            else 0.0
        )
        agent_mean_price: float = self._effective_mean_price(
            self.episode_total_cost, self.episode_total_power_consumption_mwh
        )
        baseline_mean_price: float = self._effective_mean_price(
            self.episode_baseline_cost, self.episode_baseline_power_consumption_mwh
        )
        baseline_off_mean_price: float = self._effective_mean_price(
            self.episode_baseline_cost_off, self.episode_baseline_power_consumption_off_mwh
        )
        agent_cost_per_1000_completed: float = self._safe_ratio(
            self.episode_total_cost * 1000.0, float(self.episode_jobs_completed)
        )
        baseline_cost_per_1000_completed: float = self._safe_ratio(
            self.episode_baseline_cost * 1000.0, float(self.episode_baseline_jobs_completed)
        )
        # baseline_off is a cost variant of baseline scheduling, so it uses the same completed-job count.
        baseline_off_cost_per_1000_completed: float = self._safe_ratio(
            self.episode_baseline_cost_off * 1000.0, float(self.episode_baseline_jobs_completed)
        )
        savings_vs_baseline: float = self.episode_baseline_cost - self.episode_total_cost
        savings_vs_baseline_off: float = self.episode_baseline_cost_off - self.episode_total_cost
        savings_vs_oracle: float = self.episode_total_cost - self.episode_oracle_cost

        # Proportional (per-core) power: idle_base for all on-nodes + compute delta scaled by core utilization.
        # Formula per step: COST_IDLE_MW * num_on + (COST_USED_MW - COST_IDLE_MW) * (cores_used / CORES_PER_NODE)
        _compute_delta_mw = COST_USED_MW - COST_IDLE_MW
        agent_prop_power_mwh: float = sum(
            COST_IDLE_MW * on + _compute_delta_mw * (cores / CORES_PER_NODE)
            for on, cores in zip(self.episode_on_nodes, self.episode_used_cores)
        )
        # Baseline always has all MAX_NODES on
        baseline_prop_power_mwh: float = sum(
            COST_IDLE_MW * MAX_NODES + _compute_delta_mw * (cores / CORES_PER_NODE)
            for cores in self.episode_baseline_used_cores
        )
        # Baseline_off: only used nodes are on (no idle nodes)
        baseline_off_prop_power_mwh: float = sum(
            COST_IDLE_MW * used + _compute_delta_mw * (cores / CORES_PER_NODE)
            for used, cores in zip(self.episode_baseline_used_nodes, self.episode_baseline_used_cores)
        )
        # Proportional cost: same as prop power but multiplied by price at each step
        agent_prop_cost: float = sum(
            (COST_IDLE_MW * on + _compute_delta_mw * (cores / CORES_PER_NODE)) * price
            for on, cores, price in zip(self.episode_on_nodes, self.episode_used_cores, self.episode_price_stats)
        )
        baseline_prop_cost: float = sum(
            (COST_IDLE_MW * MAX_NODES + _compute_delta_mw * (cores / CORES_PER_NODE)) * price
            for cores, price in zip(self.episode_baseline_used_cores, self.episode_price_stats)
        )
        baseline_off_prop_cost: float = sum(
            (COST_IDLE_MW * used + _compute_delta_mw * (cores / CORES_PER_NODE)) * price
            for used, cores, price in zip(self.episode_baseline_used_nodes, self.episode_baseline_used_cores, self.episode_price_stats)
        )
        savings_prop_cost_vs_baseline: float = baseline_prop_cost - agent_prop_cost
        savings_prop_cost_vs_baseline_off: float = baseline_off_prop_cost - agent_prop_cost
        dropped_jobs_per_saved_euro: float = self._safe_ratio(
            float(self.episode_jobs_dropped), savings_vs_baseline
        ) if savings_vs_baseline > 0.0 else float("nan")
        dropped_jobs_per_saved_euro_off: float = self._safe_ratio(
            float(self.episode_jobs_dropped), savings_vs_baseline_off
        ) if savings_vs_baseline_off > 0.0 else float("nan")

        self.oracle_cost += self.episode_oracle_cost
        self.oracle_contiguous_cost += self.episode_oracle_contiguous_cost
        self.oracle_contiguous_unscheduled += self.episode_oracle_contiguous_unscheduled
        self.oracle_contiguous_spillover += self.episode_oracle_contiguous_spillover

        episode_data: dict[str, float | int] = {
            'episode': current_episode,
            'agent_cost': self.episode_total_cost,
            'baseline_cost': self.episode_baseline_cost,
            'baseline_cost_off': self.episode_baseline_cost_off,
            'oracle_cost': self.episode_oracle_cost,
            'oracle_contiguous_cost': self.episode_oracle_contiguous_cost,
            'oracle_contiguous_unscheduled': self.episode_oracle_contiguous_unscheduled,
            'oracle_contiguous_spillover': self.episode_oracle_contiguous_spillover,
            'savings_vs_oracle': savings_vs_oracle,
            'agent_power_consumption_mwh': self.episode_total_power_consumption_mwh,
            'baseline_power_consumption_mwh': self.episode_baseline_power_consumption_mwh,
            'baseline_power_consumption_off_mwh': self.episode_baseline_power_consumption_off_mwh,
            'agent_prop_power_mwh': agent_prop_power_mwh,
            'baseline_prop_power_mwh': baseline_prop_power_mwh,
            'baseline_off_prop_power_mwh': baseline_off_prop_power_mwh,
            'agent_prop_cost': agent_prop_cost,
            'baseline_prop_cost': baseline_prop_cost,
            'baseline_off_prop_cost': baseline_off_prop_cost,
            'savings_prop_cost_vs_baseline': savings_prop_cost_vs_baseline,
            'savings_prop_cost_vs_baseline_off': savings_prop_cost_vs_baseline_off,
            'agent_mean_price': agent_mean_price,
            'baseline_mean_price': baseline_mean_price,
            'baseline_off_mean_price': baseline_off_mean_price,
            'savings_vs_baseline': savings_vs_baseline,
            'savings_vs_baseline_off': savings_vs_baseline_off,
            'savings_pct_baseline': ((self.episode_baseline_cost - self.episode_total_cost) / self.episode_baseline_cost) * 100 if self.episode_baseline_cost > 0 else 0.0,
            'savings_pct_baseline_off': ((self.episode_baseline_cost_off - self.episode_total_cost) / self.episode_baseline_cost_off) * 100 if self.episode_baseline_cost_off > 0 else 0.0,
            'agent_cost_per_1000_completed_jobs': agent_cost_per_1000_completed,
            'baseline_cost_per_1000_completed_jobs': baseline_cost_per_1000_completed,
            'baseline_off_cost_per_1000_completed_jobs': baseline_off_cost_per_1000_completed,
            'agent_dropped_jobs_per_saved_euro': dropped_jobs_per_saved_euro,
            'agent_dropped_jobs_per_saved_euro_off': dropped_jobs_per_saved_euro_off,
            'total_reward': self.episode_reward,
            # Agent job metrics
            'jobs_submitted': self.episode_jobs_submitted,
            'jobs_launched': self.episode_jobs_launched,
            'jobs_completed': self.episode_jobs_completed,
            'avg_wait_time': avg_wait_time,
            'completion_rate': completion_rate,
            'on_nodes_end': int(self.episode_on_nodes[-1]) if self.episode_on_nodes else 0,
            'used_nodes_end': int(self.episode_used_nodes[-1]) if self.episode_used_nodes else 0,
            'max_queue_size': self.episode_max_queue_size_reached,
            'max_backlog_size': self.episode_max_backlog_size_reached,
            'max_drop_streak': self.episode_max_drop_streak,
            'pending_jobs_end': self.episode_pending_jobs_end,
            'pending_core_demand_end': self.episode_pending_core_demand_end,
            'pending_core_hours_end': self.episode_pending_core_hours_end,
            'overdue_jobs_end': self.episode_overdue_jobs_end,
            'overdue_age_core_hours_end': self.episode_overdue_age_core_hours_end,
            # Baseline job metrics
            'baseline_jobs_submitted': self.episode_baseline_jobs_submitted,
            'baseline_jobs_launched': self.episode_baseline_jobs_launched,
            'baseline_jobs_completed': self.episode_baseline_jobs_completed,
            'baseline_avg_wait_time': baseline_avg_wait_time,
            'baseline_completion_rate': baseline_completion_rate,
            'baseline_max_queue_size': self.episode_baseline_max_queue_size_reached,
            'baseline_max_backlog_size': self.episode_baseline_max_backlog_size_reached,
            # Loss metrics: includes age expirations, queue-full rejections, and streak-triggered flushes.
            "jobs_dropped": self.episode_jobs_dropped,
            "jobs_flushed": self.episode_jobs_flushed,
            "jobs_lost_total": jobs_lost_total,
            "drop_rate": drop_rate,
            "flush_rate": flush_rate,
            "loss_rate": loss_rate,
            "jobs_rejected_queue_full": self.episode_jobs_rejected_queue_full,
            "baseline_jobs_dropped": self.episode_baseline_jobs_dropped,
            "baseline_jobs_flushed": self.episode_baseline_jobs_flushed,
            "baseline_jobs_lost_total": baseline_jobs_lost_total,
            "baseline_drop_rate": baseline_drop_rate,
            "baseline_flush_rate": baseline_flush_rate,
            "baseline_loss_rate": baseline_loss_rate,
            "baseline_jobs_rejected_queue_full": self.episode_baseline_jobs_rejected_queue_full,
        }
        self.episode_costs.append(episode_data)
        return episode_data
