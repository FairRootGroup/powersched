from src.config import COST_IDLE_MW, COST_USED_MW, CORES_PER_NODE, MAX_NODES
from stable_baselines3.common.callbacks import BaseCallback


class ComputeClusterCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        env = self.training_env.envs[0].unwrapped
        dones = self.locals.get("dones")
        if dones is None or not bool(dones[0]) or not env.metrics.episode_costs:
            return True

        episode_data = env.metrics.episode_costs[-1]

        self.logger.record("metrics/cost", float(episode_data["agent_cost"]))
        self.logger.record("metrics/savings", float(episode_data["savings_vs_baseline"]))
        self.logger.record("metrics/savings_off", float(episode_data["savings_vs_baseline_off"]))
        self.logger.record("metrics/baseline_cost", float(episode_data["baseline_cost"]))
        self.logger.record("metrics/baseline_cost_off", float(episode_data["baseline_cost_off"]))

        # Job metrics (agent)
        self.logger.record("metrics/jobs_submitted", int(episode_data["jobs_submitted"]))
        self.logger.record("metrics/jobs_launched", int(episode_data["jobs_launched"]))
        self.logger.record("metrics/jobs_completed", int(episode_data["jobs_completed"]))
        self.logger.record("metrics/completion_rate", float(episode_data["completion_rate"]))
        self.logger.record("metrics/avg_wait_hours", float(episode_data["avg_wait_time"]))
        self.logger.record("metrics/on_nodes", int(episode_data.get("on_nodes_end", 0)))
        self.logger.record("metrics/used_nodes", int(episode_data.get("used_nodes_end", 0)))
        self.logger.record("metrics/max_queue_size", int(episode_data["max_queue_size"]))
        self.logger.record("metrics/max_backlog_size", int(episode_data["max_backlog_size"]))
        self.logger.record("metrics/max_drop_streak", int(episode_data.get("max_drop_streak", 0)))
        self.logger.record("metrics/pending_jobs_end", int(episode_data.get("pending_jobs_end", 0)))
        self.logger.record("metrics/pending_core_hours_end", float(episode_data.get("pending_core_hours_end", 0.0)))
        self.logger.record("metrics/overdue_jobs_end", int(episode_data.get("overdue_jobs_end", 0)))
        self.logger.record("metrics/jobs_dropped", int(episode_data["jobs_dropped"]))
        self.logger.record("metrics/jobs_flushed", int(episode_data.get("jobs_flushed", 0)))
        self.logger.record("metrics/jobs_lost_total", int(episode_data["jobs_lost_total"]))
        self.logger.record("metrics/loss_rate", float(episode_data["loss_rate"]))
        self.logger.record("metrics/jobs_rejected_queue_full", int(episode_data["jobs_rejected_queue_full"]))

        # Job metrics (baseline)
        self.logger.record("metrics/baseline_jobs_submitted", int(episode_data["baseline_jobs_submitted"]))
        self.logger.record("metrics/baseline_jobs_launched", int(episode_data["baseline_jobs_launched"]))
        self.logger.record("metrics/baseline_jobs_completed", int(episode_data["baseline_jobs_completed"]))
        self.logger.record("metrics/baseline_completion_rate", float(episode_data["baseline_completion_rate"]))
        self.logger.record("metrics/baseline_avg_wait_hours", float(episode_data["baseline_avg_wait_time"]))
        self.logger.record("metrics/baseline_max_queue_size", int(episode_data["baseline_max_queue_size"]))
        self.logger.record("metrics/baseline_max_backlog_size", int(episode_data["baseline_max_backlog_size"]))
        self.logger.record("metrics/baseline_jobs_dropped", int(episode_data["baseline_jobs_dropped"]))
        self.logger.record("metrics/baseline_jobs_flushed", int(episode_data.get("baseline_jobs_flushed", 0)))
        self.logger.record("metrics/baseline_jobs_lost_total", int(episode_data["baseline_jobs_lost_total"]))
        self.logger.record("metrics/baseline_loss_rate", float(episode_data["baseline_loss_rate"]))
        self.logger.record("metrics/baseline_jobs_rejected_queue_full", int(episode_data["baseline_jobs_rejected_queue_full"]))

        # Proportional (per-core) power metrics
        self.logger.record("metrics/prop_power_mwh", float(episode_data["agent_prop_power_mwh"]))
        self.logger.record("metrics/baseline_prop_power_mwh", float(episode_data["baseline_prop_power_mwh"]))
        self.logger.record("metrics/baseline_off_prop_power_mwh", float(episode_data["baseline_off_prop_power_mwh"]))
        self.logger.record(
            "metrics/savings_prop_power_vs_baseline_off",
            float(episode_data["baseline_off_prop_power_mwh"]) - float(episode_data["agent_prop_power_mwh"]),
        )
        self.logger.record(
            "metrics/savings_prop_cost_vs_baseline_off",
            float(episode_data["savings_prop_cost_vs_baseline_off"]),
        )

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
