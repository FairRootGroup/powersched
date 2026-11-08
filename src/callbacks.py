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
        # Every env reports its finished episode's metrics in that step's info (environment.py),
        # so this works for any number of envs, in-process or in subprocesses.
        for info in self.locals.get("infos", ()):
            episode_data = info.get("episode_metrics")
            if episode_data is not None:
                self._record_episode(episode_data)
        return True

    def _record_episode(self, episode_data: dict) -> None:
        """Log one finished episode. record_mean averages over all episodes since the last dump."""
        self.logger.record_mean("metrics/cost", float(episode_data["agent_cost"]))
        self.logger.record_mean("metrics/savings", float(episode_data["savings_vs_baseline"]))
        self.logger.record_mean("metrics/savings_off", float(episode_data["savings_vs_baseline_off"]))
        self.logger.record_mean("metrics/baseline_cost", float(episode_data["baseline_cost"]))
        self.logger.record_mean("metrics/baseline_cost_off", float(episode_data["baseline_cost_off"]))

        # Job metrics (agent)
        self.logger.record_mean("metrics/jobs_submitted", int(episode_data["jobs_submitted"]))
        self.logger.record_mean("metrics/jobs_launched", int(episode_data["jobs_launched"]))
        self.logger.record_mean("metrics/jobs_completed", int(episode_data["jobs_completed"]))
        self.logger.record_mean("metrics/completion_rate", float(episode_data["completion_rate"]))
        self.logger.record_mean("metrics/avg_wait_hours", float(episode_data["avg_wait_time"]))
        self.logger.record_mean("metrics/on_nodes", int(episode_data.get("on_nodes_end", 0)))
        self.logger.record_mean("metrics/used_nodes", int(episode_data.get("used_nodes_end", 0)))
        self.logger.record_mean("metrics/max_queue_size", int(episode_data["max_queue_size"]))
        self.logger.record_mean("metrics/max_drop_streak", int(episode_data.get("max_drop_streak", 0)))
        self.logger.record_mean("metrics/pending_jobs_end", int(episode_data.get("pending_jobs_end", 0)))
        self.logger.record_mean("metrics/pending_core_hours_end", float(episode_data.get("pending_core_hours_end", 0.0)))
        self.logger.record_mean("metrics/overdue_jobs_end", int(episode_data.get("overdue_jobs_end", 0)))
        self.logger.record_mean("metrics/jobs_dropped", int(episode_data["jobs_dropped"]))
        self.logger.record_mean("metrics/jobs_flushed", int(episode_data.get("jobs_flushed", 0)))
        self.logger.record_mean("metrics/jobs_lost_total", int(episode_data["jobs_lost_total"]))
        self.logger.record_mean("metrics/loss_rate", float(episode_data["loss_rate"]))
        self.logger.record_mean("metrics/jobs_rejected_queue_full", int(episode_data["jobs_rejected_queue_full"]))

        # Job metrics (baseline)
        self.logger.record_mean("metrics/baseline_jobs_submitted", int(episode_data["baseline_jobs_submitted"]))
        self.logger.record_mean("metrics/baseline_jobs_launched", int(episode_data["baseline_jobs_launched"]))
        self.logger.record_mean("metrics/baseline_jobs_completed", int(episode_data["baseline_jobs_completed"]))
        self.logger.record_mean("metrics/baseline_completion_rate", float(episode_data["baseline_completion_rate"]))
        self.logger.record_mean("metrics/baseline_avg_wait_hours", float(episode_data["baseline_avg_wait_time"]))
        self.logger.record_mean("metrics/baseline_max_queue_size", int(episode_data["baseline_max_queue_size"]))
        self.logger.record_mean("metrics/baseline_jobs_dropped", int(episode_data["baseline_jobs_dropped"]))
        self.logger.record_mean("metrics/baseline_jobs_flushed", int(episode_data.get("baseline_jobs_flushed", 0)))
        self.logger.record_mean("metrics/baseline_jobs_lost_total", int(episode_data["baseline_jobs_lost_total"]))
        self.logger.record_mean("metrics/baseline_loss_rate", float(episode_data["baseline_loss_rate"]))
        self.logger.record_mean("metrics/baseline_jobs_rejected_queue_full", int(episode_data["baseline_jobs_rejected_queue_full"]))

        # Proportional (per-core) power metrics
        self.logger.record_mean("metrics/prop_power_mwh", float(episode_data["agent_prop_power_mwh"]))
        self.logger.record_mean("metrics/baseline_prop_power_mwh", float(episode_data["baseline_prop_power_mwh"]))
        self.logger.record_mean("metrics/baseline_off_prop_power_mwh", float(episode_data["baseline_off_prop_power_mwh"]))
        self.logger.record_mean(
            "metrics/savings_prop_power_vs_baseline_off",
            float(episode_data["baseline_off_prop_power_mwh"]) - float(episode_data["agent_prop_power_mwh"]),
        )
        self.logger.record_mean(
            "metrics/savings_prop_cost_vs_baseline_off",
            float(episode_data["savings_prop_cost_vs_baseline_off"]),
        )

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
