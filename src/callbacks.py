from src.config import EPISODE_HOURS, MAX_QUEUE_SIZE
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
        if env.metrics.current_hour == EPISODE_HOURS-1:
            self.logger.record("metrics/cost", env.metrics.episode_total_cost)
            self.logger.record("metrics/savings", env.metrics.episode_baseline_cost - env.metrics.episode_total_cost)
            self.logger.record("metrics/savings_off", env.metrics.episode_baseline_cost_off - env.metrics.episode_total_cost)
            #self.logger.record("metrics/queue_fill_pct", env.metrics.episode_max_queue_size_reached / MAX_QUEUE_SIZE * 100)
            self.logger.record("metrics/baseline_cost", env.metrics.episode_baseline_cost)
            self.logger.record("metrics/baseline_cost_off", env.metrics.episode_baseline_cost_off)

            # Job metrics (agent)
            completion_rate = (env.metrics.episode_jobs_completed / env.metrics.episode_jobs_submitted * 100 if env.metrics.episode_jobs_submitted > 0 else 0.0)
            avg_wait = (env.metrics.episode_total_job_wait_time / env.metrics.episode_jobs_completed if env.metrics.episode_jobs_completed > 0 else 0.0)
            self.logger.record("metrics/jobs_submitted", env.metrics.episode_jobs_submitted)
            self.logger.record("metrics/jobs_completed", env.metrics.episode_jobs_completed)
            self.logger.record("metrics/completion_rate", completion_rate)
            self.logger.record("metrics/avg_wait_hours", avg_wait)
            self.logger.record("metrics/max_queue_size", env.metrics.episode_max_queue_size_reached)
            self.logger.record("metrics/max_backlog_size", env.metrics.episode_max_backlog_size_reached)
            self.logger.record("metrics/jobs_dropped", env.metrics.episode_jobs_dropped)
            self.logger.record("metrics/jobs_rejected_queue_full", env.metrics.episode_jobs_rejected_queue_full)

            # Job metrics (baseline)
            baseline_completion_rate = (env.metrics.episode_baseline_jobs_completed / env.metrics.episode_baseline_jobs_submitted * 100 if env.metrics.episode_baseline_jobs_submitted > 0 else 0.0)
            baseline_avg_wait = (env.metrics.episode_baseline_total_job_wait_time / env.metrics.episode_baseline_jobs_completed if env.metrics.episode_baseline_jobs_completed > 0 else 0.0)
            self.logger.record("metrics/baseline_jobs_submitted", env.metrics.episode_baseline_jobs_submitted)
            self.logger.record("metrics/baseline_jobs_completed", env.metrics.episode_baseline_jobs_completed)
            self.logger.record("metrics/baseline_completion_rate", baseline_completion_rate)
            self.logger.record("metrics/baseline_avg_wait_hours", baseline_avg_wait)
            self.logger.record("metrics/baseline_max_queue_size", env.metrics.episode_baseline_max_queue_size_reached)
            self.logger.record("metrics/baseline_max_backlog_size", env.metrics.episode_baseline_max_backlog_size_reached)
            self.logger.record("metrics/baseline_jobs_dropped", env.metrics.episode_baseline_jobs_dropped)
            self.logger.record("metrics/baseline_jobs_rejected_queue_full", env.metrics.episode_baseline_jobs_rejected_queue_full)

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
