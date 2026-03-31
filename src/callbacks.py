from src.config import EPISODE_HOURS, MAX_QUEUE_SIZE, COST_IDLE_MW, COST_USED_MW, CORES_PER_NODE, MAX_NODES
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
            self.logger.record("metrics/total_reward", env.metrics.episode_reward)
            self.logger.record("metrics/reward_eff", sum(env.metrics.episode_eff_rewards) / 100)
            self.logger.record("metrics/reward_price", sum(env.metrics.episode_price_rewards) / 100)
            self.logger.record("metrics/penalty_idle", sum(env.metrics.episode_idle_penalties) / 100)
            self.logger.record("metrics/penalty_job_age", sum(env.metrics.episode_job_age_penalties) / 100)
            self.logger.record("metrics/penalty_drop", sum(env.metrics.episode_drop_penalties))

            self.logger.record("metrics/cost", env.metrics.episode_total_cost)
            self.logger.record("metrics/savings", env.metrics.episode_baseline_cost - env.metrics.episode_total_cost)
            savings_off = env.metrics.episode_baseline_cost_off - env.metrics.episode_total_cost
            self.logger.record("metrics/savings_off", savings_off)
            savings_off_clean = savings_off if env.metrics.episode_jobs_dropped == 0 else 0.0
            self.logger.record("metrics/savings_off_clean", savings_off_clean)
            #self.logger.record("metrics/queue_fill_pct", env.metrics.episode_max_queue_size_reached / MAX_QUEUE_SIZE * 100)
            self.logger.record("metrics/bl_cost", env.metrics.episode_baseline_cost)
            self.logger.record("metrics/bl_cost_off", env.metrics.episode_baseline_cost_off)

            # Job metrics (agent)
            completion_rate = (env.metrics.episode_jobs_completed / env.metrics.episode_jobs_submitted * 100 if env.metrics.episode_jobs_submitted > 0 else 0.0)
            avg_wait = (env.metrics.episode_total_job_wait_time / env.metrics.episode_jobs_completed if env.metrics.episode_jobs_completed > 0 else 0.0)
            self.logger.record("metrics/jobs_submitted", env.metrics.episode_jobs_submitted)
            self.logger.record("metrics/jobs_completed", env.metrics.episode_jobs_completed)
            self.logger.record("metrics/completion_rate", completion_rate)
            self.logger.record("metrics/avg_wait_hours", avg_wait)
            self.logger.record("metrics/nodes_on", env.metrics.episode_on_nodes[-1])
            self.logger.record("metrics/nodes_used", env.metrics.episode_used_nodes[-1])
            self.logger.record("metrics/nodes_idle", env.metrics.episode_on_nodes[-1] - env.metrics.episode_used_nodes[-1])
            self.logger.record("metrics/nodes_off", MAX_NODES - env.metrics.episode_on_nodes[-1])
            self.logger.record("metrics/max_queue_size", env.metrics.episode_max_queue_size_reached)
            self.logger.record("metrics/max_backlog_size", env.metrics.episode_max_backlog_size_reached)
            self.logger.record("metrics/jobs_dropped", env.metrics.episode_jobs_dropped)
            self.logger.record("metrics/jobs_lost_total", env.metrics.episode_jobs_dropped)
            loss_rate = (env.metrics.episode_jobs_dropped / env.metrics.episode_jobs_submitted * 100 if env.metrics.episode_jobs_submitted > 0 else 0.0)
            self.logger.record("metrics/loss_rate", loss_rate)
            self.logger.record("metrics/jobs_rejected_queue_full", env.metrics.episode_jobs_rejected_queue_full)

            # Job metrics (baseline)
            baseline_completion_rate = (env.metrics.episode_baseline_jobs_completed / env.metrics.episode_baseline_jobs_submitted * 100 if env.metrics.episode_baseline_jobs_submitted > 0 else 0.0)
            baseline_avg_wait = (env.metrics.episode_baseline_total_job_wait_time / env.metrics.episode_baseline_jobs_completed if env.metrics.episode_baseline_jobs_completed > 0 else 0.0)
            self.logger.record("metrics/bl_jobs_submitted", env.metrics.episode_baseline_jobs_submitted)
            self.logger.record("metrics/bl_jobs_completed", env.metrics.episode_baseline_jobs_completed)
            self.logger.record("metrics/bl_completion_rate", baseline_completion_rate)
            self.logger.record("metrics/bl_avg_wait_hours", baseline_avg_wait)
            self.logger.record("metrics/bl_max_queue_size", env.metrics.episode_baseline_max_queue_size_reached)
            self.logger.record("metrics/bl_max_backlog_size", env.metrics.episode_baseline_max_backlog_size_reached)
            self.logger.record("metrics/bl_jobs_dropped", env.metrics.episode_baseline_jobs_dropped)
            self.logger.record("metrics/bl_jobs_lost_total", env.metrics.episode_baseline_jobs_dropped)
            baseline_loss_rate = (env.metrics.episode_baseline_jobs_dropped / env.metrics.episode_baseline_jobs_submitted * 100 if env.metrics.episode_baseline_jobs_submitted > 0 else 0.0)
            self.logger.record("metrics/bl_loss_rate", baseline_loss_rate)
            self.logger.record("metrics/bl_jobs_rejected_queue_full", env.metrics.episode_baseline_jobs_rejected_queue_full)

            # Proportional (per-core) power metrics
            _delta = COST_USED_MW - COST_IDLE_MW
            agent_prop_power = sum(
                COST_IDLE_MW * on + _delta * (cores / CORES_PER_NODE)
                for on, cores in zip(env.metrics.episode_on_nodes, env.metrics.episode_used_cores)
            )
            baseline_prop_power = sum(
                COST_IDLE_MW * MAX_NODES + _delta * (cores / CORES_PER_NODE)
                for cores in env.metrics.episode_baseline_used_cores
            )
            baseline_off_prop_power = sum(
                COST_IDLE_MW * used + _delta * (cores / CORES_PER_NODE)
                for used, cores in zip(env.metrics.episode_baseline_used_nodes, env.metrics.episode_baseline_used_cores)
            )
            self.logger.record("metrics/prop_power_mwh", agent_prop_power)
            self.logger.record("metrics/bl_prop_power_mwh", baseline_prop_power)
            self.logger.record("metrics/bl_off_prop_power_mwh", baseline_off_prop_power)
            self.logger.record("metrics/savings_prop_power_vs_baseline_off", baseline_off_prop_power - agent_prop_power)
            agent_prop_cost = sum(
                (COST_IDLE_MW * on + _delta * (cores / CORES_PER_NODE)) * price
                for on, cores, price in zip(env.metrics.episode_on_nodes, env.metrics.episode_used_cores, env.metrics.episode_price_stats)
            )
            baseline_off_prop_cost = sum(
                (COST_IDLE_MW * used + _delta * (cores / CORES_PER_NODE)) * price
                for used, cores, price in zip(env.metrics.episode_baseline_used_nodes, env.metrics.episode_baseline_used_cores, env.metrics.episode_price_stats)
            )
            self.logger.record("metrics/savings_prop_cost_vs_baseline_off", baseline_off_prop_cost - agent_prop_cost)

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
