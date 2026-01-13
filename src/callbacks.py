from src.config import EPISODE_HOURS
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
            self.logger.record("metrics/cost", env.metrics.total_cost)
            self.logger.record("metrics/savings", env.metrics.baseline_cost - env.metrics.total_cost)
            self.logger.record("metrics/savings_off", env.metrics.baseline_cost_off - env.metrics.total_cost)

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
