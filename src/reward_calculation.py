"""Reward calculation and normalization logic for the PowerSched environment."""

import numpy as np
from src.config import (
    COST_IDLE_MW, COST_USED_MW, PENALTY_IDLE_NODE, PENALTY_WAITING_JOB,
    PENALTY_DROPPED_JOB, MAX_NODES, MAX_NEW_JOBS_PER_HOUR, MAX_JOB_AGE, MAX_QUEUE_SIZE
)


def power_cost(num_used_nodes, num_idle_nodes, current_price):
    """
    Calculate power cost based on node usage and current electricity price.

    Args:
        num_used_nodes: Number of nodes with jobs running
        num_idle_nodes: Number of idle (on but unused) nodes
        current_price: Current electricity price

    Returns:
        Total power cost
    """
    idle_cost = COST_IDLE_MW * current_price * num_idle_nodes
    usage_cost = COST_USED_MW * current_price * num_used_nodes
    total_cost = idle_cost + usage_cost
    return total_cost


class RewardCalculator:
    """Calculates rewards with pre-computed normalization bounds."""

    def __init__(self, prices):
        """
        Initialize reward calculator with normalization bounds.

        Args:
            prices: Prices object with MIN_PRICE and MAX_PRICE attributes
        """
        self.prices = prices
        self._compute_bounds()

    def _compute_bounds(self):
        """Compute min/max bounds for reward normalization."""
        # Efficiency bounds
        cost_for_min_efficiency = power_cost(0, MAX_NODES, self.prices.MAX_PRICE)
        cost_for_max_efficiency = power_cost(MAX_NODES, 0, self.prices.MIN_PRICE)

        self._min_efficiency_reward = self._reward_efficiency(0, cost_for_min_efficiency)
        self._max_efficiency_reward = max(1.0, self._reward_efficiency(MAX_NODES, cost_for_max_efficiency))

        # Price bounds
        self._min_price_reward = 0
        self._max_price_reward = self._reward_price(self.prices.MIN_PRICE, self.prices.MAX_PRICE, MAX_NEW_JOBS_PER_HOUR)

        # Idle penalty bounds
        self._min_idle_penalty = self._penalty_idle(0)
        self._max_idle_penalty = self._penalty_idle(MAX_NODES)

        # Job age penalty bounds
        self._min_job_age_penalty = 0.0
        self._max_job_age_penalty = PENALTY_WAITING_JOB * MAX_JOB_AGE * MAX_QUEUE_SIZE

    @staticmethod
    def _normalize(current, minimum, maximum):
        """Normalize a value to [0, 1] range."""
        if maximum == minimum:
            return 0.5  # Avoid division by zero
        return (current - minimum) / (maximum - minimum)

    @staticmethod
    def _reward_efficiency(num_used_nodes, total_cost):
        """Calculate efficiency reward: work done per unit cost."""
        return num_used_nodes / (total_cost + 1e-6)

    def _reward_efficiency_normalized(self, num_used_nodes, num_idle_nodes, num_unprocessed_jobs, total_cost):
        """Calculate normalized efficiency reward [0, 1]."""
        if num_used_nodes + num_idle_nodes == 0:
            if num_unprocessed_jobs == 0:
                return 1
            else:
                return np.clip(1.0 / np.log1p(num_unprocessed_jobs), a_min=None, a_max=1.0)
        else:
            current_reward = self._reward_efficiency(num_used_nodes, total_cost)
            return self._normalize(current_reward, self._min_efficiency_reward, self._max_efficiency_reward)

    def _reward_price(self, current_price, average_future_price, num_processed_jobs):
        """Calculate price-based reward for scheduling jobs at favorable prices."""
        history_avg, future_avg = self.prices.get_price_context()

        if history_avg is not None:
            # We have some history - use both past and future
            context_avg = (history_avg + future_avg) / 2
            price_diff = context_avg - current_price
        else:
            # No history yet - fall back to just using future prices
            price_diff = average_future_price - current_price

        return price_diff * num_processed_jobs

    def _reward_price_normalized(self, current_price, average_future_price, num_processed_jobs):
        """Calculate normalized price reward [0, 1]."""
        if num_processed_jobs == 0:
            return 0
        current_reward = self._reward_price(current_price, average_future_price, num_processed_jobs)
        return self._normalize(current_reward, self._min_price_reward, self._max_price_reward)

    @staticmethod
    def _penalty_idle(num_idle_nodes):
        """Calculate penalty for idle nodes."""
        return PENALTY_IDLE_NODE * num_idle_nodes

    def _penalty_idle_normalized(self, num_idle_nodes):
        """Calculate normalized idle penalty [-1, 0]."""
        current_penalty = self._penalty_idle(num_idle_nodes)
        normalized_penalty = -self._normalize(current_penalty, self._min_idle_penalty, self._max_idle_penalty)
        return np.clip(normalized_penalty, -1, 0)

    @staticmethod
    def _penalty_job_age(num_off_nodes, job_queue_2d):
        """Calculate penalty for jobs waiting in queue when nodes are off."""
        job_age_penalty = 0
        if num_off_nodes > 0:
            for job in job_queue_2d:
                job_duration, job_age, _, _ = job
                if job_duration > 0:
                    job_age_penalty += PENALTY_WAITING_JOB * job_age
        return job_age_penalty

    def _penalty_job_age_normalized(self, num_off_nodes, job_queue_2d):
        """Calculate normalized job age penalty [-1, 0]."""
        current_penalty = self._penalty_job_age(num_off_nodes, job_queue_2d)
        normalized_penalty = -self._normalize(current_penalty, self._min_job_age_penalty, self._max_job_age_penalty)
        return np.clip(normalized_penalty, -1, 0)

    def _reward_energy_efficiency_normalized(self, num_used_nodes: int, num_idle_nodes: int) -> float:
        '''Redefine meaning of "efficiency". Use purely as "energy efficiency", aka: How much of the energy (in MW) which is currently needed, gets used for work.
        NOTE: Original efficiency function was doing 3 things at once. 1. Handled Blackout logic, with (2.) penalty-ish reward delay for unprocessed jobs, while blackout. 
        But this log1p function would start to become "harsh" only for a very high number of unprocessed. This rewarded shutting everything off. 
        3. rewarded used/cost, but cost was defined in units of price. Price reward should handle this solely, otherwise double counting. 
        Hence, here new efficiency definition.'''
        used = float(num_used_nodes)
        idle = float(num_idle_nodes)
        p_used = float(COST_USED_MW)
        p_idle = float(COST_IDLE_MW)

        total_work = used * p_used + idle * p_idle
        if total_work <= 0.0:
            return 0.0  # nothing on => no "efficiency" signal
        return float(np.clip((used * p_used) / total_work, 0.0, 1.0))
    

    def _blackout_term(self, num_used_nodes: int, num_idle_nodes: int, num_unprocessed_jobs: int) -> float:
        """
        Reward/penalty for full blackout (all nodes off).
        If queue is empty, reward the blackout. If jobs are waiting, apply a smooth penalty in [-1, 0].
        """
        BLACKOUT_QUEUE_THRESHOLD = 10  # jobs waiting until penalty saturates to -1
        SATURATION_FACTOR = 2
        on_nodes = int(num_used_nodes) + int(num_idle_nodes)
        queue_waiting = int(num_unprocessed_jobs)

        if on_nodes != 0:
            return 0.0  # only care about full blackout

        if queue_waiting <= 0:
            return 1.0  # correct blackout

        ratio = queue_waiting / float(max(BLACKOUT_QUEUE_THRESHOLD, 1))
        penalty = np.exp(-ratio * SATURATION_FACTOR) - 1.0
        return float(np.clip(penalty, -1.0, 0.0))

    def calculate(self, num_used_nodes, num_idle_nodes, current_price, average_future_price,
                  num_off_nodes, num_processed_jobs, num_node_changes, job_queue_2d,
                  num_unprocessed_jobs, weights, num_dropped_this_step, env_print):
        """
        Calculate total reward by aggregating weighted components.

        Args:
            num_used_nodes: Number of nodes with jobs running
            num_idle_nodes: Number of idle nodes
            current_price: Current electricity price
            average_future_price: Average predicted future price
            num_off_nodes: Number of offline nodes
            num_processed_jobs: Number of jobs launched this step
            num_node_changes: Number of node state changes
            job_queue_2d: 2D job queue array
            num_unprocessed_jobs: Number of jobs waiting in queue
            weights: Weights object with weight values
            num_dropped_this_step: Number of jobs dropped this step
            env_print: Print function for logging

        Returns:
            Tuple of (total reward, total cost, eff_reward_norm, price_reward_norm,
                      idle_penalty_norm, job_age_penalty_norm)
        """
        # 0. Energy efficiency. Reward calculation based on Workload (used nodes) (W) / Cost (C)
        total_cost = power_cost(num_used_nodes, num_idle_nodes, current_price)
        efficiency_reward_norm = self._reward_energy_efficiency_normalized(num_used_nodes, num_idle_nodes) + self._blackout_term(num_used_nodes, num_idle_nodes, num_unprocessed_jobs)
        efficiency_reward_weighted = weights.efficiency_weight * efficiency_reward_norm

        # 2. increase reward if jobs were scheduled in this step and the current price is below average
        price_reward_norm = self._reward_price_normalized(
            current_price, average_future_price, num_processed_jobs
        )
        price_reward_weighted = weights.price_weight * price_reward_norm

        # 3. penalize delayed jobs, more if they are older. but only if there are turned off nodes
        job_age_penalty_norm = self._penalty_job_age_normalized(num_off_nodes, job_queue_2d)
        job_age_penalty_weighted = weights.job_age_weight * job_age_penalty_norm

        # 5. penalty for idling nodes
        idle_penalty_norm = self._penalty_idle_normalized(num_idle_nodes)
        idle_penalty_weighted = weights.idle_weight * idle_penalty_norm

        # 6. penalty for dropped jobs (WIP - unnormalized, weighted)
        drop_penalty = min(0, PENALTY_DROPPED_JOB * num_dropped_this_step)
        drop_penalty_weighted = weights.drop_weight * drop_penalty


        reward = (
            efficiency_reward_weighted
            + price_reward_weighted
            + job_age_penalty_weighted
            + idle_penalty_weighted
            + drop_penalty_weighted
        )

        env_print(f"    > $$$TOTAL: {reward:.4f} = {efficiency_reward_weighted:.4f} + {price_reward_weighted:.4f} + {idle_penalty_weighted:.4f} + {job_age_penalty_weighted:.4f} + {drop_penalty_weighted:.4f}")
        env_print(f"    > step cost: €{total_cost:.4f}")

        return reward, total_cost, efficiency_reward_norm, price_reward_norm, idle_penalty_norm, job_age_penalty_norm
