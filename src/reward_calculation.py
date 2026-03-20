"""Reward calculation and normalization logic for the PowerSched environment."""

from collections.abc import Callable

import numpy as np

from src.config import (
    COST_IDLE_MW, COST_USED_MW, CORES_PER_NODE, PENALTY_IDLE_NODE,
    PENALTY_DROPPED_JOB, MAX_NODES, MAX_NEW_JOBS_PER_HOUR, WEEK_HOURS
)
from src.prices import Prices
from src.weights import Weights


def power_cost(num_on_nodes: int, cores_used: int, current_price: float) -> float:
    """
    Calculate power cost based on node usage and current electricity price.

    Proportional model: all on-nodes draw idle power, plus additional compute
    power scaled linearly by actual core utilization.
    Formula: (COST_IDLE_MW * num_on + (COST_USED_MW - COST_IDLE_MW) * cores_used/CORES_PER_NODE) * price

    Args:
        num_on_nodes: Number of nodes that are on (used + idle)
        cores_used: Total number of cores actively running jobs
        current_price: Current electricity price

    Returns:
        Total power cost
    """
    return (COST_IDLE_MW * num_on_nodes + (COST_USED_MW - COST_IDLE_MW) * (cores_used / CORES_PER_NODE)) * current_price


def power_consumption_mwh(num_on_nodes: int, cores_used: int) -> float:
    """
    Calculate energy consumption for one environment step.

    One environment step equals one hour, so this is both average MW and MWh/step.
    Uses the same proportional model as power_cost.

    Args:
        num_on_nodes: Number of nodes that are on (used + idle)
        cores_used: Total number of cores actively running jobs

    Returns:
        Energy consumption in MWh for this step
    """
    return COST_IDLE_MW * num_on_nodes + (COST_USED_MW - COST_IDLE_MW) * (cores_used / CORES_PER_NODE)


class RewardCalculator:
    """Calculates rewards with pre-computed normalization bounds."""
    # Faster response so price signal reacts on the same horizon as node-efficiency actions.
    # Price scaling uses active used nodes as work proxy, matching efficiency semantics.
    PRICE_ADVANTAGE_GAIN = 4.0
    # Asymmetric node scaling: high-price execution ramps faster than low-price reward.
    PRICE_NODE_TAU_POS = 70.0
    PRICE_NODE_TAU_NEG = 40.0
    NEGATIVE_PRICE_NODE_TAU = 14.0  # fast node saturation only for negative-price overdrive
    NEGATIVE_PRICE_TAU = 8.0
    # Overdrive terms for negative prices:
    # - gain controls overdrive strength during negative-price windows
    # - floor guarantees a minimum positive drive proportional to negative-price strength and used work
    # Toggle behavior:
    # - capped mode (default): overdrive is folded into tanh, so reward stays <= 1
    # - uncapped mode: overdrive is added after tanh and can exceed 1 up to NEGATIVE_PRICE_OVERDRIVE_MAX_REWARD
    NEGATIVE_PRICE_OVERDRIVE_GAIN = 2.5
    NEGATIVE_PRICE_OVERDRIVE_FLOOR = 0.35
    NEGATIVE_PRICE_OVERDRIVE_ALLOW_ABOVE_ONE = True
    NEGATIVE_PRICE_OVERDRIVE_MAX_REWARD = 1.5

    def __init__(self, prices: Prices) -> None:
        """
        Initialize reward calculator with normalization bounds.

        Args:
            prices: Prices object with MIN_PRICE and MAX_PRICE attributes
        """
        self.prices = prices
        self._compute_bounds()

    def _compute_bounds(self) -> None:
        """Compute min/max bounds for reward normalization."""
        # Efficiency bounds (for legacy _reward_efficiency_normalized only)
        cost_for_min_efficiency = power_cost(MAX_NODES, 0, self.prices.MAX_PRICE)
        cost_for_max_efficiency = power_cost(MAX_NODES, MAX_NODES * CORES_PER_NODE, self.prices.MIN_PRICE)

        self._min_efficiency_reward = self._reward_efficiency(0, cost_for_min_efficiency)
        self._max_efficiency_reward = max(1.0, self._reward_efficiency(MAX_NODES, cost_for_max_efficiency))

        # Price bounds (legacy behavior kept for debugging/ablation).
        self._max_price_reward_legacy = self._reward_price_legacy(
            self.prices.MIN_PRICE,
            self.prices.MAX_PRICE,
            MAX_NEW_JOBS_PER_HOUR,
        )
        self._min_price_reward_legacy = -self._max_price_reward_legacy

        # Idle penalty bounds
        self._min_idle_penalty = self._penalty_idle(0)
        self._max_idle_penalty = self._penalty_idle(MAX_NODES)

        # Job age penalty bounds
        self._min_job_age_penalty = 0.0
        self._max_job_age_penalty = 1.0

    @staticmethod
    def _normalize(current: float, minimum: float, maximum: float) -> float:
        """Normalize a value to [0, 1] range."""
        if maximum == minimum:
            return 0.5  # Avoid division by zero
        return (current - minimum) / (maximum - minimum)

    @staticmethod
    def _reward_efficiency(num_used_nodes: int, total_cost: float) -> float:
        """Calculate efficiency reward: work done per unit cost."""
        return num_used_nodes / (total_cost + 1e-6)

    def _reward_efficiency_normalized(self, num_used_nodes: int, num_idle_nodes: int, num_unprocessed_jobs: int, total_cost: float) -> float:
        """Calculate normalized efficiency reward [0, 1]."""
        if num_used_nodes + num_idle_nodes == 0:
            if num_unprocessed_jobs == 0:
                return 1
            else:
                return float(np.clip(1.0 / np.log1p(num_unprocessed_jobs), a_min=None, a_max=1.0))
        else:
            current_reward = self._reward_efficiency(num_used_nodes, total_cost)
            return self._normalize(current_reward, self._min_efficiency_reward, self._max_efficiency_reward)

    def _price_context_average(self, average_future_price: float) -> float:
        """Get context price average for comparison with current price."""
        history_avg, future_avg = self.prices.get_price_context()
        if history_avg is not None:
            return (history_avg + future_avg) / 2
        return average_future_price

    def _reward_price_legacy(self, current_price: float, average_future_price: float, num_processed_jobs: int) -> float:
        """Legacy linear reward: preserved for comparison/ablation."""
        context_avg = self._price_context_average(average_future_price)
        price_diff = context_avg - current_price
        return price_diff * num_processed_jobs

    def _reward_price_normalized_legacy(self, current_price: float, average_future_price: float, num_processed_jobs: int) -> float:
        """Legacy normalized price reward [0, 1] in typical operating range."""
        if num_processed_jobs == 0:
            return 0.0
        current_reward = self._reward_price_legacy(current_price, average_future_price, num_processed_jobs)
        return self._normalize(current_reward, self._min_price_reward_legacy, self._max_price_reward_legacy)

    def _reward_price(self, current_price: float, average_future_price: float, num_used_nodes: int) -> float:
        """
        Active signed price reward with fast saturation and negative-price overdrive.

        - Saturates quickly with better-than-context prices and used nodes.
        - Always applies overdrive when current price is negative.
        """

        if num_used_nodes <= 0:
            return 0.0

        context_avg = self._price_context_average(average_future_price)
        price_span = max(self.prices.MAX_PRICE - self.prices.MIN_PRICE, 1e-6)
        relative_advantage = (context_avg - current_price) / price_span

        advantage_component = self.PRICE_ADVANTAGE_GAIN * relative_advantage
        tau = self.PRICE_NODE_TAU_POS if advantage_component >= 0.0 else self.PRICE_NODE_TAU_NEG
        node_component = 1.0 - np.exp(-num_used_nodes / tau)
        raw_reward = advantage_component * node_component

        if current_price < 0.0:
            # Negative-price overdrive:
            # - negative_strength: how strongly negative the current price is (saturates to 1).
            # - negative_node_component: how much usable work is active (used-node saturation).
            # - overdrive: combined activation of "cheap enough" and "enough work running".
            # The floor guarantees a minimum positive incentive during negative-price windows,
            # scaled by overdrive instead of a fixed constant.
            negative_strength = (1.0 - np.exp(-abs(current_price) / self.NEGATIVE_PRICE_TAU))
            negative_node_component = (1.0 - np.exp(-num_used_nodes / self.NEGATIVE_PRICE_NODE_TAU))
            overdrive = negative_node_component * negative_strength

            if self.NEGATIVE_PRICE_OVERDRIVE_ALLOW_ABOVE_ONE:
                # Uncapped mode: keep signed base in [-1, 1], then add overdrive on top.
                # This allows >1 reward in negative-price periods, up to configurable max.
                reward = np.tanh(raw_reward) + self.NEGATIVE_PRICE_OVERDRIVE_GAIN * overdrive
                reward = min(reward, self.NEGATIVE_PRICE_OVERDRIVE_MAX_REWARD)
            else:
                # Capped mode: fold overdrive into raw score before tanh, keeping reward <= 1.
                raw_reward += self.NEGATIVE_PRICE_OVERDRIVE_GAIN * overdrive
                reward = np.tanh(raw_reward)

            reward = max(reward, self.NEGATIVE_PRICE_OVERDRIVE_FLOOR * overdrive)
        else:
            reward = np.tanh(raw_reward)

        return reward

    @staticmethod
    def _penalty_idle(num_idle_nodes: int) -> float:
        """Calculate penalty for idle nodes."""
        return PENALTY_IDLE_NODE * num_idle_nodes

    def _penalty_idle_normalized(self, num_idle_nodes: int) -> float:
        """Calculate normalized idle penalty [-1, 0]."""
        current_penalty = self._penalty_idle(num_idle_nodes)
        normalized_penalty = -self._normalize(current_penalty, self._min_idle_penalty, self._max_idle_penalty)
        return float(np.clip(normalized_penalty, -1, 0))

    @staticmethod
    def _penalty_job_age(num_off_nodes: int, job_queue_2d: np.ndarray) -> float:
        """Calculate saturated penalty for jobs waiting in queue when nodes are off."""
        job_age_penalty = 0.0
        if num_off_nodes > 0:
            # Vectorized max age calculation (much faster than Python loop)
            # [:, 0] selects column 0 (duration) for all rows; > 0 creates boolean mask
            valid_mask = job_queue_2d[:, 0] > 0
            # [valid_mask, 1] selects column 1 (age) only for rows where mask is True
            max_age = job_queue_2d[valid_mask, 1].max() if valid_mask.any() else 0
            if max_age > 0:
                tau_hours = WEEK_HOURS / 2.0
                max_factor = 1.0 - np.exp(-WEEK_HOURS / tau_hours)
                factor = 1.0 - np.exp(-max_age / tau_hours)
                factor = min(factor / max_factor, 1.0)
                job_age_penalty = factor
        return job_age_penalty

    def _penalty_job_age_normalized(self, num_off_nodes: int, job_queue_2d: np.ndarray) -> float:
        """Calculate normalized job age penalty [-1, 0]."""
        current_penalty = self._penalty_job_age(num_off_nodes, job_queue_2d)
        # _penalty_job_age already returns [0, 1]; negate to get [-1, 0]
        # normalized_penalty = self._normalize(current_penalty, 0, -1)
        normalized_penalty = -current_penalty
        return float(np.clip(normalized_penalty, -1, 0))

    def _reward_energy_efficiency_normalized(self, num_on_nodes: int, cores_used: int) -> float:
        '''Energy efficiency: fraction of total power draw that goes to actual computation.

        Proportional model: total power = COST_IDLE_MW * num_on + compute_delta * cores/CORES_PER_NODE.
        Compute power = compute_delta * cores/CORES_PER_NODE (the portion above idle baseline).
        Efficiency = compute_power / total_power, scaled to [-1, 1].
        '''
        compute_power = (COST_USED_MW - COST_IDLE_MW) * (cores_used / CORES_PER_NODE)
        total_power = COST_IDLE_MW * num_on_nodes + compute_power
        if total_power <= 0.0:
            return 0.0  # nothing on => no "efficiency" signal
        return 2 * float(np.clip(compute_power / total_power, 0.0, 1.0)) - 1.0  # scale to [-1, 1]

    def _blackout_term(self, num_used_nodes: int, num_idle_nodes: int, num_unprocessed_jobs: int) -> float:
        """
        Reward/penalty for full blackout (all nodes off).
        If queue is empty, reward the blackout. If jobs are waiting, apply a smooth penalty in [-1, 0].
        """
        BLACKOUT_QUEUE_THRESHOLD = 10  # jobs waiting until penalty saturates to -1
        SATURATION_FACTOR = 2
        on_nodes = num_used_nodes + num_idle_nodes

        if on_nodes != 0:
            return 0.0  # only care about full blackout

        if num_unprocessed_jobs <= 0:
            return 1.0  # correct blackout

        ratio = num_unprocessed_jobs / max(BLACKOUT_QUEUE_THRESHOLD, 1)
        penalty = np.exp(-ratio * SATURATION_FACTOR) - 1.0
        return float(np.clip(penalty, -1.0, 0.0))

    def calculate(self, num_used_nodes: int, num_idle_nodes: int, num_used_cores: int, current_price: float, average_future_price: float,
                  num_off_nodes: int, _num_processed_jobs: int, num_node_changes: int, job_queue_2d: np.ndarray,  # noqa: ARG002 - _num_processed_jobs legacy; num_node_changes reserved for future node-change penalty
                  num_unprocessed_jobs: int, weights: Weights, num_dropped_this_step: int,
                  env_print: Callable[..., None]) -> tuple[float, float, float, float, float, float]:
        """
        Calculate total reward by aggregating weighted components.

        Args:
            num_used_nodes: Number of nodes with jobs running
            num_idle_nodes: Number of idle nodes
            current_price: Current electricity price
            average_future_price: Average predicted future price
            num_off_nodes: Number of offline nodes
            _num_processed_jobs: Number of jobs launched this step (legacy param, unused by active price reward)
            num_node_changes: Number of node state changes
            job_queue_2d: 2D job queue array
            num_unprocessed_jobs: Number of jobs waiting in queue
            weights: Weights object with weight values
            num_dropped_this_step: Number of jobs dropped this step
            env_print: Print function for logging

        Returns:
            Tuple of (total reward, total cost, eff_reward_norm, price_reward,
                      idle_penalty_norm, job_age_penalty_norm)
        """
        # 0. Energy efficiency. Reward calculation based on Workload (used nodes) (W) / Cost (C)
        num_on_nodes = num_used_nodes + num_idle_nodes
        total_cost = power_cost(num_on_nodes, num_used_cores, current_price)
        efficiency_reward_norm = self._reward_energy_efficiency_normalized(num_on_nodes, num_used_cores) + self._blackout_term(num_used_nodes, num_idle_nodes, num_unprocessed_jobs)
        efficiency_reward_weighted = weights.efficiency_weight * efficiency_reward_norm

        # 2. Increase reward if current price is favorable and currently used nodes are high.
        price_reward = self._reward_price(
            current_price, average_future_price, num_used_nodes
        )
        price_reward_weighted = weights.price_weight * price_reward

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

        return reward, total_cost, efficiency_reward_norm, price_reward, idle_penalty_norm, job_age_penalty_norm
