"""Reward calculation and normalization logic for the PowerSched environment."""

import numpy as np
from src.config import (
    COST_IDLE_MW, COST_USED_MW, PENALTY_IDLE_NODE, PENALTY_WAITING_JOB
)


def normalize(current, minimum, maximum):
    """Normalize a value to [0, 1] range."""
    if maximum == minimum:
        return 0.5  # Avoid division by zero
    return (current - minimum) / (maximum - minimum)


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


def reward_efficiency(num_used_nodes, total_cost):
    """
    Calculate efficiency reward: work done per unit cost.

    Args:
        num_used_nodes: Number of nodes with jobs running
        total_cost: Total power cost

    Returns:
        Efficiency reward (nodes / cost)
    """
    return num_used_nodes / (total_cost + 1e-6)


def reward_efficiency_normalized(num_used_nodes, num_idle_nodes, num_unprocessed_jobs,
                                 total_cost, min_efficiency_reward, max_efficiency_reward):
    """
    Calculate normalized efficiency reward [0, 1].

    Args:
        num_used_nodes: Number of nodes with jobs running
        num_idle_nodes: Number of idle nodes
        num_unprocessed_jobs: Number of jobs waiting in queue
        total_cost: Total power cost
        min_efficiency_reward: Minimum efficiency reward for normalization
        max_efficiency_reward: Maximum efficiency reward for normalization

    Returns:
        Normalized efficiency reward
    """
    current_reward = 0
    if num_used_nodes + num_idle_nodes == 0:
        if num_unprocessed_jobs == 0:
            current_reward = 1
        else:
            current_reward = np.clip(1.0 / np.log1p(num_unprocessed_jobs), a_min=None, a_max=1.0)
    else:
        current_reward = reward_efficiency(num_used_nodes, total_cost)
        current_reward = normalize(current_reward, min_efficiency_reward, max_efficiency_reward)
    return current_reward


def reward_price(current_price, average_future_price, num_processed_jobs, prices):
    """
    Calculate price-based reward for scheduling jobs at favorable prices.

    Args:
        current_price: Current electricity price
        average_future_price: Average predicted future price
        num_processed_jobs: Number of jobs launched this step
        prices: Prices object with get_price_context() method

    Returns:
        Price reward
    """
    history_avg, future_avg = prices.get_price_context()

    if history_avg is not None:
        # We have some history - use both past and future
        context_avg = (history_avg + future_avg) / 2
        price_diff = context_avg - current_price
    else:
        # No history yet - fall back to just using future prices
        price_diff = average_future_price - current_price

    price_reward = price_diff * num_processed_jobs
    return price_reward


def reward_price_normalized(current_price, average_future_price, num_processed_jobs,
                            prices, min_price_reward, max_price_reward):
    """
    Calculate normalized price reward [0, 1].

    Args:
        current_price: Current electricity price
        average_future_price: Average predicted future price
        num_processed_jobs: Number of jobs launched this step
        prices: Prices object with get_price_context() method
        min_price_reward: Minimum price reward for normalization
        max_price_reward: Maximum price reward for normalization

    Returns:
        Normalized price reward
    """
    current_reward = reward_price(current_price, average_future_price, num_processed_jobs, prices)
    if num_processed_jobs == 0:
        normalized_reward = 0
    else:
        normalized_reward = normalize(current_reward, min_price_reward, max_price_reward)
    return normalized_reward


def penalty_idle(num_idle_nodes):
    """
    Calculate penalty for idle nodes.

    Args:
        num_idle_nodes: Number of idle (on but unused) nodes

    Returns:
        Idle penalty (negative value)
    """
    idle_penalty = PENALTY_IDLE_NODE * num_idle_nodes
    return idle_penalty


def penalty_idle_normalized(num_idle_nodes, min_idle_penalty, max_idle_penalty):
    """
    Calculate normalized idle penalty [-1, 0].

    Args:
        num_idle_nodes: Number of idle nodes
        min_idle_penalty: Minimum idle penalty for normalization
        max_idle_penalty: Maximum idle penalty for normalization

    Returns:
        Normalized idle penalty
    """
    current_penalty = penalty_idle(num_idle_nodes)
    normalized_penalty = - normalize(current_penalty, min_idle_penalty, max_idle_penalty)
    normalized_penalty = np.clip(normalized_penalty, -1, 0)
    return normalized_penalty


def penalty_job_age(num_off_nodes, job_queue_2d):
    """
    Calculate penalty for jobs waiting in queue when nodes are off.

    Args:
        num_off_nodes: Number of offline nodes
        job_queue_2d: 2D job queue array

    Returns:
        Job age penalty (negative value)
    """
    job_age_penalty = 0
    if num_off_nodes > 0:
        for job in job_queue_2d:
            job_duration, job_age, _, _ = job
            if job_duration > 0:
                job_age_penalty += PENALTY_WAITING_JOB * job_age
    return job_age_penalty


def penalty_job_age_normalized(num_off_nodes, job_queue_2d, min_job_age_penalty,
                               max_job_age_penalty):
    """
    Calculate normalized job age penalty [-1, 0].

    Args:
        num_off_nodes: Number of offline nodes
        job_queue_2d: 2D job queue array
        min_job_age_penalty: Minimum job age penalty for normalization
        max_job_age_penalty: Maximum job age penalty for normalization

    Returns:
        Normalized job age penalty
    """
    current_penalty = penalty_job_age(num_off_nodes, job_queue_2d)
    normalized_penalty = - normalize(current_penalty, min_job_age_penalty, max_job_age_penalty)
    normalized_penalty = np.clip(normalized_penalty, -1, 0)
    return normalized_penalty


def calculate_reward(num_used_nodes, num_idle_nodes, current_price, average_future_price,
                    num_off_nodes, num_processed_jobs, num_node_changes, job_queue_2d,
                    num_unprocessed_jobs, weights, prices, reward_bounds, num_dropped_this_step,
                    env_print):
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
        prices: Prices object
        reward_bounds: Dictionary with min/max values for normalization
        num_dropped_this_step: Number of jobs dropped this step
        env_print: Print function for logging

    Returns:
        Tuple of (total reward, total cost)
    """
    # 0. Efficiency. Reward calculation based on Workload (used nodes) (W) / Cost (C)
    total_cost = power_cost(num_used_nodes, num_idle_nodes, current_price)
    efficiency_reward_norm = reward_efficiency_normalized(
        num_used_nodes, num_idle_nodes, num_unprocessed_jobs, total_cost,
        reward_bounds['min_efficiency_reward'], reward_bounds['max_efficiency_reward']
    )
    efficiency_reward_weighted = weights.efficiency_weight * efficiency_reward_norm

    # 2. increase reward if jobs were scheduled in this step and the current price is below average
    price_reward_norm = reward_price_normalized(
        current_price, average_future_price, num_processed_jobs, prices,
        reward_bounds['min_price_reward'], reward_bounds['max_price_reward']
    )
    price_reward_weighted = weights.price_weight * price_reward_norm

    # 3. penalize delayed jobs, more if they are older. but only if there are turned off nodes
    job_age_penalty_norm = penalty_job_age_normalized(
        num_off_nodes, job_queue_2d,
        reward_bounds['min_job_age_penalty'], reward_bounds['max_job_age_penalty']
    )
    job_age_penalty_weighted = weights.job_age_weight * job_age_penalty_norm

    # 5. penalty for idling nodes
    idle_penalty_norm = penalty_idle_normalized(
        num_idle_nodes,
        reward_bounds['min_idle_penalty'], reward_bounds['max_idle_penalty']
    )
    idle_penalty_weighted = weights.idle_weight * idle_penalty_norm

    # 6. penalty for dropped jobs (WIP - unnormalized, weighted)
    from src.config import PENALTY_DROPPED_JOB
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
