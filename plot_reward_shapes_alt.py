#!/usr/bin/env python
"""
Visualize reward component shapes from the live RewardCalculator code.

All shapes are computed by calling actual methods in src/reward_calculation.py,
so any change to constants or formulas is immediately reflected here.

Usage:
    python plot_reward_shapes.py                        # interactive window
    python plot_reward_shapes.py -o shapes.png          # save to file
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.config import CORES_PER_NODE, MAX_NODES
from src.reward_calculation import RewardCalculator


class _MockPrices:
    """Minimal Prices stand-in with a representative 24h forecast window."""

    MIN_PRICE: float = -5.24  # €/MWh
    MAX_PRICE: float = 207.98  # €/MWh

    def __init__(self) -> None:
        # A simple day-shaped forecast window so quantile-based reward terms
        # have realistic future context for the live calculator methods.
        self.predicted_prices = np.array(
            [
                80.0, 145.0, 160.0, 175.0, 188.0, 195.0,
                182.0, 168.0, 142.0, 118.0, 92.0, 70.0,
                48.0, 26.0, 12.0, -5.24, 8.0, 20.0,
                36.0, 55.0, 78.0, 102.0, 126.0, 150.0,
            ],
            dtype=np.float32,
        )

    def get_price_context(self):
        # (None, ...) makes _price_context_average fall back to average_future_price param
        return None, float(np.mean(self.predicted_prices))


def _make_calc() -> RewardCalculator:
    return RewardCalculator(_MockPrices())  # type: ignore[arg-type]


def _cheap_price_band(calc: RewardCalculator) -> tuple[float, float]:
    """Quantile band used by the cheap-hour service-pressure term."""
    future_reference = np.asarray(calc.prices.predicted_prices[1:], dtype=np.float32)
    q_low, q_high = np.quantile(
        future_reference,
        [calc.PRICE_QUANTILE_LOW, calc.PRICE_QUANTILE_HIGH],
    )
    return float(q_low), float(q_high)


# ---------------------------------------------------------------------------
# Individual panel functions — each calls the real RewardCalculator method
# ---------------------------------------------------------------------------

def _plot_efficiency(ax: plt.Axes, calc: RewardCalculator) -> None:
    """Efficiency reward vs core utilization % (node count cancels out analytically)."""
    util_pct = np.linspace(0.0, 1.0, 300)
    N = 100  # representative count; result is invariant to N for any fixed util%
    rewards = [
        calc._reward_energy_efficiency_utilization_normalized(N, int(u * N * CORES_PER_NODE))
        for u in util_pct
    ]
    all_off = calc._reward_energy_efficiency_utilization_normalized(0, 0)

    ax.plot(util_pct * 100, rewards, lw=2)
    ax.scatter([0], [all_off], color='red', zorder=5, label=f'all-off → {all_off:.2f}')
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(calc.EFFICIENCY_TARGET_RATIO * 100, color='gray', ls='--', lw=1,
               label=f'target = {calc.EFFICIENCY_TARGET_RATIO:.0%}')
    ax.set_xlabel('Core utilization %')
    ax.set_ylabel('Reward')
    ax.set_title('Efficiency')
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_price(ax: plt.Axes, calc: RewardCalculator) -> None:
    """Quantile-based price reward vs current price for several equivalent-node loads."""
    prices = np.linspace(calc.prices.MIN_PRICE, calc.prices.MAX_PRICE, 400)
    eq_nodes_list = [5, 15, 30, 100, MAX_NODES]
    q_low, q_high = _cheap_price_band(calc)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(eq_nodes_list)))  # type: ignore[attr-defined]

    for eq_nodes, color in zip(eq_nodes_list, colors):
        used_cores = int(eq_nodes * CORES_PER_NODE)
        rewards = [calc._reward_price_quantile_utilization(float(p), used_cores) for p in prices]
        ax.plot(prices, rewards, color=color, lw=1.5, label=f'{eq_nodes} eq.nodes')

    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5, ls=':')
    ax.axvline(q_low, color='gray', ls=':', lw=1, label=f'q{int(calc.PRICE_QUANTILE_LOW * 100)} = {q_low:.0f}')
    ax.axvline(q_high, color='gray', ls='--', lw=1, label=f'q{int(calc.PRICE_QUANTILE_HIGH * 100)} = {q_high:.0f}')
    ax.set_xlabel('Current price (€/MWh)')
    ax.set_ylabel('Reward')
    ax.set_title('Price timing (quantile phase)')
    ax.set_xlim(calc.prices.MIN_PRICE, calc.prices.MAX_PRICE)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _plot_idle(ax: plt.Axes, calc: RewardCalculator) -> None:
    """Idle penalty vs number of idle nodes."""
    idle_counts = np.arange(0, MAX_NODES + 1)
    penalties = [calc._penalty_idle_normalized(int(n)) for n in idle_counts]

    ax.plot(idle_counts, penalties, lw=2)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Idle nodes')
    ax.set_ylabel('Penalty')
    ax.set_title('Idle penalty')
    ax.set_xlim(0, MAX_NODES)
    ax.set_ylim(-1.1, 0.1)
    ax.grid(True, alpha=0.3)


def _plot_job_age(ax: plt.Axes, calc: RewardCalculator) -> None:
    """Legacy 'job age' slot vs current price at several service levels."""
    prices = np.linspace(calc.prices.MIN_PRICE, calc.prices.MAX_PRICE, 400)
    q_low, q_high = _cheap_price_band(calc)
    step_capacity_cores = float(MAX_NODES * CORES_PER_NODE)
    pending_core_demand = 0.50 * step_capacity_cores
    service_levels = [0.0, 0.25, 0.50, 0.75, 1.0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(service_levels)))  # type: ignore[attr-defined]

    for served_fraction, color in zip(service_levels, colors):
        used_cores = int(round(served_fraction * pending_core_demand))
        penalties = [
            calc._penalty_job_age_normalized(float(price), pending_core_demand, used_cores)
            for price in prices
        ]
        ax.plot(
            prices,
            penalties,
            color=color,
            lw=1.7,
            label=f'served {served_fraction:.0%}',
        )

    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(q_low, color='gray', ls=':', lw=1, label=f'q{int(calc.PRICE_QUANTILE_LOW * 100)} = {q_low:.0f}')
    ax.axvline(q_high, color='gray', ls='--', lw=1, label=f'q{int(calc.PRICE_QUANTILE_HIGH * 100)} = {q_high:.0f}')
    ax.set_xlabel('Current price (€/MWh)')
    ax.set_ylabel('Penalty')
    ax.set_title('Job-age slot: cheap-hour service pressure')
    ax.set_xlim(calc.prices.MIN_PRICE, calc.prices.MAX_PRICE)
    ax.set_ylim(-1.1, 0.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_blackout(ax: plt.Axes, calc: RewardCalculator) -> None:
    """Blackout term vs unprocessed job count (all nodes off)."""
    job_counts = np.arange(0, 600)
    penalties = [calc._blackout_term(0, 0, int(n)) for n in job_counts]

    ax.plot(job_counts, penalties, lw=2)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Unprocessed jobs (all nodes off)')
    ax.set_ylabel('Reward')
    ax.set_title('Blackout term')
    ax.set_xlim(0, 600)
    ax.grid(True, alpha=0.3)


def _plot_drop(ax: plt.Axes, calc: RewardCalculator) -> None:
    """Actual loss penalty vs jobs lost this step."""
    counts = np.arange(0, 26)
    penalties = [calc.loss_penalty(int(n)) for n in counts]

    for n in [1, 5, 10, 20, 25]:
        y = calc.loss_penalty(n)
        ax.annotate(f'{n}→{y:.2f}', (n, y), fontsize=7,
                    textcoords='offset points', xytext=(4, 6))

    ax.step(counts, penalties, where='mid', lw=2)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Jobs lost this step')
    ax.set_ylabel('Penalty')
    ax.set_title('Loss penalty (0.3 x raw drop penalty)')
    ax.set_xlim(0, 25)
    ax.set_ylim(min(penalties) - 0.1, 0.1)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plot_reward_shapes(output: str | None = None) -> None:
    """Generate 3×2 grid of reward component shape plots."""
    calc = _make_calc()

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle('Reward component shapes  (src/reward_calculation.py)', fontsize=13)

    _plot_efficiency(axes[0, 0], calc)
    _plot_price(axes[0, 1], calc)
    _plot_idle(axes[1, 0], calc)
    _plot_job_age(axes[1, 1], calc)
    _plot_blackout(axes[2, 0], calc)
    _plot_drop(axes[2, 1], calc)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f'Saved → {output}')
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-o', '--output', metavar='PATH',
                        help='Save to file instead of showing interactively (e.g. shapes.png)')
    args = parser.parse_args()
    plot_reward_shapes(output=args.output)


if __name__ == '__main__':
    main()
