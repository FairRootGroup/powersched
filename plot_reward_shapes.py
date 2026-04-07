#!/usr/bin/env python
"""
Visualize reward component shapes from the live RewardCalculator code.

All shapes are computed by calling actual methods in src/reward_calculation.py,
so any change to constants or formulas is immediately reflected here.

Usage:
    python plot_reward_shapes.py                        # interactive window
    python plot_reward_shapes.py -o shapes.png          # save to file
    python plot_reward_shapes.py --context-avg 120      # different price baseline
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.config import CORES_PER_NODE, MAX_NODES, WEEK_HOURS
from src.reward_calculation import RewardCalculator

class _MockPrices:
    """Minimal Prices stand-in — no real data needed for shape visualization."""
    MIN_PRICE: float = -5.24  # €/MWh
    MAX_PRICE: float = 207.98   # €/MWh

    def get_price_context(self):
        # (None, ...) makes _price_context_average fall back to average_future_price param
        return None, 0.0


def _make_calc() -> RewardCalculator:
    return RewardCalculator(_MockPrices())  # type: ignore[arg-type]


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


def _plot_price(ax: plt.Axes, calc: RewardCalculator, context_avg: float) -> None:
    """Price-timing reward vs current price for several equivalent-node loads."""
    prices = np.linspace(_MockPrices.MIN_PRICE, _MockPrices.MAX_PRICE, 400)
    eq_nodes_list = [5, 15, 30, 100, MAX_NODES]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(eq_nodes_list)))  # type: ignore[attr-defined]

    for eq_nodes, color in zip(eq_nodes_list, colors):
        used_cores = int(eq_nodes * CORES_PER_NODE)
        rewards = [calc._reward_price_utilization(float(p), context_avg, used_cores) for p in prices]
        ax.plot(prices, rewards, color=color, lw=1.5, label=f'{eq_nodes} eq.nodes')

    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5, ls=':')
    ax.axvline(context_avg, color='gray', ls='--', lw=1, label=f'ctx avg = {context_avg:.0f}')
    ax.set_xlabel('Current price (€/MWh)')
    ax.set_ylabel('Reward')
    ax.set_title('Price timing')
    ax.set_xlim(_MockPrices.MIN_PRICE, _MockPrices.MAX_PRICE)
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
    """Job age penalty vs maximum job age in the queue."""
    max_ages = np.linspace(0, 2 * WEEK_HOURS, 300)
    penalties = []
    for age in max_ages:
        fake_queue = np.array([[1.0, float(age)]])  # columns: [duration, age]
        penalties.append(calc._penalty_job_age_normalized(num_off_nodes=1, job_queue_2d=fake_queue))

    tau_h = WEEK_HOURS / 2.0
    ax.plot(max_ages, penalties, lw=2)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(tau_h, color='gray', ls=':', lw=1, label=f'τ = {tau_h:.0f}h')
    ax.axvline(WEEK_HOURS, color='gray', ls='--', lw=1, label=f'1 week = {WEEK_HOURS}h')
    ax.set_xlabel('Max job age (hours)')
    ax.set_ylabel('Penalty')
    ax.set_title('Job age penalty')
    ax.set_xlim(0, 2 * WEEK_HOURS)
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
    """Drop penalty vs jobs dropped this step."""
    counts = np.linspace(0, 150, 400)
    penalties = [calc._penalty_drop(int(n)) for n in counts]

    for n in [1, 10, 20, 50]:
        y = calc._penalty_drop(n)
        ax.annotate(f'{n}→{y:.2f}', (n, y), fontsize=7,
                    textcoords='offset points', xytext=(4, 6))

    ax.plot(counts, penalties, lw=2)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(calc.DROP_PENALTY_TAU, color='gray', ls='--', lw=1,
               label=f'TAU = {calc.DROP_PENALTY_TAU:.0f}')
    ax.set_xlabel('Jobs dropped this step')
    ax.set_ylabel('Penalty')
    ax.set_title('Drop penalty')
    ax.set_xlim(0, 150)
    ax.set_ylim(-1.1, 0.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plot_reward_shapes(output: str | None = None, context_avg: float = 80.0) -> None:
    """Generate 3×2 grid of reward component shape plots."""
    calc = _make_calc()

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle('Reward component shapes  (src/reward_calculation.py)', fontsize=13)

    _plot_efficiency(axes[0, 0], calc)
    _plot_price(axes[0, 1], calc, context_avg=context_avg)
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
    parser.add_argument('--context-avg', type=float, default=80.0, metavar='€/MWh',
                        help='Context price average used as baseline in the price panel (default: 80)')
    args = parser.parse_args()
    plot_reward_shapes(output=args.output, context_avg=args.context_avg)


if __name__ == '__main__':
    main()
