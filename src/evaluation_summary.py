from __future__ import annotations

import math
from typing import Mapping, Sequence


def _fmt_optional(value: float | int | None, precision: int = 2, thousands: bool = False) -> str:
    if value is None:
        return "n/a"

    numeric_value = float(value)
    if math.isnan(numeric_value):
        return "n/a"

    return f"{numeric_value:,.{precision}f}" if thousands else f"{numeric_value:.{precision}f}"


def mean_occupancy_pct(values: Sequence[int], capacity: int) -> float:
    if not values or capacity <= 0:
        return 0.0
    return float(sum(values) * 100.0 / (len(values) * capacity))


def build_episode_summary_line(
    episode_number: int,
    episode_data: Mapping[str, float | int],
    timeline_max_queue: int,
    agent_occupancy_cores_pct: float,
    baseline_occupancy_cores_pct: float,
    agent_occupancy_nodes_pct: float,
    baseline_occupancy_nodes_pct: float,
) -> str:
    return (
        f"  Episode {episode_number}: "
        f"Agent Cost=€{float(episode_data['agent_cost']):.0f}, "
        f"Baseline Cost=€{float(episode_data['baseline_cost']):.0f} | "
        f"Baseline Off=€{float(episode_data['baseline_cost_off']):.0f}, "
        f"Savings=€{float(episode_data['savings_vs_baseline']):.0f}/"
        f"€{float(episode_data['savings_vs_baseline_off']):.0f}, "
        f"Power={float(episode_data['agent_power_consumption_mwh']):.1f}/"
        f"{float(episode_data['baseline_power_consumption_mwh']):.1f}/"
        f"{float(episode_data['baseline_power_consumption_off_mwh']):.1f} MWh "
        f"(agent/base/base_off), "
        f"MeanPrice={float(episode_data['agent_mean_price']):.2f}/"
        f"{float(episode_data['baseline_mean_price']):.2f}/"
        f"{float(episode_data['baseline_off_mean_price']):.2f} €/MWh "
        f"(agent/base/base_off), "
        f"CostPer1kCompleted="
        f"{_fmt_optional(episode_data['agent_cost_per_1000_completed_jobs'], 1, thousands=True)}/"
        f"{_fmt_optional(episode_data['baseline_cost_per_1000_completed_jobs'], 1, thousands=True)}/"
        f"{_fmt_optional(episode_data['baseline_off_cost_per_1000_completed_jobs'], 1, thousands=True)} "
        f"€/1k (agent/base/base_off), "
        f"DroppedPerSavedEuro="
        f"{_fmt_optional(episode_data['agent_dropped_jobs_per_saved_euro'], 6)}/"
        f"{_fmt_optional(episode_data['agent_dropped_jobs_per_saved_euro_off'], 6)} "
        f"jobs/€ (vs base/base_off), "
        f"Jobs={int(episode_data['jobs_completed'])}/{int(episode_data['jobs_submitted'])} "
        f"({float(episode_data['completion_rate']):.0f}%), "
        f"AvgWait={float(episode_data['avg_wait_time']):.1f}h, "
        f"PendingEnd={int(episode_data.get('pending_jobs_end', 0))}, "
        f"OverdueEnd={int(episode_data.get('overdue_jobs_end', 0))}, "
        f"EpisodeMaxQueue={int(episode_data['max_queue_size'])}, "
        f"MaxDropStreak={int(episode_data.get('max_drop_streak', 0))}, "
        f"Lost={int(episode_data.get('jobs_lost_total', episode_data['jobs_dropped']))}, "
        f"TimelineMaxQueue={timeline_max_queue}, "
        f"Agent Occupancy (Cores)={agent_occupancy_cores_pct:.2f}%, "
        f"Baseline Occupancy (Cores)={baseline_occupancy_cores_pct:.2f}%, "
        f"Agent Occupancy (Nodes)={agent_occupancy_nodes_pct:.2f}%, "
        f"Baseline Occupancy (Nodes)={baseline_occupancy_nodes_pct:.2f}%"
    )
