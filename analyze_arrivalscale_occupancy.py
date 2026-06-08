#!/usr/bin/env python3
"""
Sweep job-arrival-scale values in jobs exact-replay mode and analyze:
1) job-arrival-scale -> agent occupancy (nodes)
2) occupancy -> proportional savings (%)
3) occupancy -> proportional savings_off (%)
4) job-arrival-scale -> completion rate
5) occupancy -> proportional effective savings (%)
6) occupancy -> proportional effective savings_off (%)
7) occupancy -> average wait delta (agent - baseline)
8) occupancy -> (baseline_off - agent) cost_per_1000_completed_jobs / baseline_off
9) occupancy -> (baseline_off - agent) proportional power / baseline_off
10) arrival-scale -> baseline and baseline_off occupancies
11) arrival-scale -> mean jobs/hour (with std)
12) arrival-scale -> dropped-jobs delta (agent - baseline)

For each scale, this script runs train.py in evaluation mode for one year
(12 months = 24 episodes), parses per-episode metrics from stdout, computes
mean/std, and fits optional polynomial trend lines.

Runs both modes:
- exact replay (aggregated): --jobs --jobs-exact-replay --jobs-exact-replay-aggregate --job-arrival-scale <scale>
- sampling mode: --hourly-jobs --job-arrival-scale <scale>

FAST DEBUG MODE:
python analyze_arrivalscale_occupancy.py \
  --jobs ./data/workload_statistics/jobs_2023.log \
  --hourly-jobs ./data/workload_statistics/jobs_2023.log \
  --eval-months 1 --scales 0.8,1.0,1.2 --no-plot-dashboard
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.analysis_naming import build_analysis_dir_name
from src import analysis_metrics as metrics
from src.analysis_metrics import RunStats


def build_train_command(args: argparse.Namespace, job_arrival_scale: float, replay_mode: str) -> list[str]:
    cmd = [
        sys.executable,
        "./train.py",
        "--prices",
        args.prices,
        "--session",
        args.session,
        "--efficiency-weight",
        str(args.efficiency_weight),
        "--price-weight",
        str(args.price_weight),
        "--idle-weight",
        str(args.idle_weight),
        "--job-age-weight",
        str(args.job_age_weight),
        "--drop-weight",
        str(args.drop_weight),
        "--evaluate-savings",
        "--eval-months",
        str(args.eval_months),
        "--model",
        str(args.model),
    ]
    if replay_mode == "exact_replay_aggregate":
        cmd.extend([
            "--jobs",
            args.jobs,
            "--jobs-exact-replay",
            "--jobs-exact-replay-aggregate",
            "--job-arrival-scale",
            f"{job_arrival_scale:.6f}",
        ])
    elif replay_mode == "sampling":
        hourly_jobs = args.hourly_jobs or args.jobs
        cmd.extend([
            "--hourly-jobs",
            hourly_jobs,
            "--job-arrival-scale",
            f"{job_arrival_scale:.6f}",
        ])
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])
    else:
        raise ValueError(f"Unsupported replay_mode: {replay_mode}")
    if args.plot_dashboard:
        cmd.append("--plot-dashboard")
    if args.oracle:
        cmd.append("--oracle")
    if args.dashboard_hours is not None:
        cmd.extend(["--dashboard-hours", str(args.dashboard_hours)])
    return cmd


def run_scale_eval(
    args: argparse.Namespace,
    project_root: Path,
    job_arrival_scale: float,
    replay_mode: str,
) -> tuple[RunStats, str]:
    command = build_train_command(args, job_arrival_scale, replay_mode)
    print(f"[run] mode={replay_mode}, scale={job_arrival_scale:.6f}: {shlex.join(command)}")
    completed = subprocess.run(
        command,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    if args.echo_train_output:
        print(combined_output)
    if completed.returncode != 0:
        raise RuntimeError(
            f"train.py failed for mode={replay_mode}, scale={job_arrival_scale:.6f} with code {completed.returncode}.\n"
            f"Last output lines:\n{metrics.os_tail(combined_output, lines=40)}"
        )

    (
        occupancy,
        baseline_occupancy,
        agent_dropped,
        savings,
        savings_off,
        completion_rate,
        avg_wait,
        agent_cost_1k,
        baseline_cost_1k,
        baseline_off_cost_1k,
        agent_power,
        baseline_off_power,
        prop_savings,
        prop_savings_off,
        agent_prop_power,
        baseline_prop_cost,
        baseline_off_prop_cost,
        baseline_off_prop_power,
    ) = metrics.parse_episode_metrics(combined_output)

    agent_wait_summary, baseline_wait_summary = metrics.parse_wait_summary(combined_output)
    if agent_wait_summary is None or baseline_wait_summary is None:
        print(f"[warn] mode={replay_mode}, scale={job_arrival_scale:.6f}: could not parse run-level wait summary; effective savings may be NaN.")
        agent_avg_wait_hours = float(np.mean(avg_wait))
        baseline_avg_wait_hours = float("nan")
    else:
        agent_avg_wait_hours = float(agent_wait_summary)
        baseline_avg_wait_hours = float(baseline_wait_summary)
    arrivals_per_hour_mean, arrivals_per_hour_std = metrics.parse_arrivals_summary(combined_output)
    if arrivals_per_hour_mean is None or arrivals_per_hour_std is None:
        print(f"[warn] mode={replay_mode}, scale={job_arrival_scale:.6f}: could not parse run-level arrivals/hour summary; values set to NaN.")
        arrivals_per_hour_mean = float("nan")
        arrivals_per_hour_std = float("nan")
    dropped_jobs_agent_total, dropped_jobs_baseline_total = metrics.parse_dropped_totals_summary(combined_output)
    if dropped_jobs_agent_total is None:
        dropped_jobs_agent_total = float(np.sum(agent_dropped))
        print(f"[warn] mode={replay_mode}, scale={job_arrival_scale:.6f}: could not parse run-level agent dropped total; using sum of episode Dropped= values.")
    if dropped_jobs_baseline_total is None:
        dropped_jobs_baseline_total = 0.0
        print(f"[warn] mode={replay_mode}, scale={job_arrival_scale:.6f}: could not parse run-level baseline dropped total; defaulting to 0.")

    stats = metrics.make_run_stats(
        sweep_key=job_arrival_scale,
        replay_mode=replay_mode,
        eval_months=args.eval_months,
        command=command,
        occupancy=occupancy,
        baseline_occupancy=baseline_occupancy,
        agent_dropped=agent_dropped,
        savings=savings,
        savings_off=savings_off,
        completion_rate=completion_rate,
        agent_avg_wait_hours=agent_avg_wait_hours,
        baseline_avg_wait_hours=baseline_avg_wait_hours,
        agent_cost_1k=agent_cost_1k,
        baseline_cost_1k=baseline_cost_1k,
        baseline_off_cost_1k=baseline_off_cost_1k,
        agent_power=agent_power,
        baseline_off_power=baseline_off_power,
        prop_savings=prop_savings,
        prop_savings_off=prop_savings_off,
        agent_prop_power=agent_prop_power,
        baseline_prop_cost=baseline_prop_cost,
        baseline_off_prop_cost=baseline_off_prop_cost,
        baseline_off_prop_power=baseline_off_prop_power,
        arrivals_per_hour_mean=arrivals_per_hour_mean,
        arrivals_per_hour_std=arrivals_per_hour_std,
        dropped_jobs_agent_total=dropped_jobs_agent_total,
        dropped_jobs_baseline_total=dropped_jobs_baseline_total,
    )
    print(
        f"[ok ] mode={replay_mode}, scale={job_arrival_scale:.6f}: "
        f"occupancy={stats.occupancy_mean:.2f}%±{stats.occupancy_std:.2f}, "
        f"baseline_occ={stats.baseline_occupancy_mean:.2f}%±{stats.baseline_occupancy_std:.2f}, "
        f"arrivals/h={stats.arrivals_per_hour_mean:.2f}±{stats.arrivals_per_hour_std:.2f}, "
        f"dropped_delta={stats.dropped_jobs_delta_total:.0f}, "
        f"completion={stats.completion_rate_mean:.2f}%±{stats.completion_rate_std:.2f}, "
        f"prop_savings={stats.prop_savings_mean:.0f}±{stats.prop_savings_std:.0f}, "
        f"prop_savings_off={stats.prop_savings_off_mean:.0f}±{stats.prop_savings_off_std:.0f}, "
        f"prop_eval_savings={stats.prop_evaluation_savings:.0f}/{stats.prop_evaluation_savings_off:.0f}, "
        f"prop_annualized_savings={stats.prop_annualized_savings:.0f}/{stats.prop_annualized_savings_off:.0f}, "
        f"wait_delta={stats.wait_delta_hours:.3f}h"
    )
    return stats, combined_output


def write_summary_csv(path: Path, all_stats: list[RunStats]) -> None:
    fieldnames = ["replay_mode", "job_arrival_scale"] + metrics.CSV_COMMON_FIELDNAMES
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in sorted(all_stats, key=lambda x: (x.replay_mode, x.sweep_key)):
            writer.writerow({
                "replay_mode": s.replay_mode,
                "job_arrival_scale": f"{s.sweep_key:.6f}",
                **metrics.csv_common_row(s),
            })


def make_plot(
    path: Path,
    all_stats: list[RunStats],
    replay_mode: str,
    fit: bool = False,
    individual_dir: Path | None = None,
) -> None:
    ordered = sorted(all_stats, key=lambda x: x.sweep_key)
    if not ordered:
        return

    scales = np.array([s.sweep_key for s in ordered], dtype=float)
    occ_mean = np.array([s.occupancy_mean for s in ordered], dtype=float)
    occ_std = np.array([s.occupancy_std for s in ordered], dtype=float)
    occ_minmax = metrics.minmax_error_array([s.occupancy_samples for s in ordered], occ_mean)
    baseline_occ_mean = np.array([s.baseline_occupancy_mean for s in ordered], dtype=float)
    baseline_occ_std = np.array([s.baseline_occupancy_std for s in ordered], dtype=float)
    baseline_occ_minmax = metrics.minmax_error_array([s.baseline_occupancy_samples for s in ordered], baseline_occ_mean)
    baseline_off_occ_mean = np.array([s.baseline_off_occupancy_mean for s in ordered], dtype=float)
    baseline_off_occ_std = np.array([s.baseline_off_occupancy_std for s in ordered], dtype=float)
    baseline_off_occ_minmax = metrics.minmax_error_array(
        [s.baseline_off_occupancy_samples for s in ordered],
        baseline_off_occ_mean,
    )
    arrivals_per_hour_mean = np.array([s.arrivals_per_hour_mean for s in ordered], dtype=float)
    arrivals_per_hour_std = np.array([s.arrivals_per_hour_std for s in ordered], dtype=float)
    dropped_jobs_delta_total = np.array([s.dropped_jobs_delta_total for s in ordered], dtype=float)
    prop_sav_pct_mean = np.array([s.prop_savings_pct_mean for s in ordered], dtype=float)
    prop_sav_pct_std = np.array([s.prop_savings_pct_std for s in ordered], dtype=float)
    prop_sav_pct_off_mean = np.array([s.prop_savings_pct_off_mean for s in ordered], dtype=float)
    prop_sav_pct_off_std = np.array([s.prop_savings_pct_off_std for s in ordered], dtype=float)
    completion_mean = np.array([s.completion_rate_mean for s in ordered], dtype=float)
    completion_std = np.array([s.completion_rate_std for s in ordered], dtype=float)
    prop_eff_sav_pct_mean = np.array([s.prop_effective_savings_pct_mean for s in ordered], dtype=float)
    prop_eff_sav_pct_std = np.array([s.prop_effective_savings_pct_std for s in ordered], dtype=float)
    prop_eff_sav_pct_off_mean = np.array([s.prop_effective_savings_pct_off_mean for s in ordered], dtype=float)
    prop_eff_sav_pct_off_std = np.array([s.prop_effective_savings_pct_off_std for s in ordered], dtype=float)
    wait_delta_hours = np.array([s.wait_delta_hours for s in ordered], dtype=float)
    cost_per_1k_delta_base_off_mean = np.array([s.cost_per_1k_delta_pct_baseline_off_mean for s in ordered], dtype=float)
    cost_per_1k_delta_base_off_std = np.array([s.cost_per_1k_delta_pct_baseline_off_std for s in ordered], dtype=float)
    prop_power_delta_base_off_mean = np.array([s.prop_power_delta_pct_baseline_off_mean for s in ordered], dtype=float)
    prop_power_delta_base_off_std = np.array([s.prop_power_delta_pct_baseline_off_std for s in ordered], dtype=float)

    scale_min = float(np.min(scales))
    scale_max = float(np.max(scales))
    if scale_max <= scale_min:
        scale_max = scale_min + 1.0
    norm = matplotlib.colors.Normalize(vmin=scale_min, vmax=scale_max)
    cmap = plt.get_cmap("turbo")
    point_colors = cmap(norm(scales))
    colorbar_label = "Job arrival scale (point color)"

    def plot_colored_points(
        ax: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        xerr: np.ndarray | None = None,
        yerr: np.ndarray | None = None,
        xerr_range: np.ndarray | None = None,
        yerr_range: np.ndarray | None = None,
    ) -> None:
        for i, (xi, yi, c) in enumerate(zip(x, y, point_colors)):
            if not (np.isfinite(xi) and np.isfinite(yi)):
                continue
            metrics.draw_point(
                ax,
                float(xi),
                float(yi),
                c,
                xerr_std=metrics.error_at(xerr, i),
                yerr_std=metrics.error_at(yerr, i),
                xerr_range=metrics.error_at(xerr_range, i),
                yerr_range=metrics.error_at(yerr_range, i),
            )

    def draw_baseline_occupancy_pair(ax: plt.Axes) -> None:
        for i, (xv, c) in enumerate(zip(scales, point_colors)):
            y_base = float(baseline_occ_mean[i])
            y_base_off = float(baseline_off_occ_mean[i])
            if np.isfinite(xv) and np.isfinite(y_base):
                metrics.draw_point(
                    ax, xv, y_base, c, marker="o",
                    yerr_std=metrics.error_at(baseline_occ_std, i),
                    yerr_range=metrics.error_at(baseline_occ_minmax, i),
                )
            if np.isfinite(xv) and np.isfinite(y_base_off):
                metrics.draw_point(
                    ax, xv, y_base_off, c, marker="^",
                    yerr_std=metrics.error_at(baseline_off_occ_std, i),
                    yerr_range=metrics.error_at(baseline_off_occ_minmax, i),
                )
        ax.scatter([], [], marker="o", color="black", label="Baseline")
        ax.scatter([], [], marker="^", color="black", label="Baseline_off")
        ax.legend()

    panel_specs: list[tuple[str, Callable[[plt.Axes], None]]] = []

    def _panel(slug: str, title: str, xlabel: str, ylabel: str, draw_body: Callable[[plt.Axes], None]) -> None:
        def _draw(ax: plt.Axes) -> None:
            draw_body(ax)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
        panel_specs.append((slug, _draw))

    _panel(
        "01_scale_vs_agent_occupancy",
        "Arrival Scale vs Occupancy/Episode",
        "Job arrival scale",
        "Agent Occupancy (Nodes, %) / Episode",
        lambda ax: (
            plot_colored_points(ax, scales, occ_mean, yerr=occ_std, yerr_range=occ_minmax),
            metrics.maybe_plot_fit(ax, scales, occ_mean, fit),
        ),
    )
    _panel(
        "02_occupancy_vs_prop_savings",
        "Occupancy/Episode vs Proportional Savings (%)",
        "Agent Occupancy (Nodes, %) / Episode",
        "Prop Savings vs Baseline (%)",
        lambda ax: (
            plot_colored_points(ax, occ_mean, prop_sav_pct_mean, xerr=occ_std, yerr=prop_sav_pct_std),
            metrics.maybe_plot_fit(ax, occ_mean, prop_sav_pct_mean, fit),
        ),
    )
    _panel(
        "03_occupancy_vs_prop_savings_off",
        "Occupancy vs Proportional Savings_off (%)",
        "Agent Occupancy (Nodes, %) / Episode",
        "Prop Savings vs Baseline_off (%)",
        lambda ax: (
            plot_colored_points(ax, occ_mean, prop_sav_pct_off_mean, xerr=occ_std, yerr=prop_sav_pct_off_std),
            metrics.maybe_plot_fit(ax, occ_mean, prop_sav_pct_off_mean, fit),
        ),
    )
    _panel(
        "04_scale_vs_completion_rate",
        "Arrival Scale vs Agent Completion Rate",
        "Job arrival scale",
        "Completion Rate (%)",
        lambda ax: (plot_colored_points(ax, scales, completion_mean, yerr=completion_std), metrics.maybe_plot_fit(ax, scales, completion_mean, fit)),
    )
    _panel(
        "05_occupancy_vs_prop_effective_savings",
        "Occupancy vs Proportional Effective Savings (%)",
        "Agent Occupancy (Nodes, %) / Episode",
        "Prop Effective Savings vs Baseline (% adjusted)",
        lambda ax: (
            plot_colored_points(ax, occ_mean, prop_eff_sav_pct_mean, xerr=occ_std, yerr=prop_eff_sav_pct_std),
            metrics.maybe_plot_fit(ax, occ_mean, prop_eff_sav_pct_mean, fit),
        ),
    )
    _panel(
        "06_occupancy_vs_prop_effective_savings_off",
        "Occupancy vs Proportional Effective Savings_off (%)",
        "Agent Occupancy (Nodes, %) / Episode",
        "Prop Effective Savings vs Baseline_off (% adjusted)",
        lambda ax: (
            plot_colored_points(ax, occ_mean, prop_eff_sav_pct_off_mean, xerr=occ_std, yerr=prop_eff_sav_pct_off_std),
            metrics.maybe_plot_fit(ax, occ_mean, prop_eff_sav_pct_off_mean, fit),
        ),
    )
    _panel(
        "07_occupancy_vs_average_wait_delta",
        "Occupancy vs Average Wait Delta",
        "Agent Occupancy (Nodes, %) / Episode",
        "Average Wait Delta (Agent - Baseline, hours)",
        lambda ax: (
            plot_colored_points(ax, occ_mean, wait_delta_hours, xerr=occ_std),
            metrics.maybe_plot_fit(ax, occ_mean, wait_delta_hours, fit),
        ),
    )
    _panel(
        "08_occupancy_vs_cost_per_1k_delta_baseline_off",
        "Occupancy vs Cost/1k Delta vs Baseline_off",
        "Agent Occupancy (Nodes, %) / Episode",
        "(Baseline_off - Agent) / Baseline_off  [%]",
        lambda ax: (
            plot_colored_points(ax, occ_mean, cost_per_1k_delta_base_off_mean, xerr=occ_std, yerr=cost_per_1k_delta_base_off_std),
            metrics.maybe_plot_fit(ax, occ_mean, cost_per_1k_delta_base_off_mean, fit),
        ),
    )
    _panel(
        "09_occupancy_vs_prop_power_delta_baseline_off",
        "Occupancy vs Prop Power Delta vs Baseline_off",
        "Agent Occupancy (Nodes, %) / Episode",
        "Prop Power Delta vs Baseline_off (%)",
        lambda ax: (
            plot_colored_points(ax, occ_mean, prop_power_delta_base_off_mean, xerr=occ_std, yerr=prop_power_delta_base_off_std),
            metrics.maybe_plot_fit(ax, occ_mean, prop_power_delta_base_off_mean, fit),
        ),
    )
    _panel(
        "10_scale_vs_baseline_occupancies",
        "Arrival Scale vs Baseline Occupancies",
        "Job arrival scale",
        "Baseline Occupancy (Nodes, %) / Episode",
        draw_baseline_occupancy_pair,
    )
    _panel(
        "11_scale_vs_jobs_per_hour",
        "Arrival Scale vs Job Arrivals/Hour",
        "Job arrival scale",
        "Job Arrivals/Hour (mean ± std)",
        lambda ax: (plot_colored_points(ax, scales, arrivals_per_hour_mean, yerr=arrivals_per_hour_std), metrics.maybe_plot_fit(ax, scales, arrivals_per_hour_mean, fit)),
    )
    _panel(
        "12_scale_vs_dropped_jobs_delta",
        "Arrival Scale vs Dropped Jobs Delta",
        "Job arrival scale",
        "Dropped Jobs Delta (Agent - Baseline)",
        lambda ax: (plot_colored_points(ax, scales, dropped_jobs_delta_total), metrics.maybe_plot_fit(ax, scales, dropped_jobs_delta_total, fit)),
    )

    fig, axes = plt.subplots(4, 3, figsize=(22, 22), constrained_layout=True)
    for ax, (_, draw_fn) in zip(axes.ravel(), panel_specs):
        draw_fn(ax)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.02)
    cbar.set_label(colorbar_label)

    fig.suptitle(f"Job-Arrival-Scale Sweep ({replay_mode})", fontsize=14)
    fig.savefig(path, dpi=220)
    plt.close(fig)

    if individual_dir is not None:
        individual_dir.mkdir(parents=True, exist_ok=True)
        for slug, draw_fn in panel_specs:
            panel_path = individual_dir / f"{slug}.png"
            fig_i, ax_i = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
            draw_fn(ax_i)
            sm_i = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm_i.set_array([])
            cbar_i = fig_i.colorbar(sm_i, ax=ax_i, pad=0.02)
            cbar_i.set_label(colorbar_label)
            fig_i.savefig(panel_path, dpi=220)
            plt.close(fig_i)


def normalize_scale(value: float) -> float:
    return float(f"{value:.6f}")


def parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def unique_scales_sorted(values: list[float]) -> list[float]:
    return sorted({normalize_scale(float(v)) for v in values})


def build_scale_grid(min_scale: float, max_scale: float, n: int) -> list[float]:
    if n <= 1 or abs(max_scale - min_scale) < 1e-12:
        return [normalize_scale(min_scale)]
    vals = np.linspace(min_scale, max_scale, n)
    return unique_scales_sorted([float(v) for v in vals])


def select_scale_schedule(args: argparse.Namespace) -> list[float]:
    if args.scales:
        return unique_scales_sorted(parse_float_list(args.scales))
    return build_scale_grid(args.min_scale, args.max_scale, args.num_points)


def with_scale_one_first(scales: list[float]) -> list[float]:
    one = normalize_scale(1.0)
    return [one] + [s for s in scales if abs(s - one) > 1e-12]


def format_scale_for_filename(scale: float) -> str:
    return f"{scale:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare --jobs exact replay aggregate vs --hourly-jobs sampling (--job-arrival-scale) and fit occupancy trend lines."
    )

    # Core train.py params.
    parser.add_argument("--prices", default="")
    parser.add_argument("--jobs", required=True, help="Path forwarded to train.py --jobs")
    parser.add_argument(
        "--hourly-jobs",
        default="",
        help="Path forwarded to train.py --hourly-jobs. Defaults to --jobs when omitted.",
    )
    parser.add_argument("--session", default="")
    parser.add_argument("--efficiency-weight", type=float, default=0.6)
    parser.add_argument("--price-weight", type=float, default=0.1)
    parser.add_argument("--idle-weight", type=float, default=0.1)
    parser.add_argument("--job-age-weight", type=float, default=0.2)
    parser.add_argument("--drop-weight", type=float, default=0.0)
    parser.add_argument("--eval-months", type=int, default=12)
    parser.add_argument("--model", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=None, help="Forwarded to train.py (sampling mode only).")

    parser.add_argument("--plot-dashboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oracle", action="store_true", default=False, help="Enable oracle for evaluation")
    parser.add_argument("--dashboard-hours", type=int, default=None)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Run only exact replay aggregate mode when selected alone.",
    )
    parser.add_argument(
        "--sampler",
        action="store_true",
        help="Run only hourly sampler mode when selected alone.",
    )

    # Scale sweep controls.
    parser.add_argument(
        "--scales",
        type=str,
        default="",
        help="Explicit comma-separated --job-arrival-scale values. If set, min/max/num-points are ignored.",
    )
    parser.add_argument("--min-scale", type=float, default=0.5)
    parser.add_argument("--max-scale", type=float, default=1.5)
    parser.add_argument("--num-points", type=int, default=7)

    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--save-logs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--echo-train-output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fit", action="store_true", default=False, help="Enable polynomial fitting of datasets")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.num_points < 2 and not args.scales:
        parser.error("--num-points must be >= 2 when --scales is not provided")
    if args.eval_months <= 0:
        parser.error("--eval-months must be > 0")
    if args.min_scale < 0.0:
        parser.error("--min-scale must be >= 0")
    if args.max_scale < args.min_scale:
        parser.error("--max-scale must be >= --min-scale")

    project_root = Path(__file__).resolve().parent
    train_py = project_root / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"Could not find train.py at: {train_py}")

    jobs_path = Path(args.jobs).expanduser()
    if not jobs_path.exists():
        raise FileNotFoundError(f"Could not find jobs file: {jobs_path}")
    if args.replay or args.sampler:
        mode_names = []
        if args.replay:
            mode_names.append("exact_replay_aggregate")
        if args.sampler:
            mode_names.append("sampling")
    else:
        mode_names = ["exact_replay_aggregate", "sampling"]

    if "sampling" in mode_names:
        if not args.hourly_jobs:
            args.hourly_jobs = str(jobs_path)
            print(f"[info] sampling mode will reuse --jobs as --hourly-jobs: {args.hourly_jobs}")
        hourly_jobs_path = Path(args.hourly_jobs).expanduser()
        if not hourly_jobs_path.exists():
            raise FileNotFoundError(f"Could not find hourly jobs file: {hourly_jobs_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir_name = build_analysis_dir_name(
            prefix="arrivalscale_occupancy_sweep",
            timestamp=timestamp,
            model=args.model,
            efficiency_weight=args.efficiency_weight,
            price_weight=args.price_weight,
            idle_weight=args.idle_weight,
            job_age_weight=args.job_age_weight,
        )
        out_dir = project_root / "analysis" / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = out_dir / "logs"
    if args.save_logs:
        logs_dir.mkdir(parents=True, exist_ok=True)

    selected_scales = select_scale_schedule(args)
    # Always warm-start each mode at scale=1.0, then continue with requested sweep scales.
    scales = with_scale_one_first(selected_scales)

    all_stats: list[RunStats] = []

    for replay_mode in mode_names:
        for scale in scales:
            stats, raw_output = run_scale_eval(args, project_root, scale, replay_mode)
            all_stats.append(stats)
            if args.save_logs:
                scale_part = format_scale_for_filename(scale)
                log_path = logs_dir / f"{replay_mode}_scale_{scale_part}.log"
                log_path.write_text(raw_output)

    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"
    write_summary_csv(csv_path, all_stats)
    with json_path.open("w") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(),
                "selected_scales": selected_scales,
                "scales": scales,
                "modes": mode_names,
                "args": vars(args),
                "results": [asdict(s) for s in all_stats],
            },
            f,
            indent=2,
        )

    plot_paths: list[Path] = []
    individual_plot_dirs: list[Path] = []
    for replay_mode in mode_names:
        mode_stats = [s for s in all_stats if s.replay_mode == replay_mode]
        plot_path = out_dir / f"trendlines_{replay_mode}.png"
        individual_dir = out_dir / "plots_individual" / replay_mode
        make_plot(plot_path, mode_stats, replay_mode, fit=args.fit, individual_dir=individual_dir)
        plot_paths.append(plot_path)
        individual_plot_dirs.append(individual_dir)

    print("\nSweep complete.")
    print(f"  Scales: {scales}")
    print(f"  Modes: {mode_names}")
    print(f"  Evaluation months: {args.eval_months}")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    for p in plot_paths:
        print(f"  Plot: {p}")
    for p in individual_plot_dirs:
        print(f"  Individual Plots: {p}")


if __name__ == "__main__":
    main()
