#!/usr/bin/env python3
"""
Sweep random seeds in --hourly-jobs mode (fixed --job-arrival-scale 1.0) and analyze:
1) seed -> agent occupancy (nodes)
2) occupancy -> proportional savings (%)
3) occupancy -> proportional savings_off (%)
4) seed -> completion rate
5) occupancy -> proportional effective savings (%)
6) occupancy -> proportional effective savings_off (%)
7) occupancy -> average wait delta (agent - baseline)
8) occupancy -> (baseline_off - agent) cost_per_1000_completed_jobs / baseline_off
9) occupancy -> (baseline_off - agent) proportional power / baseline_off
10) seed -> baseline and baseline_off occupancies
11) seed -> mean jobs/hour (with std)
12) seed -> dropped-jobs delta (agent - baseline)

For each seed, this script runs train.py in evaluation mode for one year
(12 months = 24 episodes), parses per-episode metrics from stdout, computes
mean/std, and fits optional polynomial trend lines.

FAST DEBUG MODE:
python analyze_seed_occupancy.py \
  --hourly-jobs ./data/allusers-gpu-30.log \
  --eval-months 1 --seeds 1,2,3 --no-plot-dashboard
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


FIXED_JOB_ARRIVAL_SCALE = 1.0


def build_train_command(args: argparse.Namespace, seed: int) -> list[str]:
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
        "--hourly-jobs",
        args.hourly_jobs,
        "--job-arrival-scale",
        f"{FIXED_JOB_ARRIVAL_SCALE:.1f}",
        "--seed",
        str(seed),
        "--seed-path",
        args.seed_path
    ]
    if args.plot_dashboard:
        cmd.append("--plot-dashboard")
    if args.dashboard_hours is not None:
        cmd.extend(["--dashboard-hours", str(args.dashboard_hours)])
    return cmd


def run_seed_eval(args: argparse.Namespace, project_root: Path, seed: int) -> tuple[RunStats, str]:
    command = build_train_command(args, seed)
    print(f"[run] seed={seed}: {shlex.join(command)}")
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
            f"train.py failed for seed={seed} with code {completed.returncode}.\n"
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
        print(f"[warn] seed={seed}: could not parse run-level wait summary; effective savings may be NaN.")
        agent_avg_wait_hours = float(np.mean(avg_wait))
        baseline_avg_wait_hours = float("nan")
    else:
        agent_avg_wait_hours = float(agent_wait_summary)
        baseline_avg_wait_hours = float(baseline_wait_summary)
    arrivals_per_hour_mean, arrivals_per_hour_std = metrics.parse_arrivals_summary(combined_output)
    if arrivals_per_hour_mean is None or arrivals_per_hour_std is None:
        print(f"[warn] seed={seed}: could not parse run-level arrivals/hour summary; values set to NaN.")
        arrivals_per_hour_mean = float("nan")
        arrivals_per_hour_std = float("nan")
    dropped_jobs_agent_total, dropped_jobs_baseline_total = metrics.parse_dropped_totals_summary(combined_output)
    if dropped_jobs_agent_total is None:
        dropped_jobs_agent_total = float(np.sum(agent_dropped))
        print(f"[warn] seed={seed}: could not parse run-level agent dropped total; using sum of episode Dropped= values.")
    if dropped_jobs_baseline_total is None:
        dropped_jobs_baseline_total = float("nan")
        print(f"[warn] seed={seed}: could not parse run-level baseline dropped total; defaulting to NaN.")
    stats = metrics.make_run_stats(
        sweep_key=float(seed),
        replay_mode="",
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
        f"[ok ] seed={seed}: "
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


def write_summary_csv(path: Path, stats_by_seed: list[RunStats]) -> None:
    fieldnames = ["seed"] + metrics.CSV_COMMON_FIELDNAMES
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in sorted(stats_by_seed, key=lambda x: x.sweep_key):
            writer.writerow({
                "seed": int(s.sweep_key),
                **metrics.csv_common_row(s),
            })


def make_plot(
    path: Path,
    stats_by_seed: list[RunStats],
    fit: bool = False,
    individual_dir: Path | None = None,
) -> None:
    ordered = sorted(stats_by_seed, key=lambda x: x.sweep_key)
    if not ordered:
        return

    seeds = np.array([s.sweep_key for s in ordered], dtype=float)
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

    seed_min = float(np.min(seeds))
    seed_max = float(np.max(seeds))
    if seed_max <= seed_min:
        seed_max = seed_min + 1.0
    norm = matplotlib.colors.Normalize(vmin=seed_min, vmax=seed_max)
    cmap = plt.get_cmap("turbo")
    point_colors = cmap(norm(seeds))
    colorbar_label = "Random seed (point color)"

    def _apply_seed_ticks(ax: plt.Axes) -> None:
        if seeds.size <= 15:
            ax.set_xticks(seeds.tolist())

    def plot_colored_points(
        ax: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        xerr: np.ndarray | None = None,
        yerr: np.ndarray | None = None,
        xerr_range: np.ndarray | None = None,
        yerr_range: np.ndarray | None = None,
        seed_x_axis: bool = False,
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
        if seed_x_axis:
            _apply_seed_ticks(ax)

    def draw_baseline_occupancy_pair(ax: plt.Axes) -> None:
        for i, (xv, c) in enumerate(zip(seeds, point_colors)):
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
        _apply_seed_ticks(ax)
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
        "01_seed_vs_agent_occupancy",
        "Seed vs Occupancy/Episode",
        "Random seed",
        "Agent Occupancy (Nodes, %) / Episode",
        lambda ax: (
            plot_colored_points(ax, seeds, occ_mean, yerr=occ_std, yerr_range=occ_minmax, seed_x_axis=True),
            metrics.maybe_plot_fit(ax, seeds, occ_mean, fit),
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
        "04_seed_vs_completion_rate",
        "Seed vs Agent Completion Rate",
        "Random seed",
        "Completion Rate (%)",
        lambda ax: (plot_colored_points(ax, seeds, completion_mean, yerr=completion_std, seed_x_axis=True), metrics.maybe_plot_fit(ax, seeds, completion_mean, fit)),
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
        "10_seed_vs_baseline_occupancies",
        "Seed vs Baseline Occupancies",
        "Random seed",
        "Baseline Occupancy (Nodes, %) / Episode",
        draw_baseline_occupancy_pair,
    )
    _panel(
        "11_seed_vs_jobs_per_hour",
        "Seed vs Job Arrivals/Hour",
        "Random seed",
        "Job Arrivals/Hour (mean ± std)",
        lambda ax: (plot_colored_points(ax, seeds, arrivals_per_hour_mean, yerr=arrivals_per_hour_std, seed_x_axis=True), metrics.maybe_plot_fit(ax, seeds, arrivals_per_hour_mean, fit)),
    )
    _panel(
        "12_seed_vs_dropped_jobs_delta",
        "Seed vs Dropped Jobs Delta",
        "Random seed",
        "Dropped Jobs Delta (Agent - Baseline)",
        lambda ax: (plot_colored_points(ax, seeds, dropped_jobs_delta_total, seed_x_axis=True), metrics.maybe_plot_fit(ax, seeds, dropped_jobs_delta_total, fit)),
    )

    fig, axes = plt.subplots(4, 3, figsize=(22, 22), constrained_layout=True)
    for ax, (_, draw_fn) in zip(axes.ravel(), panel_specs):
        draw_fn(ax)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.02)
    if seeds.size <= 15:
        cbar.set_ticks(seeds.tolist())
    cbar.set_label(colorbar_label)

    fig.suptitle("Hourly-Jobs Seed Sweep", fontsize=14)
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
            if seeds.size <= 15:
                cbar_i.set_ticks(seeds.tolist())
            cbar_i.set_label(colorbar_label)
            fig_i.savefig(panel_path, dpi=220)
            plt.close(fig_i)


def build_seed_schedule(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return metrics.unique_ints_sorted(metrics.parse_int_list(args.seeds))
    return metrics.unique_ints_sorted(list(range(args.min_seed, args.max_seed + 1, args.seed_step)))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep random seeds in --hourly-jobs mode and fit occupancy/savings trend lines."
    )

    parser.add_argument("--prices", default="./data/prices_2023.csv")
    parser.add_argument("--hourly-jobs", required=True, help="Path forwarded to train.py --hourly-jobs")
    parser.add_argument("--session", default="")
    parser.add_argument("--efficiency-weight", type=float, default=0.6)
    parser.add_argument("--price-weight", type=float, default=0.1)
    parser.add_argument("--idle-weight", type=float, default=0.1)
    parser.add_argument("--job-age-weight", type=float, default=0.2)
    parser.add_argument("--drop-weight", type=float, default=0.0)
    parser.add_argument("--eval-months", type=int, default=12)
    parser.add_argument("--model", type=int, default=1000000)

    parser.add_argument("--plot-dashboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dashboard-hours", type=int, default=None)

    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Explicit comma-separated seed list. If set, min/max/step are ignored.",
    )
    parser.add_argument("--min-seed", type=int, default=100)
    parser.add_argument("--max-seed", type=int, default=700)
    parser.add_argument("--seed-step", type=int, default=50)

    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--save-logs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--echo-train-output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fit", action="store_true", default=False, help="Enable polynomial fitting of datasets")
    parser.add_argument("--seed-path", default="", help="Path if models are saved by seed (forwarded to train.py)")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.eval_months <= 0:
        parser.error("--eval-months must be > 0")
    if not args.seeds:
        if args.seed_step <= 0:
            parser.error("--seed-step must be > 0")
        if args.max_seed < args.min_seed:
            parser.error("--max-seed must be >= --min-seed")

    project_root = Path(__file__).resolve().parent
    train_py = project_root / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"Could not find train.py at: {train_py}")

    hourly_jobs_path = Path(args.hourly_jobs).expanduser()
    if not hourly_jobs_path.exists():
        raise FileNotFoundError(f"Could not find hourly jobs file: {hourly_jobs_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        prefix = "hourlyjobs_seed_occupancy_sweep"
        if args.seed_path != "":
            prefix += "_train" + args.seed_path
        out_dir_name = build_analysis_dir_name(
            prefix=prefix,
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

    selected_seeds = build_seed_schedule(args)
    if not selected_seeds:
        parser.error("No seeds selected; provide --seeds or a valid --min-seed/--max-seed range.")
    all_stats: list[RunStats] = []

    if args.seed_path != "":
        args.session = f"{args.session}/{args.seed_path}"

    for seed in selected_seeds:
        stats, raw_output = run_seed_eval(args, project_root, seed)
        all_stats.append(stats)
        if args.save_logs:
            log_path = logs_dir / f"seed_{seed}.log"
            log_path.write_text(raw_output)

    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"
    plot_path = out_dir / "trendlines.png"
    individual_plots_dir = out_dir / "plots_individual"

    write_summary_csv(csv_path, all_stats)
    with json_path.open("w") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(),
                "selected_seeds": selected_seeds,
                "job_arrival_scale": FIXED_JOB_ARRIVAL_SCALE,
                "args": vars(args),
                "results": [asdict(s) for s in all_stats],
            },
            f,
            indent=2,
        )
    make_plot(plot_path, all_stats, fit=args.fit, individual_dir=individual_plots_dir)

    print("\nSweep complete.")
    print(f"  Seeds: {selected_seeds}")
    print(f"  Job arrival scale: {FIXED_JOB_ARRIVAL_SCALE:.1f}")
    print(f"  Evaluation months: {args.eval_months}")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  Plot: {plot_path}")
    print(f"  Individual Plots: {individual_plots_dir}")


if __name__ == "__main__":
    main()
