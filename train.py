from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from torchinfo import summary
import os
from src.environment import ComputeClusterEnv, Weights, PlottingComplete
from src.plot_config import PlotConfig
from src.callbacks import ComputeClusterCallback
from src.plotter import plot_dashboard, plot_cumulative_savings, plot_episode_summary
import re
import glob
import argparse
import sys
import pandas as pd
from src.workloadgen import WorkloadGenerator
from src.workloadgen_cli import add_workloadgen_args, build_workloadgen_config
from src.config import MAX_NODES, CORES_PER_NODE, EPISODE_HOURS
import time


# Train.py passes strings; the env treats "" as falsy in some places and truthy in others.
# To be safe: normalize "" -> None here.
def norm_path(x):
    return None if (x is None or str(x).strip() == "") else x


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """Return numerator/denominator, or None when denominator is not positive."""
    return (numerator / denominator) if denominator > 0 else None


def fmt_optional(value: float | None, precision: int = 2, thousands: bool = False) -> str:
    """Format float values for logs, using 'n/a' when value is undefined."""
    if value is None:
        return "n/a"
    return f"{value:,.{precision}f}" if thousands else f"{value:.{precision}f}"


STEPS_PER_ITERATION = 100000


def main():
    parser = argparse.ArgumentParser(description="Run the Compute Cluster Environment with optional rendering.")
    parser.add_argument('--render', type=str, default='none', choices=['human', 'none'], help='Render mode for the environment (default: none).')
    parser.add_argument('--quick-plot', action='store_true', help='In "human" render mode, skip quickly to the plot (default: False).')
    parser.add_argument('--plot-once', action='store_true', help='In "human" render mode, exit after the first plot.')
    parser.add_argument('--prices', type=str, nargs='?', const="", default="", help='Path to the CSV file containing electricity prices (Date,Price)')
    parser.add_argument('--job-durations', type=str, nargs='?', const="", default="", help='Path to a file containing job duration samples (for use with durations_sampler)')
    parser.add_argument('--jobs', type=str, nargs='?', const="", default="", help='Path to a file containing job samples (for use with jobs_sampler)')
    parser.add_argument('--hourly-jobs', type=str, nargs='?', const="", default="", help='Path to Slurm log file for hourly statistical sampling (for use with hourly_sampler)')
    parser.add_argument('--job-arrival-scale', type=float, default=1.0, help='Scale sampled arrivals per step (1.0 = unchanged).')
    parser.add_argument('--jobs-exact-replay', action='store_true', help='For --jobs mode, replay raw jobs in timeline order (no template aggregation).')
    parser.add_argument('--jobs-exact-replay-aggregate', action='store_true', help='With --jobs-exact-replay, aggregate each sampled raw time-bin before enqueueing.')
    parser.add_argument('--plot-rewards', action='store_true', help='Per step, plot rewards for all possible num_idle_nodes & num_used_nodes (default: False).')
    parser.add_argument('--plot-eff-reward', action=argparse.BooleanOptionalAction, default=True, help='Include efficiency reward in the plot (dashed line).')
    parser.add_argument('--plot-price-reward', action=argparse.BooleanOptionalAction, default=True, help='Include price reward in the plot (dashed line).')
    parser.add_argument('--plot-idle-penalty', action=argparse.BooleanOptionalAction, default=True, help='Include idle penalty in the plot (dashed line).')
    parser.add_argument('--plot-job-age-penalty', action=argparse.BooleanOptionalAction, default=True, help='Include job age penalty in the plot (dashed line).')
    parser.add_argument('--plot-total-reward', action=argparse.BooleanOptionalAction, default=True, help='Include total reward per step in the dashboard (raw values).')
    parser.add_argument('--plot-price', action=argparse.BooleanOptionalAction, default=True, help='Plot electricity price.')
    parser.add_argument('--plot-online-nodes', action=argparse.BooleanOptionalAction, default=True, help='Plot online nodes.')
    parser.add_argument('--plot-used-nodes', action=argparse.BooleanOptionalAction, default=True, help='Plot used nodes.')
    parser.add_argument('--plot-job-queue', action=argparse.BooleanOptionalAction, default=True, help='Plot job queue.')
    parser.add_argument('--ent-coef', type=float, default=0.0, help='Entropy coefficient for the loss calculation (default: 0.0) (Passed to PPO).')
    parser.add_argument("--efficiency-weight", type=float, default=0.7, help="Weight for efficiency reward")
    parser.add_argument("--price-weight", type=float, default=0.2, help="Weight for price reward")
    parser.add_argument("--idle-weight", type=float, default=0.1, help="Weight for idle penalty")
    parser.add_argument("--job-age-weight", type=float, default=0.0, help="Weight for job age penalty")
    parser.add_argument("--drop-weight", type=float, default=0.0, help="Weight for dropped jobs penalty (WIP - default 0.0)")
    parser.add_argument("--iter-limit", type=int, default=0, help=f"Max number of training iterations (1 iteration = {STEPS_PER_ITERATION} steps)")
    parser.add_argument("--session", default="default", help="Session ID")
    parser.add_argument("--evaluate-savings", action='store_true', help="Load latest model and evaluate long-term savings (no training)")
    parser.add_argument("--eval-months", type=int, default=12, help="Months to evaluate for savings analysis (default: 12, only used with --evaluate-savings)")
    add_workloadgen_args(parser)
    parser.add_argument("--plot-dashboard", action="store_true", help="Generate dashboard plot (per-hour panels + cumulative savings).")
    parser.add_argument("--dashboard-hours", type=int, default=24*14, help="Hours to show in dashboard time-series panels (default: 336).")
    parser.add_argument("--model", type=int, default=None, help="Load a specific model by timestep number (e.g. 5000000 loads 5000000.zip).")
    parser.add_argument("--net-arch", type=str, default="64,64", help="Hidden layer sizes for policy and value networks (comma-separated, e.g., '256,128' or '512,256,128')")
    parser.add_argument("--device", type=str, default="auto", help="Device for training: 'auto' (default, uses CUDA if available), 'cuda', 'cpu'")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (seeds environment, numpy, torch, and PPO)")
    parser.add_argument("--print-policy", action="store_true", help="Print structure of the policy network.")

    args = parser.parse_args()
    if args.job_arrival_scale < 0.0:
        parser.error("--job-arrival-scale must be >= 0.0")
    if args.jobs_exact_replay and not norm_path(args.jobs):
        parser.error("--jobs-exact-replay requires --jobs")
    if args.jobs_exact_replay_aggregate and not args.jobs_exact_replay:
        parser.error("--jobs-exact-replay-aggregate requires --jobs-exact-replay")
    if args.workload_gen and args.job_arrival_scale != 1.0:
        print(
            "Warning: --job-arrival-scale is not allowed with --workload-gen; "
            "resetting it to 1.0. Use workload generator arrival settings instead.",
            file=sys.stderr,
        )
        args.job_arrival_scale = 1.0

    prices_file_path = args.prices
    job_durations_file_path = args.job_durations
    jobs_file_path = args.jobs
    hourly_jobs_file_path = args.hourly_jobs

    # Set random seed for reproducibility
    if args.seed is not None:
        set_random_seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    if norm_path(prices_file_path):
        df = pd.read_csv(prices_file_path, parse_dates=['Date'])
        prices = df['Price'].values.tolist()
        print(f"Loaded {len(prices)} prices from CSV: {prices_file_path}")
        # print("First few prices:", prices[:30])
    else:
        prices = None
        print("No CSV file provided. Using default price generation.")

    weights = Weights(
        efficiency_weight=args.efficiency_weight,
        price_weight=args.price_weight,
        idle_weight=args.idle_weight,
        job_age_weight=args.job_age_weight,
        drop_weight=args.drop_weight
    )

    weights_prefix = f"e{weights.efficiency_weight}_p{weights.price_weight}_i{weights.idle_weight}_a{weights.job_age_weight}_d{weights.drop_weight}"

    models_dir = f"sessions/{args.session}/models/{weights_prefix}/"
    log_dir = f"sessions/{args.session}/logs/{weights_prefix}/"
    plots_dir = f"sessions/{args.session}/plots/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Load Workload Generator:

    wg_cfg = build_workloadgen_config(args)
    workload_gen = WorkloadGenerator(wg_cfg) if wg_cfg is not None else None

    plot_config = PlotConfig(
        quick_plot=args.quick_plot,
        plot_once=args.plot_once,
        plot_eff_reward=args.plot_eff_reward,
        plot_price_reward=args.plot_price_reward,
        plot_idle_penalty=args.plot_idle_penalty,
        plot_job_age_penalty=args.plot_job_age_penalty,
        plot_total_reward=args.plot_total_reward,
        plot_price=args.plot_price,
        plot_online_nodes=args.plot_online_nodes,
        plot_used_nodes=args.plot_used_nodes,
        plot_job_queue=args.plot_job_queue,
    )

    env = ComputeClusterEnv(weights=weights,
                            session=args.session,
                            render_mode=args.render,
                            external_prices=prices,
                            external_durations=norm_path(job_durations_file_path),
                            external_jobs=norm_path(jobs_file_path),
                            external_hourly_jobs=norm_path(hourly_jobs_file_path),
                            plot_config=plot_config,
                            steps_per_iteration=STEPS_PER_ITERATION,
                            evaluation_mode=args.evaluate_savings,
                            workload_gen=workload_gen,
                            job_arrival_scale=args.job_arrival_scale,
                            jobs_exact_replay=args.jobs_exact_replay,
                            jobs_exact_replay_aggregate=args.jobs_exact_replay_aggregate)
    env.reset(seed=args.seed)

    # Check if there are any saved models in models_dir
    model_files = glob.glob(models_dir + "*.zip")
    latest_model_file = None
    if model_files:
        # Sort the files by extracting the timestep number from the filename and converting it to an integer
        model_files.sort(key=lambda filename: int(re.match(r"(\d+)", os.path.basename(filename)).group()))
        if args.model is not None:
            selected = os.path.join(models_dir, f"{args.model}.zip")
            if os.path.exists(selected):
                latest_model_file = selected
            else:
                print(f"Requested model not found: {selected}. Falling back to latest model.")
                latest_model_file = model_files[-1]
        else:
            latest_model_file = model_files[-1]  # Get the last file after sorting, which should be the one with the most timesteps
        print(f"Found a saved model: {latest_model_file}")
        model = PPO.load(latest_model_file, env=env, tensorboard_log=log_dir, n_steps=64, batch_size=64, device=args.device)
    else:
        print(f"Starting a new model training...")
        # Parse network architecture from comma-separated string (e.g., "256,128" -> [256, 128])
        net_arch_layers = [int(x) for x in args.net_arch.split(',')]
        policy_kwargs = dict(
            # pi = policy (actor) network, vf = value function (critic) network
            net_arch=dict(pi=net_arch_layers, vf=net_arch_layers)
        )
        print(f"Network architecture: {net_arch_layers}")
        model = PPO('MultiInputPolicy', env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir, ent_coef=args.ent_coef, n_steps=64, batch_size=64, device=args.device, verbose=1)

    print(f"Device: {model.device}")
    if args.print_policy:
        print(model.policy)
        summary(model.policy, depth=4)

    iters = 0

    # If we're continuing from a saved model, adjust iters so that filenames continue sequentially
    if latest_model_file:
        try:
            # Assumes the filename format is "{models_dir}/{STEPS_PER_ITERATION * iters}.zip"
            iters = int(os.path.basename(latest_model_file).split('.')[0]) // STEPS_PER_ITERATION
        except ValueError:
            # If the filename doesn't follow expected format, default to 0
            iters = 0

    env.set_progress(iters)

    if args.evaluate_savings:
        if not latest_model_file:
            print("Error: No trained model found for evaluation!")
            print(f"Expected model files in: {models_dir}")
            print("Train a model first, then run evaluation mode.")
            return

        print(f"=== EVALUATION MODE ===")
        print(f"Evaluation period: {args.eval_months} months ({args.eval_months * 2} episodes, Each episode = 2 weeks)")

        num_episodes = args.eval_months * 2 # 2 episodes per month
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            step_count = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                step_count += 1
                if step_count%1000==0:
                    print(f"Episode {episode + 1}, Step {step_count}, Action: {action}, Reward: {reward:.2f}, Total Reward: {episode_reward:.2f}, Total Cost: €{env.metrics.total_cost:.2f}")
                done = terminated or truncated

            savings_vs_baseline = env.metrics.baseline_cost - env.metrics.total_cost
            savings_vs_baseline_off = env.metrics.baseline_cost_off - env.metrics.total_cost
            completion_rate = (env.metrics.jobs_completed / env.metrics.jobs_submitted * 100) if env.metrics.jobs_submitted > 0 else 0
            avg_wait = env.metrics.total_job_wait_time / env.metrics.jobs_completed if env.metrics.jobs_completed > 0 else 0
            agent_power_mwh = env.metrics.episode_total_power_consumption_mwh
            baseline_power_mwh = env.metrics.episode_baseline_power_consumption_mwh
            baseline_power_off_mwh = env.metrics.episode_baseline_power_consumption_off_mwh
            agent_mean_price = (env.metrics.episode_total_cost / agent_power_mwh) if agent_power_mwh > 0 else 0.0
            baseline_mean_price = (env.metrics.episode_baseline_cost / baseline_power_mwh) if baseline_power_mwh > 0 else 0.0
            baseline_off_mean_price = (env.metrics.episode_baseline_cost_off / baseline_power_off_mwh) if baseline_power_off_mwh > 0 else 0.0
            agent_cost_per_1000_completed = safe_ratio(env.metrics.total_cost * 1000.0, env.metrics.jobs_completed)
            baseline_cost_per_1000_completed = safe_ratio(env.metrics.baseline_cost * 1000.0, env.metrics.baseline_jobs_completed)
            # baseline_off is a cost variant of baseline scheduling, so it uses the same completed-job count.
            baseline_off_cost_per_1000_completed = safe_ratio(env.metrics.baseline_cost_off * 1000.0, env.metrics.baseline_jobs_completed)
            dropped_jobs_per_saved_euro = safe_ratio(env.metrics.episode_jobs_dropped, savings_vs_baseline) if savings_vs_baseline > 0 else None
            dropped_jobs_per_saved_euro_off = safe_ratio(env.metrics.episode_jobs_dropped, savings_vs_baseline_off) if savings_vs_baseline_off > 0 else None
            print(f"  Episode {episode + 1}: "
                f"Agent Cost=€{env.metrics.total_cost:.0f}, "
                f"Baseline Cost=€{env.metrics.baseline_cost:.0f} | Baseline Off=€{env.metrics.baseline_cost_off:.0f}, "
                f"Savings=€{savings_vs_baseline:.0f}/€{savings_vs_baseline_off:.0f}, "
                f"Power={agent_power_mwh:.1f}/{baseline_power_mwh:.1f}/{baseline_power_off_mwh:.1f} MWh (agent/base/base_off), "
                f"MeanPrice={agent_mean_price:.2f}/{baseline_mean_price:.2f}/{baseline_off_mean_price:.2f} €/MWh (agent/base/base_off), "
                f"CostPer1kCompleted={fmt_optional(agent_cost_per_1000_completed, 1, thousands=True)}/{fmt_optional(baseline_cost_per_1000_completed, 1, thousands=True)}/{fmt_optional(baseline_off_cost_per_1000_completed, 1, thousands=True)} €/1k (agent/base/base_off), "
                f"DroppedPerSavedEuro={fmt_optional(dropped_jobs_per_saved_euro, 6)}/{fmt_optional(dropped_jobs_per_saved_euro_off, 6)} jobs/€ (vs base/base_off), "
                f"Jobs={env.metrics.jobs_completed}/{env.metrics.jobs_submitted} ({completion_rate:.0f}%), "
                f"AvgWait={avg_wait:.1f}h, "
                f"EpisodeMaxQueue={env.metrics.episode_max_queue_size_reached}, Dropped={env.metrics.episode_jobs_dropped}, "
                f"MaxQueue={env.metrics.max_queue_size_reached}, "
                f"Agent Occupancy (Cores)={env.metrics.episode_used_cores[-1]*100/(CORES_PER_NODE*MAX_NODES) if env.metrics.episode_used_cores else 0 :.2f}%, Baseline Occupancy (Cores)={env.metrics.episode_baseline_used_cores[-1]*100/(CORES_PER_NODE*MAX_NODES) if env.metrics.episode_baseline_used_cores else 0 :.2f}%, "
                f"Agent Occupancy (Nodes)={env.metrics.episode_used_nodes[-1]*100/MAX_NODES if env.metrics.episode_used_nodes else 0 :.2f}%, Baseline Occupancy (Nodes)={env.metrics.episode_baseline_used_nodes[-1]*100/MAX_NODES if env.metrics.episode_baseline_used_nodes else 0 :.2f}% " )

        print(f"\nEvaluation complete! Generated {num_episodes} episodes of cost data.")

        # Generate cumulative savings plot
        session_dir = f"sessions/{args.session}"
        try:
            results = plot_cumulative_savings(env, env.metrics.episode_costs, session_dir, save=True, show=args.render == 'human')
            plot_episode_summary(env, env.metrics.episode_costs, session_dir, save=True, show=args.render == 'human', suffix=f"eval_{args.eval_months}m")
            if results:
                print(f"\n=== CUMULATIVE SAVINGS ANALYSIS ===")
                print(f"\nVs Baseline (with idle nodes):")
                print(f"  Total Savings: €{results['total_savings']:,.0f}")
                print(f"  Average Monthly Reduction: {results['avg_monthly_savings_pct']:.1f}%")
                print(f"  Annual Savings Rate: €{results['total_savings'] * 12 / args.eval_months:,.0f}/year")

                print(f"\nVs Baseline_off (no idle nodes):")
                print(f"  Total Savings: €{results['total_savings_off']:,.0f}")
                print(f"  Average Monthly Reduction: {results['avg_monthly_savings_pct_off']:.1f}%")
                print(f"  Annual Savings Rate: €{results['total_savings_off'] * 12 / args.eval_months:,.0f}/year")

                # Calculate job metrics across all episodes
                total_jobs_submitted = sum(ep['jobs_submitted'] for ep in env.metrics.episode_costs)
                total_jobs_completed = sum(ep['jobs_completed'] for ep in env.metrics.episode_costs)
                total_baseline_submitted = sum(ep['baseline_jobs_submitted'] for ep in env.metrics.episode_costs)
                total_baseline_completed = sum(ep['baseline_jobs_completed'] for ep in env.metrics.episode_costs)
                avg_wait_time = sum(ep['avg_wait_time'] * ep['jobs_completed'] for ep in env.metrics.episode_costs) / total_jobs_completed if total_jobs_completed > 0 else 0
                avg_baseline_wait_time = sum(ep['baseline_avg_wait_time'] * ep['baseline_jobs_completed'] for ep in env.metrics.episode_costs) / total_baseline_completed if total_baseline_completed > 0 else 0
                avg_max_queue = sum(ep['max_queue_size'] for ep in env.metrics.episode_costs) / len(env.metrics.episode_costs)
                avg_baseline_max_queue = sum(ep['baseline_max_queue_size'] for ep in env.metrics.episode_costs) / len(env.metrics.episode_costs)
                total_agent_cost = sum(float(ep['agent_cost']) for ep in env.metrics.episode_costs)
                total_baseline_cost = sum(float(ep['baseline_cost']) for ep in env.metrics.episode_costs)
                total_baseline_off_cost = sum(float(ep['baseline_cost_off']) for ep in env.metrics.episode_costs)
                total_jobs_dropped = sum(int(ep.get('jobs_dropped', 0)) for ep in env.metrics.episode_costs)
                total_agent_power_mwh = sum(float(ep.get('agent_power_consumption_mwh', 0.0)) for ep in env.metrics.episode_costs)
                total_baseline_power_mwh = sum(float(ep.get('baseline_power_consumption_mwh', 0.0)) for ep in env.metrics.episode_costs)
                total_baseline_off_power_mwh = sum(float(ep.get('baseline_power_consumption_off_mwh', 0.0)) for ep in env.metrics.episode_costs)
                total_agent_mean_price = (total_agent_cost / total_agent_power_mwh) if total_agent_power_mwh > 0 else 0.0
                total_baseline_mean_price = (total_baseline_cost / total_baseline_power_mwh) if total_baseline_power_mwh > 0 else 0.0
                total_baseline_off_mean_price = (total_baseline_off_cost / total_baseline_off_power_mwh) if total_baseline_off_power_mwh > 0 else 0.0
                total_agent_completion_rate = (total_jobs_completed / total_jobs_submitted * 100) if total_jobs_submitted > 0 else 0.0
                total_baseline_completion_rate = (total_baseline_completed / total_baseline_submitted * 100) if total_baseline_submitted > 0 else 0.0
                total_savings_vs_baseline = total_baseline_cost - total_agent_cost
                total_savings_vs_baseline_off = total_baseline_off_cost - total_agent_cost
                total_agent_cost_per_1000_completed = safe_ratio(total_agent_cost * 1000.0, total_jobs_completed)
                total_baseline_cost_per_1000_completed = safe_ratio(total_baseline_cost * 1000.0, total_baseline_completed)
                # baseline_off is a cost variant of baseline scheduling, so it uses the same completed-job count.
                total_baseline_off_cost_per_1000_completed = safe_ratio(total_baseline_off_cost * 1000.0, total_baseline_completed)
                total_dropped_jobs_per_saved_euro = safe_ratio(total_jobs_dropped, total_savings_vs_baseline) if total_savings_vs_baseline > 0 else None
                total_dropped_jobs_per_saved_euro_off = safe_ratio(total_jobs_dropped, total_savings_vs_baseline_off) if total_savings_vs_baseline_off > 0 else None
                arrivals_per_hour_by_episode = [float(ep['jobs_submitted']) / float(EPISODE_HOURS) for ep in env.metrics.episode_costs]
                mean_arrivals_per_hour = (sum(arrivals_per_hour_by_episode) / len(arrivals_per_hour_by_episode)) if arrivals_per_hour_by_episode else 0.0
                arrivals_variance = (
                    sum((x - mean_arrivals_per_hour) ** 2 for x in arrivals_per_hour_by_episode) / len(arrivals_per_hour_by_episode)
                ) if arrivals_per_hour_by_episode else 0.0
                std_arrivals_per_hour = arrivals_variance ** 0.5

                print(f"\n=== JOB PROCESSING METRICS ===")
                print(f"\nAgent:")
                print(f"  Jobs Completed: {total_jobs_completed:,} / {total_jobs_submitted:,} ({total_agent_completion_rate:.1f}%)")
                print(f"  Average Wait Time: {avg_wait_time:.1f} hours")
                print(f"  Average Max Queue Size: {avg_max_queue:.0f}")
                print(f"  Total Cost: €{total_agent_cost:,.0f}")
                print(f"  Job Arrivals/Hour (mean ± std): {mean_arrivals_per_hour:.2f} ± {std_arrivals_per_hour:.2f}")

                print(f"\nBaseline:")
                print(f"  Jobs Completed: {total_baseline_completed:,} / {total_baseline_submitted:,} ({total_baseline_completion_rate:.1f}%)")
                print(f"  Average Wait Time: {avg_baseline_wait_time:.1f} hours")
                print(f"  Average Max Queue Size: {avg_baseline_max_queue:.0f}")
                print(f"  Baseline Total Cost: €{total_baseline_cost:,.0f}")
                print(f"  Baseline_off Total Cost: €{total_baseline_off_cost:,.0f}")

                print(f"\n=== COST PER 1,000 COMPLETED JOBS ===")
                print(f"  Agent:        {fmt_optional(total_agent_cost_per_1000_completed, 2, thousands=True)} €/1k jobs")
                print(f"  Baseline:     {fmt_optional(total_baseline_cost_per_1000_completed, 2, thousands=True)} €/1k jobs")
                print(f"  Baseline_off: {fmt_optional(total_baseline_off_cost_per_1000_completed, 2, thousands=True)} €/1k jobs")

                print(f"\n=== AGENT DROPPED JOBS PER SAVED EURO ===")
                print(f"  Total Dropped Jobs (Agent): {total_jobs_dropped:,}")
                print(f"  Vs Baseline:     {fmt_optional(total_dropped_jobs_per_saved_euro, 6)} jobs/€")
                print(f"  Vs Baseline_off: {fmt_optional(total_dropped_jobs_per_saved_euro_off, 6)} jobs/€")


                print(f"\n=== POWER & PRICE METRICS (TOTAL OVER EVALUATION) ===")
                print(f"  Agent:        Power={total_agent_power_mwh:,.1f} MWh, Mean Price={total_agent_mean_price:.2f} €/MWh")
                print(f"  Baseline:     Power={total_baseline_power_mwh:,.1f} MWh, Mean Price={total_baseline_mean_price:.2f} €/MWh")
                print(f"  Baseline_off: Power={total_baseline_off_power_mwh:,.1f} MWh, Mean Price={total_baseline_off_mean_price:.2f} €/MWh")
        except Exception as e:
            print(f"Could not generate cumulative savings plot: {e}")

        # Optional: single dashboard plot combining the per-hour traces from the LAST episode
        # and cumulative savings across all evaluated episodes.
        if args.plot_dashboard:
            try:
                plot_dashboard(
                    env,
                    num_hours=args.dashboard_hours,
                    max_nodes=335,
                    save=True,
                    show=(args.render == "human"),
                    suffix=f"eval_{args.eval_months}m",
                )
            except Exception as e:
                print(f"Could not generate dashboard plot: {e}")

        print("\nEvaluation complete!")
        env.close()
        return

    try:
        while True:
            print(f"Training iteration {iters + 1} ({STEPS_PER_ITERATION * (iters + 1)} steps)...")
            iters += 1
            t0 = time.time()
            if (iters+1)%10==0:
                print(f"Running... at {iters + 1} of {STEPS_PER_ITERATION * (iters + 1)} steps")
            if args.iter_limit > 0 and iters > args.iter_limit:
                print(f"iterations limit ({args.iter_limit}) reached: {iters}.")
                break
            try:
                model.learn(total_timesteps=STEPS_PER_ITERATION, reset_num_timesteps=False, tb_log_name=f"PPO", callback=ComputeClusterCallback())
                print(f"Iteration {iters} finished in {time.time()-t0:.2f}s")
                model.save(f"{models_dir}/{STEPS_PER_ITERATION * iters}.zip")

                if args.plot_dashboard:
                    try:
                        plot_dashboard(
                            env,
                            num_hours=args.dashboard_hours,
                            max_nodes=335,
                            save=True,
                            show=False,
                            suffix=STEPS_PER_ITERATION * iters,
                        )
                    except Exception as e:
                        print(f"Dashboard plot failed (non-fatal): {e}")

            except PlottingComplete:
                print("Plotting complete, terminating training...")
                break
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        print("Exiting training...")
        env.close()

if __name__ == "__main__":
    main()
