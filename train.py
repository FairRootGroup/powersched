from stable_baselines3 import PPO
import os
from src.environment import ComputeClusterEnv, Weights, PlottingComplete
from src.callbacks import ComputeClusterCallback
from src.plot import plot_cumulative_savings
import re
import glob
import argparse
import pandas as pd
from src.workloadgen import WorkloadGenerator, WorkloadGenConfig

# Import environment constants from config module:
from src.config import (
    MAX_JOB_DURATION,
    MIN_NODES_PER_JOB, MAX_NODES_PER_JOB,
    MIN_CORES_PER_JOB,
    CORES_PER_NODE,
)


# Train.py passes strings; the env treats "" as falsy in some places and truthy in others.
# To be safe: normalize "" -> None here.
def norm_path(x):
    return None if (x is None or str(x).strip() == "") else x

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
    parser.add_argument('--plot-rewards', action='store_true', help='Per step, plot rewards for all possible num_idle_nodes & num_used_nodes (default: False).')
    parser.add_argument('--plot-eff-reward', action='store_true', help='Include efficiency reward in the plot (dashed line).')
    parser.add_argument('--plot-price-reward', action='store_true', help='Include price reward in the plot (dashed line).')
    parser.add_argument('--plot-idle-penalty', action='store_true', help='Include idle penalty in the plot (dashed line).')
    parser.add_argument('--plot-job-age-penalty', action='store_true', help='Include job age penalty in the plot (dashed line).')
    parser.add_argument('--skip-plot-price', action='store_true', help='Skip electricity price in the plot (blue line).')
    parser.add_argument('--skip-plot-online-nodes', action='store_true', help='Skip online nodes in the plot (blue line).')
    parser.add_argument('--skip-plot-used-nodes', action='store_true', help='Skip used nodes in the plot (blue line).')
    parser.add_argument('--skip-plot-job-queue', action='store_true', help='Skip job queue in the plot (blue line).')
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
    parser.add_argument("--workload-gen", type=str, default="", choices=["", "flat", "poisson", "uniform"], help="Enable workload generator (default: disabled).",)
    parser.add_argument("--wg-poisson-lambda", type=float, default=200.0, help="Poisson lambda for jobs/hour.")
    parser.add_argument("--wg-max-jobs-hour", type=int, default=1500, help="Cap jobs/hour for generator.")


    args = parser.parse_args()
    prices_file_path = args.prices
    job_durations_file_path = args.job_durations
    jobs_file_path = args.jobs
    hourly_jobs_file_path = args.hourly_jobs

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

    weights_prefix = f"e{weights.efficiency_weight}_p{weights.price_weight}_i{weights.idle_weight}_d{weights.job_age_weight}"

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

    workload_gen = None
    if args.workload_gen:
        cfg = WorkloadGenConfig(
            arrivals=args.workload_gen,
            poisson_lambda=args.wg_poisson_lambda,
            max_new_jobs_per_hour=args.wg_max_jobs_hour,
            min_duration=1,
            max_duration=MAX_JOB_DURATION,
            min_nodes=MIN_NODES_PER_JOB,
            max_nodes=MAX_NODES_PER_JOB,
            min_cores=MIN_CORES_PER_JOB,
            max_cores=CORES_PER_NODE,
        )
        workload_gen = WorkloadGenerator(cfg)

    env = ComputeClusterEnv(weights=weights,
                            session=args.session,
                            render_mode=args.render,
                            quick_plot=args.quick_plot,
                            external_prices=prices,
                            external_durations=norm_path(job_durations_file_path),
                            external_jobs=norm_path(jobs_file_path),
                            external_hourly_jobs=norm_path(hourly_jobs_file_path),
                            plot_rewards=args.plot_rewards,
                            plots_dir=plots_dir,
                            plot_once=args.plot_once,
                            plot_eff_reward=args.plot_eff_reward,
                            plot_price_reward=args.plot_price_reward,
                            plot_idle_penalty=args.plot_idle_penalty,
                            plot_job_age_penalty=args.plot_job_age_penalty,
                            skip_plot_price=args.skip_plot_price,
                            skip_plot_online_nodes=args.skip_plot_online_nodes,
                            skip_plot_used_nodes=args.skip_plot_used_nodes,
                            skip_plot_job_queue=args.skip_plot_job_queue,
                            steps_per_iteration=STEPS_PER_ITERATION,
                            evaluation_mode=args.evaluate_savings,
                            workload_gen=workload_gen)
    env.reset()

    # Check if there are any saved models in models_dir
    model_files = glob.glob(models_dir + "*.zip")
    latest_model_file = None
    if model_files:
        # Sort the files by extracting the timestep number from the filename and converting it to an integer
        model_files.sort(key=lambda filename: int(re.match(r"(\d+)", os.path.basename(filename)).group()))
        latest_model_file = model_files[-1]  # Get the last file after sorting, which should be the one with the most timesteps
        print(f"Found a saved model: {latest_model_file}")
        model = PPO.load(latest_model_file, env=env, tensorboard_log=log_dir)
    else:
        print(f"Starting a new model training...")
        model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_dir, ent_coef=args.ent_coef)

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
                # print(f"Episode {episode + 1}, Step {step_count}, Action: {action}, Reward: {reward:.2f}, Total Reward: {episode_reward:.2f}, Total Cost: €{env.total_cost:.2f}")
                done = terminated or truncated

            savings_vs_baseline = env.baseline_cost - env.total_cost
            savings_vs_baseline_off = env.baseline_cost_off - env.total_cost
            completion_rate = (env.jobs_completed / env.jobs_submitted * 100) if env.jobs_submitted > 0 else 0
            avg_wait = env.total_job_wait_time / env.jobs_completed if env.jobs_completed > 0 else 0
            print(f"  Episode {episode + 1}: "
                f"Cost=€{env.total_cost:.0f}, "
                f"Savings=€{savings_vs_baseline:.0f}/€{savings_vs_baseline_off:.0f}, "
                f"Jobs={env.jobs_completed}/{env.jobs_submitted} ({completion_rate:.0f}%), "
                f"AvgWait={avg_wait:.1f}h, "
                f"MaxQueue={env.max_queue_size_reached}")

        print(f"\nEvaluation complete! Generated {num_episodes} episodes of cost data.")

        # Generate cumulative savings plot
        session_dir = f"sessions/{args.session}"
        try:
            results = plot_cumulative_savings(env, env.episode_costs, session_dir, months=args.eval_months, save=True, show=args.render == 'human')
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
                total_jobs_submitted = sum(ep['jobs_submitted'] for ep in env.episode_costs)
                total_jobs_completed = sum(ep['jobs_completed'] for ep in env.episode_costs)
                total_baseline_submitted = sum(ep['baseline_jobs_submitted'] for ep in env.episode_costs)
                total_baseline_completed = sum(ep['baseline_jobs_completed'] for ep in env.episode_costs)
                avg_wait_time = sum(ep['avg_wait_time'] * ep['jobs_completed'] for ep in env.episode_costs) / total_jobs_completed if total_jobs_completed > 0 else 0
                avg_baseline_wait_time = sum(ep['baseline_avg_wait_time'] * ep['baseline_jobs_completed'] for ep in env.episode_costs) / total_baseline_completed if total_baseline_completed > 0 else 0
                avg_max_queue = sum(ep['max_queue_size'] for ep in env.episode_costs) / len(env.episode_costs)
                avg_baseline_max_queue = sum(ep['baseline_max_queue_size'] for ep in env.episode_costs) / len(env.episode_costs)

                print(f"\n=== JOB PROCESSING METRICS ===")
                print(f"\nAgent:")
                print(f"  Jobs Completed: {total_jobs_completed:,} / {total_jobs_submitted:,} ({total_jobs_completed/total_jobs_submitted*100:.1f}%)")
                print(f"  Average Wait Time: {avg_wait_time:.1f} hours")
                print(f"  Average Max Queue Size: {avg_max_queue:.0f}")

                print(f"\nBaseline:")
                print(f"  Jobs Completed: {total_baseline_completed:,} / {total_baseline_submitted:,} ({total_baseline_completed/total_baseline_submitted*100:.1f}%)")
                print(f"  Average Wait Time: {avg_baseline_wait_time:.1f} hours")
                print(f"  Average Max Queue Size: {avg_baseline_max_queue:.0f}")
        except Exception as e:
            print(f"Could not generate cumulative savings plot: {e}")

        print("\nEvaluation complete!")
        env.close()
        return

    try:
        while True:
            print(f"Training iteration {iters + 1} ({STEPS_PER_ITERATION * (iters + 1)} steps)...")
            iters += 1
            if args.iter_limit > 0 and iters > args.iter_limit:
                print(f"iterations limit ({args.iter_limit}) reached: {iters}.")
                break
            try:
                model.learn(total_timesteps=STEPS_PER_ITERATION, reset_num_timesteps=False, tb_log_name=f"PPO", callback=ComputeClusterCallback())
                model.save(f"{models_dir}/{STEPS_PER_ITERATION * iters}.zip")
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
