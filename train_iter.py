import numpy as np
import subprocess
import itertools
import argparse
import os
import sys
import time
from src.arrival_scale import validate_job_arrival_scale
from src.workloadgen_cli import add_workloadgen_args, build_workloadgen_cli_args


def norm_path(x):
    return None if (x is None or str(x).strip() == "") else x


def generate_weight_combinations(step=0.1, fixed_weights=None):
    weights = np.linspace(0, 1, num=int(1/step) + 1, endpoint=True)
    combinations = []
    weight_names = ['efficiency', 'price', 'idle', 'job-age', 'drop']

    if fixed_weights:
        # Get the names of weights that aren't fixed
        variable_weights = [w for w in weight_names if w not in fixed_weights]
        fixed_sum = sum(fixed_weights.values())

        if len(variable_weights) == 0:
            # If all weights are fixed, return that single combination
            if abs(fixed_sum - 1.0) < 1e-9:  # Allow for floating point rounding
                combo = [0, 0, 0, 0, 0]
                for weight_name, value in fixed_weights.items():
                    combo[weight_names.index(weight_name)] = value
                combinations.append(tuple(combo))

        elif len(variable_weights) == 1:
            # If all but one weight is fixed, there's only one possible value
            remaining = round(1 - fixed_sum, 2)
            if 0 <= remaining <= 1:
                combo = [0, 0, 0, 0, 0]  # Initialize with five zeros
                # Set fixed weights
                for weight_name, value in fixed_weights.items():
                    combo[weight_names.index(weight_name)] = value
                # Set the remaining weight
                combo[weight_names.index(variable_weights[0])] = remaining
                combinations.append(tuple(combo))

        elif len(variable_weights) == 2:
            # If three weights are fixed, vary the other two
            for w in weights:
                remaining = round(1 - fixed_sum - w, 2)
                if 0 <= remaining <= 1:
                    combo = [0, 0, 0, 0, 0]  # Initialize with five zeros
                    # Set fixed weights
                    for weight_name, value in fixed_weights.items():
                        combo[weight_names.index(weight_name)] = value
                    # Set variable weights
                    combo[weight_names.index(variable_weights[0])] = round(w, 2)
                    combo[weight_names.index(variable_weights[1])] = remaining
                    combinations.append(tuple(combo))

        elif len(variable_weights) == 3:
            # If two weights are fixed, vary the other three
            for w1, w2 in itertools.product(weights, repeat=2):
                remaining = round(1 - fixed_sum - w1 - w2, 2)
                if 0 <= remaining <= 1:
                    combo = [0, 0, 0, 0, 0]  # Initialize with five zeros
                    # Set fixed weights
                    for weight_name, value in fixed_weights.items():
                        combo[weight_names.index(weight_name)] = value
                    # Set variable weights
                    combo[weight_names.index(variable_weights[0])] = round(w1, 2)
                    combo[weight_names.index(variable_weights[1])] = round(w2, 2)
                    combo[weight_names.index(variable_weights[2])] = remaining
                    combinations.append(tuple(combo))

        elif len(variable_weights) == 4:
            # If one weight is fixed, vary the other four
            for w1, w2, w3 in itertools.product(weights, repeat=3):
                remaining = round(1 - fixed_sum - w1 - w2 - w3, 2)
                if 0 <= remaining <= 1:
                    combo = [0, 0, 0, 0, 0]  # Initialize with five zeros
                    # Set fixed weights
                    for weight_name, value in fixed_weights.items():
                        combo[weight_names.index(weight_name)] = value
                    # Set variable weights
                    combo[weight_names.index(variable_weights[0])] = round(w1, 2)
                    combo[weight_names.index(variable_weights[1])] = round(w2, 2)
                    combo[weight_names.index(variable_weights[2])] = round(w3, 2)
                    combo[weight_names.index(variable_weights[3])] = remaining
                    combinations.append(tuple(combo))

    else:
        # If no weight is fixed, generate all combinations
        for e, p, i, ja in itertools.product(weights, repeat=4):
            d = round(1 - e - p - i - ja, 2)  # drop weight
            if 0 <= d <= 1:
                combinations.append((round(e, 2), round(p, 2), round(i, 2), round(ja, 2), round(d, 2)))

    return combinations

def build_command(
    efficiency_weight,
    price_weight,
    idle_weight,
    job_age_weight,
    drop_weight,
    iter_limit_per_step,
    session,
    prices,
    job_durations,
    jobs,
    hourly_jobs,
    job_arrival_scale,
    jobs_exact_replay,
    jobs_exact_replay_aggregate,
    plot_dashboard=False,
    dashboard_hours=24 * 14,
    seed=None,
    evaluate_savings=False,
    eval_months=0,
    workloadgen_args=None,
):
    python_executable = sys.executable
    command = [
        python_executable, "train.py",
        "--efficiency-weight", f"{efficiency_weight:.2f}",
        "--price-weight", f"{price_weight:.2f}",
        "--idle-weight", f"{idle_weight:.2f}",
        "--job-age-weight", f"{job_age_weight:.2f}",
        "--drop-weight", f"{drop_weight:.2f}",
        "--iter-limit", f"{iter_limit_per_step}",
        "--prices", f"{prices}",
        "--job-durations", f"{job_durations}",
        "--jobs", f"{jobs}",
        "--hourly-jobs", f"{hourly_jobs}",
        "--job-arrival-scale", f"{job_arrival_scale}",
        "--session", f"{session}"
    ]
    if jobs_exact_replay:
        command += ["--jobs-exact-replay"]
    if jobs_exact_replay_aggregate:
        command += ["--jobs-exact-replay-aggregate"]
    if plot_dashboard:
        command += ["--plot-dashboard", "--dashboard-hours", str(dashboard_hours)]
    if seed is not None:
        command += ["--seed", str(seed)]
    if evaluate_savings:
        command += ["--evaluate-savings", "--eval-months", str(eval_months)]
    if workloadgen_args:
        command += workloadgen_args
    return command


def run_all_parallel(combinations, max_parallel, iter_limit_per_step, session, prices,
                     job_durations, jobs, hourly_jobs, job_arrival_scale, jobs_exact_replay, jobs_exact_replay_aggregate, plot_dashboard, dashboard_hours,
                     seed, evaluate_savings, eval_months, workloadgen_args):
    active = []  # list of (proc, label)
    current_env = os.environ.copy()
    failure_count = 0

    for combo in combinations:
        efficiency_weight, price_weight, idle_weight, job_age_weight, drop_weight = combo
        label = f"efficiency={efficiency_weight}, price={price_weight}, idle={idle_weight}, job_age={job_age_weight}, drop={drop_weight}"

        # Wait until a slot is free
        while len(active) >= max_parallel:
            still_running = []
            for proc, lbl in active:
                if proc.poll() is None:
                    still_running.append((proc, lbl))
                else:
                    rc = proc.returncode
                    if rc != 0:
                        failure_count += 1
                    status = "done" if rc == 0 else f"error (rc={rc})"
                    print(f"[run] {status}: {lbl}")
            active = still_running
            if len(active) >= max_parallel:
                time.sleep(1)

        command = build_command(
            efficiency_weight, price_weight, idle_weight, job_age_weight, drop_weight,
            iter_limit_per_step, session, prices, job_durations, jobs, hourly_jobs, job_arrival_scale, jobs_exact_replay, jobs_exact_replay_aggregate,
            plot_dashboard, dashboard_hours, seed,
            evaluate_savings, eval_months,
            workloadgen_args,
        )
        print(f"[run] starting: {label}")
        proc = subprocess.Popen(command, env=current_env)
        active.append((proc, label))

    # Wait for all remaining processes
    for proc, label in active:
        proc.wait()
        rc = proc.returncode
        if rc != 0:
            failure_count += 1
        status = "done" if rc == 0 else f"error (rc={rc})"
        print(f"[run] {status}: {label}")

    return failure_count

def parse_fixed_weights(fix_weights_str, fix_values_str):
    if not fix_weights_str or not fix_values_str:
        return None

    weights = fix_weights_str.split(',')
    values = [float(v) for v in fix_values_str.split(',')]

    if len(weights) != len(values):
        raise ValueError("Number of fixed weights must match number of fixed values")

    fixed_weights = dict(zip(weights, values))
    total = sum(fixed_weights.values())

    if total > 1:
        raise ValueError("Sum of fixed weights cannot exceed 1")

    return fixed_weights


def main():
    parser = argparse.ArgumentParser(description="Run parameter sweep for weights")
    parser.add_argument("--step", type=float, default=0.1, help="Step size for weight combinations")
    parser.add_argument('--prices', type=str, nargs='?', const="", default="", help='Path to the CSV file containing electricity prices (Date,Price)')
    parser.add_argument('--job-durations', type=str, nargs='?', const="", default="", help='Path to a file containing job duration samples (for use with duration_sampler)')
    parser.add_argument('--jobs', type=str, nargs='?', const="", default="", help='Path to a file containing jobs samples (for use with jobs_sampler)')
    parser.add_argument('--hourly-jobs', type=str, nargs='?', const="", default="", help='Path to Slurm log file for hourly statistical sampling (for use with hourly_sampler)')
    parser.add_argument('--job-arrival-scale', type=float, default=1.0, help='Scale sampled arrivals per step (forwarded to train.py).')
    parser.add_argument('--jobs-exact-replay', action='store_true', help='Forward to train.py: replay raw jobs in timeline order for --jobs mode.')
    parser.add_argument('--jobs-exact-replay-aggregate', action='store_true', help='Forward to train.py: aggregate per-step raw jobs in exact replay mode.')
    parser.add_argument("--fix-weights", type=str, help="Comma-separated list of weights to fix (efficiency,price,idle,job-age,drop)")
    parser.add_argument("--fix-values", type=str, help="Comma-separated list of values for fixed weights")
    parser.add_argument("--iter-limit-per-step", type=int, help="Max number of training iterations per step (1 iteration = {TIMESTEPS} steps)")
    parser.add_argument("--plot-dashboard", action="store_true", help="Forward to train.py to generate dashboard plots.")
    parser.add_argument("--dashboard-hours", type=int, default=24*14, help="Forward to train.py.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (forwarded to train.py)")
    parser.add_argument("--parallel", type=int, default=1, metavar="N", help="Number of training runs to execute in parallel (default: 1, sequential)")
    parser.add_argument("--evaluate-savings", action="store_true", help="Forward to train.py to evaluate savings compared to baseline.")
    parser.add_argument("--eval-months", type=int, default=6, help="Number of months to evaluate savings over (forwarded to train.py)")
    add_workloadgen_args(parser)

    parser.add_argument("--session", help="Session ID")

    args = parser.parse_args()

    if args.parallel < 1:
        parser.error("--parallel must be at least 1")
    try:
        args.job_arrival_scale = validate_job_arrival_scale(args.job_arrival_scale)
    except ValueError as exc:
        parser.error(str(exc))
    if args.jobs_exact_replay and not norm_path(args.jobs):
        parser.error("--jobs-exact-replay requires --jobs")
    if args.jobs_exact_replay_aggregate and not args.jobs_exact_replay:
        parser.error("--jobs-exact-replay-aggregate requires --jobs-exact-replay")
    if args.workload_gen and args.job_arrival_scale != 1.0:
        parser.error("--job-arrival-scale is not supported with --workload-gen. Use workload generator arrival settings instead.")

    try:
        fixed_weights = parse_fixed_weights(args.fix_weights, args.fix_values)
    except ValueError as e:
        parser.error(str(e))

    combinations = generate_weight_combinations(step=args.step, fixed_weights=fixed_weights)
    workloadgen_args = build_workloadgen_cli_args(args)

    if not combinations:
        print("No valid weight combinations found with the given constraints")
        return

    print(f"Execution preview:")
    for combo in combinations:
        efficiency_weight, price_weight, idle_weight, job_age_weight, drop_weight = combo
        print(f"    efficiency={efficiency_weight}, price={price_weight}, idle={idle_weight}, job_age={job_age_weight}, drop={drop_weight}")

    print(f"Running {len(combinations)} combinations with up to {args.parallel} parallel processes")
    failures = run_all_parallel(
        combinations,
        max_parallel=args.parallel,
        iter_limit_per_step=args.iter_limit_per_step,
        session=args.session,
        prices=args.prices,
        job_durations=args.job_durations,
        jobs=args.jobs,
        hourly_jobs=args.hourly_jobs,
        job_arrival_scale=args.job_arrival_scale,
        jobs_exact_replay=args.jobs_exact_replay,
        jobs_exact_replay_aggregate=args.jobs_exact_replay_aggregate,
        plot_dashboard=args.plot_dashboard,
        dashboard_hours=args.dashboard_hours,
        seed=args.seed,
        evaluate_savings=args.evaluate_savings,
        eval_months=args.eval_months,
        workloadgen_args=workloadgen_args,
    )
    if failures:
        print(f"{failures} run(s) failed")
        sys.exit(failures)

if __name__ == "__main__":
    main()
