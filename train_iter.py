import numpy as np
import subprocess
import itertools
import argparse
import os
import sys

def generate_weight_combinations(step=0.1, fixed_weights=None):
    weights = np.linspace(0, 1, num=int(1/step) + 1, endpoint=True)
    combinations = []
    weight_names = ['efficiency', 'price', 'idle', 'job-age', 'drop']

    if fixed_weights:
        # Get the names of weights that aren't fixed
        variable_weights = [w for w in weight_names if w not in fixed_weights]
        fixed_sum = sum(fixed_weights.values())

        if len(variable_weights) == 1:
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

def run(
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
    plot_dashboard=False,
    dashboard_hours=24 * 14,
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
        "--session", f"{session}"
    ]
    print(f"executing: {command}")
    current_env = os.environ.copy()
    result = subprocess.run(command, text=True, env=current_env)
    if result.returncode != 0:
        print("Error occurred: train.py returned a non-zero exit code.")
    return result.stdout

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
    parser.add_argument("--fix-weights", type=str, help="Comma-separated list of weights to fix (efficiency,price,idle,job-age,drop)")
    parser.add_argument("--fix-values", type=str, help="Comma-separated list of values for fixed weights")
    parser.add_argument("--iter-limit-per-step", type=int, help="Max number of training iterations per step (1 iteration = {TIMESTEPS} steps)")
    parser.add_argument("--session", help="Session ID")

    args = parser.parse_args()

    try:
        fixed_weights = parse_fixed_weights(args.fix_weights, args.fix_values)
    except ValueError as e:
        parser.error(str(e))

    combinations = generate_weight_combinations(step=args.step, fixed_weights=fixed_weights)

    if not combinations:
        print("No valid weight combinations found with the given constraints")
        return

    print(f"Execution preview:")
    for combo in combinations:
        efficiency_weight, price_weight, idle_weight, job_age_weight, drop_weight = combo
        print(f"    efficiency={efficiency_weight}, price={price_weight}, idle={idle_weight}, job_age={job_age_weight}, drop={drop_weight}")

    for combo in combinations:
        efficiency_weight, price_weight, idle_weight, job_age_weight, drop_weight = combo
        print(f"Running with weights: efficiency={efficiency_weight}, price={price_weight}, idle={idle_weight}, job_age={job_age_weight}, drop={drop_weight}")
        run(efficiency_weight, price_weight, idle_weight, job_age_weight, drop_weight, args.iter_limit_per_step, args.session, args.prices, args.job_durations, args.jobs, args.hourly_jobs)

if __name__ == "__main__":
    main()
