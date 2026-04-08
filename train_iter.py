import numpy as np
import subprocess
import itertools
import argparse
import os
import sys
import time
import threading
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
    plot_dashboard=False,
    dashboard_hours=24 * 14,
    seed=None,
    seed_sweep=False,
    evaluate_savings=False,
    eval_months=0,
    flush_after_drop_streak=0,
    workloadgen_args=None,
    output_dir=None,
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
    if plot_dashboard:
        command += ["--plot-dashboard", "--dashboard-hours", str(dashboard_hours)]
    if seed is not None:
        command += ["--seed", str(seed)]
    if seed_sweep:
        command += ["--seed-sweep"]
    if evaluate_savings:
        command += ["--evaluate-savings", "--eval-months", str(eval_months)]
    if flush_after_drop_streak > 0:
        command += ["--flush-after-drop-streak", str(flush_after_drop_streak)]
    if workloadgen_args:
        command += workloadgen_args
    if output_dir is not None:
        command += ["--output-dir", output_dir]
    return command


def make_log_dir(session, output_dir="sessions"):
    ts = str(int(time.time()))
    if session:
        log_dir = os.path.join(output_dir, session, "proc_logs", ts)
    else:
        log_dir = os.path.join(output_dir, "proc_logs", ts)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def label_to_filename(label):
    return label.replace(", ", "_").replace("=", "") + ".log"


def _elapsed_str(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _run_plain(tasks, max_parallel, log_dir, launch):
    pending = list(tasks)
    active = []  # (proc, label, log_fh, start_time)
    done_log = []
    failure_count = 0
    total = len(pending)

    print(f"[run] logs -> {log_dir}/")

    try:
        while pending or active:
            while pending and len(active) < max_parallel:
                combo, seed = pending.pop(0)
                proc, label, fh, t0 = launch(combo, seed)
                print(f"[run] starting ({len(done_log) + len(active) + 1}/{total}): {label}")
                active.append((proc, label, fh, t0))

            still_running = []
            for proc, label, fh, t0 in active:
                if proc.poll() is not None:
                    fh.close()
                    rc = proc.returncode
                    if rc != 0:
                        failure_count += 1
                    elapsed = time.time() - t0
                    done_log.append((label, rc, elapsed))
                    status = "done" if rc == 0 else f"FAILED (rc={rc})"
                    print(f"[run] [{len(done_log)}/{total}] {status}: {label}  ({_elapsed_str(elapsed)})")
                else:
                    still_running.append((proc, label, fh, t0))
            active = still_running

            if active:
                time.sleep(1)
    finally:
        for proc, label, fh, t0 in active:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            except OSError:
                pass
            try:
                fh.close()
            except OSError:
                pass

    return failure_count


def _draw_tui(stdscr, active, done_log, n_pending, total, log_dir, input_buf=""):
    import curses as _curses
    try:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        row = 0

        hdr = (f"train_iter  [{len(done_log)}/{total} done | {len(active)} running | "
               f"{n_pending} queued]  logs: {log_dir}/")
        stdscr.addstr(row, 0, hdr[:w - 1], _curses.A_BOLD)
        row += 1
        stdscr.addstr(row, 0, "-" * min(w - 1, 80))
        row += 1

        # Reserve last line for the terminate prompt
        body_end = h - 2

        if active and row < body_end:
            stdscr.addstr(row, 0, "Running:")
            row += 1
            for i, (_, label, _, t0) in enumerate(active):
                if row >= body_end:
                    break
                line = f"  [{i + 1}] {_elapsed_str(time.time() - t0)}  {label}"
                stdscr.addstr(row, 0, line[:w - 1])
                row += 1
            row += 1

        max_show = body_end - row
        if done_log and max_show > 1 and row < body_end:
            stdscr.addstr(row, 0, "Completed:")
            row += 1
            max_show -= 1
            for label, rc, elapsed in done_log[-max_show:]:
                if row >= body_end:
                    break
                if rc == 0:
                    status = "done"
                elif rc == -1:
                    status = "terminated"
                else:
                    status = f"FAILED(rc={rc})"
                stdscr.addstr(row, 0, f"  {status}: {label}  ({_elapsed_str(elapsed)})"[:w - 1])
                row += 1

        # Terminate prompt at the last line
        prompt = f"Terminate #: {input_buf}_"
        stdscr.addstr(h - 1, 0, prompt[:w - 1])

        stdscr.refresh()
    except Exception:
        pass


def _run_tui(stdscr, tasks, max_parallel, log_dir, launch):
    import curses as _curses
    _curses.curs_set(0)
    stdscr.nodelay(True)

    pending = list(tasks)
    active = []   # (proc, label, log_fh, start_time)
    done_log = [] # (label, rc, elapsed)
    failure_count = 0
    total = len(pending)
    input_buf = ""

    while pending or active:
        while pending and len(active) < max_parallel:
            combo, seed = pending.pop(0)
            proc, label, fh, t0 = launch(combo, seed)
            active.append((proc, label, fh, t0))

        still_running = []
        for proc, label, fh, t0 in active:
            if proc.poll() is not None:
                fh.close()
                rc = proc.returncode
                if rc != 0:
                    failure_count += 1
                done_log.append((label, rc, time.time() - t0))
            else:
                still_running.append((proc, label, fh, t0))
        active = still_running

        _draw_tui(stdscr, active, done_log, len(pending), total, log_dir, input_buf)

        try:
            key = stdscr.getkey()
            if key in ("\n", "\r", "KEY_ENTER"):
                try:
                    idx = int(input_buf) - 1
                    if 0 <= idx < len(active):
                        proc, label, fh, t0 = active.pop(idx)
                        elapsed = time.time() - t0
                        proc.terminate()
                        def _reap(proc, label, fh, elapsed):
                            try:
                                proc.wait()
                            except OSError:
                                pass
                            try:
                                fh.close()
                            except OSError:
                                pass
                            done_log.append((label, -1, elapsed))
                        threading.Thread(target=_reap, args=(proc, label, fh, elapsed), daemon=True).start()
                        failure_count += 1
                except ValueError:
                    pass
                input_buf = ""
            elif key in ("KEY_BACKSPACE", "\x7f", "\b"):
                input_buf = input_buf[:-1]
            elif key == "\x1b":  # ESC
                input_buf = ""
            elif key.isdigit():
                input_buf += key
        except _curses.error:
            pass

        time.sleep(0.25)

    _draw_tui(stdscr, [], done_log, 0, total, log_dir)
    try:
        h, w = stdscr.getmaxyx()
        summary = f"All {total} runs done. {failure_count} failure(s). Press any key to exit."
        stdscr.addstr(h - 1, 0, summary[:w - 1], _curses.A_BOLD)
        stdscr.refresh()
    except Exception:
        pass
    if sys.stdin.isatty():
        stdscr.nodelay(False)
        stdscr.getch()

    return failure_count


def run_all_parallel(combinations, max_parallel, iter_limit_per_step, session, prices,
                     job_durations, jobs, hourly_jobs, job_arrival_scale, jobs_exact_replay,
                     plot_dashboard, dashboard_hours,
                     seeds, seed_sweep, evaluate_savings, eval_months, flush_after_drop_streak, workloadgen_args,
                     no_tui=False):
    multi_seed = len(seeds) > 1
    current_env = os.environ.copy()
    log_dir = make_log_dir(session, output_dir or "sessions")
    tasks = list(itertools.product(combinations, seeds))

    def launch(combo, seed):
        efficiency_weight, price_weight, idle_weight, job_age_weight, drop_weight = combo
        label = f"efficiency={efficiency_weight+0}, price={price_weight+0}, idle={idle_weight+0}, job_age={job_age_weight+0}, drop={drop_weight+0}"
        if multi_seed:
            label += f", seed={seed}"
        command = build_command(
            efficiency_weight, price_weight, idle_weight, job_age_weight, drop_weight,
            iter_limit_per_step, session, prices, job_durations, jobs, hourly_jobs,
            job_arrival_scale, jobs_exact_replay,
            plot_dashboard, dashboard_hours, seed, seed_sweep,
            evaluate_savings, eval_months, flush_after_drop_streak, workloadgen_args,
        )
        log_path = os.path.join(log_dir, label_to_filename(label))
        log_fh = open(log_path, "w")
        try:
            proc = subprocess.Popen(command, env=current_env, stdout=log_fh, stderr=subprocess.STDOUT)
        except OSError:
            log_fh.close()
            raise
        return proc, label, log_fh, time.time()

    if not no_tui and sys.stdout.isatty():
        import curses
        failure_count = [0]
        def _run(stdscr):
            failure_count[0] = _run_tui(stdscr, tasks, max_parallel, log_dir, launch)
        curses.wrapper(_run)
        return failure_count[0]
    else:
        return _run_plain(tasks, max_parallel, log_dir, launch)

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
    parser.add_argument("--fix-weights", type=str, help="Comma-separated list of weights to fix (efficiency,price,idle,job-age,drop)")
    parser.add_argument("--fix-values", type=str, help="Comma-separated list of values for fixed weights")
    parser.add_argument("--iter-limit-per-step", type=int, help="Max number of training iterations per step (1 iteration = {TIMESTEPS} steps)")
    parser.add_argument("--plot-dashboard", action="store_true", help="Forward to train.py to generate dashboard plots.")
    parser.add_argument("--dashboard-hours", type=int, default=24*14, help="Forward to train.py.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (forwarded to train.py)")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated list of seeds to iterate over (e.g. 42,123,456); overrides --seed")
    parser.add_argument("--parallel", type=int, default=1, metavar="N", help="Number of training runs to execute in parallel (default: 1, sequential)")
    parser.add_argument("--evaluate-savings", action="store_true", help="Forward to train.py to evaluate savings compared to baseline.")
    parser.add_argument("--eval-months", type=int, default=6, help="Number of months to evaluate savings over (forwarded to train.py)")
    parser.add_argument(
        "--flush-after-drop-streak",
        type=int,
        default=0,
        help="Forward to train.py: flush at episode end only after this many consecutive dropped-job steps (0 disables).",
    )
    parser.add_argument("--no-tui", action="store_true", help="Disable interactive TUI; print plain progress lines instead (auto-disabled when not a TTY)")
    add_workloadgen_args(parser)

    parser.add_argument("--session", help="Session ID")
    parser.add_argument("--output-dir", default=None, help="Base directory for all output (models, logs, plots). Defaults to 'sessions'.")

    args = parser.parse_args()

    if args.parallel < 1:
        parser.error("--parallel must be at least 1")
    try:
        args.job_arrival_scale = validate_job_arrival_scale(args.job_arrival_scale)
    except ValueError as exc:
        parser.error(str(exc))
    if args.jobs_exact_replay and not norm_path(args.jobs):
        parser.error("--jobs-exact-replay requires --jobs")
    if args.workload_gen and args.job_arrival_scale != 1.0:
        parser.error("--job-arrival-scale is not supported with --workload-gen. Use workload generator arrival settings instead.")

    try:
        fixed_weights = parse_fixed_weights(args.fix_weights, args.fix_values)
    except ValueError as e:
        parser.error(str(e))

    if args.seeds is not None:
        try:
            seeds = [int(s.strip()) for s in args.seeds.split(",")]
        except ValueError:
            parser.error("--seeds must be a comma-separated list of integers (e.g. 42,123,456)")
    else:
        seeds = [args.seed]  # may be None (no seed)

    combinations = generate_weight_combinations(step=args.step, fixed_weights=fixed_weights)
    workloadgen_args = build_workloadgen_cli_args(args)

    if not combinations:
        print("No valid weight combinations found with the given constraints")
        return

    print(f"Execution preview:")
    for combo, seed in itertools.product(combinations, seeds):
        efficiency_weight, price_weight, idle_weight, job_age_weight, drop_weight = combo
        seed_str = f", seed={seed}" if len(seeds) > 1 else ""
        print(f"    efficiency={efficiency_weight+0}, price={price_weight+0}, idle={idle_weight+0}, job_age={job_age_weight+0}, drop={drop_weight+0}{seed_str}")

    total_runs = len(combinations) * len(seeds)
    print(f"Running {total_runs} combinations with up to {args.parallel} parallel processes")
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
        plot_dashboard=args.plot_dashboard,
        dashboard_hours=args.dashboard_hours,
        seeds=seeds,
        seed_sweep=(args.seeds is not None),
        evaluate_savings=args.evaluate_savings,
        eval_months=args.eval_months,
        workloadgen_args=workloadgen_args,
        flush_after_drop_streak=args.flush_after_drop_streak,
        no_tui=args.no_tui,
        output_dir=args.output_dir,
    )
    if failures:
        print(f"{failures} run(s) failed")
        sys.exit(failures)

if __name__ == "__main__":
    main()
