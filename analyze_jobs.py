"""
Report jobs-per-hour statistics for a given log file across three views:
  1. Raw       – jobs as parsed, one entry per actual job
  2. Aggregated – jobs grouped by (nodes, cores, duration), one entry per unique profile
  3. Hourly    – aggregated jobs converted to 1-hour equivalents (what the env receives)

Also reports job duration statistics (min, max, mean, median) across all raw jobs.
"""

import argparse
import sys
import statistics

from src.sampler_jobs import DurationSampler
from src.config import MAX_NODES_PER_JOB, CORES_PER_NODE


def summarize_jobs_per_hour(counts: dict[str, int], bin_minutes: int) -> tuple[str, float, float, float]:
    rates = [count * 60.0 / bin_minutes for count in counts.values()]
    max_period = max(counts, key=counts.get)
    return max_period, max(rates), statistics.mean(rates), statistics.pstdev(rates)


def count_hourly_instances(jobs: list[dict[str, int]]) -> int:
    return sum(max(1, int(job.get("instances", 1))) for job in jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report jobs-per-hour statistics from a Slurm log file.")
    parser.add_argument("--file-path", required=True, help="Path to the job log file")
    parser.add_argument("--bin-minutes", type=int, default=60, help="Bin size in minutes (default: 60)")
    parser.add_argument("--cores-per-node", type=int, default=CORES_PER_NODE, help=f"Cores per node (default: {CORES_PER_NODE})")
    parser.add_argument("--max-nodes-per-job", type=int, default=MAX_NODES_PER_JOB, help=f"Max nodes per job (default: {MAX_NODES_PER_JOB})")
    parser.add_argument("--verbose", action="store_true", help="Print top-N hours for each view")
    parser.add_argument("--top", type=int, default=5, help="Number of top hours to show with --verbose (default: 5)")
    args = parser.parse_args()

    s = DurationSampler()
    result = s.parse_jobs(args.file_path, args.bin_minutes)
    if result is None:
        print("Failed to parse jobs file.", file=sys.stderr)
        sys.exit(1)

    # --- Raw ---
    raw_counts = {period: len(jobs) for period, jobs in s.jobs.items()}
    max_raw_period, max_raw, mean_raw, std_raw = summarize_jobs_per_hour(raw_counts, args.bin_minutes)
    total_hours_raw = len(raw_counts)

    # --- Aggregated ---
    agg_counts = {period: len(jobs) for period, jobs in s.aggregated_jobs.items()}
    max_agg_period, max_agg, mean_agg, std_agg = summarize_jobs_per_hour(agg_counts, args.bin_minutes)

    # --- Hourly-converted ---
    s.precalculate_hourly_jobs(args.cores_per_node, args.max_nodes_per_job)
    hourly_counts = {period: count_hourly_instances(jobs) for period, jobs in s.hourly_jobs.items()}
    max_hourly_period, max_hourly, mean_hourly, std_hourly = summarize_jobs_per_hour(hourly_counts, args.bin_minutes)

    # --- Duration stats (from all raw jobs) ---
    all_durations = [job["duration_minutes"] for jobs in s.jobs.values() for job in jobs]
    dur_min = min(all_durations)
    dur_max = max(all_durations)
    dur_mean = statistics.mean(all_durations)
    dur_median = statistics.median(all_durations)
    total_jobs = len(all_durations)

    print(f"File          : {args.file_path}")
    print(f"Bin size      : {args.bin_minutes} min")
    print(f"Total periods : {total_hours_raw}")
    print(f"Total jobs    : {total_jobs}")
    print(f"Cores/node    : {args.cores_per_node}  |  Max nodes/job: {args.max_nodes_per_job}")
    print()
    print(f"{'View':<12}  {'Max jobs/hour':>14}  {'Mean jobs/hour':>15}  {'Std jobs/hour':>14}  {'At period'}")
    print("-" * 92)
    print(f"{'Raw':<12}  {max_raw:>14.2f}  {mean_raw:>15.2f}  {std_raw:>14.2f}  {max_raw_period}")
    print(f"{'Aggregated':<12}  {max_agg:>14.2f}  {mean_agg:>15.2f}  {std_agg:>14.2f}  {max_agg_period}")
    print(f"{'Hourly':<12}  {max_hourly:>14.2f}  {mean_hourly:>15.2f}  {std_hourly:>14.2f}  {max_hourly_period}")
    print()
    print(f"{'Job duration':<10}  {'minutes':>10}  {'hours':>8}")
    print("-" * 32)
    print(f"{'Min':<10}  {dur_min:>10}  {dur_min / 60:>8.2f}")
    print(f"{'Max':<10}  {dur_max:>10}  {dur_max / 60:>8.2f}")
    print(f"{'Mean':<10}  {dur_mean:>10.1f}  {dur_mean / 60:>8.2f}")
    print(f"{'Median':<10}  {dur_median:>10.1f}  {dur_median / 60:>8.2f}")

    if args.verbose:
        for label, counts in [("Raw", raw_counts), ("Aggregated", agg_counts), ("Hourly", hourly_counts)]:
            top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[: args.top]
            print(f"\nTop {args.top} periods — {label}:")
            for period, count in top:
                print(f"  {period}  {count:>6} jobs")


if __name__ == "__main__":
    main()
