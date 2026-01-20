from src.sampler_hourly import hourly_sampler
import argparse
import numpy as np

def main():
    rng = np.random.default_rng()
    parser = argparse.ArgumentParser(description='Test the hourly job sampler with Slurm logs.')
    parser.add_argument('--file-path', required=True, help='Path to the Slurm log file')
    parser.add_argument('--print-stats', action='store_true', help='Print summary statistics.')
    parser.add_argument('--test-samples', type=int, default=0, help='Number of hours to sample (default: 0)')
    parser.add_argument('--test-day', action='store_true', help='Sample a full 24-hour day')
    args = parser.parse_args()

    # Initialize sampler
    print(f"Loading Slurm log: {args.file_path}")
    hourly_sampler.parse_jobs(args.file_path)

    if args.print_stats:
        print("\n=== Summary Statistics ===")
        stats = hourly_sampler.get_stats()
        for hour, hour_stats in stats.items():
            print(f"\nHour {hour:2d}:00")
            print(f"  Avg jobs/hour: {hour_stats['avg_jobs_per_hour']:.2f}")
            print(f"  Max jobs/hour: {hour_stats['max_jobs_per_hour']}")
            print(f"  Zero-job %: {hour_stats['zero_job_percentage']:.1f}%")
            print(f"  Total jobs observed: {hour_stats['total_jobs_observed']}")
            print(f"  Avg duration: {hour_stats['avg_duration_minutes']:.1f} min")
            print(f"  Avg nodes: {hour_stats['avg_nodes']:.2f}")
            print(f"  Avg cores/node: {hour_stats['avg_cores_per_node']:.2f}")

    if args.test_samples > 0:
        print(f"\n=== Test Sampling {args.test_samples} Hours ===")
        for i in range(args.test_samples):
            hour = i % 24
            jobs = hourly_sampler.sample(hour, rng)
            print(f"\nHour {hour:2d}:00 - Sampled {len(jobs)} jobs:")
            for j, job in enumerate(jobs[:5]):  # Show first 5 jobs
                print(f"  Job {j+1}: {job['duration']} min, {job['nodes']} nodes, {job['cores_per_node']} cores/node")
            if len(jobs) > 5:
                print(f"  ... and {len(jobs) - 5} more jobs")

    if args.test_day:
        print("\n=== Sampling Full 24-Hour Day ===")
        total_jobs = 0
        for hour in range(24):
            jobs = hourly_sampler.sample(hour, rng)
            total_jobs += len(jobs)
            print(f"Hour {hour:2d}:00 - {len(jobs):3d} jobs", end="")
            if len(jobs) == 0:
                print(" (no jobs)")
            else:
                avg_dur = sum(j['duration'] for j in jobs) / len(jobs)
                avg_nodes = sum(j['nodes'] for j in jobs) / len(jobs)
                print(f" (avg: {avg_dur:.0f} min, {avg_nodes:.1f} nodes)")
        print(f"\nTotal jobs in 24 hours: {total_jobs}")

if __name__ == "__main__":
    main()