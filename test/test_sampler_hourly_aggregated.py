from src.sampler_hourly import hourly_sampler
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Test hourly sampler aggregation functionality.')
    parser.add_argument('--file-path', required=True, help='Path to the Slurm log file')
    parser.add_argument('--cores-per-node', type=int, default=96, help='Cores per node in simulation')
    parser.add_argument('--max-nodes-per-job', type=int, default=16, help='Maximum nodes per job')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples per hour to test')
    parser.add_argument('--test-hours', type=str, default=None, help='Comma-separated list of hours to test (default: all)')
    args = parser.parse_args()

    # Parse the jobs file
    hourly_sampler.parse_jobs(args.file_path)

    # Precalculate hourly templates
    print()
    hourly_sampler.precalculate_hourly_templates(args.cores_per_node, args.max_nodes_per_job)

    # Initialize RNG
    rng = np.random.default_rng(args.seed)

    # Determine which hours to test
    if args.test_hours:
        test_hours = [int(h) for h in args.test_hours.split(',')]
    else:
        # Test hours with significant activity
        test_hours = []
        for hour in range(24):
            dist = hourly_sampler.hour_distributions[hour]
            if np.mean(dist["job_count"]) > 0.5:
                test_hours.append(hour)
        if not test_hours:
            test_hours = list(range(24))

    print(f"\n=== Testing sample_aggregated for hours: {test_hours} ===")

    for hour in test_hours:
        dist = hourly_sampler.hour_distributions[hour]
        tmpl = hourly_sampler.hourly_templates[hour]

        print(f"\n===== Hour {hour:02d} =====")
        print(f"  Original distribution: {len(dist['durations'])} total jobs observed")
        print(f"  Avg jobs/hour: {np.mean(dist['job_count']):.1f}, Max: {np.max(dist['job_count'])}")
        print(f"  Templates: {len(tmpl['templates'])} ({tmpl['sub_hour_bins']} sub-hour bins, {tmpl['hourly_jobs']} hourly+ jobs)")
        print(f"  Total weight (original job count): {tmpl['total_weight']}")

        if len(tmpl['templates']) == 0:
            print("  No templates to sample from")
            continue

        # Show template details
        print(f"  Template details (up to 5):")
        for i, t in enumerate(tmpl['templates'][:5]):
            if 'hourly_job_count' in t:
                print(f"    [{i}] Sub-hour: ({t['nodes']} nodes, {t['cores_per_node']} cores) -> {t['hourly_job_count']} hourly jobs (from {t['original_job_count']} original)")
            else:
                print(f"    [{i}] Hourly+: ({t['nodes']} nodes, {t['cores_per_node']} cores) x {t['duration_hours']}h")

        # Sample multiple times and show statistics
        print(f"\n  Sampling {args.num_samples} times:")
        total_jobs = []
        total_hours = []
        for i in range(args.num_samples):
            jobs = hourly_sampler.sample_aggregated(hour, rng)
            job_count = len(jobs)
            job_hours = sum(j['duration_hours'] for j in jobs) if jobs else 0
            total_jobs.append(job_count)
            total_hours.append(job_hours)

            if i < 3:  # Show first 3 samples in detail
                if jobs:
                    # Group by (nodes, cores, duration)
                    job_types = {}
                    for j in jobs:
                        key = (j['nodes'], j['cores_per_node'], j['duration_hours'])
                        job_types[key] = job_types.get(key, 0) + 1
                    type_strs = [f"{cnt}x({n}n,{c}c,{d}h)" for (n, c, d), cnt in sorted(job_types.items())]
                    print(f"    Sample {i+1}: {job_count} jobs, {job_hours} job-hours: {', '.join(type_strs[:5])}{'...' if len(type_strs) > 5 else ''}")
                else:
                    print(f"    Sample {i+1}: 0 jobs")

        print(f"  Summary over {args.num_samples} samples:")
        print(f"    Jobs: avg={np.mean(total_jobs):.1f}, min={min(total_jobs)}, max={max(total_jobs)}")
        print(f"    Job-hours: avg={np.mean(total_hours):.1f}, min={min(total_hours)}, max={max(total_hours)}")

    # Test consistency: verify that sampling respects the distribution
    print("\n=== Consistency Check ===")
    print("Comparing sampled job counts vs expected distribution...")

    for hour in test_hours[:3]:  # Check first 3 hours
        dist = hourly_sampler.hour_distributions[hour]
        expected_avg = np.mean(dist["job_count"])

        # Sample many times
        sampled_counts = []
        for _ in range(100):
            jobs = hourly_sampler.sample_aggregated(hour, rng)
            sampled_counts.append(len(jobs))

        sampled_avg = np.mean(sampled_counts)
        # The sampled average should be in the same ballpark as expected
        # (aggregation changes individual job counts but overall scaling should be similar)
        print(f"  Hour {hour:02d}: expected avg jobs/hour={expected_avg:.1f}, sampled avg={sampled_avg:.1f}")

if __name__ == "__main__":
    main()
