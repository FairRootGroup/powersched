from src.sampler_jobs import jobs_sampler
import argparse

def main():
    parser = argparse.ArgumentParser(description='Test job aggregation functionality.')
    parser.add_argument('--file-path', required=True, help='Path to the job duration log file')
    parser.add_argument('--bin-minutes', type=int, default=60, help='Bin duration in minutes (default: 60)')
    parser.add_argument('--cores-per-node', type=int, default=96, help='Cores per node in simulation')
    parser.add_argument('--max-nodes-per-job', type=int, default=16, help='Maximum nodes per job')
    parser.add_argument('--num-samples', type=int, default=1, help='Number of samples to take from the job data')
    args = parser.parse_args()

    # Parse the jobs file - this precalculates aggregations
    jobs_sampler.parse_jobs(args.file_path, args.bin_minutes)

    # Precalculate hourly jobs conversions
    jobs_sampler.precalculate_hourly_jobs(args.cores_per_node, args.max_nodes_per_job)

    # Get comprehensive data for sampled periods
    sampled_data = jobs_sampler.sample_hourly(args.num_samples, wrap=True)
    sampled_periods = list(sampled_data.keys())

    print(f"Sampled {args.num_samples} periods: {sampled_periods}")

    for period in sampled_periods:
        period_data = sampled_data[period]
        raw_jobs = period_data['raw_jobs']
        aggregated_jobs = period_data['aggregated_jobs']
        hourly_jobs = period_data['hourly_jobs']

        print(f"===== Period: {period}: raw jobs: {len(raw_jobs)} =====")

        if len(raw_jobs) == 0:
            continue

        # Get resource statistics before and after aggregation
        before_stats = jobs_sampler.calculate_resource_hours(raw_jobs)
        after_stats = jobs_sampler.calculate_resource_hours(aggregated_jobs)

        print(f"  Before aggregation:")
        print(f"    Total jobs: {before_stats['total_jobs']}, Node-hours: {before_stats['total_node_hours']:.2f}, Core-hours: {before_stats['total_core_hours']:.2f}")

        print(f"  After aggregation:")
        print(f"    Unique job types: {len(aggregated_jobs)}, Total jobs represented: {after_stats['total_jobs']}, Node-hours: {after_stats['total_node_hours']:.2f}, Core-hours: {after_stats['total_core_hours']:.2f}")

        # Show the top 5 aggregated jobs
        top_count = min(5, len(aggregated_jobs))
        print(f"  Aggregated job types (Top {top_count} out of {len(aggregated_jobs)}):")
        for i, job in enumerate(aggregated_jobs[:5]):
            print(f"    Type {i+1}: {job['count']} jobs with {job['nnodes']} nodes, {job['cores_per_node']} cores/node, {job['duration_minutes']} minutes each. Total core-minutes: {job['total_core_minutes']}")

        # Show the precalculated hourly jobs
        top_hourly = min(5, len(hourly_jobs))
        print(f"  Converted to hourly simulation jobs (Top {top_hourly} out of {len(hourly_jobs)}):")
        for i, job in enumerate(hourly_jobs[:5]):
            print(f"    Job {i+1}: {job['nnodes']} nodes, {job['cores_per_node']} cores/node, {job['duration_hours']} hours (represents {job['original_job_count']} original jobs)")

        print(f"Max jobs per hour: {jobs_sampler.max_new_jobs_per_hour}")
        print(f"Max job duration: {jobs_sampler.max_job_duration}")

if __name__ == "__main__":
    main()
