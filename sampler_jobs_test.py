from sampler_jobs import jobs_sampler
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze job durations from a log file.')
    parser.add_argument('--file-path', default="", help='Path to the job duration log file')
    parser.add_argument('--bin-minutes', type=int, default=60, help='Bin duration in minutes (default: 60)')
    args = parser.parse_args()

    jobs_sampler.parse_jobs(args.file_path, args.bin_minutes)
    all_aggregated_jobs = jobs_sampler.get_all_aggregated_jobs()

    for period, jobs in all_aggregated_jobs.items():
        print(f"period: {period}, jobs: {len(jobs)}")
        for i, job in enumerate(jobs):
            print(f"  Job {i+1}: Nodes={job['nnodes']}, Cores per node={job['cores_per_node']}, Duration={job['duration_minutes']} minutes")

    samples = jobs_sampler.sample_aggregated(14)
    if samples is None:
        print("No jobs data found.")
        sys.exit(1)

    # samples is a dictionary with keys as start time string and values are array of jobs
    print(f"Sampled {len(samples)} jobs:")
    for i, key in enumerate(samples):
        print(f"  {i+1}: {key} - {len(samples[key])} jobs")


