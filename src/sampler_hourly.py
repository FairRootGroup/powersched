import re
import datetime
import math
from collections import defaultdict
import numpy as np

class HourlySampler:
    def __init__(self):
        self.hour_distributions = {}
        self.initialized = False

    def parse_jobs(self, filepath):
        """Parse Slurm log file and build hourly distributions."""
        if not filepath:
            print("No jobs file path provided.")
            return None

        with open(filepath, 'r') as f:
            data_text = f.read()

        lines = data_text.strip().split('\n')
        # Skip header lines (first 2 lines)
        job_lines = [line for line in lines[2:] if line.strip() and not line.strip().startswith('-')]

        # Collect jobs by hour of day (0-23)
        hourly_jobs = defaultdict(list)
        hourly_job_counts = defaultdict(list)

        # Track which hours have data (to calculate zero-job hours properly)
        hours_with_data = defaultdict(set)

        # Count jobs per hour for each date to get distribution of job counts
        # This includes zero-job hours
        job_counts_by_hour = defaultdict(lambda: defaultdict(int))

        # Parse all job lines (single pass)
        for line in job_lines:
            parts = re.split(r'\s+', line.strip())

            submit_time = parts[3]
            elapsed_raw = int(parts[-3])
            ncpus = int(parts[-2])
            nnodes = int(parts[-1])

            # Parse submit time
            submit_datetime = datetime.datetime.strptime(submit_time, "%Y-%m-%dT%H:%M:%S")
            hour_of_day = submit_datetime.hour
            date_key = submit_datetime.strftime("%Y-%m-%d")

            # Calculate job metrics
            cores_per_node = ncpus // nnodes if nnodes > 0 else 0
            duration_minutes = max(1, math.ceil(elapsed_raw / 60))

            hourly_jobs[hour_of_day].append({
                "duration": duration_minutes,
                "nodes": nnodes,
                "cores_per_node": cores_per_node
            })

            hours_with_data[date_key].add(hour_of_day)
            job_counts_by_hour[date_key][hour_of_day] += 1

        # Build distributions for each hour of day (0-23)
        for hour in range(24):
            durations = []
            nodes = []
            cores_per_node = []
            job_counts = []

            # Collect job characteristics for this hour
            for job in hourly_jobs[hour]:
                durations.append(job["duration"])
                nodes.append(job["nodes"])
                cores_per_node.append(job["cores_per_node"])

            # Collect job counts (including zeros) for this hour across all dates
            for date_key, hours_set in hours_with_data.items():
                # Only include dates that have any data
                if date_key in job_counts_by_hour:
                    # Add the count for this hour (0 if not present)
                    count = job_counts_by_hour[date_key].get(hour, 0)
                    job_counts.append(count)

            self.hour_distributions[hour] = {
                "job_count": np.array(job_counts if job_counts else [0]),
                "durations": np.array(durations if durations else [1]),
                "nodes": np.array(nodes if nodes else [1]),
                "cores_per_node": np.array(cores_per_node if cores_per_node else [1])
            }

        self.initialized = True
        print(f"Parsed {len(job_lines)} jobs from {filepath}")
        print(f"Built distributions for 24 hours of day")

        # Print statistics
        for hour in range(24):
            dist = self.hour_distributions[hour]
            avg_count = np.mean(dist["job_count"])
            zero_pct = (np.count_nonzero(dist["job_count"] == 0) / len(dist["job_count"]) * 100) if len(dist["job_count"]) > 0 else 0
            print(f"  Hour {hour:2d}: avg={avg_count:.1f} jobs/hour, {zero_pct:.0f}% zero-job samples, {len(dist['durations'])} total jobs")

    def sample(self, hour_of_day: int, rng, max_jobs: int | None = None):
        """
        Sample jobs for a given hour of day.

        Args:
            hour_of_day (int): Hour of day (0-23)

        Returns:
            list: List of job dictionaries with keys: duration, nodes, cores_per_node
                  Can be empty list if no jobs sampled for this hour
        """
        if not self.initialized:
            raise RuntimeError("Sampler not initialized. Call parse_jobs() first.")

        hour_of_day = hour_of_day % 24  # Wrap around

        dist = self.hour_distributions[hour_of_day]

        # Sample number of jobs for this hour (can be 0)
        num_jobs = rng.choice(dist["job_count"])

        if num_jobs <= 0:
            return []

        if max_jobs is not None:
            num_jobs = min(num_jobs, int(max_jobs))
            if num_jobs <= 0:
                return []
        # Batch sample all job attributes at once (much faster than looping)
        durations = rng.choice(dist["durations"], size=num_jobs)
        nodes = rng.choice(dist["nodes"], size=num_jobs)
        cores = rng.choice(dist["cores_per_node"], size=num_jobs)

        return [
            {"duration": int(d), "nodes": int(n), "cores_per_node": int(c)}
            for d, n, c in zip(durations, nodes, cores)
        ]

    def get_stats(self):
        """Return summary statistics of the sampler."""
        if not self.initialized:
            raise RuntimeError("Sampler not initialized. Call parse_jobs() first.")

        stats = {}
        for hour in range(24):
            dist = self.hour_distributions[hour]
            stats[hour] = {
                "avg_jobs_per_hour": np.mean(dist["job_count"]),
                "max_jobs_per_hour": np.max(dist["job_count"]),
                "zero_job_percentage": (np.count_nonzero(dist["job_count"] == 0) / len(dist["job_count"]) * 100),
                "total_jobs_observed": len(dist["durations"]),
                "avg_duration_minutes": np.mean(dist["durations"]) if len(dist["durations"]) > 0 else 0,
                "avg_nodes": np.mean(dist["nodes"]) if len(dist["nodes"]) > 0 else 0,
                "avg_cores_per_node": np.mean(dist["cores_per_node"]) if len(dist["cores_per_node"]) > 0 else 0
            }
        return stats

# Create a singleton instance
hourly_sampler = HourlySampler()