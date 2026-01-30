import re
import math
from collections import defaultdict
import numpy as np

class HourlySampler:
    def __init__(self):
        self.hour_distributions = {}
        self.hourly_templates = {}  # Precalculated aggregated hourly job templates
        self.initialized = False
        self.aggregation_initialized = False

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

            # Extract hour and date directly from string (much faster than datetime parsing)
            # Format is fixed: "YYYY-MM-DDTHH:MM:SS"
            hour_of_day = int(submit_time[11:13])
            date_key = submit_time[:10]

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

    def sample(self, hour_of_day: int, rng):
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

    def precalculate_hourly_templates(self, cores_per_node: int, max_nodes_per_job: int, verbose: bool = True):
        """
        Precalculate aggregated hourly job templates for each hour of day.

        This converts raw job distributions into hourly-equivalent job templates.
        Sub-hour jobs are aggregated by their resource profile (nodes, cores_per_node),
        preserving the original resource characteristics while converting total runtime
        to equivalent 1-hour jobs.

        Note: This approach converts time to number of jobs while keeping resources
        unchanged. For sparse hours where sub-hour jobs don't sum to a full hour,
        rounding up can inflate total core-hours slightly.

        Args:
            cores_per_node: Number of cores per node in the simulation
            max_nodes_per_job: Maximum nodes a job can use in the simulation
            verbose: Whether to print aggregation statistics (default: True)
        """
        if not self.initialized:
            raise RuntimeError("Sampler not initialized. Call parse_jobs() first.")

        for hour in range(24):
            dist = self.hour_distributions[hour]

            if len(dist["durations"]) == 0:
                self.hourly_templates[hour] = {
                    "templates": [],
                    "total_weight": 0,
                    "sub_hour_bins": 0,
                    "hourly_jobs": 0
                }
                continue

            # Separate sub-hour and hourly+ jobs
            # Sub-hour jobs are aggregated by resource profile
            sub_hour_bins = {}  # key: (nodes, cores) -> total_minutes, count
            hourly_jobs = []    # jobs >= 1 hour kept individually

            durations = dist["durations"]
            nodes = dist["nodes"]
            cores = dist["cores_per_node"]

            for i in range(len(durations)):
                duration_min = int(durations[i])
                n_nodes = min(int(nodes[i]), max_nodes_per_job)
                n_cores = min(int(cores[i]), cores_per_node)

                if duration_min < 60:
                    # Sub-hour job: aggregate by resource profile
                    key = (n_nodes, n_cores)
                    if key not in sub_hour_bins:
                        sub_hour_bins[key] = {
                            "total_minutes": 0,
                            "count": 0
                        }
                    sub_hour_bins[key]["total_minutes"] += duration_min
                    sub_hour_bins[key]["count"] += 1
                else:
                    # Hourly+ job: keep as-is with rounded duration
                    duration_hours = max(1, int(math.ceil(duration_min / 60)))
                    hourly_jobs.append({
                        "nodes": n_nodes,
                        "cores_per_node": n_cores,
                        "duration_hours": duration_hours,
                        "original_job_count": 1
                    })

            # Convert sub-hour bins to hourly templates
            # Each bin becomes ceil(total_minutes / 60) jobs with duration=1 hour
            templates = []
            original_job_counts = []

            for (n_nodes, n_cores), bin_data in sub_hour_bins.items():
                # Number of 1-hour jobs needed to represent this work
                num_hourly_jobs = max(1, int(math.ceil(bin_data["total_minutes"] / 60)))

                templates.append({
                    "nodes": n_nodes,
                    "cores_per_node": n_cores,
                    "duration_hours": 1,
                    "hourly_job_count": num_hourly_jobs,  # How many 1-hour jobs this becomes
                    "original_job_count": bin_data["count"]
                })
                original_job_counts.append(bin_data["count"])

            # Add hourly+ jobs as individual templates
            for job in hourly_jobs:
                templates.append(job)
                original_job_counts.append(1)

            # total_weight is the sum of original job counts, used for scaling during sampling
            total_weight = sum(original_job_counts)

            self.hourly_templates[hour] = {
                "templates": templates,
                "total_weight": total_weight,
                "sub_hour_bins": len(sub_hour_bins),
                "hourly_jobs": len(hourly_jobs)
            }

        self.aggregation_initialized = True

        if verbose:
            print("Precalculated hourly job templates:")
            total_orig = 0
            total_templates = 0
            for hour in range(24):
                tmpl = self.hourly_templates[hour]
                orig_jobs = len(self.hour_distributions[hour]["durations"])
                total_orig += orig_jobs
                agg_templates = len(tmpl["templates"])
                total_templates += agg_templates
                print(f"  Hour {hour:2d}: {orig_jobs:5d} jobs -> {agg_templates:3d} templates "
                      f"({tmpl['sub_hour_bins']} sub-hour bins, {tmpl['hourly_jobs']} hourly+ jobs)")
            print(f"  Total: {total_orig} jobs -> {total_templates} templates")

    def sample_aggregated(self, hour_of_day: int, rng):
        """
        Sample aggregated hourly jobs for a given hour of day.

        Returns hourly-equivalent jobs that preserve resource profiles (nodes, cores)
        while aggregating sub-hour jobs by time.

        Args:
            hour_of_day: Hour of day (0-23)
            rng: NumPy random generator

        Returns:
            list: List of job dictionaries with keys: nodes, cores_per_node, duration_hours
                  Can be empty list if no jobs sampled for this hour
        """
        if not self.aggregation_initialized:
            raise RuntimeError("Aggregation not initialized. Call precalculate_hourly_templates() first.")

        hour_of_day = hour_of_day % 24

        dist = self.hour_distributions[hour_of_day]
        tmpl = self.hourly_templates[hour_of_day]

        # Sample number of jobs for this hour (can be 0)
        num_jobs = rng.choice(dist["job_count"])

        if num_jobs <= 0 or len(tmpl["templates"]) == 0:
            return []

        # Calculate scale factor based on sampled vs total original jobs
        if tmpl["total_weight"] > 0:
            scale_factor = num_jobs / tmpl["total_weight"]
        else:
            return []

        # Build result by scaling each template proportionally
        result = []
        for template in tmpl["templates"]:
            # For sub-hour aggregated bins, scale the hourly_job_count directly
            # (it already represents total work for all original jobs in this bin)
            if "hourly_job_count" in template:
                hourly_count = template["hourly_job_count"]
                expected_count = hourly_count * scale_factor
            else:
                # Hourly+ jobs: scale by original frequency
                expected_count = template["original_job_count"] * scale_factor

            # Use probabilistic rounding
            count = int(expected_count)
            if rng.random() < (expected_count - count):
                count += 1

            for _ in range(count):
                result.append({
                    "nodes": template["nodes"],
                    "cores_per_node": template["cores_per_node"],
                    "duration_hours": template.get("duration_hours", 1)
                })

        return result

# Create a singleton instance
hourly_sampler = HourlySampler()