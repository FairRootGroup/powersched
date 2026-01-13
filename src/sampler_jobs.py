import re
import datetime
import math
import random
from collections import defaultdict

class DurationSampler:
    def __init__(self):
        self.jobs = {}
        self.aggregated_jobs = {}
        self.hourly_jobs = {}
        self.keys = []
        self.current_position = 0

        self.max_new_jobs_per_hour = 0
        self.max_job_duration = 0

    def parse_jobs(self, filepath, bin_minutes=60):
        if filepath:
            with open(filepath, 'r') as f:
                data_text = f.read()
        else:
            print("No jobs file path provided.")
            return None

        lines = data_text.strip().split('\n')
        job_lines = [line for line in lines[2:] if line.strip() and not line.strip().startswith('-')]
        jobs_temp = defaultdict(list)
        all_timestamps = []

        # Determine the time format based on bin_minutes
        if bin_minutes >= 1440:  # Daily or longer (≥ 24 hours)
            time_format = "%Y-%m-%d"
            time_delta = datetime.timedelta(days=1)
        elif bin_minutes >= 60:  # Hourly or longer
            time_format = "%Y-%m-%d %H:00"
            time_delta = datetime.timedelta(hours=1)
        else:  # Less than an hour
            time_format = "%Y-%m-%d %H:%M"
            time_delta = datetime.timedelta(minutes=bin_minutes)

        self.time_format = time_format
        self.time_delta = time_delta

        # Parse all job lines and collect timestamps
        for line in job_lines:
            parts = re.split(r'\s+', line.strip())

            submit_time = parts[3]
            elapsed_raw = int(parts[-3])
            ncpus = int(parts[-2])
            nnodes = int(parts[-1])

            # Parse submit time
            submit_datetime = datetime.datetime.strptime(submit_time, "%Y-%m-%dT%H:%M:%S")
            all_timestamps.append(submit_datetime)

            # Calculate bin start time based on bin_minutes
            minutes_since_epoch = int(submit_datetime.timestamp() / 60)
            bin_start_minutes = (minutes_since_epoch // bin_minutes) * bin_minutes
            bin_start_datetime = datetime.datetime.fromtimestamp(bin_start_minutes * 60)

            # Format bin key
            bin_key = bin_start_datetime.strftime(time_format)

            # Calculate job metrics
            cores_per_node = ncpus // nnodes if nnodes > 0 else 0
            duration_minutes = max(1, math.ceil(elapsed_raw / 60))

            jobs_temp[bin_key].append({
                "nnodes": nnodes,
                "cores_per_node": cores_per_node,
                "duration_minutes": duration_minutes
            })

        # Create a continuous timeline with all time periods
        self.jobs = {}
        self.aggregated_jobs = {}  # Store precalculated aggregations

        if all_timestamps:
            min_time = min(all_timestamps)
            max_time = max(all_timestamps)

            # Round min_time down and max_time up to bin boundaries
            minutes_since_epoch = int(min_time.timestamp() / 60)
            bin_start_minutes = (minutes_since_epoch // bin_minutes) * bin_minutes
            min_bin_time = datetime.datetime.fromtimestamp(bin_start_minutes * 60)

            minutes_since_epoch = int(max_time.timestamp() / 60)
            bin_end_minutes = ((minutes_since_epoch // bin_minutes) + 1) * bin_minutes
            max_bin_time = datetime.datetime.fromtimestamp(bin_end_minutes * 60)

            # Generate all period keys and precalculate aggregations
            current_time = min_bin_time

            while current_time <= max_bin_time:
                period_key = current_time.strftime(time_format)
                # Use jobs from parsed data or empty list
                raw_jobs = jobs_temp.get(period_key, [])
                self.jobs[period_key] = raw_jobs

                # Precalculate aggregation for this period
                if raw_jobs:
                    self.aggregated_jobs[period_key] = self.aggregate_jobs(raw_jobs)
                else:
                    self.aggregated_jobs[period_key] = []

                current_time += time_delta

        # Store keys for sampling
        self.keys = list(self.jobs.keys())
        self.current_position = 0
        self.bin_minutes = bin_minutes

        return self

    def get_all_jobs(self):
        return self.jobs

    def get_all_aggregated_jobs(self):
        return self.aggregated_jobs

    def sample(self, n=1, wrap=True):
        """
        Sample n consecutive time periods, including empty ones.

        Parameters:
        - n: Number of time periods to sample
        - wrap: Whether to wrap around to the beginning when reaching the end

        Returns:
        - Dictionary mapping period keys to job lists (empty list for periods with no jobs)
        """
        if not self.keys:
            return {}

        result = {}
        periods_sampled = 0

        # Get starting position
        start_pos = self.current_position
        if start_pos >= len(self.keys):
            if wrap:
                start_pos = 0
            else:
                return {}

        # Sample n consecutive periods
        while periods_sampled < n:
            if (start_pos + periods_sampled) >= len(self.keys):
                if wrap:
                    # Wrap around to the beginning
                    current_pos = (start_pos + periods_sampled) % len(self.keys)
                else:
                    # Stop sampling if we reach the end
                    break
            else:
                current_pos = start_pos + periods_sampled

            # Add this period to results
            current_key = self.keys[current_pos]
            result[current_key] = self.jobs[current_key]
            periods_sampled += 1

        # Update current position for next sample
        self.current_position = (start_pos + periods_sampled) % len(self.keys) if wrap else min(start_pos + periods_sampled, len(self.keys))

        return result

    def sample_aggregated(self, n=1, wrap=True):
        """
        Efficiently sample n time periods of aggregated jobs using precalculated data.

        Parameters:
        - n: Number of time periods to sample
        - wrap: Whether to wrap around to the beginning when reaching the end

        Returns:
        - Dictionary mapping period keys to lists of aggregated jobs
        """
        if not self.keys:
            return {}

        result = {}
        periods_sampled = 0

        # Get starting position
        start_pos = self.current_position
        if start_pos >= len(self.keys):
            if wrap:
                start_pos = 0
            else:
                return {}

        # Sample n consecutive periods
        while periods_sampled < n:
            if (start_pos + periods_sampled) >= len(self.keys):
                if wrap:
                    # Wrap around to the beginning
                    current_pos = (start_pos + periods_sampled) % len(self.keys)
                else:
                    # Stop sampling if we reach the end
                    break
            else:
                current_pos = start_pos + periods_sampled

            # Add the precalculated aggregated jobs for this period
            current_key = self.keys[current_pos]
            result[current_key] = self.aggregated_jobs[current_key]
            periods_sampled += 1

        # Update current position for next sample
        self.current_position = (start_pos + periods_sampled) % len(self.keys) if wrap else min(start_pos + periods_sampled, len(self.keys))

        return result

    def reset_position(self):
        """Reset the current sampling position to the beginning."""
        self.current_position = 0
        return self

    def sample_random(self, n=1):
        """Sample n random time bins from the parsed data."""
        if not self.keys:
            return {}

        selected_keys = random.sample(self.keys, min(n, len(self.keys)))
        return {key: self.jobs[key] for key in selected_keys}

    def aggregate_jobs(self, jobs_list):
        """
        Aggregate similar jobs to reduce the total number of job objects.

        Parameters:
        - jobs_list: List of job dictionaries

        Returns:
        - List of aggregated job dictionaries with additional 'count' field
        """
        if not jobs_list:
            return []

        # Create bins for jobs with similar characteristics
        job_bins = {}

        for job in jobs_list:
            # Create a key based on job characteristics
            key = (job['nnodes'], job['cores_per_node'], job['duration_minutes'])

            if key not in job_bins:
                # First job with these characteristics - create a new bin
                job_bins[key] = {
                    'nnodes': job['nnodes'],
                    'cores_per_node': job['cores_per_node'],
                    'duration_minutes': job['duration_minutes'],
                    'count': 1,
                    'total_core_minutes': job['nnodes'] * job['cores_per_node'] * job['duration_minutes']
                }
            else:
                # Add to existing bin
                job_bins[key]['count'] += 1
                job_bins[key]['total_core_minutes'] += job['nnodes'] * job['cores_per_node'] * job['duration_minutes']

        # Convert bins to a list of aggregated jobs
        aggregated_jobs = list(job_bins.values())

        # Sort by total resource usage (largest first)
        aggregated_jobs.sort(key=lambda j: j['total_core_minutes'], reverse=True)

        return aggregated_jobs

    def calculate_resource_hours(self, jobs):
        """
        Calculate the total resource-hours (node-hours, core-hours) for a set of jobs.
        Useful for validating that aggregation preserves the overall workload.

        Parameters:
        - jobs: List of job dictionaries (raw or aggregated)

        Returns:
        - Dictionary with resource usage statistics
        """
        stats = {
            'total_jobs': 0,
            'total_node_hours': 0,
            'total_core_hours': 0
        }

        for job in jobs:
            # Check if this is an aggregated job
            count = job.get('count', 1)
            stats['total_jobs'] += count

            # Convert minutes to hours for calculation
            duration_hours = job['duration_minutes'] / 60

            stats['total_node_hours'] += job['nnodes'] * duration_hours * count
            stats['total_core_hours'] += job['nnodes'] * job['cores_per_node'] * duration_hours * count

        return stats

    def convert_to_hourly_jobs(self, aggregated_jobs, cores_per_node, max_nodes_per_job):
        """
        Convert aggregated jobs to hourly simulation jobs.

        Parameters:
        - aggregated_jobs: List of aggregated job dictionaries
        - cores_per_node: Number of cores per node in the simulation
        - max_nodes_per_job: Maximum nodes a job can use in the simulation

        Returns:
        - List of hourly simulation job dictionaries
        """
        hourly_jobs = []

        for agg_job in aggregated_jobs:
            # Calculate total compute resources needed
            total_core_minutes = agg_job['total_core_minutes']
            total_core_hours = total_core_minutes / 60

            # If the job is sub-hour, scale it appropriately
            if agg_job['duration_minutes'] < 60:
                # Calculate how many cores would be needed to do the same work in an hour
                # This preserves the total core-hours
                equivalent_cores = max(1, int(total_core_hours))

                # Convert to nodes based on cores_per_node
                equivalent_nodes = max(1, math.ceil(equivalent_cores / cores_per_node))
                equivalent_nodes = min(equivalent_nodes, max_nodes_per_job)

                # Calculate cores per node based on distribution
                cores_needed = min(equivalent_cores, equivalent_nodes * cores_per_node)
                cores_per_node_needed = min(cores_per_node, math.ceil(cores_needed / equivalent_nodes))

                # Create the hourly equivalent job
                hourly_job = {
                    'nnodes': equivalent_nodes,
                    'cores_per_node': cores_per_node_needed,
                    'duration_hours': 1,  # 1 hour
                    'original_job_count': agg_job['count']
                }
                hourly_jobs.append(hourly_job)
            else:
                # For longer jobs, keep the original structure but convert to hours
                # with appropriate scaling
                duration_hours = math.ceil(agg_job['duration_minutes'] / 60)

                hourly_job = {
                    'nnodes': agg_job['nnodes'],
                    'cores_per_node': agg_job['cores_per_node'],
                    'duration_hours': duration_hours,
                    'original_job_count': agg_job['count']
                }
                hourly_jobs.append(hourly_job)

            if hourly_job['duration_hours'] > self.max_job_duration:
                self.max_job_duration = hourly_job['duration_hours']

        if len(hourly_jobs) > self.max_new_jobs_per_hour:
            self.max_new_jobs_per_hour = len(hourly_jobs)

        return hourly_jobs

    def precalculate_hourly_jobs(self, cores_per_node, max_nodes_per_job):
        """
        Precalculate hourly job conversions for all time periods.

        Parameters:
        - cores_per_node: Number of cores per node in the simulation
        - max_nodes_per_job: Maximum nodes a job can use in the simulation

        Returns:
        - self for method chaining
        """
        self.hourly_jobs = {}

        for period_key, aggregated_jobs in self.aggregated_jobs.items():
            if aggregated_jobs:
                self.hourly_jobs[period_key] = self.convert_to_hourly_jobs(aggregated_jobs, cores_per_node, max_nodes_per_job)
            else:
                self.hourly_jobs[period_key] = []

        return self

    def sample_hourly(self, n=1, wrap=True):
        """
        Sample n time periods and return comprehensive data for each period.

        Parameters:
        - n: Number of time periods to sample
        - wrap: Whether to wrap around to the beginning when reaching the end

        Returns:
        - Dictionary mapping period keys to dictionaries containing:
        - 'raw_jobs': List of raw jobs
        - 'aggregated_jobs': List of aggregated jobs
        - 'hourly_jobs': List of hourly jobs
        """
        if not hasattr(self, 'hourly_jobs') or not self.hourly_jobs:
            raise ValueError("Hourly jobs not precalculated. Call precalculate_hourly_jobs first.")

        # Sample the periods
        sample_periods = self.sample(n, wrap)

        # Build comprehensive results
        results = {}
        for period_key in sample_periods:
            results[period_key] = {
                'raw_jobs': self.jobs.get(period_key, []),
                'aggregated_jobs': self.aggregated_jobs.get(period_key, []),
                'hourly_jobs': self.hourly_jobs.get(period_key, [])
            }

        return results

    def sample_one_hourly(self, wrap=True):
        """
        Sample one time period and return comprehensive data for each period.

        Parameters:
        - wrap: Whether to wrap around to the beginning when reaching the end

        Returns:
        - Dictionary containing:
        - 'raw_jobs': List of raw jobs
        - 'aggregated_jobs': List of aggregated jobs
        - 'hourly_jobs': List of hourly jobs
        """
        if not hasattr(self, 'hourly_jobs') or not self.hourly_jobs:
            raise ValueError("Hourly jobs not precalculated. Call precalculate_hourly_jobs first.")

        # Sample the periods
        sample_periods = self.sample(1, wrap)

        # Build comprehensive results
        results = {}
        for period_key in sample_periods:
            results = {
                'raw_jobs': self.jobs.get(period_key, []),
                'aggregated_jobs': self.aggregated_jobs.get(period_key, []),
                'hourly_jobs': self.hourly_jobs.get(period_key, [])
            }

        return results

jobs_sampler = DurationSampler()
