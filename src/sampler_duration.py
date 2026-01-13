import numpy as np
import matplotlib.pyplot as plt

class DurationSampler:
    def __init__(self):
        self.hours_data = None
        self.sample_values = None
        self.probabilities = None

    def _parse_duration(self, duration_str):
        """Parse duration string in format HH:MM:SS or DD-HH:MM:SS into hours."""
        duration_str = duration_str.strip()
        days = 0
        hours = 0
        minutes = 0
        seconds = 0

        parts = duration_str.split(':')
        if len(parts) == 3:
            if '-' in parts[0]:
                day_hour = parts[0].split('-')
                days = int(day_hour[0])
                hours = int(day_hour[1])
            else:
                hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])

        total_hours = days * 24 + hours + minutes/60 + seconds/3600
        return total_hours

    def _create_sampler(self, hours):
        """Create sampling function from duration data."""
        rounded_hours = np.ceil(np.maximum(hours, 1))
        max_hour = int(max(rounded_hours))
        bins = range(1, max_hour + 2)
        counts, edges = np.histogram(rounded_hours, bins=bins)
        probabilities = counts / len(rounded_hours)
        self.sample_values = edges[:-1]
        self.probabilities = probabilities

    def init(self, file_path):
        """Initialize the sampler with duration data from a file."""
        try:
            with open(file_path, 'r') as file:
                durations = file.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_path}' not found.")
        except Exception as e:
            raise Exception(f"Error reading file: {e}")

        self.hours_data = [self._parse_duration(duration) for duration in durations]
        self._create_sampler(self.hours_data)
        return self

    def sample(self, n=1):
        """
        Sample n durations from the empirical distribution.

        Args:
            n (int): Number of samples to generate

        Returns:
            float or numpy.ndarray: Single float if n=1, otherwise array of floats
        """
        if self.sample_values is None or self.probabilities is None:
            raise RuntimeError("Sampler not initialized. Call init() first.")

        samples = np.random.choice(self.sample_values, size=n, p=self.probabilities)
        return samples.astype(int)

    def get_stats(self):
        """Return summary statistics of the duration data."""
        if self.hours_data is None:
            raise RuntimeError("No data available. Call init() first.")

        stats = {
            'total_jobs': len(self.hours_data),
            'average_duration': np.mean(self.hours_data),
            'max_duration': max(self.hours_data),
            'min_duration': min(self.hours_data),
            'hour_breakdown': {}
        }

        # Create hour-by-hour breakdown
        for hour in self.hours_data:
            bin_hour = int(hour) + 1
            stats['hour_breakdown'][bin_hour] = stats['hour_breakdown'].get(bin_hour, 0) + 1

        return stats

    def plot(self):
        """Create and display a histogram of job durations."""
        if self.hours_data is None:
            raise RuntimeError("No data available. Call init() first.")

        plt.figure(figsize=(12, 6))

        # Calculate max hour (rounded up) for proper binning
        max_hour = int(max(self.hours_data)) + 1
        bins = range(0, max_hour + 2)  # +2 to include the last hour

        # Create histogram
        plt.hist(self.hours_data, bins=bins, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Job Run Times')
        plt.xlabel('Duration (hours)')
        plt.ylabel('Number of Jobs')
        plt.grid(True, alpha=0.3)

        # Add exact counts above each bar
        counts, edges = np.histogram(self.hours_data, bins=bins)
        for i in range(len(counts)):
            if counts[i] > 0:  # Only show counts > 0
                plt.text(edges[i] + 0.5, counts[i], str(counts[i]),
                        ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

# Create a singleton instance
durations_sampler = DurationSampler()