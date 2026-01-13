from src.sampler_duration import durations_sampler
import argparse

def main():
    parser = argparse.ArgumentParser(description='Analyze job durations from a log file.')
    parser.add_argument('--file-path', default="data/durations_2024-09-01--14.txt", help='Path to the job duration log file')
    parser.add_argument('--plot', action='store_true', help='Plot histogram.')
    parser.add_argument("--test-samples", type=int, default=0, help='Number of test samples to generate.')
    parser.add_argument("--print-stats", action='store_true', help='Print summary statistics.')
    args = parser.parse_args()

    durations_sampler.init(args.file_path)

    if args.print_stats:
        stats = durations_sampler.get_stats()
        print("\nSummary statistics:")
        print(stats)

    if args.plot:
        durations_sampler.plot()

    if args.test_samples:
        samples = durations_sampler.sample(n=args.test_samples)
        print("\nTest samples:")
        print(samples)

if __name__ == "__main__":
    main()
