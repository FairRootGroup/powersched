#!/usr/bin/env python3
"""Run all tests sequentially."""

import subprocess
import sys

TESTS = [
    ["python", "-m", "test.test_checkenv"],
    ["python", "-m", "test.test_env"],
    ["python", "-m", "test.test_plotter"],
    ["python", "-m", "test.test_sanity_workloadgen"],
    ["python", "-m", "test.test_determinism_workloadgen"],
    ["python", "-m", "test.test_price_history"],
    ["python", "-m", "test.test_prices_cycling"],
    ["python", "-m", "test.test_sanity_env", "--steps", "200"],
    ["python", "-m", "test.test_sanity_env", "--check-gym", "--check-determinism", "--steps", "300"],
    ["python", "-m", "test.test_sanity_env", "--prices", "data/prices_2023.csv", "--hourly-jobs", "data/allusers-gpu-30.log", "--steps", "300"],
    ["python", "-m", "test.test_sanity_env", "--prices", "data/prices_2023.csv", "--hourly-jobs", "data/allusers-gpu-30.log", "--steps", "300", "--carry-over-state"],
    ["python", "-m", "test.test_sampler_duration", "--print-stats", "--test-samples", "10"],
    ["python", "-m", "test.test_sampler_hourly", "--file-path", "data/allusers-gpu-30.log", "--test-day"],
    ["python", "-m", "test.test_sampler_jobs", "--file-path", "data/allusers-gpu-30.log"],
    ["python", "-m", "test.test_sampler_jobs_aggregated", "--file-path", "data/allusers-gpu-30.log"],
]

def main():
    failed = []
    for cmd in TESTS:
        name = cmd[2]  # test module name
        print(f"\n{'='*60}")
        print(f"Running: {' '.join(cmd)}")
        print('='*60)

        result = subprocess.run(cmd)
        if result.returncode != 0:
            failed.append(name)
            print(f"FAILED: {name}")
        else:
            print(f"PASSED: {name}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Total: {len(TESTS)}, Passed: {len(TESTS) - len(failed)}, Failed: {len(failed)}")

    if failed:
        print(f"\nFailed tests:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
