#!/usr/bin/env python3
"""
Merge multiple Slurm partition log files into one chronologically sorted file.

Usage:
    python merge_logs.py [--output merged.log] [file1.log file2.log ...]

If no files are given, defaults to all allusers-*-30.log files in the script's directory.
Sort key is the Submit timestamp (4th whitespace-delimited column).
"""

import argparse
import glob
import os
import sys


def parse_submit(line: str):
    """Return the Submit timestamp string for sorting, or None for non-data lines."""
    parts = line.split()
    if len(parts) >= 4:
        ts = parts[3]
        # Submit timestamps look like 2025-04-13T16:35:12
        if len(ts) == 19 and ts[4] == '-' and 'T' in ts:
            return ts
    return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Merge Slurm partition logs chronologically.")
    parser.add_argument(
        'files', nargs='*',
        help='Log files to merge (default: all allusers-*-30.log in script directory)'
    )
    parser.add_argument(
        '--output', '-o', default=os.path.join(script_dir, 'merged-30.log'),
        help='Output file path (default: merged-30.log in script directory)'
    )
    args = parser.parse_args()

    input_files = args.files or sorted(glob.glob(os.path.join(script_dir, 'allusers-*-30.log')))

    if not input_files:
        print("No input files found.", file=sys.stderr)
        sys.exit(1)

    output_abs = os.path.realpath(args.output)
    if any(os.path.realpath(p) == output_abs for p in input_files):
        print(f"Error: output file '{args.output}' is also listed as an input — aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"Merging {len(input_files)} files:")
    for f in input_files:
        print(f"  {f}")

    header = None
    data_lines = []

    for path in input_files:
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                h1 = fh.readline()
                h2 = fh.readline()
                if not h1 or not h2:
                    print(f"Error: '{path}' has fewer than 2 header lines.", file=sys.stderr)
                    sys.exit(1)
                file_header = h1 + (h2 if h2.endswith('\n') else h2 + '\n')
                if header is None:
                    header = file_header
                elif header != file_header:
                    print(
                        f"Warning: header in {path} differs from first file — using first file's header.",
                        file=sys.stderr,
                    )
                # Collect data lines (skip blank lines)
                for line in fh:
                    if line.strip():
                        data_lines.append(line if line.endswith('\n') else line + '\n')
        except OSError as e:
            print(f"Error reading '{path}': {e}", file=sys.stderr)
            sys.exit(1)

    def _submit_key(line: str):
        ts = parse_submit(line)
        return (ts is None, ts or "")

    print(f"Sorting {len(data_lines):,} job records by Submit timestamp...")
    data_lines.sort(key=_submit_key)

    try:
        with open(args.output, 'w') as out:
            out.write(header)
            out.writelines(data_lines)
    except OSError as e:
        print(f"Error writing '{args.output}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Written to: {args.output}")


if __name__ == '__main__':
    main()
