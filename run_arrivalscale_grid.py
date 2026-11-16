#!/usr/bin/env python3
r"""Run arrival-scale analyses sequentially for three sessions and three years.

Each session/year is evaluated with sampler seeds 101, 201, and 301. Results
are kept in analysis/v2_<weights>/s<session>_pr<year>_seed<seed>_jobs23/.
The existing analyzer writes eval_main.csv, summary.csv, plots, and scale logs
inside each run directory. Idle and drop weights are always 0.0.

Example (run with the project's Python environment):
    python run_arrivalscale_grid.py \
        --session-root /path/to/sessions/v2 \
        --efficiency-weight 0.3 --price-weight 0.3 --job-age-weight 0.4 \
        --dry-run

Remove --dry-run to execute. All nine exact checkpoints and input files must
exist before execution starts. Existing run directories are never overwritten.
Once started, failed runs are recorded in batch_status.csv and the batch
continues. All paths supplied as relative paths are relative to this script.
"""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


SESSION_SEEDS = (1, 2, 3)
TEST_YEARS = (2023, 2024, 2025)
EVAL_SEEDS = (101, 201, 301)
DEFAULT_SCALES = "0.25,0.5,0.75,1,1.5,2,2.5,3,4,5"


@dataclass(frozen=True)
class Run:
    session_seed: int
    test_year: int
    eval_seed: int
    session: Path
    prices: Path
    checkpoint: Path
    out_dir: Path
    command: tuple[str, ...]


def nonnegative_weight(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("weights must be finite and nonnegative")
    return value


def weight_tag(efficiency: float, price: float, job_age: float) -> str:
    """Encode weights in tenths: 0.3 -> 3, 0.25 -> 2.5."""
    def token(value: float) -> str:
        if value == 0:
            return "0"
        return format((Decimal(str(value)) * 10).normalize(), "f")

    return f"e{token(efficiency)}p{token(price)}s{token(job_age)}"


def resolve_path(path: Path, project_root: Path) -> Path:
    path = path.expanduser()
    return (project_root / path).resolve()


def build_command(
    args: argparse.Namespace,
    project_root: Path,
    session: Path,
    prices: Path,
    eval_seed: int,
    out_dir: Path,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-u",
        str(project_root / "analyze_arrivalscale_occupancy.py"),
        "--sampler",
        "--hourly-jobs", str(resolve_path(args.hourly_jobs, project_root)),
        "--scales", args.scales,
        "--prices", str(prices),
        "--dispatcher", "easy",
        "--seed", str(eval_seed),
        "--efficiency-weight", str(args.efficiency_weight),
        "--price-weight", str(args.price_weight),
        "--job-age-weight", str(args.job_age_weight),
        "--idle-weight", "0.0",
        "--drop-weight", "0.0",
        "--oracle",
        "--session", str(session),
        "--model", str(args.model),
        "--eval-months", "13",
        "--out-dir", str(out_dir),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--session-root", required=True, type=Path,
        help="Parent of de_<weights>_s<1|2|3>_test<2023|2024|2025> session directories.",
    )
    parser.add_argument("--efficiency-weight", required=True, type=nonnegative_weight)
    parser.add_argument("--price-weight", required=True, type=nonnegative_weight)
    parser.add_argument("--job-age-weight", required=True, type=nonnegative_weight)
    parser.add_argument("--model", type=int, default=8_000_000)
    parser.add_argument("--hourly-jobs", type=Path, default=Path("data-internal/allusers-main-30.log"))
    parser.add_argument("--prices-dir", type=Path, default=Path("data-external"))
    parser.add_argument("--out-root", type=Path, default=Path("analysis"))
    parser.add_argument("--scales", default=DEFAULT_SCALES)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print all 27 commands without checking input files, creating outputs, or running analyses.",
    )
    return parser


def build_runs(args: argparse.Namespace, project_root: Path) -> list[Run]:
    tag = weight_tag(args.efficiency_weight, args.price_weight, args.job_age_weight)
    session_root = resolve_path(args.session_root, project_root)
    prices_dir = resolve_path(args.prices_dir, project_root)
    group_dir = resolve_path(args.out_root, project_root) / f"v2_{tag}"
    # Match train.py's float formatting exactly; it can otherwise fall back to
    # another checkpoint when the requested model is absent.
    model_weights = (
        f"e{args.efficiency_weight}_p{args.price_weight}"
        f"_i0.0_a{args.job_age_weight}_d0.0"
    )
    runs = []
    for session_seed in SESSION_SEEDS:
        for test_year in TEST_YEARS:
            session = session_root / f"de_{tag}_s{session_seed}_test{test_year}"
            prices = prices_dir / f"prices_{test_year}.csv"
            checkpoint = session / "models" / model_weights / f"{args.model}.zip"
            for eval_seed in EVAL_SEEDS:
                out_dir = group_dir / f"s{session_seed}_pr{test_year % 100:02d}_seed{eval_seed}_jobs23"
                runs.append(Run(
                    session_seed=session_seed,
                    test_year=test_year,
                    eval_seed=eval_seed,
                    session=session,
                    prices=prices,
                    checkpoint=checkpoint,
                    out_dir=out_dir,
                    command=build_command(args, project_root, session, prices, eval_seed, out_dir),
                ))
    return runs


def preflight(runs: list[Run], args: argparse.Namespace, project_root: Path) -> list[str]:
    """Check the entire batch before any outputs or subprocesses are created."""
    required_files = [
        project_root / "analyze_arrivalscale_occupancy.py",
        project_root / "train.py",
        resolve_path(args.hourly_jobs, project_root),
    ]
    required_files.extend(run.prices for run in runs)
    required_files.extend(run.checkpoint for run in runs)
    errors = []
    for path in dict.fromkeys(required_files):
        if not path.is_file():
            errors.append(f"Missing required file: {path}")
            continue
        try:
            with path.open("rb") as handle:
                handle.read(1)
        except OSError as exc:
            errors.append(f"Cannot read {path}: {exc}")
    for run in runs:
        if run.out_dir.exists() or run.out_dir.is_symlink():
            errors.append(f"Output already exists (choose another --out-root): {run.out_dir}")
    return errors


def write_batch_status(path: Path, rows: list[dict]) -> None:
    # Replace the small manifest atomically so it remains readable during runs.
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_grid(runs: list[Run], project_root: Path) -> int:
    group_dir = runs[0].out_dir.parent
    group_dir.mkdir(parents=True, exist_ok=True)
    status_path = group_dir / "batch_status.csv"
    rows = [dict(
        session_seed=run.session_seed, test_year=run.test_year, eval_seed=run.eval_seed,
        session=str(run.session), prices=str(run.prices), out_dir=str(run.out_dir),
        status="pending", returncode="", error="",
    ) for run in runs]
    write_batch_status(status_path, rows)
    failed = 0
    for index, (run, row) in enumerate(zip(runs, rows), start=1):
        print(f"\n[{index}/{len(runs)}] {run.out_dir.name}", flush=True)
        print(shlex.join(run.command), flush=True)
        row["status"] = "running"
        write_batch_status(status_path, rows)
        try:
            run.out_dir.mkdir(exist_ok=False)
            (run.out_dir / "command.txt").write_text(shlex.join(run.command) + "\n", encoding="utf-8")
            completed = subprocess.run(run.command, cwd=project_root, check=False)
            row["returncode"] = completed.returncode
            row["status"] = "completed" if completed.returncode == 0 else "failed"
            if completed.returncode != 0:
                row["error"] = f"Analysis exited with status {completed.returncode}"
        except OSError as exc:
            row.update(status="failed", returncode=1, error=str(exc))
        except KeyboardInterrupt:
            row.update(status="interrupted", returncode=130, error="Interrupted by user")
            write_batch_status(status_path, rows)
            raise
        write_batch_status(status_path, rows)
        if row["status"] == "failed":
            failed += 1
            print(f"[failed] {run.out_dir.name}: {row['error']}; continuing.", file=sys.stderr, flush=True)
    print(f"\nBatch complete: {len(runs) - failed} succeeded, {failed} failed. Status: {status_path}", flush=True)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.model <= 0:
        parser.error("--model must be positive")
    try:
        scales = [float(value) for value in args.scales.split(",")]
    except ValueError:
        parser.error("--scales must be a comma-separated list of numbers")
    if any(not math.isfinite(value) or value < 0 for value in scales):
        parser.error("--scales values must be finite and nonnegative")
    project_root = Path(__file__).resolve().parent
    runs = build_runs(args, project_root)
    if args.dry_run:
        print(f"Dry run: {len(runs)} sequential analyses; no files checked or written.")
        for index, run in enumerate(runs, start=1):
            print(f"[{index}/{len(runs)}] {shlex.join(run.command)}")
        return 0
    errors = preflight(runs, args, project_root)
    if errors:
        print("Preflight failed; no analyses started:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1
    try:
        return run_grid(runs, project_root)
    except KeyboardInterrupt:
        print("\nBatch interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
