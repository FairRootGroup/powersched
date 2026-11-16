"""Tests for grid orchestration; no model evaluations are launched."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
import io
from itertools import product
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_arrivalscale_grid as grid


def option(command, name):
    return command[command.index(name) + 1]


class GridCommandTests(unittest.TestCase):
    def test_command_preserves_evaluation_settings_and_paths_with_spaces(self):
        root = Path("/tmp/grid project")
        session = root / "sessions/de_e3p3s4_s1_test2025"
        prices = root / "data-external/prices_2025.csv"
        output = root / "analysis/v2_e3p3s4/s1_pr25_seed101_jobs23"
        args = argparse.Namespace(
            hourly_jobs=Path("data-internal/allusers-main-30.log"),
            scales="0.25,0.5,0.75,1,1.5,2,2.5,3,4,5",
            efficiency_weight=0.3,
            price_weight=0.3,
            job_age_weight=0.4,
            model=8000000,
        )

        command = grid.build_command(args, root, session, prices, 101, output)

        self.assertEqual(command[0], sys.executable)
        self.assertIn("--sampler", command)
        self.assertIn("--oracle", command)
        expected = {
            "--session": str(session),
            "--prices": str(prices),
            "--out-dir": str(output),
            "--hourly-jobs": str(root / args.hourly_jobs),
            "--scales": args.scales,
            "--seed": "101",
            "--dispatcher": "easy",
            "--model": "8000000",
            "--eval-months": "13",
            "--efficiency-weight": "0.3",
            "--price-weight": "0.3",
            "--job-age-weight": "0.4",
            "--idle-weight": "0.0",
            "--drop-weight": "0.0",
        }
        for flag, value in expected.items():
            with self.subTest(flag=flag):
                self.assertEqual(option(command, flag), value)

    def test_weight_tag_matches_requested_directory_convention(self):
        self.assertEqual(grid.weight_tag(0.3, 0.3, 0.4), "e3p3s4")
        self.assertEqual(grid.weight_tag(0.0, 0.4, 0.6), "e0p4s6")


class GridBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cli = [
            "--session-root", str(self.root / "sessions"),
            "--out-root", str(self.root / "analysis"),
            "--efficiency-weight", "0.3",
            "--price-weight", "0.3",
            "--job-age-weight", "0.4",
        ]
        self.args = grid.build_arg_parser().parse_args(self.cli)
        self.runs = grid.build_runs(self.args, self.root)

    def create_inputs(self):
        paths = {
            self.root / "analyze_arrivalscale_occupancy.py",
            self.root / "train.py",
            self.root / "data-internal/allusers-main-30.log",
            *(run.prices for run in self.runs),
            *(run.checkpoint for run in self.runs),
        }
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")

    def test_grid_has_all_27_distinct_outputs_and_matching_inputs(self):
        expected_cells = list(product((1, 2, 3), (2023, 2024, 2025), (101, 201, 301)))
        self.assertEqual(
            [(run.session_seed, run.test_year, run.eval_seed) for run in self.runs],
            expected_cells,
        )
        self.assertEqual(len({run.out_dir for run in self.runs}), 27)
        self.assertEqual(len({run.session for run in self.runs}), 9)
        for run in self.runs:
            with self.subTest(session=run.session_seed, year=run.test_year, seed=run.eval_seed):
                expected_name = (
                    f"s{run.session_seed}_pr{str(run.test_year)[2:]}"
                    f"_seed{run.eval_seed}_jobs23"
                )
                self.assertEqual(
                    run.out_dir,
                    self.root / "analysis" / "v2_e3p3s4" / expected_name,
                )
                self.assertEqual(
                    run.session,
                    self.root / "sessions" / f"de_e3p3s4_s{run.session_seed}_test{run.test_year}",
                )
                self.assertEqual(run.prices, self.root / "data-external" / f"prices_{run.test_year}.csv")
                self.assertEqual(option(run.command, "--session"), str(run.session))
                self.assertEqual(option(run.command, "--out-dir"), str(run.out_dir))
                self.assertEqual(option(run.command, "--prices"), str(run.prices))
                self.assertEqual(option(run.command, "--seed"), str(run.eval_seed))
                self.assertEqual(
                    option(run.command, "--hourly-jobs"),
                    str(self.root / "data-internal/allusers-main-30.log"),
                )

    def test_preflight_requires_exact_checkpoint_even_when_another_exists(self):
        self.create_inputs()
        self.assertEqual(grid.preflight(self.runs, self.args, self.root), [])
        checkpoint = self.runs[0].checkpoint
        self.assertEqual(
            checkpoint.relative_to(self.runs[0].session).as_posix(),
            "models/e0.3_p0.3_i0.0_a0.4_d0.0/8000000.zip",
        )
        checkpoint.rename(checkpoint.with_name("9000000.zip"))

        errors = grid.preflight(self.runs, self.args, self.root)

        self.assertTrue(any(str(checkpoint) in error for error in errors), errors)
        self.assertFalse((self.root / "analysis").exists())

    def test_preflight_rejects_existing_output_without_changing_it(self):
        self.create_inputs()
        existing = self.runs[0].out_dir
        existing.mkdir(parents=True)
        result = existing / "eval_main.csv"
        result.write_text("previous result\n")

        errors = grid.preflight(self.runs, self.args, self.root)

        self.assertTrue(any(str(existing) in error for error in errors), errors)
        self.assertEqual(result.read_text(), "previous result\n")

    def test_dry_run_lists_commands_without_inputs_or_outputs(self):
        output = io.StringIO()
        with patch.object(grid.subprocess, "run") as launch:
            with redirect_stdout(output), redirect_stderr(io.StringIO()):
                result = grid.main(self.cli + ["--dry-run"])

        self.assertEqual(result, 0)
        launch.assert_not_called()
        self.assertEqual(sum("--sampler" in line for line in output.getvalue().splitlines()), 27)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_batch_runs_in_order_and_continues_after_exit_and_launch_failures(self):
        runs = self.runs[:4]
        calls = []

        def fake_run(command, **kwargs):
            index = len(calls)
            calls.append(tuple(command))
            self.assertEqual(tuple(command), runs[index].command)
            self.assertEqual(Path(kwargs["cwd"]), self.root)
            if index == 1:
                return subprocess.CompletedProcess(command, 7)
            if index == 2:
                raise OSError("simulated process launch failure")
            runs[index].out_dir.mkdir(parents=True, exist_ok=True)
            for filename in ("eval_main.csv", "summary.csv"):
                (runs[index].out_dir / filename).write_text("header\nresult\n")
            return subprocess.CompletedProcess(command, 0)

        with patch.object(grid.subprocess, "run", side_effect=fake_run):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = grid.run_grid(runs, self.root)

        self.assertEqual(result, 1)
        self.assertEqual(calls, [run.command for run in runs])
        with (runs[0].out_dir.parent / "batch_status.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["status"] for row in rows], ["completed", "failed", "failed", "completed"])


if __name__ == "__main__":
    unittest.main()
