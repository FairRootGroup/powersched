"""Exercise the existing figure pipeline without writing raster/PDF artifacts."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from test.test_multi_run_analysis import write_grid


PLOTTER = Path(__file__).resolve().parents[1] / "meta-eval" / "plot_eval_oracle.py"
FIGURES = ("fig1", "fig1b", "fig2", "fig3", "fig3b", "fig4")


def plot_inputs(root):
    directories = write_grid(root)
    for path in directories[0].glob("*/eval_main.csv"):
        frame = pd.read_csv(path)
        values = frame.cost_agent
        frame["cost_baseline_off"] = values + 20
        frame["savings_vs_baseline_off_pct"] = 100 * 20 / (values + 20)
        frame["cost_per_1k_agent"] = values / 10
        frame["cost_per_1k_baseline_off"] = (values + 20) / 10
        frame["completion_pct"] = 98 + values / 10000
        frame["mean_wait_h"] = values / 100
        frame["dropped_jobs"] = values / 20
        frame["power_agent_mwh"] = values / 5
        frame["power_baseline_off_mwh"] = values / 4
        frame["eur_per_mwh_agent"] = values / 2
        frame["eur_per_mwh_baseline_off"] = values / 2 + 10
        frame["oracle_contiguous_eur"] = values / 3
        frame["oracle_capture_pct"] = values / 10
        # Some runs lose money, even though the grand mean savings is positive.
        frame["savings_abs_vs_baseline_off_eur"] = -100 if "_seed9_" in path.parent.name else 100
        frame["price_savings_pct"] = 70.0
        frame["power_savings_pct"] = 30.0
        frame.to_csv(path, index=False)
    return directories


class MultiRunPlottingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.directories = plot_inputs(cls.root)
        argv = [str(PLOTTER), "--multi-run", "--input-dirs", str(cls.directories[0]),
                "--output-dir", str(cls.root / "multi_output")]
        with patch.object(sys, "argv", argv), patch.object(Figure, "savefig"), redirect_stdout(io.StringIO()):
            cls.module = runpy.run_path(str(PLOTTER))

    @classmethod
    def tearDownClass(cls):
        plt.close("all")
        cls.temp.cleanup()

    def test_each_existing_figure_contains_descriptive_error_bars(self):
        for name in FIGURES:
            with self.subTest(figure=name):
                containers = [container for ax in self.module[name].axes for container in ax.containers]
                errors = [item for item in containers if isinstance(item, ErrorbarContainer)]
                self.assertTrue(errors)
                self.assertTrue(all(item.has_yerr for item in errors))
                self.assertFalse(any(item.has_xerr for item in errors))

    def test_plotting_table_consumes_grand_means_and_total_sd(self):
        means = self.module["multi_run_result"].grand_total
        plotted = self.module["df"]
        for metric in self.module["multi_run_result"].metrics:
            np.testing.assert_allclose(plotted[metric], means[f"{metric}_mean"])
            np.testing.assert_allclose(plotted[f"{metric}_std_total"], means[f"{metric}_std_total"])
        np.testing.assert_allclose(plotted.signed_price_savings_pct, 70 / 3)
        np.testing.assert_allclose(plotted.signed_price_savings_pct_std_total,
                                   np.std([-70] * 9 + [70] * 18, ddof=1))

    def assert_reference_markers(self, module, actual_scale):
        reference = module["ref"].set_index("scale")
        lower_x = reference.loc[0.5, "arrivals"]
        upper_x = reference.loc[1.0, "arrivals"]
        category_x = (actual_scale - 0.5) / 0.5
        arrival_x = lower_x + category_x * (upper_x - lower_x)
        self.assertAlmostEqual(module["scale_one_arrival"], arrival_x)
        for name in FIGURES:
            expected_x = category_x if name in ("fig2", "fig3", "fig3b") else arrival_x
            markers = [line for ax in module[name].axes for line in ax.lines
                       if line.get_label() == "scale = 1"]
            with self.subTest(figure=name):
                self.assertTrue(markers, "reference marker disappeared")
                for marker in markers:
                    np.testing.assert_allclose(marker.get_xdata(), [expected_x, expected_x])
                for ax in module[name].axes:
                    if any(line.get_label() == "scale = 1" for line in ax.lines):
                        lower, upper = sorted(ax.get_xlim())
                        self.assertLessEqual(lower, expected_x)
                        self.assertGreaterEqual(upper, expected_x)
                scale_labels = [line.get_label() for ax in module[name].axes for line in ax.lines
                                if "scale =" in line.get_label()]
                self.assertEqual(set(scale_labels), {"scale = 1"})
        text = [text for ax in module["fig4"].axes for text in ax.texts
                if text.get_text() == "scale = 1"]
        self.assertEqual(len(text), 1)
        self.assertAlmostEqual(text[0].get_position()[0], arrival_x)

    def test_default_reference_scale_keeps_existing_marker_positions(self):
        self.assert_reference_markers(self.module, actual_scale=1.0)

    def test_defined_value_maps_actual_scale_to_all_plot_coordinate_systems(self):
        source = sorted(self.directories[0].glob("*/eval_main.csv"))[0]
        for multi_run in (False, True):
            for actual_scale in (0.5, 0.6, 1.5):
                with self.subTest(multi_run=multi_run, actual_scale=actual_scale):
                    output = self.root / f"marker_{multi_run}_{actual_scale}"
                    argv = [str(PLOTTER), "--output-dir", str(output)]
                    argv += (["--multi-run", "--input-dirs", str(self.directories[0])]
                             if multi_run else [str(source)])
                    with patch.object(sys, "argv", argv), patch.object(Figure, "savefig"), redirect_stdout(io.StringIO()):
                        module = runpy.run_path(str(PLOTTER), init_globals={"DEFINED_VALUE": actual_scale})
                    self.assert_reference_markers(module, actual_scale)
                    if multi_run:
                        pd.testing.assert_frame_equal(module["multi_run_result"].grand_total,
                                                      self.module["multi_run_result"].grand_total)

    def test_linear_unit_conversion_scales_errors_and_leaves_x_uncertainty_out(self):
        rows = pd.DataFrame({"arrivals": [300, 310], "cost_agent": [10, 20],
                             "cost_agent_std_total": [3, 4], "arrivals_std_total": [8, 9]})
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        with patch.object(ax, "errorbar", wraps=ax.errorbar) as errorbar:
            lines = self.module["plot_measurement"](
                ax, rows, "cost_agent", transform=lambda values: values / 1000,
                color="blue", marker="o",
            )
        np.testing.assert_allclose(lines[0].get_xdata(), [300, 310])
        np.testing.assert_allclose(lines[0].get_ydata(), [0.01, 0.02])
        np.testing.assert_allclose(errorbar.call_args.kwargs["yerr"], [0.003, 0.004])
        self.assertNotIn("xerr", errorbar.call_args.kwargs)

    def test_dropped_job_whiskers_stop_at_zero_without_changing_values(self):
        rows = pd.DataFrame({"arrivals": [100.0, 200.0, 300.0],
                             "dropped_jobs": [0.0, 2.0, 10.0],
                             "dropped_jobs_std_total": [0.0, 5.0, 3.0]})
        original = rows.copy(deep=True)
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        plot = self.module["plot_dropped_jobs"]
        with patch.dict(plot.__globals__, {"mrow": lambda model: rows}), \
                patch.object(ax, "errorbar", wraps=ax.errorbar) as errorbar:
            plot(ax)
        yerr = np.asarray(errorbar.call_args.kwargs["yerr"])
        np.testing.assert_allclose(yerr, [[0, 2, 3], [0, 5, 3]])
        policy = next(line for line in ax.lines if line.get_label() == self.module["MODELS"][0])
        np.testing.assert_allclose(policy.get_ydata(), [0, 2, 10])
        self.assertNotIn(policy.get_marker(), (None, "None", ""))
        self.assertEqual(ax.get_yscale(), "symlog")
        self.assertEqual(ax.get_ylim()[0], 0)
        container = next(item for item in ax.containers if isinstance(item, ErrorbarContainer))
        whiskers = np.asarray(container.lines[2][0].get_segments())
        np.testing.assert_allclose(whiskers[:, :, 1], [[0, 0], [0, 7], [7, 13]])
        pd.testing.assert_frame_equal(rows, original)

    def test_reference_curves_use_bands_without_whiskers_or_extra_legend_items(self):
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        original = self.module["df"].copy(deep=True)
        with patch.object(ax, "errorbar", wraps=ax.errorbar) as errorbar:
            self.module["ref_line"](ax, "cost_baseline_off", ":", "baseline-off",
                                    transform=lambda values: values / 1000)
        errorbar.assert_not_called()
        bands = [item for item in ax.collections if isinstance(item, PolyCollection)]
        self.assertEqual(len(bands), 1)
        self.assertGreater(bands[0].get_alpha(), 0)
        self.assertLessEqual(bands[0].get_alpha(), 0.15)
        self.assertLess(bands[0].get_zorder(), ax.lines[0].get_zorder())
        self.assertEqual(ax.get_legend_handles_labels()[1], ["baseline-off"])
        ref = self.module["ref"]
        np.testing.assert_allclose(ax.lines[0].get_ydata(), ref.cost_baseline_off / 1000)
        vertices = np.concatenate([path.vertices for path in bands[0].get_paths()])
        for row in ref.itertuples():
            ys = vertices[np.isclose(vertices[:, 0], row.arrivals), 1]
            self.assertAlmostEqual(ys.min(), (row.cost_baseline_off - row.cost_baseline_off_std_total) / 1000)
            self.assertAlmostEqual(ys.max(), (row.cost_baseline_off + row.cost_baseline_off_std_total) / 1000)
        pd.testing.assert_frame_equal(self.module["df"], original)

    def test_reference_bands_leave_gaps_for_missing_x_mean_or_sd(self):
        for missing in ("x", "mean", "std"):
            with self.subTest(missing=missing):
                values = {"x": np.arange(5, dtype=float), "mean": np.full(5, 10.0),
                          "std": np.ones(5)}
                values[missing][2] = np.nan
                fig, ax = plt.subplots()
                self.addCleanup(plt.close, fig)
                self.module["plot_reference_band"](ax, values["x"], values["mean"],
                                                   values["std"], color="gray")
                bands = [item for item in ax.collections if isinstance(item, PolyCollection)]
                self.assertEqual(len(bands), 1)
                paths = bands[0].get_paths()
                self.assertEqual(len(paths), 2)
                extents = sorted((path.vertices[:, 0].min(), path.vertices[:, 0].max())
                                 for path in paths)
                np.testing.assert_allclose(extents, [[0, 1], [3, 4]])
                self.assertTrue(all(np.isfinite(path.vertices).all() for path in paths))
                self.assertEqual(ax.get_legend_handles_labels()[1], [])

    def test_reference_band_bounds_apply_only_when_requested_and_require_sd(self):
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        band = self.module["plot_reference_band"]
        band(ax, [0, 1], [-2, -2], [np.nan, np.nan], color="gray")
        self.assertEqual(len(ax.collections), 0)
        band(ax, [0, 1], [-2, -2], [5, 5], color="gray")
        vertices = ax.collections[0].get_paths()[0].vertices
        self.assertEqual(vertices[:, 1].min(), -7)
        self.assertEqual(vertices[:, 1].max(), 3)
        band(ax, [0, 1, 2], [0, 2, 10], [0, 5, 3], color="blue", lower_bound=0)
        vertices = ax.collections[1].get_paths()[0].vertices
        self.assertEqual(vertices[:, 1].min(), 0)
        self.assertEqual(vertices[:, 1].max(), 13)

    def test_reference_band_clipping_respects_metric_units(self):
        rows = pd.DataFrame({"arrivals": [100, 200], "cost_baseline_off": [-2, -2],
                             "cost_baseline_off_std_total": [5, 5],
                             "power_baseline_off_mwh": [0, 2],
                             "power_baseline_off_mwh_std_total": [0, 5]})
        ref_line = self.module["ref_line"]
        for metric, expected_lower, expected_upper in (("cost_baseline_off", -7, 3),
                                                       ("power_baseline_off_mwh", 0, 7)):
            with self.subTest(metric=metric):
                fig, ax = plt.subplots()
                self.addCleanup(plt.close, fig)
                with patch.dict(ref_line.__globals__, {"ref": rows}):
                    ref_line(ax, metric, ":", "reference")
                band = next(item for item in ax.collections if isinstance(item, PolyCollection))
                vertices = np.concatenate([path.vertices for path in band.get_paths()])
                self.assertEqual(vertices[:, 1].min(), expected_lower)
                self.assertEqual(vertices[:, 1].max(), expected_upper)

    def test_cost_evolution_uses_reference_bands_and_keeps_policy_whiskers(self):
        ax = self.module["fig4"].axes[0]
        bands = [item for item in ax.collections if isinstance(item, PolyCollection)]
        self.assertEqual(len(bands), 2)
        errors = [item for item in ax.containers if isinstance(item, ErrorbarContainer)]
        self.assertEqual(len(errors), len(self.module["COST_EVOLUTION_MODELS"]))
        for error, model in zip(errors, self.module["COST_EVOLUTION_MODELS"]):
            rows = self.module["mrow"](model)
            segments = np.asarray(error.lines[2][0].get_segments())
            expected = np.column_stack([rows.cost_agent - rows.cost_agent_std_total,
                                        rows.cost_agent + rows.cost_agent_std_total]) / 1000
            np.testing.assert_allclose(segments[:, :, 1], expected)
        labels = ax.get_legend_handles_labels()[1]
        self.assertEqual(labels.count("baseline-off"), 1)
        self.assertEqual(labels.count("contiguous oracle"), 1)
        self.assertTrue(all(band.get_label().startswith("_") for band in bands))

    def test_blank_metrics_and_arrivals_are_preserved_in_tables_and_skipped_in_plots(self):
        inputs = plot_inputs(self.root / "partial_inputs")
        paths = sorted(inputs[0].glob("*/eval_main.csv"))
        for index, path in enumerate(paths):
            frame = pd.read_csv(path)
            frame.loc[frame.scale.eq(0.5), "arrivals"] = np.nan
            frame["mean_wait_h"] = np.nan
            if index == 0:
                frame.loc[frame.scale.eq(1.0), "mean_wait_h"] = 42.0
                frame.loc[frame.scale.eq(1.0), "savings_abs_vs_baseline_off_eur"] = np.nan
            frame.to_csv(path, index=False)
        output = self.root / "partial_output"
        argv = [str(PLOTTER), "--multi-run", "--input-dirs", str(inputs[0]),
                "--output-dir", str(output)]
        with patch.object(sys, "argv", argv), patch.object(Figure, "savefig") as savefig, redirect_stdout(io.StringIO()):
            module = runpy.run_path(str(PLOTTER))
        self.assertEqual(savefig.call_count, 12)
        self.assertEqual(module["df"].scale.tolist(), [0.5, 1.0])
        self.assertEqual(module["mrow"]("e2p2s6").scale.tolist(), [1.0])
        totals = module["multi_run_result"].grand_total.set_index("scale")
        self.assertEqual(totals.loc[0.5, "arrivals_n_valid"], 0)
        self.assertTrue(np.isnan(totals.loc[0.5, "arrivals_mean"]))
        self.assertEqual(totals.loc[0.5, "mean_wait_h_n_valid"], 0)
        self.assertTrue(np.isnan(totals.loc[0.5, "mean_wait_h_mean"]))
        self.assertEqual(totals.loc[1.0, "mean_wait_h_n_valid"], 1)
        self.assertEqual(totals.loc[1.0, "mean_wait_h_mean"], 42)
        self.assertTrue(np.isnan(totals.loc[1.0, "mean_wait_h_std_total"]))
        self.assertEqual(totals.loc[1.0, "signed_price_savings_pct_n_valid"], 26)
        # A missing sign must not be silently treated as a positive sign.
        available = module["raw"].loc[module["raw"].scale.eq(1.0), "signed_price_savings_pct"]
        self.assertEqual(available.isna().sum(), 1)
        grand_csv = next(path for path in output.rglob("*.csv") if "std_total" in path.read_text())
        exported = pd.read_csv(grand_csv).set_index("scale")
        self.assertEqual(exported.loc[0.5, "arrivals_n_valid"], 0)
        self.assertEqual(exported.loc[1.0, "signed_price_savings_pct_n_valid"], 26)

    def test_all_missing_arrivals_and_costs_still_export_and_render_figures(self):
        inputs = plot_inputs(self.root / "unavailable_inputs")
        for path in inputs[0].glob("*/eval_main.csv"):
            frame = pd.read_csv(path)
            for metric in ("arrivals", "cost_agent", "cost_baseline_off", "oracle_contiguous_eur"):
                frame[metric] = np.nan
            frame.to_csv(path, index=False)
        argv = [str(PLOTTER), "--multi-run", "--input-dirs", str(inputs[0]),
                "--output-dir", str(self.root / "unavailable_output")]
        with patch.object(sys, "argv", argv), patch.object(Figure, "savefig") as savefig, redirect_stdout(io.StringIO()):
            module = runpy.run_path(str(PLOTTER))
        self.assertEqual(savefig.call_count, 12)
        self.assertEqual(len(module["df"]), 2)
        self.assertTrue(module["mrow"]("e2p2s6").empty)
        totals = module["multi_run_result"].grand_total
        self.assertTrue(totals.cost_agent_mean.isna().all())
        self.assertTrue(totals.cost_agent_std_total.isna().all())
        self.assertTrue(totals.cost_agent_n_valid.eq(0).all())

    def test_disabled_mode_keeps_existing_figures_without_error_bars(self):
        source = sorted(self.directories[0].glob("*/eval_main.csv"))[0]
        argv = [str(PLOTTER), str(source), "--output-dir", str(self.root / "single_output")]
        with patch.object(sys, "argv", argv), patch.object(Figure, "savefig") as savefig, redirect_stdout(io.StringIO()):
            module = runpy.run_path(str(PLOTTER))
        self.assertFalse(module["args"].multi_run)
        self.assertEqual(module["name_suffix"], "main")
        self.assertEqual(savefig.call_count, 12)
        self.assertIsNone(module["measurement_error"](module["df"], "cost_agent"))
        self.assertFalse((self.root / "single_output" / "multi_run_tables").exists())
        for name in FIGURES:
            self.assertFalse(any(isinstance(container, ErrorbarContainer)
                                 for ax in module[name].axes for container in ax.containers))
            self.assertFalse(any(isinstance(collection, PolyCollection)
                                 for ax in module[name].axes for collection in ax.collections))

    def test_jobs_selector_requires_multi_run(self):
        source = sorted(self.directories[0].glob("*/eval_main.csv"))[0]
        argv = [str(PLOTTER), str(source), "--jobs", "23"]
        error = io.StringIO()
        with patch.object(sys, "argv", argv), redirect_stderr(error), \
                self.assertRaises(SystemExit) as failure:
            runpy.run_path(str(PLOTTER))
        self.assertEqual(failure.exception.code, 2)
        self.assertIn("--multi-run", error.getvalue())

    def test_jobs_selector_isolates_plot_data_and_default_output_names(self):
        project = self.root / "jobs_cli"
        inputs = plot_inputs(project / "inputs")
        # Run the real script from a temporary project so default output paths
        # can be exercised without writing generated artifacts into the repo.
        plotter = project / "meta-eval" / PLOTTER.name
        plotter.parent.mkdir()
        shutil.copy2(PLOTTER, plotter)

        def run_with_selector(selector):
            argv = [str(plotter), "--multi-run", "--input-dirs", str(inputs[0])]
            if selector is not None:
                argv += ["--jobs", selector]
            with patch.object(sys, "argv", argv), patch.object(Figure, "savefig") as savefig, redirect_stdout(io.StringIO()):
                module = runpy.run_path(str(plotter))
            return module, [Path(call.args[0]) for call in savefig.call_args_list]

        automatic, automatic_paths = run_with_selector(None)
        self.assertEqual(automatic["name_suffix"], "multi_run")
        self.assertEqual(automatic["OUTPUT_DIR"], project / "analysis" / "multi_run_meta_analysis")
        self.assertTrue(all(path.stem.endswith("_multi_run") for path in automatic_paths))

        for source in list(inputs[0].glob("*_jobs23/eval_main.csv")):
            destination = source.parent.with_name(source.parent.name.replace("jobs23", "jobs26"))
            destination.mkdir()
            frame = pd.read_csv(source)
            frame["cost_agent"] += 1000
            frame.to_csv(destination / "eval_main.csv", index=False)

        selected = {}
        for selector, normalized in (("23", "23"), ("jobs26", "26")):
            with self.subTest(selector=selector):
                module, paths = run_with_selector(selector)
                selected[normalized] = module["multi_run_result"].grand_total
                self.assertEqual(module["name_suffix"], f"multi_run_jobs{normalized}")
                self.assertEqual(module["OUTPUT_DIR"],
                                 project / "analysis" / f"multi_run_meta_analysis_jobs{normalized}")
                self.assertEqual(len(paths), 12)
                self.assertTrue(all(path.stem.endswith(f"_multi_run_jobs{normalized}") for path in paths))
                self.assertTrue((module["OUTPUT_DIR"] / "multi_run_tables").is_dir())
        pd.testing.assert_frame_equal(selected["23"], automatic["multi_run_result"].grand_total)
        np.testing.assert_allclose(selected["26"].cost_agent_mean - selected["23"].cost_agent_mean, 1000)


if __name__ == "__main__":
    unittest.main()
