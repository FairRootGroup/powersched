"""Exercise the existing figure pipeline without writing raster/PDF artifacts."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    unittest.main()
