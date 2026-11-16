"""Numerical and input-grid regression tests for multi-run meta-analysis."""

from itertools import product
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.multi_run_analysis import (
    aggregate_multi_run,
    export_multi_run,
    load_multi_run,
    parse_run_directory,
)


YEARS = (2023, 2024, 2025)
TRAINING_SEEDS = (4, 11, 29)
EVALUATION_SEEDS = (9, 55, 701)


def balanced_frame(models=("e2p2s6",), scales=(0.5, 1.0)):
    """Distinct scale/model offsets expose accidental pooling of either key."""
    rows = []
    for model_index, model in enumerate(models):
        for scale, (j, year), (i, train), (k, seed) in product(
            scales, enumerate(YEARS), enumerate(TRAINING_SEEDS),
            enumerate(EVALUATION_SEEDS),
        ):
            value = 1000 * model_index + 100 * scale + 20 * j + 5 * i + (-2, 0, 7)[k]
            rows.append({
                "model": model,
                "scale": scale,
                "eval_year": year,
                "training_seed": train,
                "eval_seed": seed,
                "arrivals": 300 * scale + i + 2 * j + 0.25 * k,
                "cost_agent": value,
                "cost_baseline": 2 * value + k + 1,
                "savings_vs_baseline_pct": 100 * (1 - value / (2 * value + k + 1)),
            })
    return pd.DataFrame(rows)


def write_grid(root, models=("e2p2s6",), scales=(0.5, 1.0)):
    frame = balanced_frame(models=models, scales=scales)
    directories = []
    for model in models:
        directory = root / f"v2_{model}"
        directories.append(directory)
        for year, train, seed in product(YEARS, TRAINING_SEEDS, EVALUATION_SEEDS):
            path = directory / f"s{train}_pr{year % 100}_seed{seed}_jobs23" / "eval_main.csv"
            path.parent.mkdir(parents=True)
            selection = frame.loc[
                frame.model.eq(model) & frame.eval_year.eq(year)
                & frame.training_seed.eq(train) & frame.eval_seed.eq(seed)
            ]
            selection.drop(columns=["eval_year", "training_seed", "eval_seed"]).to_csv(path, index=False)
    return directories


class HierarchicalAggregationTests(unittest.TestCase):
    def test_all_intermediate_means_and_sample_standard_deviations(self):
        raw = balanced_frame(models=("e4p4s2", "e2p2s6"))
        result = aggregate_multi_run(raw.sample(frac=1, random_state=22))

        self.assertEqual(len(result.eval_seed), 36)
        self.assertEqual(len(result.training_seed), 12)
        self.assertEqual(len(result.year), 4)
        self.assertEqual(len(result.grand_total), 4)
        self.assertIn("arrivals", result.metrics)
        keys = list(result.grand_total[["model", "scale"]].itertuples(index=False, name=None))
        self.assertEqual(keys, sorted(keys))

        for metric in result.metrics:
            for row in result.eval_seed.to_dict("records"):
                values = raw.loc[
                    raw.model.eq(row["model"]) & raw.scale.eq(row["scale"])
                    & raw.eval_year.eq(row["eval_year"])
                    & raw.training_seed.eq(row["training_seed"]), metric,
                ].to_numpy()
                self.assertAlmostEqual(row[f"{metric}_mean"], np.mean(values))
                self.assertAlmostEqual(row[f"{metric}_std_eval"], np.std(values, ddof=1))
                self.assertEqual(row[f"{metric}_n_valid"], 3)
            for row in result.training_seed.to_dict("records"):
                values = result.eval_seed.loc[
                    result.eval_seed.model.eq(row["model"])
                    & result.eval_seed.scale.eq(row["scale"])
                    & result.eval_seed.eval_year.eq(row["eval_year"]), f"{metric}_mean",
                ].to_numpy()
                self.assertAlmostEqual(row[f"{metric}_mean"], np.mean(values))
                self.assertAlmostEqual(row[f"{metric}_std_train"], np.std(values, ddof=1))
                self.assertEqual(row[f"{metric}_n_valid"], 3)
            for row in result.year.to_dict("records"):
                values = result.training_seed.loc[
                    result.training_seed.model.eq(row["model"])
                    & result.training_seed.scale.eq(row["scale"]), f"{metric}_mean",
                ].to_numpy()
                direct = raw.loc[raw.model.eq(row["model"]) & raw.scale.eq(row["scale"]), metric]
                self.assertAlmostEqual(row[f"{metric}_mean"], np.mean(values))
                self.assertAlmostEqual(row[f"{metric}_mean"], direct.mean())
                self.assertAlmostEqual(row[f"{metric}_std_year"], np.std(values, ddof=1))
                self.assertEqual(row[f"{metric}_n_valid"], 3)
            for row in result.grand_total.to_dict("records"):
                direct = raw.loc[raw.model.eq(row["model"]) & raw.scale.eq(row["scale"]), metric]
                self.assertAlmostEqual(row[f"{metric}_mean"], direct.mean())
                self.assertAlmostEqual(row[f"{metric}_std_total"], np.std(direct, ddof=1))
                self.assertEqual(row[f"{metric}_n_valid"], 27)

    def test_percentages_are_averaged_per_run(self):
        raw = balanced_frame(scales=(1.0,))
        raw["cost_baseline"] = np.arange(1, 28, dtype=float) ** 2
        raw["cost_agent"] = np.arange(1, 28, dtype=float)
        raw["savings_vs_baseline_pct"] = 100 * (1 - raw.cost_agent / raw.cost_baseline)
        row = aggregate_multi_run(raw).grand_total.iloc[0]
        self.assertAlmostEqual(row.savings_vs_baseline_pct_mean, raw.savings_vs_baseline_pct.mean())
        ratio_of_means = 100 * (1 - row.cost_agent_mean / row.cost_baseline_mean)
        self.assertGreater(abs(row.savings_vs_baseline_pct_mean - ratio_of_means), 1)

    def test_zero_spread_and_variation_at_each_level(self):
        for varying_key, active_sd in (
            (None, None), ("training_seed", "train"),
            ("eval_year", "year"), ("eval_seed", "eval"),
        ):
            with self.subTest(varying_key=varying_key):
                raw = balanced_frame(scales=(1.0,))
                raw["cost_agent"] = 10.0
                if varying_key:
                    values = sorted(raw[varying_key].unique())
                    raw["cost_agent"] += raw[varying_key].map(dict(zip(values, (-1.0, 0.0, 1.0))))
                result = aggregate_multi_run(raw)
                for level, frame in (
                    ("eval", result.eval_seed), ("train", result.training_seed), ("year", result.year),
                ):
                    np.testing.assert_allclose(frame[f"cost_agent_std_{level}"], 1 if level == active_sd else 0)
                row = result.grand_total.iloc[0]
                self.assertAlmostEqual(row.cost_agent_mean, 10)
                self.assertAlmostEqual(row.cost_agent_std_total, np.sqrt(18 / 26) if varying_key else 0)

    def test_direct_api_rejects_an_incomplete_grid(self):
        with self.assertRaisesRegex(ValueError, "(?i)missing|incomplete|27|grid"):
            aggregate_multi_run(balanced_frame().iloc[1:])

    def test_seed_ids_can_differ_between_trained_models_and_years(self):
        raw = balanced_frame()
        raw["training_seed"] += 100 * (raw.eval_year - 2023)
        raw["eval_seed"] += 10000 * raw.training_seed
        result = aggregate_multi_run(raw)
        self.assertEqual(len(result.eval_seed), 18)
        self.assertEqual(len(result.grand_total), 2)

    def test_wholly_unavailable_metrics_remain_missing_with_zero_counts(self):
        raw = balanced_frame()
        raw["oracle_contiguous_eur"] = np.nan
        raw["cost_agent"] = np.nan
        result = aggregate_multi_run(raw)
        for frame, level in ((result.eval_seed, "eval"), (result.training_seed, "train"),
                             (result.year, "year"), (result.grand_total, "total")):
            for metric in ("oracle_contiguous_eur", "cost_agent"):
                self.assertTrue(frame[f"{metric}_mean"].isna().all())
                self.assertTrue(frame[f"{metric}_std_{level}"].isna().all())
                self.assertTrue(frame[f"{metric}_n_valid"].eq(0).all())

    def test_partial_optional_metrics_skip_blanks(self):
        raw = balanced_frame()
        raw["oracle_contiguous_eur"] = 10.0
        raw.loc[0, "oracle_contiguous_eur"] = np.nan
        result = aggregate_multi_run(raw)
        np.testing.assert_allclose(result.grand_total.oracle_contiguous_eur_mean, 10)
        np.testing.assert_allclose(result.grand_total.oracle_contiguous_eur_std_total, 0)
        self.assertEqual(result.grand_total.oracle_contiguous_eur_n_valid.tolist(), [26, 27])

    def test_unbalanced_blanks_keep_equal_weight_hierarchy_and_valid_denominator(self):
        raw = balanced_frame(scales=(1.0,))
        # First year has two available training-cell means (0, 30), second
        # year has one (100), and the third year has no available values.
        raw["cost_agent"] = [0, np.nan, np.nan, 30, 30, 30, np.nan, np.nan, np.nan,
                             100, 100, np.nan, *([np.nan] * 15)]
        result = aggregate_multi_run(raw)
        cells = result.eval_seed.set_index(["eval_year", "training_seed"])
        self.assertEqual(cells.loc[(2023, 4), "cost_agent_n_valid"], 1)
        self.assertTrue(np.isnan(cells.loc[(2023, 4), "cost_agent_std_eval"]))
        self.assertEqual(cells.loc[(2023, 29), "cost_agent_n_valid"], 0)
        self.assertTrue(np.isnan(cells.loc[(2023, 29), "cost_agent_mean"]))
        self.assertEqual(cells.loc[(2024, 4), "cost_agent_n_valid"], 2)
        self.assertEqual(cells.loc[(2024, 4), "cost_agent_std_eval"], 0)

        years = result.training_seed.set_index("eval_year")
        self.assertAlmostEqual(years.loc[2023, "cost_agent_mean"], 15)
        self.assertAlmostEqual(years.loc[2023, "cost_agent_std_train"], np.std([0, 30], ddof=1))
        self.assertEqual(years.loc[2023, "cost_agent_n_valid"], 2)
        self.assertAlmostEqual(years.loc[2024, "cost_agent_mean"], 100)
        self.assertTrue(np.isnan(years.loc[2024, "cost_agent_std_train"]))
        self.assertEqual(years.loc[2024, "cost_agent_n_valid"], 1)
        self.assertTrue(np.isnan(years.loc[2025, "cost_agent_mean"]))
        self.assertEqual(years.loc[2025, "cost_agent_n_valid"], 0)

        grand_mean = (15 + 100) / 2
        self.assertAlmostEqual(result.year.iloc[0].cost_agent_mean, grand_mean)
        self.assertAlmostEqual(result.year.iloc[0].cost_agent_std_year, np.std([15, 100], ddof=1))
        self.assertEqual(result.year.iloc[0].cost_agent_n_valid, 2)
        row = result.grand_total.iloc[0]
        observed = np.array([0, 30, 30, 30, 100, 100], dtype=float)
        self.assertAlmostEqual(row.cost_agent_mean, grand_mean)
        self.assertNotAlmostEqual(row.cost_agent_mean, observed.mean())
        self.assertEqual(row.cost_agent_n_valid, 6)
        self.assertAlmostEqual(row.cost_agent_std_total,
                               np.sqrt(np.sum((observed - grand_mean) ** 2) / (len(observed) - 1)))
        self.assertNotAlmostEqual(row.cost_agent_std_total, np.std(observed, ddof=1))
        self.assertEqual(row.arrivals_n_valid, 27)
        self.assertAlmostEqual(row.arrivals_mean, raw.arrivals.mean())

    def test_one_available_value_has_mean_but_no_sample_standard_deviation(self):
        raw = balanced_frame(scales=(1.0,))
        raw["cost_agent"] = np.nan
        raw.loc[0, "cost_agent"] = 42.0
        result = aggregate_multi_run(raw)
        for frame, level in ((result.eval_seed, "eval"), (result.training_seed, "train"),
                             (result.year, "year"), (result.grand_total, "total")):
            self.assertEqual(frame.cost_agent_mean.dropna().tolist(), [42])
            self.assertTrue(frame[f"cost_agent_std_{level}"].isna().all())
            self.assertEqual(frame.cost_agent_n_valid.sum(), 1)


class MultiRunLoadingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.inputs = write_grid(self.root)
        self.path = sorted(self.inputs[0].glob("*/eval_main.csv"))[0]

    def edit_first(self, edit):
        frame = pd.read_csv(self.path)
        edit(frame).to_csv(self.path, index=False)

    def test_parse_uses_arbitrary_seed_identifiers(self):
        self.assertEqual(parse_run_directory(Path("v2_e2p2s6/s29_pr24_seed701_jobs23")), (2024, 29, 701))
        self.assertEqual(parse_run_directory(Path("e4p4s2/s11_pr25_seed55_jobs26")), (2025, 11, 55))

    def test_parse_rejects_a_malformed_run_directory(self):
        with self.assertRaises(ValueError):
            parse_run_directory(Path("e4p4s2/s11_pr25_jobs26"))

    def test_loading_aligns_by_scale_and_keeps_models_separate(self):
        second = write_grid(self.root, models=("e4p4s2",))
        self.edit_first(lambda frame: frame.iloc[::-1])
        raw = load_multi_run(self.inputs + second)
        self.assertEqual(len(raw), 108)
        self.assertEqual(set(raw.model), {"e2p2s6", "e4p4s2"})
        self.assertEqual(set(raw.training_seed), set(TRAINING_SEEDS))
        self.assertEqual(set(raw.eval_seed), set(EVALUATION_SEEDS))
        self.assertEqual(set(raw.eval_year), set(YEARS))
        self.assertGreater(raw.loc[raw.scale.eq(1.0), "arrivals"].nunique(), 3)
        self.assertEqual(len(aggregate_multi_run(raw).grand_total), 4)

    def test_missing_evaluation_file(self):
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "(?i)missing.*eval_main|eval_main.*missing"):
            load_multi_run(self.inputs)

    def test_missing_evaluation_seed_combination(self):
        shutil.rmtree(self.path.parent)
        with self.assertRaisesRegex(ValueError, "(?i)missing|incomplete|27|grid"):
            load_multi_run(self.inputs)

    def test_missing_training_seed(self):
        for path in self.inputs[0].glob("s29_pr25_*"):
            shutil.rmtree(path)
        with self.assertRaisesRegex(ValueError, "(?i)missing|training|grid"):
            load_multi_run(self.inputs)

    def test_duplicate_job_suffix_is_not_an_additional_measurement(self):
        duplicate = self.path.parent.with_name(self.path.parent.name.replace("jobs23", "jobs26"))
        shutil.copytree(self.path.parent, duplicate)
        with self.assertRaisesRegex(ValueError, "(?i)duplicate"):
            load_multi_run(self.inputs)

    def test_duplicate_scale_in_one_file(self):
        self.edit_first(lambda frame: pd.concat([frame, frame.iloc[:1]], ignore_index=True))
        with self.assertRaisesRegex(ValueError, "(?i)duplicate"):
            load_multi_run(self.inputs)

    def test_missing_workload_scale(self):
        self.edit_first(lambda frame: frame.iloc[:1])
        with self.assertRaisesRegex(ValueError, "(?i)scale|missing|incomplete"):
            load_multi_run(self.inputs)

    def test_conflicting_model_identifier(self):
        self.edit_first(lambda frame: frame.assign(model="e4p4s2"))
        with self.assertRaisesRegex(ValueError, "(?i)model"):
            load_multi_run(self.inputs)

    def test_inconsistent_schema(self):
        self.edit_first(lambda frame: frame.drop(columns="cost_agent"))
        with self.assertRaisesRegex(ValueError, "(?i)schema|column"):
            load_multi_run(self.inputs)

    def test_blank_metrics_load_without_dropping_run_rows(self):
        original = self.path.read_bytes()
        for value in (np.nan, "", " \t "):
            with self.subTest(value=value):
                self.path.write_bytes(original)
                self.edit_first(lambda frame: frame.assign(cost_agent=value))
                raw = load_multi_run(self.inputs)
                self.assertEqual(len(raw), 54)
                self.assertEqual(raw.cost_agent.isna().sum(), 2)
                result = aggregate_multi_run(raw)
                self.assertEqual(result.grand_total.cost_agent_n_valid.tolist(), [26, 26])
                self.assertEqual(result.grand_total.arrivals_n_valid.tolist(), [27, 27])

    def test_nonfinite_and_malformed_metrics_are_not_silently_skipped(self):
        original = self.path.read_bytes()
        for value in (np.inf, -np.inf, "not-a-number"):
            with self.subTest(value=value):
                self.path.write_bytes(original)
                self.edit_first(lambda frame: frame.assign(cost_agent=value))
                with self.assertRaisesRegex(ValueError, "(?i)cost_agent|finite|numeric|missing"):
                    load_multi_run(self.inputs)

    def test_blank_scale_still_invalidates_the_experimental_grid(self):
        self.edit_first(lambda frame: frame.assign(scale=np.nan))
        with self.assertRaisesRegex(ValueError, "(?i)scale|finite|missing"):
            load_multi_run(self.inputs)

    def test_exports_retain_precision_and_do_not_modify_sources(self):
        before = {path: path.read_bytes() for path in self.inputs[0].glob("*/eval_main.csv")}
        result = aggregate_multi_run(load_multi_run(self.inputs))
        destination = self.root / "output"
        export_multi_run(result, destination)
        csvs = list(destination.rglob("*.csv"))
        texs = list(destination.rglob("*.tex"))
        self.assertGreaterEqual(len(csvs), 4)
        self.assertGreaterEqual(len(texs), 4)
        self.assertTrue(any("std_total" in path.read_text() for path in csvs))
        self.assertTrue(any("\\begin{tabular}" in path.read_text() for path in texs))
        self.assertTrue(any("cost\\_agent" in path.read_text() for path in texs))
        self.assertTrue(any("Valid n" in path.read_text() for path in texs))
        grand = next(path for path in csvs if "std_total" in path.read_text())
        np.testing.assert_allclose(pd.read_csv(grand).cost_agent_mean, result.grand_total.cost_agent_mean, rtol=1e-14)
        self.assertEqual(pd.read_csv(grand).cost_agent_n_valid.tolist(), [27, 27])
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
