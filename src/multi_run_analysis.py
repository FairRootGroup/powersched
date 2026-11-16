"""Balanced evaluation-grid aggregation for the PowerSched meta-analysis.

Each model/scale has three evaluation years, three training seeds and three
evaluation seeds. Intermediate SDs describe their respective levels; the final
spread describes available raw observations around the hierarchical grand mean,
without standard-error scaling or quadrature. Blank metrics are skipped
independently, and each table records the number of valid inputs per metric.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EVALUATION_YEARS = (2023, 2024, 2025)
RUN_DIRECTORY = re.compile(
    r"s(?P<training_seed>\d+)_pr(?P<year>\d{2}|\d{4})_"
    r"seed(?P<eval_seed>\d+)_jobs(?P<jobs>[^/]+)"
)
GROUP_KEYS = ["model", "scale"]
RUN_KEYS = ["eval_year", "training_seed", "eval_seed"]
IDENTIFIER_COLUMNS = {*GROUP_KEYS, *RUN_KEYS, "source_csv"}


@dataclass
class MultiRunAggregation:
    """Full-precision tables, with distinct names for every SD level."""

    eval_seed: pd.DataFrame
    training_seed: pd.DataFrame
    year: pd.DataFrame
    grand_total: pd.DataFrame
    metrics: list[str]
    raw: pd.DataFrame


def parse_run_directory(path: Path | str) -> tuple[int, int, int]:
    """Return (evaluation year, training seed, evaluation seed), ignoring jobs."""
    name = Path(path).name
    match = RUN_DIRECTORY.fullmatch(name)
    if match is None:
        raise ValueError(
            f"Invalid evaluation directory {name!r}; expected "
            "s<train>_pr<year>_seed<eval>_jobs<workload>"
        )
    year = int(match["year"])
    if len(match["year"]) == 2:
        year += 2000
    return year, int(match["training_seed"]), int(match["eval_seed"])


def _run_directories(path: Path) -> list[Path]:
    return sorted(
        child for child in path.iterdir()
        if child.is_dir() and RUN_DIRECTORY.fullmatch(child.name)
    )


def _model_directories(input_dirs: list[Path]) -> list[Path]:
    """Accept explicit model directories or their common analysis directory."""
    models = []
    for supplied in input_dirs:
        path = Path(supplied).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"Input directory does not exist: {path}")
        if _run_directories(path):
            models.append(path)
            continue
        children = [
            child for child in sorted(path.iterdir())
            if child.is_dir() and _run_directories(child)
        ]
        if not children:
            raise ValueError(f"No evaluation run directories found under {path}")
        models.extend(children)
    if not models:
        raise ValueError("No model input directories supplied")
    # Repeated/overlapping inputs are an error, rather than silently deduplicated.
    if len(set(models)) != len(models):
        raise ValueError("Duplicate model directories in --input-dirs")
    return sorted(models)


def load_multi_run(input_dirs: list[Path]) -> pd.DataFrame:
    """Discover and validate original eval_main.csv files without modifying them."""
    frames = []
    schema = None
    schema_source = None
    seen_runs = {}
    for model_dir in _model_directories(input_dirs):
        # v2_ is an experiment-version prefix, not part of CSV model identifiers.
        expected_model = re.sub(r"^v\d+_", "", model_dir.name)
        for run_dir in _run_directories(model_dir):
            year, training_seed, eval_seed = parse_run_directory(run_dir)
            run_key = (expected_model, year, training_seed, eval_seed)
            if run_key in seen_runs:
                raise ValueError(
                    f"Duplicate model/year/training-seed/evaluation-seed {run_key}: "
                    f"{seen_runs[run_key]} and {run_dir} (jobs suffix is ignored)"
                )
            seen_runs[run_key] = run_dir
            source = run_dir / "eval_main.csv"
            if not source.is_file():
                raise ValueError(f"Missing evaluation file for {run_key}: {source}")
            try:
                with source.open(newline="", encoding="utf-8-sig") as stream:
                    header = next(csv.reader(stream), [])
                if not header or len(set(header)) != len(header):
                    raise ValueError("empty or duplicate CSV column labels")
                if schema is None:
                    schema, schema_source = set(header), source
                elif set(header) != schema:
                    raise ValueError(
                        f"Inconsistent CSV schema relative to {schema_source}; "
                        f"missing={sorted(schema - set(header))}, "
                        f"extra={sorted(set(header) - schema)}"
                    )
                required = {"model", "scale", "arrivals"}
                if not required.issubset(header):
                    raise ValueError(f"missing required columns: {sorted(required - set(header))}")
                reserved = set(header) & (set(RUN_KEYS) | {"source_csv"})
                if reserved:
                    raise ValueError(f"reserved metadata columns in input CSV: {sorted(reserved)}")
                frame = pd.read_csv(source, dtype={"model": "string"})
                if frame.empty:
                    raise ValueError("evaluation CSV has no workload-scale rows")
                frame["model"] = frame["model"].str.strip()
                if frame["model"].isna().any() or not frame["model"].eq(expected_model).all():
                    raise ValueError(
                        f"Conflicting model identifiers: directory expects {expected_model!r}, "
                        f"CSV contains {frame['model'].unique().tolist()}"
                    )
                for column in frame.columns.difference(["model"]):
                    try:
                        frame[column] = pd.to_numeric(
                            frame[column].replace(r"^\s*$", np.nan, regex=True),
                            errors="raise",
                        )
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"Non-numeric metric {column!r}: {exc}") from exc
                frame["eval_year"] = year
                frame["training_seed"] = training_seed
                frame["eval_seed"] = eval_seed
                frame["source_csv"] = str(source)
                frames.append(frame)
            except (OSError, ValueError) as exc:
                raise ValueError(f"{source}: {exc}") from exc
    raw = pd.concat(frames, ignore_index=True)
    validate_multi_run(raw)
    return raw.sort_values(GROUP_KEYS + RUN_KEYS).reset_index(drop=True)


def _describe_rows(rows: pd.DataFrame, limit: int = 6) -> str:
    columns = [*GROUP_KEYS, *RUN_KEYS]
    if "source_csv" in rows:
        columns.append("source_csv")
    return rows[columns].head(limit).to_string(index=False)


def validate_multi_run(raw: pd.DataFrame) -> list[str]:
    """Reject incomplete grids and invalid values; allow blank metric values."""
    required = {*GROUP_KEYS, *RUN_KEYS}
    if raw.empty or not required.issubset(raw.columns):
        raise ValueError(f"Empty data or missing aggregation identifiers: {sorted(required - set(raw.columns))}")
    if raw.columns.duplicated().any():
        raise ValueError("Duplicate CSV column labels")
    if raw["model"].isna().any() or raw["model"].astype(str).str.strip().eq("").any():
        raise ValueError("Missing model identifiers")
    for key in ["scale", *RUN_KEYS]:
        values = pd.to_numeric(raw[key], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Missing/non-finite aggregation identifier {key}")
        if key in RUN_KEYS and not (values == np.floor(values)).all():
            raise ValueError(f"Non-integer run identifier {key}")
    duplicate = raw.duplicated(GROUP_KEYS + RUN_KEYS, keep=False)
    if duplicate.any():
        raise ValueError(f"Duplicate evaluation combinations:\n{_describe_rows(raw[duplicate])}")

    scales = set(raw["scale"].unique())
    for model, model_rows in raw.groupby("model", sort=True):
        years = set(model_rows["eval_year"].unique())
        issues = []
        if years != set(EVALUATION_YEARS):
            issues.append(
                f"missing years={sorted(set(EVALUATION_YEARS) - years)}, "
                f"unexpected years={sorted(years - set(EVALUATION_YEARS))}"
            )
        missing_scales = scales - set(model_rows["scale"].unique())
        if missing_scales:
            issues.append(f"missing workload scales={sorted(missing_scales)}")
        if issues:
            raise ValueError(f"Incomplete evaluation grid for model={model}: {'; '.join(issues)}")
        expected = set()
        for year, year_rows in model_rows.groupby("eval_year", sort=True):
            training = sorted(year_rows["training_seed"].unique())
            if len(training) != 3:
                raise ValueError(
                    f"Incomplete evaluation grid for model={model}, year={year}: "
                    f"expected 3 training seeds; found {training}"
                )
            for training_seed, seed_rows in year_rows.groupby("training_seed", sort=True):
                evaluation = sorted(seed_rows["eval_seed"].unique())
                if len(evaluation) != 3:
                    raise ValueError(
                        f"Incomplete evaluation grid for model={model}, year={year}, "
                        f"training_seed={training_seed}: expected 3 evaluation seeds; "
                        f"found {evaluation}"
                    )
                expected.update((year, training_seed, seed) for seed in evaluation)
        for scale, rows in model_rows.groupby("scale", sort=True):
            actual = set(rows[RUN_KEYS].itertuples(index=False, name=None))
            if actual != expected:
                raise ValueError(
                    f"Incomplete evaluation grid for model={model}, scale={scale}: "
                    f"expected 27 results, found {len(rows)}; "
                    f"missing (year, training_seed, eval_seed)={sorted(expected - actual)}, "
                    f"unexpected={sorted(actual - expected)}"
                )

    metrics = sorted(set(raw.columns) - IDENTIFIER_COLUMNS)
    if not metrics:
        raise ValueError("No numerical metrics found")
    for metric in metrics:
        if not pd.api.types.is_numeric_dtype(raw[metric]):
            raise ValueError(f"Non-numeric metric {metric!r}")
        infinite = np.isinf(raw[metric].to_numpy(dtype=float))
        if infinite.any():
            raise ValueError(f"Non-finite metric {metric!r}:\n{_describe_rows(raw[infinite])}")
    return metrics


def _aggregate_level(
    frame: pd.DataFrame, keys: list[str], metrics: list[str], sd_label: str,
    *, input_suffix: str = "",
) -> pd.DataFrame:
    columns = [metric + input_suffix for metric in metrics]
    grouped = frame.groupby(keys, sort=True, observed=True)
    means = grouped[columns].mean().rename(
        columns={column: f"{metric}_mean" for column, metric in zip(columns, metrics)}
    )
    stds = grouped[columns].std(ddof=1).rename(
        columns={column: f"{metric}_std_{sd_label}" for column, metric in zip(columns, metrics)}
    )
    counts = grouped[columns].count().rename(
        columns={column: f"{metric}_n_valid" for column, metric in zip(columns, metrics)}
    )
    result = pd.concat([means, stds, counts, grouped.size().rename("n_observations")], axis=1)
    return result.copy().reset_index()


def aggregate_multi_run(raw: pd.DataFrame) -> MultiRunAggregation:
    """Apply evaluation-seed, training-seed, year, then observed-spread levels."""
    metrics = validate_multi_run(raw)
    eval_seed = _aggregate_level(raw, GROUP_KEYS + ["eval_year", "training_seed"], metrics, "eval")
    training_seed = _aggregate_level(
        eval_seed, GROUP_KEYS + ["eval_year"], metrics, "train", input_suffix="_mean"
    )
    year = _aggregate_level(training_seed, GROUP_KEYS, metrics, "year", input_suffix="_mean")
    total = _aggregate_level(raw, GROUP_KEYS, metrics, "total")
    mean_columns = [metric + "_mean" for metric in metrics]
    # Equal-weight hierarchy and direct means agree only for fully observed
    # metrics. Missing values never remove a run's other metric observations.
    complete = total[[metric + "_n_valid" for metric in metrics]].to_numpy() == 27
    if not np.allclose(
        year[mean_columns].to_numpy()[complete], total[mean_columns].to_numpy()[complete],
        rtol=1e-12, atol=1e-12, equal_nan=True,
    ):
        raise ValueError("Hierarchical grand means disagree with direct means of the balanced grid")
    # The reported center is explicitly Level 3, while spread uses all raw runs.
    total[mean_columns] = year[mean_columns]
    # Preserve the reporting convention's center for partially observed metrics,
    # using their actual n_valid - 1 denominator. With no blanks this is the
    # ordinary sample SD of all 27 runs; fewer than two values leave SD undefined.
    centers = raw[GROUP_KEYS].merge(
        year[GROUP_KEYS + mean_columns], on=GROUP_KEYS, how="left",
        sort=False, validate="many_to_one",
    )
    squared = raw[GROUP_KEYS].reset_index(drop=True).copy()
    squared[metrics] = (
        raw[metrics].to_numpy(dtype=float) - centers[mean_columns].to_numpy(dtype=float)
    ) ** 2
    squared_sums = squared.groupby(GROUP_KEYS, sort=True, observed=True)[metrics].sum(min_count=1).reset_index()
    for metric in metrics:
        counts = total[f"{metric}_n_valid"]
        total[f"{metric}_std_total"] = np.sqrt(
            squared_sums[metric] / (counts - 1).where(counts > 1)
        )
    return MultiRunAggregation(
        eval_seed=eval_seed, training_seed=training_seed, year=year,
        grand_total=total, metrics=metrics,
        raw=raw.sort_values(GROUP_KEYS + RUN_KEYS).reset_index(drop=True).copy(),
    )


def _latex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def export_multi_run(result: MultiRunAggregation, out_dir: Path) -> Path:
    """Write full-precision CSVs and one narrow LaTeX table per metric/level."""
    table_dir = Path(out_dir) / "multi_run_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    result.raw.to_csv(table_dir / "raw_measurements.csv", index=False)
    levels = [
        ("evaluation_seed", result.eval_seed, GROUP_KEYS + ["eval_year", "training_seed"], "eval"),
        ("training_seed", result.training_seed, GROUP_KEYS + ["eval_year"], "train"),
        ("year", result.year, GROUP_KEYS, "year"),
        ("grand_total", result.grand_total, GROUP_KEYS, "total"),
    ]
    identifiers = {"model": "Model", "scale": "Scale", "eval_year": "Year", "training_seed": "Training seed"}
    safe_names = {metric: re.sub(r"[^A-Za-z0-9_.-]", "_", metric) for metric in result.metrics}
    if len(set(safe_names.values())) != len(safe_names):
        raise ValueError("Metric names collide when converted to LaTeX output filenames")
    for level, frame, keys, sd_label in levels:
        frame.to_csv(table_dir / f"{level}.csv", index=False)
        for metric in result.metrics:
            columns = keys + [f"{metric}_mean", f"{metric}_std_{sd_label}", f"{metric}_n_valid"]
            headers = [identifiers[key] for key in keys] + ["Mean", f"SD ({sd_label})", "Valid n"]
            lines = [
                "% Metric: " + _latex_escape(metric),
                "% Blanks are skipped; -- means unavailable (SD requires at least two values).",
                "% SD uses n_valid - 1; total spread is centered on the hierarchical grand mean.",
                r"\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}",
                r"\hline",
                r"\multicolumn{" + str(len(columns)) + r"}{l}{\texttt{" + _latex_escape(metric) + r"}} \\",
                " & ".join(_latex_escape(header) for header in headers) + r" \\",
                r"\hline",
            ]
            for values in frame[columns].itertuples(index=False, name=None):
                formatted = []
                for key, value in zip(columns, values):
                    if key == "model":
                        formatted.append(_latex_escape(value))
                    elif pd.isna(value):
                        formatted.append("--")
                    else:
                        formatted.append(f"{value:.6g}")
                lines.append(" & ".join(formatted) + r" \\")
            lines.extend([r"\hline", r"\end{tabular}", ""])
            (table_dir / f"{level}_{safe_names[metric]}.tex").write_text("\n".join(lines), encoding="utf-8")
    return table_dir
