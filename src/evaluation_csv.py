"""Reference-compatible CSV reporting for evaluation sweep scripts."""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.config import EPISODE_HOURS


EVALUATION_CSV_METRIC_FIELDS = (
    "cost_agent",
    "cost_baseline",
    "cost_baseline_off",
    "savings_vs_baseline_pct",
    "savings_vs_baseline_off_pct",
    "cost_per_1k_agent",
    "cost_per_1k_baseline",
    "cost_per_1k_baseline_off",
    "completion_pct",
    "mean_wait_h",
    "dropped_jobs",
    "dropped_per_eur",
    "power_agent_mwh",
    "power_baseline_mwh",
    "power_baseline_off_mwh",
    "eur_per_mwh_agent",
    "eur_per_mwh_baseline",
    "eur_per_mwh_baseline_off",
    "oracle_liquid_eur",
    "oracle_contiguous_eur",
    "oracle_capture_pct",
    "savings_abs_vs_baseline_eur",
    "savings_abs_vs_baseline_off_eur",
    "delta_power_mwh",
    "savings_model_eur",
    "delta_eur_per_mwh",
    "cost_per_1k_delta",
    "price_savings_eur",
    "price_savings_pct",
    "power_savings_pct",
)

_NUMBER = r"-?[\d,]+(?:\.\d+)?"
_EPISODE_LINE_RE = re.compile(r"^\s*Episode\s+(?P<episode>\d+):")
_JOBS_RE = re.compile(
    rf"\bJobs=(?P<completed>{_NUMBER})/(?P<submitted>{_NUMBER})"
)
_WAIT_RE = re.compile(rf"\bAvgWait=(?P<value>{_NUMBER})h")
_DROPPED_RE = re.compile(rf"\b(?:Dropped|Lost)=(?P<value>{_NUMBER})")
_PROP_POWER_RE = re.compile(
    rf"\bPropPower=(?P<agent>{_NUMBER})/"
    rf"(?P<baseline>{_NUMBER})/"
    rf"(?P<baseline_off>{_NUMBER})\s*MWh"
)
_PROP_COST_RE = re.compile(
    rf"\bPropCost=€(?P<agent>{_NUMBER})/"
    rf"€(?P<baseline>{_NUMBER})/"
    rf"€(?P<baseline_off>{_NUMBER})\b"
)
_COST_PER_1K_RE = re.compile(
    rf"\bCostPer1kCompleted=(?:{_NUMBER}|n/a)/"
    rf"(?P<baseline>{_NUMBER}|n/a)/"
    rf"(?:{_NUMBER}|n/a)\s*€/1k"
)
_ORACLE_LIQUID_RE = re.compile(rf"\bOracleLiq=€(?P<value>{_NUMBER})")
_ORACLE_CONTIGUOUS_RE = re.compile(rf"\bOracleCon=€(?P<value>{_NUMBER})")

_JOB_PROCESSING_TOTALS_RE = re.compile(
    rf"=== JOB PROCESSING METRICS ===.*?"
    rf"\bAgent:\s*"
    rf"Jobs Launched:\s*(?P<agent_jobs_launched>{_NUMBER})\s*/\s*{_NUMBER}.*?"
    rf"Jobs Completed:\s*(?P<agent_jobs_completed>{_NUMBER})\s*/\s*"
    rf"(?P<agent_jobs_submitted>{_NUMBER})\s*\([^)]*\).*?"
    rf"Average Wait Time:\s*(?P<agent_mean_wait>{_NUMBER})\s*hours.*?"
    rf"\bBaseline:\s*"
    rf"Jobs Launched:\s*{_NUMBER}\s*/\s*{_NUMBER}.*?"
    rf"Jobs Completed:\s*(?P<baseline_jobs_completed>{_NUMBER})\s*/",
    re.DOTALL,
)
_AGGREGATE_COST_TOTALS_RE = re.compile(
    rf"=== JOB PROCESSING METRICS ===.*?"
    rf"\bAgent:.*?"
    rf"^\s*Total Cost:\s*€(?P<agent>{_NUMBER})\s*$.*?"
    rf"\bBaseline:.*?"
    rf"^\s*Baseline Total Cost:\s*€(?P<baseline>{_NUMBER})\s*$.*?"
    rf"^\s*Baseline_off Total Cost:\s*€(?P<baseline_off>{_NUMBER})\s*$",
    re.DOTALL | re.MULTILINE,
)
_AGGREGATE_POWER_TOTALS_RE = re.compile(
    rf"=== POWER & PRICE METRICS \(TOTAL OVER EVALUATION\) ===.*?"
    rf"^\s*Agent:\s*Power=(?P<agent>{_NUMBER})\s*MWh.*?"
    rf"^\s*Baseline:\s*Power=(?P<baseline>{_NUMBER})\s*MWh.*?"
    rf"^\s*Baseline_off:\s*Power=(?P<baseline_off>{_NUMBER})\s*MWh",
    re.DOTALL | re.MULTILINE,
)
_ORACLE_LIQUID_TOTAL_RE = re.compile(
    rf"\bLiquid Oracle\b.*?^\s*Total Cost:\s*€(?P<value>{_NUMBER})\s*$",
    re.DOTALL | re.MULTILINE,
)
_ORACLE_CONTIGUOUS_TOTAL_RE = re.compile(
    rf"\bContiguous Oracle\b.*?^\s*Total Cost:\s*€(?P<value>{_NUMBER})\s*$",
    re.DOTALL | re.MULTILINE,
)


def evaluation_csv_fieldnames(sweep_parameter: str) -> tuple[str, ...]:
    """Return the established evaluation columns for a scale or lambda sweep."""

    if sweep_parameter not in {"scale", "lambda"}:
        raise ValueError("sweep_parameter must be 'scale' or 'lambda'")
    return ("model", "arrivals", sweep_parameter, *EVALUATION_CSV_METRIC_FIELDS)


def _number(raw: str) -> float:
    return float(raw.replace(",", ""))


def _require(
    pattern: re.Pattern[str],
    line: str,
    field_description: str,
) -> re.Match[str]:
    match = pattern.search(line)
    if match is None:
        episode = _EPISODE_LINE_RE.search(line)
        episode_number = episode.group("episode") if episode else "unknown"
        raise RuntimeError(
            f"Episode {episode_number} summary is missing {field_description}; "
            "cannot build the evaluation CSV."
        )
    return match


def _parse_episode_lines(stdout: str) -> list[dict[str, float | int]]:
    episodes: list[dict[str, float | int]] = []
    for line in stdout.splitlines():
        if _EPISODE_LINE_RE.search(line) is None:
            continue

        jobs = _require(_JOBS_RE, line, "Jobs")
        wait = _require(_WAIT_RE, line, "AvgWait")
        dropped = _require(_DROPPED_RE, line, "Dropped/Lost")
        power = _require(_PROP_POWER_RE, line, "PropPower")
        cost = _require(_PROP_COST_RE, line, "PropCost")
        cost_per_1k = _require(_COST_PER_1K_RE, line, "CostPer1kCompleted")
        oracle_liquid = _ORACLE_LIQUID_RE.search(line)
        oracle_contiguous = _ORACLE_CONTIGUOUS_RE.search(line)

        jobs_submitted = int(_number(jobs.group("submitted")))
        baseline_cost = _number(cost.group("baseline"))
        baseline_cost_per_1k_raw = cost_per_1k.group("baseline")
        if baseline_cost_per_1k_raw == "n/a":
            baseline_jobs_completed_estimate = jobs_submitted
        else:
            baseline_cost_per_1k = _number(baseline_cost_per_1k_raw)
            baseline_jobs_completed_estimate = (
                int(round(baseline_cost * 1000.0 / baseline_cost_per_1k))
                if baseline_cost_per_1k > 0.0
                else jobs_submitted
            )

        episodes.append(
            {
                "agent_prop_cost": _number(cost.group("agent")),
                "baseline_prop_cost": baseline_cost,
                "baseline_off_prop_cost": _number(cost.group("baseline_off")),
                "agent_prop_power_mwh": _number(power.group("agent")),
                "baseline_prop_power_mwh": _number(power.group("baseline")),
                "baseline_off_prop_power_mwh": _number(
                    power.group("baseline_off")
                ),
                "jobs_submitted": jobs_submitted,
                "jobs_completed": int(_number(jobs.group("completed"))),
                "baseline_jobs_completed_estimate": (
                    baseline_jobs_completed_estimate
                ),
                "avg_wait_time": _number(wait.group("value")),
                "jobs_lost_total": int(_number(dropped.group("value"))),
                "oracle_cost": (
                    _number(oracle_liquid.group("value"))
                    if oracle_liquid is not None
                    else 0.0
                ),
                "oracle_contiguous_cost": (
                    _number(oracle_contiguous.group("value"))
                    if oracle_contiguous is not None
                    else 0.0
                ),
            }
        )

    if not episodes:
        raise RuntimeError(
            "Could not parse evaluation CSV metrics from train.py output; "
            "no complete episode summaries were found."
        )
    return episodes


def _sum_metric(
    episodes: Sequence[Mapping[str, float | int]],
    fieldname: str,
) -> float:
    return sum(float(episode[fieldname]) for episode in episodes)


def _aggregate_or_episode_totals(
    aggregate_match: re.Match[str] | None,
    episodes: Sequence[Mapping[str, float | int]],
    fieldnames: tuple[str, str, str],
) -> tuple[float, float, float]:
    if aggregate_match is not None:
        return tuple(
            _number(aggregate_match.group(group))
            for group in ("agent", "baseline", "baseline_off")
        )
    return tuple(_sum_metric(episodes, fieldname) for fieldname in fieldnames)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def _ratio_or_none(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator != 0.0 else None


def build_evaluation_csv_row(
    stdout: str,
    *,
    sweep_parameter: str,
    sweep_value: float,
    model: str,
    oracle_enabled: bool,
) -> dict[str, str | float | int | None]:
    """Aggregate one ``train.py`` run into the reference evaluation schema."""

    evaluation_csv_fieldnames(sweep_parameter)
    episodes = _parse_episode_lines(stdout)
    job_totals = _JOB_PROCESSING_TOTALS_RE.search(stdout)

    # In current train.py the aggregate Total Cost values and PropCost use the
    # same per-core power formula. Prefer those once-rounded run totals; fall
    # back to the explicitly proportional episode fields for older output.
    cost_agent, cost_baseline, cost_baseline_off = _aggregate_or_episode_totals(
        _AGGREGATE_COST_TOTALS_RE.search(stdout),
        episodes,
        ("agent_prop_cost", "baseline_prop_cost", "baseline_off_prop_cost"),
    )
    power_agent, power_baseline, power_baseline_off = (
        _aggregate_or_episode_totals(
            _AGGREGATE_POWER_TOTALS_RE.search(stdout),
            episodes,
            (
                "agent_prop_power_mwh",
                "baseline_prop_power_mwh",
                "baseline_off_prop_power_mwh",
            ),
        )
    )

    jobs_submitted = _sum_metric(episodes, "jobs_submitted")
    jobs_completed = _sum_metric(episodes, "jobs_completed")
    if job_totals is not None:
        baseline_jobs_completed = _number(
            job_totals.group("baseline_jobs_completed")
        )
        mean_wait_h = _number(job_totals.group("agent_mean_wait"))
    else:
        # train.py prints its aggregate job block inside the plotting/reporting
        # try-block. If plotting fails after all episodes finish, retain a row
        # using the same legacy fallbacks as the reference helper: completed
        # jobs weight the per-episode waits, while baseline completions are
        # recovered from each displayed cost-per-1k value.
        baseline_jobs_completed = _sum_metric(
            episodes,
            "baseline_jobs_completed_estimate",
        )
        mean_wait_h = (
            sum(
                float(episode["avg_wait_time"])
                * float(episode["jobs_completed"])
                for episode in episodes
            )
            / jobs_completed
            if jobs_completed > 0.0
            else None
        )
    dropped_jobs = sum(int(episode["jobs_lost_total"]) for episode in episodes)

    savings_vs_baseline = cost_baseline - cost_agent
    savings_vs_baseline_off = cost_baseline_off - cost_agent
    cost_per_1k_agent = _safe_ratio(cost_agent * 1000.0, jobs_completed)
    cost_per_1k_baseline = _safe_ratio(
        cost_baseline * 1000.0, baseline_jobs_completed
    )
    cost_per_1k_baseline_off = _safe_ratio(
        cost_baseline_off * 1000.0, baseline_jobs_completed
    )
    price_agent = _safe_ratio(cost_agent, power_agent)
    price_baseline = _safe_ratio(cost_baseline, power_baseline)
    price_baseline_off = _safe_ratio(cost_baseline_off, power_baseline_off)

    delta_power = power_baseline_off - power_agent
    delta_price = None
    cost_per_1k_delta = None
    price_savings = None
    power_savings = None
    savings_model = None
    if price_agent is not None and price_baseline_off is not None:
        delta_price = price_baseline_off - price_agent
        price_savings = power_baseline_off * delta_price
        power_savings = delta_power * price_agent
        savings_model = price_savings + power_savings
    if cost_per_1k_agent is not None and cost_per_1k_baseline_off is not None:
        cost_per_1k_delta = cost_per_1k_baseline_off - cost_per_1k_agent

    oracle_liquid = None
    oracle_contiguous = None
    oracle_capture = None
    if oracle_enabled:
        liquid_total = _ORACLE_LIQUID_TOTAL_RE.search(stdout)
        contiguous_total = _ORACLE_CONTIGUOUS_TOTAL_RE.search(stdout)
        oracle_liquid = (
            _number(liquid_total.group("value"))
            if liquid_total is not None
            else _sum_metric(episodes, "oracle_cost")
        )
        oracle_contiguous = (
            _number(contiguous_total.group("value"))
            if contiguous_total is not None
            else _sum_metric(episodes, "oracle_contiguous_cost")
        )
        oracle_capture = _safe_ratio(
            savings_vs_baseline_off * 100.0,
            cost_baseline_off - oracle_contiguous,
        )

    return {
        "model": model,
        "arrivals": jobs_submitted / (len(episodes) * EPISODE_HOURS),
        sweep_parameter: sweep_value,
        "cost_agent": cost_agent,
        "cost_baseline": cost_baseline,
        "cost_baseline_off": cost_baseline_off,
        "savings_vs_baseline_pct": _safe_ratio(
            savings_vs_baseline * 100.0, cost_baseline
        ),
        "savings_vs_baseline_off_pct": _safe_ratio(
            savings_vs_baseline_off * 100.0, cost_baseline_off
        ),
        "cost_per_1k_agent": cost_per_1k_agent,
        "cost_per_1k_baseline": cost_per_1k_baseline,
        "cost_per_1k_baseline_off": cost_per_1k_baseline_off,
        "completion_pct": _safe_ratio(jobs_completed * 100.0, jobs_submitted),
        "mean_wait_h": mean_wait_h,
        "dropped_jobs": dropped_jobs,
        "dropped_per_eur": (
            dropped_jobs / savings_vs_baseline_off
            if savings_vs_baseline_off > 0.0
            else None
        ),
        "power_agent_mwh": power_agent,
        "power_baseline_mwh": power_baseline,
        "power_baseline_off_mwh": power_baseline_off,
        "eur_per_mwh_agent": price_agent,
        "eur_per_mwh_baseline": price_baseline,
        "eur_per_mwh_baseline_off": price_baseline_off,
        "oracle_liquid_eur": oracle_liquid,
        "oracle_contiguous_eur": oracle_contiguous,
        "oracle_capture_pct": oracle_capture,
        "savings_abs_vs_baseline_eur": savings_vs_baseline,
        "savings_abs_vs_baseline_off_eur": savings_vs_baseline_off,
        "delta_power_mwh": delta_power,
        "savings_model_eur": savings_model,
        "delta_eur_per_mwh": delta_price,
        "cost_per_1k_delta": cost_per_1k_delta,
        "price_savings_eur": price_savings,
        "price_savings_pct": (
            _ratio_or_none(price_savings * 100.0, savings_vs_baseline_off)
            if price_savings is not None
            else None
        ),
        "power_savings_pct": (
            _ratio_or_none(power_savings * 100.0, savings_vs_baseline_off)
            if power_savings is not None
            else None
        ),
    }


def write_evaluation_csv(
    output_path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    sweep_parameter: str,
) -> None:
    """Rewrite accumulated rows in the reference evaluation CSV format."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=evaluation_csv_fieldnames(sweep_parameter),
        )
        writer.writeheader()
        writer.writerows(rows)
