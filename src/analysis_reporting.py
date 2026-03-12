from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def validate_eval_months(eval_months: int) -> int:
    if eval_months <= 0:
        raise ValueError(f"eval_months must be > 0, got {eval_months}")
    return eval_months


def compute_savings_totals(
    savings: Sequence[float] | np.ndarray,
    eval_months: int,
) -> tuple[float, float]:
    months = validate_eval_months(eval_months)
    evaluation_savings = float(np.sum(np.asarray(savings, dtype=float)))
    annualized_savings = float(evaluation_savings * 12.0 / months)
    return evaluation_savings, annualized_savings
