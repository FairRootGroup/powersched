from __future__ import annotations

import math


def validate_job_arrival_scale(job_arrival_scale: float) -> float:
    """Return a normalized arrival scale or raise for invalid values."""
    scale = float(job_arrival_scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("--job-arrival-scale must be finite and >= 0.0")
    return scale
