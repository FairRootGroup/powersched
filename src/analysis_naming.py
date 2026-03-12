from __future__ import annotations


def _format_slug_number(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def build_weight_slug(
    efficiency_weight: float,
    price_weight: float,
    idle_weight: float,
    job_age_weight: float,
) -> str:
    return (
        f"e{_format_slug_number(efficiency_weight)}"
        f"_p{_format_slug_number(price_weight)}"
        f"_i{_format_slug_number(idle_weight)}"
        f"_ja{_format_slug_number(job_age_weight)}"
    )


def build_analysis_dir_name(
    prefix: str,
    timestamp: str,
    model: int | None,
    efficiency_weight: float,
    price_weight: float,
    idle_weight: float,
    job_age_weight: float,
) -> str:
    parts = [prefix]
    if model is not None:
        parts.append(f"m{int(model)}")
    parts.append(
        build_weight_slug(
            efficiency_weight=efficiency_weight,
            price_weight=price_weight,
            idle_weight=idle_weight,
            job_age_weight=job_age_weight,
        )
    )
    parts.append(timestamp)
    return "_".join(parts)


def build_model_weight_dir_name(
    model: int | None,
    efficiency_weight: float,
    price_weight: float,
    idle_weight: float,
    job_age_weight: float,
) -> str:
    parts = []
    if model is not None:
        parts.append(f"m{int(model)}")
    parts.append(
        build_weight_slug(
            efficiency_weight=efficiency_weight,
            price_weight=price_weight,
            idle_weight=idle_weight,
            job_age_weight=job_age_weight,
        )
    )
    return "_".join(parts)
