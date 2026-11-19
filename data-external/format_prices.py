#!/usr/bin/env python3
"""Convert a semicolon-separated Date;Price CSV to a formatted Date,Price CSV.

Input dates use the format "Jan 01, 2024 12:00 AM" (any year). Output dates
use ISO 8601 with Europe/Berlin UTC offsets.

Usage: python format_prices_2024.py input.csv --output output.csv

Choose any input and output paths. If ``--output`` is omitted, the source
file is reformatted in place.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo


SOURCE_DATE_FORMAT = "%b %d, %Y %I:%M %p"
LOCAL_TIMEZONE = ZoneInfo("Europe/Berlin")
UTC = timezone.utc


def _parse_price(value: str, row_number: int) -> str:
    """Validate a price and remove unnecessary trailing decimal zeros."""
    try:
        price = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"row {row_number}: invalid price {value!r}") from exc

    if not price.is_finite():
        raise ValueError(f"row {row_number}: price must be finite, got {value!r}")

    result = format(price, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return "0" if result == "-0" else result


def _localize_timestamp(
    value: str, previous_utc: datetime | None, row_number: int
) -> tuple[datetime, datetime]:
    """Attach the correct Berlin UTC offset, including both repeated DST hours."""
    try:
        local_time = datetime.strptime(value, SOURCE_DATE_FORMAT)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: invalid date {value!r}") from exc

    candidates: list[tuple[datetime, datetime]] = []
    seen_utc: set[datetime] = set()
    for fold in (0, 1):
        aware = local_time.replace(tzinfo=LOCAL_TIMEZONE, fold=fold)
        utc_time = aware.astimezone(UTC)

        # zoneinfo allows attaching an offset to a nonexistent wall-clock time.
        # A UTC round trip filters that invalid candidate out.
        round_trip = utc_time.astimezone(LOCAL_TIMEZONE).replace(tzinfo=None)
        if round_trip == local_time and utc_time not in seen_utc:
            candidates.append((utc_time, aware))
            seen_utc.add(utc_time)

    candidates.sort(key=lambda candidate: candidate[0])
    if previous_utc is not None:
        candidates = [candidate for candidate in candidates if candidate[0] > previous_utc]

    if not candidates:
        raise ValueError(
            f"row {row_number}: date {value!r} is nonexistent, duplicated too many "
            "times, or out of chronological order"
        )

    utc_time, aware = candidates[0]
    if previous_utc is not None and utc_time - previous_utc != timedelta(hours=1):
        raise ValueError(
            f"row {row_number}: expected the hour after {previous_utc.isoformat()}, "
            f"got {utc_time.isoformat()}"
        )

    return aware, utc_time


def _read_rows(source: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    previous_utc: datetime | None = None

    with source.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter=";")
        if reader.fieldnames != ["Date", "Price"]:
            raise ValueError(
                f"expected semicolon-separated columns Date;Price, got "
                f"{reader.fieldnames!r}"
            )

        for row_number, row in enumerate(reader, start=2):
            if None in row or row["Date"] is None or row["Price"] is None:
                raise ValueError(f"row {row_number}: expected exactly two columns")

            aware, previous_utc = _localize_timestamp(
                row["Date"].strip(), previous_utc, row_number
            )
            price = _parse_price(row["Price"].strip(), row_number)
            rows.append((aware.isoformat(timespec="minutes"), price))

    if not rows:
        raise ValueError("input file contains no price rows")
    return rows


def _write_rows(destination: Path, rows: list[tuple[str, str]], mode: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(
            file_descriptor, "w", encoding="utf-8-sig", newline=""
        ) as output_file:
            writer = csv.writer(output_file, lineterminator="\n")
            writer.writerow(("Date", "Price"))
            writer.writerows(rows)
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def format_prices(source: Path, destination: Path) -> int:
    """Format source into destination and return the number of price rows."""
    rows = _read_rows(source)
    _write_rows(destination, rows, source.stat().st_mode & 0o777)
    return len(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "source",
        type=Path,
        help="input CSV path (semicolon-separated Date;Price columns)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output path (default: replace the source atomically)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    destination = args.output if args.output is not None else args.source
    row_count = format_prices(args.source, destination)
    print(f"Formatted {row_count} price rows in {destination}")


if __name__ == "__main__":
    main()
