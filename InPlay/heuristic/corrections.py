"""Validate and finalize manually corrected rally candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .segment import CANONICAL_FIELDS

ALLOWED_DECISIONS = {"", "accept", "reject"}


def read_candidates(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(CANONICAL_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"candidate CSV missing columns: {sorted(missing)}")
        return list(reader)


def validate_rows(
    rows: list[dict[str, str]], source_start: int | None = None, source_end: int | None = None
) -> list[str]:
    errors: list[str] = []
    intervals: dict[str, list[tuple[int, int, int]]] = {}
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, 2):
        decision = row["manual_decision"].strip().lower()
        if decision not in ALLOWED_DECISIONS:
            errors.append(f"row {row_number}: invalid manual_decision {decision!r}")
        if (
            decision != "accept"
            and (row["manual_start_frame"].strip() or row["manual_end_frame"].strip())
        ):
            errors.append(
                f"row {row_number}: corrected boundaries require manual_decision 'accept'"
            )
        rally_id = row["rally_id"]
        if rally_id in seen_ids:
            errors.append(f"row {row_number}: duplicate rally_id {rally_id!r}")
        seen_ids.add(rally_id)
        try:
            automatic_start, automatic_end = int(row["start_frame"]), int(row["end_frame"])
            start = int(row["manual_start_frame"]) if row["manual_start_frame"].strip() else automatic_start
            end = int(row["manual_end_frame"]) if row["manual_end_frame"].strip() else automatic_end
        except ValueError:
            errors.append(f"row {row_number}: boundaries must be integers")
            continue
        if start < 0 or end < start:
            errors.append(f"row {row_number}: invalid corrected boundaries {start}-{end}")
        if source_start is not None and start < source_start:
            errors.append(f"row {row_number}: start is outside source range")
        if source_end is not None and end > source_end:
            errors.append(f"row {row_number}: end is outside source range")
        if decision != "reject":
            intervals.setdefault(row["source_id"], []).append((start, end, row_number))
    for source, values in intervals.items():
        values.sort()
        for left, right in zip(values, values[1:]):
            if right[0] <= left[1]:
                errors.append(
                    f"rows {left[2]} and {right[2]}: overlapping intervals for {source!r}"
                )
    return errors


def finalize(rows: list[dict[str, str]], fps: float) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        decision = row["manual_decision"].strip().lower()
        if decision == "reject" or (
            not decision and row["status"].strip().lower() != "accepted"
        ):
            continue
        clean = dict(row)
        if row["manual_start_frame"].strip():
            clean["start_frame"] = row["manual_start_frame"].strip()
        if row["manual_end_frame"].strip():
            clean["end_frame"] = row["manual_end_frame"].strip()
        clean["start_time"] = str(int(clean["start_frame"]) / fps)
        clean["end_time"] = str(int(clean["end_frame"]) / fps)
        clean["status"] = "accepted"
        output.append(clean)
    return output


def _write(path: str | Path, rows: list[dict[str, str]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def validate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate manual rally corrections")
    parser.add_argument("--input", required=True)
    parser.add_argument("--source-start", type=int)
    parser.add_argument("--source-end", type=int)
    args = parser.parse_args(argv)
    errors = validate_rows(read_candidates(args.input), args.source_start, args.source_end)
    if errors:
        for error in errors:
            print(error)
        return 1
    print("corrections are valid")
    return 0


def finalize_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply manual rally corrections")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--source-start", type=int)
    parser.add_argument("--source-end", type=int)
    args = parser.parse_args(argv)
    if args.fps <= 0:
        parser.error("--fps must be positive")
    rows = read_candidates(args.input)
    errors = validate_rows(rows, args.source_start, args.source_end)
    if errors:
        parser.error("; ".join(errors))
    _write(args.output, finalize(rows, args.fps))
    return 0
