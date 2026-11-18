"""Identity-table schema and join preconditions."""

import csv
import gzip
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ..storage_layout import TableScanError


IDENTITY_COLUMN_AMBIGUOUS = "identity_column_ambiguous"


def resolve_identity_columns(
    available: Iterable[str], logical_names: Iterable[str], *, table: str,
) -> dict[str, str]:
    """Map each present logical name to its unique physical spelling."""
    logical_by_fold = {name.casefold(): name for name in logical_names}
    resolved = {}
    for name in available:
        logical = logical_by_fold.get(name.casefold())
        if logical is None:
            continue
        if logical in resolved:
            raise TableScanError(
                IDENTITY_COLUMN_AMBIGUOUS, table, f"multiple columns match {logical!r}",
            )
        resolved[logical] = name
    return resolved


def read_identity_csv_header(path: Path) -> list[str]:
    """Read physical names before pandas can rename duplicate CSV headers."""
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return next(
            (row for row in csv.reader(handle) if len(row) > 1 or (row and row[0].strip())),
            [],
        )


def require_unique_keys(frame: pd.DataFrame, keys: list[str], *, table: str) -> None:
    if any(key not in frame for key in keys):
        raise ValueError(f"{table}: required identity columns unavailable")
    missing = frame[keys].isna().any(axis=1)
    duplicate = frame.duplicated(keys, keep=False)
    if missing.any() or duplicate.any():
        raise ValueError(
            f"{table}: identity keys must be non-missing and unique; "
            f"missing_rows={int(missing.sum())}, duplicate_rows={int(duplicate.sum())}"
        )
