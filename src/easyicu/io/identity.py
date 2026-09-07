"""Identity-table join preconditions; diagnostics never contain patient keys."""

import pandas as pd


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
