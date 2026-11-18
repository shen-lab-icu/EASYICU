"""Source admission shared by the CareVue-only concept owners."""

from __future__ import annotations

import re
from pathlib import Path

from ..storage_layout import (
    _visible_parquet,
    match_table_entries,
    refuse_broken_presence,
    table_decision,
)

_CAREVUE_VIEW = re.compile(r"mimic_cv_(?:flat_r13|dir(?:_r13)?|bkt_s[0-9a-f]{9})")


def _full_mimic_path(path: Path) -> bool:
    parts = tuple(part.casefold() for part in path.parts)
    return ("databases", "mimiciii") in zip(parts, parts[1:])


def _view_root(path: Path) -> bool:
    return not _full_mimic_path(path) and (
        _CAREVUE_VIEW.fullmatch(path.name) is not None
        or tuple(part.casefold() for part in path.parts[-2:]) == ("mimicview", "raw")
    )


def is_carevue_view(path: Path | None) -> bool:
    """Require the configured root and its resolved destination to be views."""
    return path is not None and _view_root(path) and _view_root(path.resolve())


def _in_view(path: Path) -> bool:
    target = path.resolve()
    return not _full_mimic_path(target) and any(
        _view_root(parent) for parent in (target, *target.parents)
    )


def carevue_view_has_table(data_source, table):
    """Admit only Parquet inputs contained in an established CareVue view."""
    matches = match_table_entries(data_source, table)
    for match in (*matches.flat, *matches.bucket, *matches.broken):
        # Check link destinations before the storage owner probes Parquet.
        path = match.path
        if not _in_view(path) or (
            path.is_dir() and any(not _in_view(part) for part in _visible_parquet(path))
        ):
            return False
    decision = table_decision(data_source, table)
    refuse_broken_presence(table, decision)
    return bool(decision.files) and all(_in_view(path) for path in decision.files)
