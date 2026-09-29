"""What an empty figure source-data projection can verify.

Owner
-----
This module owns the figure source audit's decision on an empty source-data
table.  A step that reports a failed-closed decision may bind parents that are
empty by design, such as the characterization tables of a clustering that
froze no class.  Its projection of such a parent is empty too and names only
that parent's columns: the parent has no value to verify, and the projection
credits no result family.  Any other empty projection cannot authenticate a
rendered result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Set, Tuple

import pandas as pd

from ..schema import ValidationFinding

__all__ = [
    "PROJECTION_PROVENANCE_COLUMNS",
    "empty_parent_projection",
    "empty_source_data_outcome",
]

#: Columns a source-data projection adds to name its parent rows.
PROJECTION_PROVENANCE_COLUMNS = frozenset(
    {"source_row_index", "source_table", "source_step_id"}
)


def empty_parent_projection(
    source_df: pd.DataFrame,
    *,
    step_summary: Mapping[str, Any],
    table_frames: Mapping[Path, pd.DataFrame],
    parent_paths: Set[Path],
) -> Set[Path]:
    """The empty bound parents an empty source table projects, in a no-result step."""

    if step_summary.get("scientific_status") != "failed_closed":
        return set()
    columns = set(source_df.columns) - PROJECTION_PROVENANCE_COLUMNS
    if not columns:
        return set()
    return {
        path
        for path in parent_paths
        if (frame := table_frames.get(path)) is not None
        and frame.empty
        and columns <= set(frame.columns)
    }


def empty_source_data_outcome(
    *,
    validator: str,
    step_id: str,
    source_path: Path,
    source_df: pd.DataFrame,
    step_summary: Mapping[str, Any],
    table_frames: Mapping[Path, pd.DataFrame],
    parent_paths: Set[Path],
) -> Tuple[Set[Path], List[ValidationFinding]]:
    """The parents an empty projection verifies, or the finding that it verifies none."""

    verified = empty_parent_projection(
        source_df,
        step_summary=step_summary,
        table_frames=table_frames,
        parent_paths=parent_paths,
    )
    if verified:
        return verified, []
    return set(), [
        ValidationFinding(
            validator=validator,
            severity="error",
            message=(
                f"Figure source-data table {source_path.name} is "
                "empty and cannot authenticate a rendered result."
            ),
            detail={
                "step_id": step_id,
                "source_table": source_path.name,
                "reason": "source_data_empty",
            },
        )
    ]
