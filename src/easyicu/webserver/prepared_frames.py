"""Reading one prepared module frame off a converted export.

Cohort Statistics and Patient Review each carried a copy of this stack inside
modules of 3211 and 2615 lines. The copies were the same algorithm — locate the
module's file in the descriptor, resolve which column identifies an ICU stay,
project the wanted columns, canonicalize the id column — with Patient Review's
version additionally pushing an entity-id predicate into the read.

Two things stay with the callers because they are genuinely per-consumer, not
duplication:

* the **column allow-list**. Cohort Statistics tolerates many spellings of the
  outcome columns because it summarises exports of different vintages; Patient
  Review reads a fixed set for a drilldown.
* the **skip policy**. Cohort Statistics reads whole modules, so it refuses a
  multi-million-row time-indexed table that is not parquet; Patient Review
  filters to a handful of stays and has no such ceiling.

Both are passed in. What is shared is the mechanism, and that lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from easyicu.webserver import entity_ids as entity_id_contract


# Decides whether a module file is too expensive to read for this caller.
# Receives the descriptor entry and the resolved path; True means "skip".
SkipPolicy = Callable[[Dict[str, Any], Path], bool]


def module_file_meta(desc: Dict[str, Any], module: str) -> Dict[str, Any] | None:
    """The descriptor entry for one module, or None if the export lacks it."""

    return next(
        (
            entry
            for entry in desc.get("files") or []
            if entry.get("module") == module
        ),
        None,
    )


def _is_number_like(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _entity_id_filters(
    path: Path, entity_column: str, stay_ids: Iterable[str]
) -> List[Tuple[str, str, List[Any]]] | None:
    """Build a parquet predicate typed like the column it filters.

    A string predicate against an int64 id column matches nothing silently,
    which reads as an empty cohort rather than as an error, so the values are
    cast to whatever the file's own schema says.
    """

    values: List[Any]
    try:
        import pyarrow.parquet as pq
        import pyarrow.types as pat

        field = pq.ParquetFile(path).schema_arrow.field(entity_column)
        if pat.is_integer(field.type):
            values = [int(value) for value in stay_ids if str(value).isdigit()]
        elif pat.is_floating(field.type):
            values = [float(value) for value in stay_ids if _is_number_like(value)]
        else:
            values = [str(value) for value in stay_ids if str(value)]
    except Exception:
        values = [str(value) for value in stay_ids if str(value)]
    if not values:
        return None
    return [(entity_column, "in", values)]


def read_selected_columns(
    path: Path,
    columns: Sequence[str],
    stay_ids: set[str] | None = None,
    *,
    entity_column: str = entity_id_contract.CANONICAL_ENTITY_ID,
) -> Any:
    """Read `columns` from a prepared file, optionally only for `stay_ids`.

    With `stay_ids=None` this is a plain projecting read of the whole file.
    """

    import pandas as pd

    selected = list(columns)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        filters = (
            _entity_id_filters(path, entity_column, stay_ids)
            if stay_ids and entity_column in selected
            else None
        )
        if filters:
            return pd.read_parquet(path, columns=selected, filters=filters)
        return pd.read_parquet(path, columns=selected)
    if suffix == ".xlsx":
        frame = pd.read_excel(path, usecols=selected)
    else:
        frame = pd.read_csv(path, usecols=selected)
    # Only parquet can push the predicate down; the rest filter after the read.
    if stay_ids and entity_column in frame.columns:
        frame = frame.copy()
        frame[entity_column] = frame[entity_column].map(
            entity_id_contract.normalize_entity_id
        )
        frame = frame[frame[entity_column].isin(stay_ids)]
    return frame


def read_module_frame(
    path: Path,
    desc: Dict[str, Any],
    module: str,
    columns: Sequence[str],
    *,
    stay_ids: set[str] | None = None,
    skip: SkipPolicy | None = None,
) -> Any:
    """One module's frame, keyed on the canonical stay id, or None.

    None means "this export cannot answer for this module" — the descriptor
    has no such file, it carries no resolvable stay id, or the caller's skip
    policy declined it. It never means "empty".
    """

    file_meta = module_file_meta(desc, module)
    if not file_meta:
        return None
    file_path = path / str(file_meta.get("file") or "")
    if skip is not None and skip(file_meta, file_path):
        return None
    available = [str(column) for column in file_meta.get("columns") or []]
    entity_column = entity_id_contract.resolve_entity_id_column(available)
    if not entity_column:
        return None
    selected = [entity_column] + [
        column
        for column in columns
        if column in available and column != entity_column
    ]
    frame = read_selected_columns(
        file_path, selected, stay_ids=stay_ids, entity_column=entity_column
    )
    return entity_id_contract.canonicalize_entity_frame(frame, entity_column)


def fallback_entity_frame(path: Path, desc: Dict[str, Any]) -> Any:
    """Every stay id in the export, from whichever file carries one."""

    file_meta = next(
        (
            entry
            for entry in desc.get("files") or []
            if entity_id_contract.resolve_entity_id_column(entry.get("columns") or [])
        ),
        None,
    )
    if not file_meta:
        return None
    entity_column = entity_id_contract.resolve_entity_id_column(
        file_meta.get("columns") or []
    )
    if not entity_column:
        return None
    frame = read_selected_columns(
        path / str(file_meta.get("file") or ""),
        [str(entity_column)],
        entity_column=str(entity_column),
    )
    return entity_id_contract.canonicalize_entity_frame(frame, str(entity_column))
