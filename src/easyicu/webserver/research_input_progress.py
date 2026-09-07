"""Path-free, non-authorizing progress for a pipeline's research input.

Source registration and a zero-row planning catalog do not prove extraction.
Read only the typed context and Parquet footer when the run owner projects a
receipt; never scan patient values or recompute scientific results for the UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from easyicu.research_agent.research_context.typed import parse_research_context_json

ResearchInputState = Literal["metadata_only", "prepared", "unavailable"]


def project_research_input_state(value: object) -> ResearchInputState:
    """Bound persisted progress to this owner's path-free vocabulary."""

    if value == "metadata_only":
        return "metadata_only"
    if value == "prepared":
        return "prepared"
    return "unavailable"


def research_input_state(run_dir: Path | None) -> ResearchInputState:
    """Physical preparation only, not plan approval or clinical validation."""

    if run_dir is None:
        return "unavailable"
    context_path = Path(run_dir) / "research_context.json"
    cohort_path = Path(run_dir) / "cohort.parquet"
    try:
        if any(not path.is_file() or path.is_symlink() for path in (context_path, cohort_path)):
            return "unavailable"
        if context_path.stat().st_size > 4 * 1024 * 1024:
            return "unavailable"
        context = parse_research_context_json(context_path.read_text(encoding="utf-8"))
        if not context.cohort_parquet or Path(context.cohort_parquet).resolve() != cohort_path.resolve():
            return "unavailable"
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(cohort_path)
        provenance = context.cohort.provenance or {}
        row_count = provenance.get("analysis_row_count", context.cohort.n_stays)
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count != parquet.metadata.num_rows
            or not {variable.name for variable in context.variables}.issubset(parquet.schema_arrow.names)
        ):
            return "unavailable"
        if provenance.get("evidence_stage") == "metadata_only_planning":
            return (
                "metadata_only"
                if row_count == 0 and provenance.get("patient_rows_read") is False
                else "unavailable"
            )
        return "prepared" if row_count > 0 else "unavailable"
    except (OSError, UnicodeDecodeError, ValueError):
        # Optional progress never grants execution or replaces its validation
        # errors. An unreadable/mismatched input cannot get a completion mark.
        return "unavailable"


__all__ = ["ResearchInputState", "project_research_input_state", "research_input_state"]
