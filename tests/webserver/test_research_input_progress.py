from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.schema import ResearchContext
from easyicu.webserver.research_input_progress import (
    project_research_input_state,
    research_input_state,
)


def _write_input(root: Path, *, rows: int = 2, metadata_only: bool = False) -> Path:
    root.mkdir(exist_ok=True)
    cohort = root / "cohort.parquet"
    pd.DataFrame({"value": range(rows)}).to_parquet(cohort, index=False)
    context = ResearchContext(
        research_question="Describe the supplied research input.",
        cohort={
            "database": "miiv", "cohort_name": "Bound input fixture", "n_stays": rows,
            "provenance": {
                "analysis_row_count": rows,
                **({"evidence_stage": "metadata_only_planning", "patient_rows_read": False}
                   if metadata_only else {}),
            },
        },
        variables=[{"name": "value", "dtype": "int64"}],
        cohort_parquet=str(cohort),
    )
    (root / "research_context.json").write_text(context.model_dump_json(), encoding="utf-8")
    return root


@pytest.mark.parametrize(("rows", "metadata_only", "expected"), [
    (2, False, "prepared"), (0, True, "metadata_only"),
    (0, False, "unavailable"), (2, True, "unavailable"),
])
def test_input_progress_uses_bound_physical_metadata_only(tmp_path, monkeypatch, rows, metadata_only, expected):
    root = _write_input(tmp_path / "run", rows=rows, metadata_only=metadata_only)
    def forbidden_patient_scan(*args, **kwargs):
        raise AssertionError("Progress must not read patient values.")
    monkeypatch.setattr(pd, "read_parquet", forbidden_patient_scan)
    import pyarrow.parquet as pq
    monkeypatch.setattr(pq, "read_table", forbidden_patient_scan)
    monkeypatch.setattr(pq.ParquetFile, "read", forbidden_patient_scan)

    assert research_input_state(root) == expected


@pytest.mark.parametrize("mutation", [
    "missing_context", "missing_cohort", "invalid_context", "invalid_cohort",
    "context_symlink", "cohort_symlink", "other_cohort", "row_count", "column",
    "unknown_context_schema", "metadata_patient_rows", "oversized_context",
])
def test_input_progress_withholds_completion_on_missing_or_inconsistent_input(tmp_path, mutation):
    root = _write_input(tmp_path / "run")
    context_path = root / "research_context.json"
    cohort_path = root / "cohort.parquet"
    if mutation in {"missing_context", "missing_cohort"}:
        (context_path if mutation.endswith("context") else cohort_path).unlink()
    elif mutation in {"context_symlink", "cohort_symlink"}:
        target = context_path if mutation.startswith("context") else cohort_path
        other = tmp_path / target.name
        target.rename(other)
        target.symlink_to(other)
    elif mutation == "invalid_context":
        context_path.write_text("{}", encoding="utf-8")
    elif mutation == "invalid_cohort":
        cohort_path.write_bytes(b"not a parquet file")
    elif mutation == "oversized_context":
        context_path.write_text(" " * (4 * 1024 * 1024 + 1), encoding="utf-8")
    else:
        context = json.loads(context_path.read_text())
        if mutation == "other_cohort":
            context["cohort_parquet"] = str(tmp_path / "different.parquet")
        elif mutation == "row_count":
            context["cohort"]["provenance"]["analysis_row_count"] = 4
        elif mutation == "column":
            context["variables"][0]["name"] = "missing_input"
        elif mutation == "unknown_context_schema":
            context["schema_version"] = "unknown"
        elif mutation == "metadata_patient_rows":
            context["cohort"]["provenance"].update({
                "evidence_stage": "metadata_only_planning", "patient_rows_read": True,
            })
        context_path.write_text(json.dumps(context), encoding="utf-8")

    assert research_input_state(root) == "unavailable"


def test_input_progress_without_a_run_is_unavailable():
    assert research_input_state(None) == "unavailable"


@pytest.mark.parametrize("value", [None, True, 1, [], {}, "/private/data", "unexpected"])
def test_persisted_input_progress_does_not_project_unknown_fields(value):
    assert project_research_input_state(value) == "unavailable"


@pytest.mark.parametrize("value", ["metadata_only", "prepared", "unavailable"])
def test_persisted_input_progress_preserves_known_states(value):
    assert project_research_input_state(value) == value
