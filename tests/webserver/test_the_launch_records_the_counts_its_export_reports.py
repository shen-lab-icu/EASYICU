"""A launch records the bound export's own count of the stays its selection kept.

Data Extraction writes its count report (``cohort_report``) beside the cohort
contract the launch already reads; nothing carried it into the run, so the
Writer could not state how many stays each criterion excluded.  The launch
now reads it once, verbatim, from the manifest it binds, and records it in
``data_constraints.source_selection.export_report`` when the selection is
recorded: only then do its counts describe this study's rows.  Fixtures are
generic.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.agent_pipeline_runs import _research_user_preferences
from easyicu.webserver.research_launch_scientific import bound_export_selection_report
from tests.webserver.copilot.research_workflow_fixtures import complete_study

_REPORT = {
    "mode": "adult_first",
    "count_unit": "icu_stay",
    "source_total": 9,
    "selected": 4,
    "demographic_steps": [
        {
            "criterion": "age",
            "parameters": {"age_min": 18},
            "n_before": 9,
            "n_excluded": 5,
            "n_remaining": 4,
            "n_excluded_missing": 1,
        }
    ],
}


def _export(path: Path, manifest: dict) -> str:
    path.mkdir(parents=True, exist_ok=True)
    (path / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return str(path)


def test_the_launch_reads_the_report_the_export_wrote(tmp_path: Path) -> None:
    export = _export(tmp_path / "miiv", {"files": [], "cohort_report": _REPORT})

    assert bound_export_selection_report(export) == _REPORT


def test_an_export_without_a_report_records_none(tmp_path: Path) -> None:
    unreported = _export(tmp_path / "old", {"files": []})
    malformed = _export(tmp_path / "bad", {"files": [], "cohort_report": [1, 2]})

    assert bound_export_selection_report(unreported) is None
    assert bound_export_selection_report(malformed) is None
    assert bound_export_selection_report(str(tmp_path / "missing")) is None
    assert bound_export_selection_report(None) is None


def test_only_a_recorded_selection_carries_the_report() -> None:
    study = complete_study()

    def selection(**kwargs) -> dict:
        return json.loads(
            _research_user_preferences(study, **kwargs)["data_constraints"]
        )["source_selection"]

    recorded = selection(
        source_selection_basis="export_contract", source_selection_report=_REPORT
    )
    unrecorded = selection(
        source_selection_basis="unrecorded", source_selection_report=_REPORT
    )
    unreported = selection(source_selection_basis="export_contract")

    assert recorded["export_report"] == _REPORT
    assert recorded["basis"] == "export_contract"
    assert "export_report" not in unrecorded
    assert "export_report" not in unreported


def test_the_launch_passes_the_bound_exports_report_to_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run declares its context through one owner, which reads the export it binds."""

    export = _export(tmp_path / "miiv", {"files": [], "cohort_report": _REPORT})
    study = complete_study()
    # The export's contract selected this study's rows.
    monkeypatch.setattr(
        agent_pipeline_runs,
        "bound_export_selection_basis",
        lambda _study, _export_path: "export_contract",
    )
    scientific = SimpleNamespace(
        study=study,
        materialization_study=study,
        patient_grouping=None,
        cohort_window=(0.0, 24.0),
        metadata_planning_coordinates={},
    )

    declared = agent_pipeline_runs.research_context_declarations(
        scientific, export_path=export
    )

    constraints = json.loads(declared["user_preferences"]["data_constraints"])
    assert constraints["source_selection"]["export_report"] == _REPORT
