"""A study binds the extraction it asked for.

The workflow tells a study whose bound export cannot serve its cohort to
extract that cohort first.  The extraction ran and registered a new export,
but the study stayed bound to the old one, so the workflow kept withholding
the plan and naming the same remedy.  The extraction's terminal write now
binds the export the study's setup asked for, in the same StudyContext write
that clears the job.  An export made for another setup, a study that changed
meanwhile, a cancelled job and a superseded job keep the binding.  A study
that states no modules took its bound export's modules as its data; Copilot
could not extract its cohort at all and now keeps those modules.  Fixtures
are synthetic.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import dataio
from easyicu.webserver import sources as source_store
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import PiSessionRecord, ToolExecutionContext
from easyicu.webserver.pi_copilot.extraction_handoff import (
    compile_study_cohort,
    study_binding_for_export,
    submit_study_extraction,
)
from easyicu.webserver.research_launch_scientific import bound_export_cohort_refusal
from easyicu.webserver.routes import jobs as jobs_route
from easyicu.webserver.routes.jobs import jobs_extract

_MODULES = ["demographics", "outcome"]
_COHORT = {
    "preset": "icd",
    "age_min": 18,
    "age_max": 100,
    "exclude_readmissions": False,
    "include_diagnoses": ["A41"],
}


@pytest.fixture(autouse=True)
def _isolated_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        context_store, "_CONFIG_PATH", tmp_path / "cfg" / "study-contexts.json"
    )
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "source-cfg")
    monkeypatch.setattr(
        source_store, "_CONFIG_PATH", tmp_path / "source-cfg" / "sources.json"
    )
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])


def _write_export(
    root: Path,
    raw: Path,
    cohort: dict[str, Any],
    *,
    modules: list[str] = _MODULES,
    record_modules: bool = True,
) -> Path:
    root.mkdir(parents=True)
    files = []
    for module in modules:
        pd.DataFrame({"stay_id": [1, 2], "value": [0.5, 1.5]}).to_csv(
            root / f"{module}.csv", index=False
        )
        row = {"file": f"{module}.csv", "rows": 2}
        files.append({**row, "module": module} if record_modules else row)
    contract = dataio.normalize_export_cohort_contract(cohort)
    manifest = {
        "schema_version": "easyicu_native_export_v2",
        "database": "miiv",
        "data_path": str(raw),
        "format": "csv",
        "generated": "2026-10-06T12:00:00Z",
        "patient_count": 2,
        "cohort_contract": contract,
        "cohort_execution": dataio.export_cohort_execution(contract),
        "files": files,
    }
    (root / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _register(export: Path) -> dict[str, Any]:
    registry = source_store.register_source(str(export), active=True, crossdb=True)
    path = context_store.normalize_path(str(export))
    return next(row for row in registry["sources"] if row["path"] == path)


def _study(
    export: Path, *, modules: list[str] = _MODULES, **confirmations: bool
) -> dict[str, Any]:
    return context_store.upsert_context(
        {
            "id": "study_extract_bind",
            "title": "Sepsis mortality study",
            "question": "Among adults with sepsis, is early lactate associated with death?",
            "data_source": {
                "path": str(export),
                "database": "miiv",
                "label": "Earlier export",
            },
            "cohort": dict(_COHORT),
            "modules": list(modules),
            "time_window": {"observation_hours": 24, "anchor": "ICU admission"},
            "export_format": "csv",
            "confirmations": {"extraction_completed": True, **confirmations},
        },
        active=True,
    )


@pytest.fixture(params=["stated modules", "no modules"])
def stale_study(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """A study bound to an all-ICU export, which its launch refuses."""

    raw = tmp_path / "raw"
    raw.mkdir()
    old = _write_export(
        tmp_path / "all_icu", raw, {"preset": "all_icu", "observation_window_hours": 720}
    )
    registered = _register(old)
    study = _study(old, modules=_MODULES if request.param == "stated modules" else [])
    assert bound_export_cohort_refusal(study, registered["path"]) is not None
    return study, registered, raw


def _fake_extraction(
    monkeypatch: pytest.MonkeyPatch, out_dir: Path, *, cancel: bool = False
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    def make_export_runner(**kwargs: Any):
        requests.append(kwargs)

        def runner(job: Any) -> dict[str, Any]:
            _write_export(
                out_dir,
                Path(kwargs["data_path"]),
                dict(kwargs["cohort"]),
                modules=list(kwargs["modules"]),
            )
            if cancel:
                job.request_cancel()
            return {"out_dir": str(out_dir), "manifest": "_manifest.json", "files": []}

        return runner

    monkeypatch.setattr(dataio, "make_export_runner", make_export_runner)
    return requests


def _wait(job_id: str) -> dict[str, Any]:
    client = TestClient(app)
    deadline = time.time() + 10
    while time.time() < deadline:
        snapshot = client.get(f"/api/jobs/{job_id}").json()
        if snapshot["status"] != "running":
            return snapshot
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish")


def test_the_study_binds_the_export_its_extraction_produced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale_study
) -> None:
    study, registered, _raw = stale_study
    new = tmp_path / "study_export"
    requests = _fake_extraction(monkeypatch, new)

    transaction = submit_study_extraction(
        study=study, registered_source=registered, submit=jobs_extract
    )
    snapshot = _wait(transaction.submitted["job_id"])

    assert snapshot["status"] == "done"
    assert requests[0]["cohort"] == compile_study_cohort(study)
    # A study stating no modules keeps those of the export it was bound to.
    assert sorted(requests[0]["modules"]) == sorted(_MODULES)
    current = context_store.get_context(study["id"])
    assert current["data_source"] == {
        "path": context_store.normalize_path(str(new)),
        "database": "miiv",
        "label": "Sepsis mortality study",
    }
    assert current["confirmations"]["extraction_completed"] is True
    assert current["active_job_id"] is None
    assert current["current_stage"] == "extract_review"
    assert snapshot["result"]["study_context_rebound"] is True
    assert snapshot["result"]["study_context_revision"] == current["revision"]
    # The launch, which the workflow reads before offering a plan, accepts it.
    assert bound_export_cohort_refusal(current, current["data_source"]["path"]) is None


def test_an_extraction_for_another_setup_keeps_the_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale_study
) -> None:
    study, registered, raw = stale_study
    _fake_extraction(monkeypatch, tmp_path / "other_export")

    submitted = jobs_extract(
        {
            "path": str(raw),
            "registered_export_path": registered["path"],
            "database": "miiv",
            "modules": list(_MODULES),
            "format": "csv",
            "cohort": {"preset": "all_icu"},
            "study_context_id": study["id"],
            "study_context_revision": study["revision"],
        }
    )
    snapshot = _wait(submitted["job_id"])

    assert snapshot["status"] == "done"
    assert snapshot["result"]["registered_source"]["ok"] is True
    current = context_store.get_context(study["id"])
    assert current["data_source"]["path"] == registered["path"]
    assert current["active_job_id"] is None
    assert current["current_stage"] == "extract_review"
    assert snapshot["result"]["study_context_rebound"] is False


def test_a_cancelled_extraction_keeps_the_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale_study
) -> None:
    study, registered, _raw = stale_study
    _fake_extraction(monkeypatch, tmp_path / "cancelled_export", cancel=True)

    transaction = submit_study_extraction(
        study=study, registered_source=registered, submit=jobs_extract
    )
    snapshot = _wait(transaction.submitted["job_id"])

    assert snapshot["status"] == "cancelled"
    current = context_store.get_context(study["id"])
    assert current["data_source"]["path"] == registered["path"]
    assert current["current_stage"] == "extract_cancelled"
    assert current["active_job_id"] is None


def test_the_owner_binds_only_for_the_active_job_at_the_revision_it_read(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    old = _write_export(tmp_path / "old", raw, dict(_COHORT))
    # A plan-first study records no completed extraction until one binds.
    study = _study(old, extraction_completed=False, cohort_reviewed=True)
    binding = {"path": str(tmp_path / "new"), "database": "miiv", "label": "Study export"}

    context_store.handoff_context(study["id"], active_job_id="job-current")
    held = context_store.get_context(study["id"])
    stale = context_store.clear_active_job_if(
        study["id"],
        "job-current",
        current_stage="extract_review",
        last_route="extract",
        bind_export=(held["revision"] - 1, binding),
    )
    assert (stale["cleared"], stale["rebound"]) == (True, False)
    assert stale["context"]["data_source"]["path"] == context_store.normalize_path(str(old))
    assert stale["context"]["confirmations"]["extraction_completed"] is False

    context_store.handoff_context(study["id"], active_job_id="job-next")
    held = context_store.get_context(study["id"])
    superseded = context_store.clear_active_job_if(
        study["id"],
        "job-current",
        current_stage="extract_review",
        last_route="extract",
        bind_export=(held["revision"], binding),
    )
    assert (superseded["cleared"], superseded["rebound"]) == (False, False)
    assert context_store.get_context(study["id"]) == held

    bound = context_store.clear_active_job_if(
        study["id"],
        "job-next",
        current_stage="extract_review",
        last_route="extract",
        bind_export=(held["revision"], binding),
    )
    assert (bound["cleared"], bound["rebound"]) == (True, True)
    context = bound["context"]
    assert context["data_source"] == {
        **binding,
        "path": context_store.normalize_path(binding["path"]),
    }
    assert context["confirmations"] == {"extraction_completed": True, "cohort_reviewed": True}
    assert context["active_job_id"] is None


def test_only_an_export_holding_the_study_setup_is_its_extraction(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    study = {
        "id": "study_setup",
        "data_source": {"path": str(raw), "database": "miiv", "label": "Raw MIMIC-IV"},
        "cohort": dict(_COHORT),
        "modules": list(_MODULES),
        "time_window": {"observation_hours": 24, "anchor": "ICU admission"},
        "export_format": "csv",
    }
    exact = _register(_write_export(tmp_path / "exact", raw, compile_study_cohort(study)))

    assert study_binding_for_export(study, exact) == {
        "path": exact["path"],
        "database": "miiv",
        "label": exact["label"],
    }
    assert study_binding_for_export(study, {**exact, "label": ""})["label"] == "Raw MIMIC-IV"
    assert study_binding_for_export(study, {**exact, "ok": False}) is None
    assert study_binding_for_export(study, None) is None
    assert study_binding_for_export(study, {**exact, "path": str(tmp_path / "gone")}) is None
    assert study_binding_for_export({**study, "export_format": "parquet"}, exact) is None
    other_cohort = _write_export(tmp_path / "all_icu", raw, {"preset": "all_icu"})
    assert study_binding_for_export(study, _register(other_cohort)) is None
    fewer = _write_export(
        tmp_path / "fewer", raw, compile_study_cohort(study), modules=["demographics"]
    )
    assert study_binding_for_export(study, _register(fewer)) is None


def _tool_context() -> ToolExecutionContext:
    return ToolExecutionContext(
        session=PiSessionRecord(session_id="pi-extract-bound-modules"),
        allowed_actions={"extract"},
    )


def test_copilot_extracts_a_study_stating_no_modules_with_its_exports_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    old = _write_export(
        tmp_path / "all_icu",
        raw,
        {"preset": "all_icu", "observation_window_hours": 720},
        modules=["demographics", "outcome", "vitals"],
    )
    _register(old)
    study = _study(old, modules=[])
    submitted: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: study)
    monkeypatch.setattr(
        jobs_route,
        "jobs_extract",
        lambda body: submitted.append(dict(body))
        or {"job_id": "extract-cohort", "kind": "extract", "status": "running"},
    )

    result = tool_module.execute_tool("easyicu_start_extraction", {}, _tool_context())

    assert result["code"] == "easyicu_extraction_submitted"
    assert submitted[0]["modules"] == ["demographics", "outcome", "vitals"]
    assert submitted[0]["cohort"] == compile_study_cohort(study)
    assert submitted[0]["path"] == context_store.normalize_path(str(raw))


def test_an_export_recording_no_modules_is_not_extracted_blindly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    old = _write_export(
        tmp_path / "unrecorded",
        raw,
        {"preset": "all_icu", "observation_window_hours": 720},
        record_modules=False,
    )
    _register(old)
    study = _study(old, modules=[])
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: study)
    monkeypatch.setattr(
        jobs_route,
        "jobs_extract",
        lambda body: pytest.fail("an extraction without modules must not be submitted"),
    )

    result = tool_module.execute_tool("easyicu_start_extraction", {}, _tool_context())

    assert result["status"] == "blocked"
    assert result["code"] == "registered_export_modules_unrecorded"

