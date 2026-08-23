"""Focused contracts for the conversational Data Workbench boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.webserver import cohort_review, patient_drilldown
from easyicu.webserver.copilot_data_workbench import (
    CopilotDataWorkbenchError,
    CopilotDataWorkbenchSnapshotStore,
    build_snapshot,
)
from easyicu.webserver.pi_copilot import tools as tool_owner
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiSessionRecord,
    ToolExecutionContext,
)


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-data-workbench",
            project_id="project-data-workbench",
            binding=AuthorityBinding(),
        )
    )


def _registry(source_path: Path) -> dict:
    return {
        "active_path": str(source_path),
        "sources": [
            {
                "id": "source-mimic",
                "label": "MIMIC-IV export",
                "database": "miiv",
                "path": str(source_path),
                "ok": True,
                "modules": ["vitals", "outcome"],
            }
        ],
    }


def _cohort_payload() -> dict:
    return {
        "ok": True,
        "source": {"id": "source-mimic", "label": "MIMIC-IV export", "database": "miiv"},
        "summary": {"cohort_size": 120, "modules": 2, "mortality_pct": 12.5},
        "feature_catalog": {
            "modules": [
                {
                    "module": "vitals",
                    "features": [
                        {"id": "vitals:hr", "module": "vitals", "column": "hr", "label": "Heart rate"}
                    ],
                }
            ]
        },
        "feature_selection": {"selected": [{"id": "vitals:hr"}]},
        "selected_feature_distributions": [
            {
                "id": "vitals:hr",
                "module": "vitals",
                "label": "Heart rate",
                "kind": "numeric",
                "observed": 110,
                "observed_pct": 91.7,
                "bins": [{"low": 40, "high": 80, "count": 45}],
            }
        ],
        "groups": {"supported": []},
        "coverage": [],
        "quality": {"median_coverage_pct": 91.7},
        "survival_analysis": {"status": "blocked"},
        "blocked_features": [],
        "provenance": {"payload_scope": "cohort_aggregate_only"},
        "privacy": {"raw_rows_returned": False, "direct_identifiers_returned": False},
    }


def test_snapshot_store_is_project_scoped_digest_bound_and_path_free(tmp_path: Path) -> None:
    store = CopilotDataWorkbenchSnapshotStore(tmp_path / "snapshots")
    snapshot = build_snapshot(
        project_id="project-a",
        view="cohort_summary",
        title="Cohort",
        payload={"summary": {"cohort_size": 42}},
    )
    store.persist(snapshot)

    loaded = store.load(
        project_id="project-a", digest=snapshot["snapshot_sha256"]
    )
    assert loaded == snapshot
    with pytest.raises(CopilotDataWorkbenchError) as mismatch:
        store.load(project_id="project-b", digest=snapshot["snapshot_sha256"])
    assert mismatch.value.code == "copilot_data_workbench_snapshot_not_found"

    with pytest.raises(CopilotDataWorkbenchError) as blocked:
        build_snapshot(
            project_id="project-a",
            view="cohort_summary",
            title="Cohort",
            payload={"source_path": "/private/mimic"},
        )
    assert blocked.value.code == "copilot_data_workbench_path_forbidden"


def test_selected_feature_distributions_are_bounded_stay_level_aggregates() -> None:
    distributions = cohort_review._selected_feature_distributions(
        ["stay-a", "stay-b", "stay-c", "stay-d"],
        [
            {
                "id": "vitals:hr",
                "module": "vitals",
                "column": "hr",
                "label": "Heart rate",
                "kind": "numeric",
                "aggregation": "median",
                "mapping": {"stay-a": 60, "stay-b": 80, "stay-c": 100},
            },
            {
                "id": "outcome:mortality",
                "module": "outcome",
                "column": "mortality",
                "label": "Mortality",
                "kind": "binary",
                "aggregation": "any",
                "mapping": {"stay-a": True, "stay-b": False},
            },
        ],
    )

    numeric, binary = distributions
    assert numeric["denominator"] == 4
    assert numeric["observed"] == 3
    assert numeric["summary"]["median"] == 80
    assert sum(row["count"] for row in numeric["bins"]) == 3
    assert len(numeric["bins"]) <= 8
    assert binary["categories"] == [
        {"label": "Positive", "count": 1},
        {"label": "Negative", "count": 1},
        {"label": "Unknown", "count": 2},
    ]
    assert all("mapping" not in row for row in distributions)


def test_cohort_tool_resolves_feature_and_returns_snapshot_coordinate_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "export"
    export.mkdir()
    store = CopilotDataWorkbenchSnapshotStore(tmp_path / "snapshots")
    calls: list[dict] = []

    monkeypatch.setattr(tool_owner.sources, "load_registry", lambda: _registry(export))
    monkeypatch.setattr(
        tool_owner.cohort_review,
        "cohort_review_summary",
        lambda body: calls.append(dict(body)) or _cohort_payload(),
    )
    monkeypatch.setattr(
        tool_owner.dataio,
        "describe_export_source",
        lambda _path: {"ok": True, "summary": {"stays": 120}},
    )
    monkeypatch.setattr(
        "easyicu.webserver.patient_drilldown.eligibility._eligibility_flow_payload",
        lambda *_args: {
            "steps": [
                {"id": "source_total", "count": 150},
                {"id": "final_cohort", "count": 120},
            ],
            "initial_count": 150,
            "final_count": 120,
        },
    )
    monkeypatch.setattr(tool_owner, "CopilotDataWorkbenchSnapshotStore", lambda: store)

    result = tool_owner._review_cohort(
        _context(), {"source_id": "source-mimic", "features": ["hr"]}
    )

    assert result["code"] == "easyicu_feature_distribution_ready"
    assert result["details"]["selected_features"] == ["vitals:hr"]
    assert calls[-1]["selected_features"] == ["vitals:hr"]
    resource = result["details"]["resource"]
    assert resource["kind"] == "data_workbench_snapshot"
    assert "payload" not in resource
    assert str(export) not in json.dumps(result)
    snapshot = store.load(
        project_id="project-data-workbench", digest=resource["snapshot_sha256"]
    )
    assert snapshot["payload"]["eligibility_flow"]["final_count"] == 120
    assert snapshot["payload"]["selected_feature_distributions"][0]["id"] == "vitals:hr"


def test_patient_timeline_stays_browser_only_and_uses_pseudonymous_ordinal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "export"
    export.mkdir()
    store = CopilotDataWorkbenchSnapshotStore(tmp_path / "snapshots")
    monkeypatch.setattr(tool_owner.sources, "load_registry", lambda: _registry(export))
    monkeypatch.setattr(
        patient_drilldown,
        "patient_review_entity_page",
        lambda _body: {
            "navigation": {"options": [{"ordinal": 3, "ref": "ent_browser_only"}]}
        },
    )
    monkeypatch.setattr(
        patient_drilldown,
        "patient_review_drilldown",
        lambda _body: {
            "source": {"id": "source-mimic", "label": "MIMIC-IV export"},
            "summary": {"entities": 120},
            "selected": {"label": "Entity 3", "ref": "ent_browser_only"},
            "time_lanes": [
                {
                    "lane": "vitals",
                    "status": "ready",
                    "signals": [
                        {"feature": "hr", "times": [0, 1], "values": [80, 92]}
                    ],
                }
            ],
            "patient_overview": {},
            "trajectory_review": {},
            "quality_metrics": {},
            "eligibility_flow": {},
            "blocked_features": [],
            "provenance": {"payload_scope": "aggregate_plus_one_entity"},
            "privacy": {"direct_identifiers_returned": False},
        },
    )
    monkeypatch.setattr(tool_owner, "CopilotDataWorkbenchSnapshotStore", lambda: store)

    result = tool_owner._review_patient_timeline(
        _context(), {"source_id": "source-mimic", "entity_ordinal": 3}
    )

    model_receipt = json.dumps(result)
    assert result["code"] == "easyicu_patient_timeline_ready"
    assert "ent_browser_only" not in model_receipt
    assert '"times"' not in model_receipt
    resource = result["details"]["resource"]
    snapshot = store.load(
        project_id="project-data-workbench", digest=resource["snapshot_sha256"]
    )
    assert snapshot["payload"]["selected"]["ref"] == "ent_browser_only"
    assert snapshot["payload"]["time_lanes"][0]["signals"][0]["values"] == [80, 92]


def test_crossdb_tool_keeps_paths_host_side_and_seals_aggregate_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "mimic"
    second = tmp_path / "eicu"
    first.mkdir()
    second.mkdir()
    registry = {
        "sources": [
            {"id": "mimic", "path": str(first), "ok": True},
            {"id": "eicu", "path": str(second), "ok": True},
        ]
    }
    store = CopilotDataWorkbenchSnapshotStore(tmp_path / "snapshots")
    captured: list[dict] = []
    monkeypatch.setattr(tool_owner.sources, "load_registry", lambda: registry)
    monkeypatch.setattr(
        tool_owner.crossdb_review,
        "crossdb_review_summary",
        lambda body: captured.append(dict(body))
        or {
            "source_count": 2,
            "sources": [{"id": "mimic"}, {"id": "eicu"}],
            "selection_receipt": {"selection_digest": "abc"},
            "rows": [{"key": "cohort_size", "values": [100, 80], "delta": 20}],
            "availability": [],
            "feature_density": [],
            "feature_distributions": [],
            "shared_modules": ["vitals"],
            "all_modules": ["vitals"],
            "compatibility_gate": {"status": "compatible"},
            "blocked_features": [],
            "provenance": {"payload_scope": "cross_database_aggregate_only"},
            "privacy": {"raw_rows_returned": False},
        },
    )
    monkeypatch.setattr(tool_owner, "CopilotDataWorkbenchSnapshotStore", lambda: store)

    result = tool_owner._compare_data_sources(
        _context(), {"source_ids": ["mimic", "eicu"]}
    )

    assert captured == [{"paths": [str(first), str(second)]}]
    assert result["code"] == "easyicu_crossdb_comparison_ready"
    assert str(first) not in json.dumps(result)
    resource = result["details"]["resource"]
    snapshot = store.load(
        project_id="project-data-workbench", digest=resource["snapshot_sha256"]
    )
    assert snapshot["payload"]["rows"][0]["delta"] == 20


def test_demo_preparation_requires_extract_grant_before_job_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_owner.demo_sources,
        "get_source",
        lambda source_id: type("Demo", (), {"id": source_id})(),
    )
    submitted: list[str] = []
    monkeypatch.setattr(
        tool_owner.jobs.MANAGER,
        "submit",
        lambda kind, _runner: submitted.append(kind)
        or type("Job", (), {"id": "job-demo", "kind": kind, "status": "queued"})(),
    )
    monkeypatch.setattr(tool_owner.demo_sources, "make_prepare_runner", lambda _id: object())

    blocked = tool_owner._prepare_demo_source(
        _context(), {"source_id": "mimic_iv_demo_v2_2"}
    )
    assert blocked["code"] == "pi_action_authorization_required"
    assert submitted == []

    granted = ToolExecutionContext(
        session=_context().session, allowed_actions=["extract"]
    )
    result = tool_owner._prepare_demo_source(
        granted, {"source_id": "mimic_iv_demo_v2_2"}
    )
    assert result["code"] == "easyicu_demo_source_preparation_submitted"
    assert submitted == ["demo-source-prepare"]
