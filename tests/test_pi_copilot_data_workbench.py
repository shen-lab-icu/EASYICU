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
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiCopilotError,
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


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("easyicu_review_cohort", {}),
        ("easyicu_open_data_download", {}),
        ("easyicu_preview_icd_cohort", {"include_codes": ["A41"]}),
        ("easyicu_review_patient_timeline", {}),
    ],
)
def test_single_source_workbench_tools_require_exact_source_id(
    tool_name: str,
    arguments: dict,
) -> None:
    with pytest.raises(PiCopilotError) as exc_info:
        tool_owner.execute_tool(tool_name, arguments, _context())

    assert exc_info.value.code == "pi_data_source_selection_required"


def test_copilot_schema_and_prompt_forbid_implicit_source_fallback() -> None:
    main = (
        Path(__file__).parents[1]
        / "src/easyicu/webserver/pi_copilot/node_app/src/main.mjs"
    ).read_text(encoding="utf-8")

    for tool_name in (
        "easyicu_review_cohort",
        "easyicu_open_data_download",
        "easyicu_preview_icd_cohort",
        "easyicu_review_patient_timeline",
    ):
        tool = main.split(f'name: "{tool_name}"', 1)[1].split("hostTool", 1)[0]
        assert "source_id: Type.String" in tool
        assert "source_id: Type.Optional" not in tool
    assert (
        "the first reply must say it is ambiguous between MIMIC-III v1.4 "
        "and MIMIC-IV v3.1" in main
    )
    assert "ask one direct choice before any cohort" in main
    assert "call easyicu_list_data_sources directly with that database key" in main
    assert "Never treat a bound, active, demo, or sample source as implicit consent" in main
    assert "ask one direct Extraction authorization question and stop without calling" in main


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


def test_completed_analysis_prepares_native_feature_workbench_from_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "export"
    export.mkdir()
    wrapper = tmp_path / "wrapper"
    run_id = "run_completed"
    run_dir = wrapper / "pipeline" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "analysis_plan.json").write_text(
        json.dumps(
            {
                "design_selection": {
                    "candidates": [
                        {
                            "disposition": "selected",
                            "required_variables": [
                                "stay_id",
                                "lact_max",
                                "death",
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    store = CopilotDataWorkbenchSnapshotStore(tmp_path / "snapshots")
    service = object.__new__(PiCopilotService)
    service.project_store = type(
        "ProjectStore", (), {"resolve": lambda _self, _project: "study-a"}
    )()
    service.data_workbench_snapshot_store = store
    service._assert_project_initialized = lambda project_id: project_id
    service._latest_run_id = lambda _study_id, project_id: run_id
    service._research_run_row = lambda _project_id, _run_id: {
        "project_dir": str(wrapper)
    }
    monkeypatch.setattr(
        "easyicu.webserver.pi_copilot.service.study_contexts.get_context",
        lambda _study_id: {
            "id": "study-a",
            "data_source": {"path": str(export), "database": "miiv"},
        },
    )
    monkeypatch.setattr(
        "easyicu.webserver.dataio.describe_export_source",
        lambda _path: {
            "files": [
                {"module": "blood_gas", "columns": ["stay_id", "lact"]},
                {"module": "outcome", "columns": ["stay_id", "death"]},
            ]
        },
    )
    calls: list[dict] = []
    monkeypatch.setattr(
        "easyicu.webserver.cohort_review.cohort_review_summary",
        lambda body: calls.append(dict(body)) or _cohort_payload(),
    )

    prepared = service.prepare_data_workbench_snapshot(project_id="project-a")

    assert calls == [
        {
            "source_path": str(export),
            "selected_features": ["blood_gas:lact", "outcome:death"],
        }
    ]
    resource = prepared["resource"]
    assert resource["kind"] == "data_workbench_snapshot"
    assert resource["view"] == "feature_distribution"
    snapshot = store.load(
        project_id="project-a", digest=resource["snapshot_sha256"]
    )
    assert snapshot["payload"]["summary"]["cohort_size"] == 120


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


def test_selected_hospital_death_distribution_matches_cohort_owner_semantics() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "stay_id": ["stay-a", "stay-b", "stay-c", "stay-d"],
            "death": [True, None, False, None],
        }
    )

    profile = cohort_review._selected_feature_profile(
        frame,
        {
            "id": "outcome:death",
            "module": "outcome",
            "column": "death",
            "label": "Death",
        },
    )
    [distribution] = cohort_review._selected_feature_distributions(
        ["stay-a", "stay-b", "stay-c", "stay-d"], [profile]
    )

    assert distribution["observed"] == 4
    assert distribution["observed_pct"] == 100.0
    assert distribution["categories"] == [
        {"label": "Positive", "count": 1},
        {"label": "Negative", "count": 3},
        {"label": "Unknown", "count": 0},
    ]


def test_other_binary_feature_missingness_remains_unknown() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "stay_id": ["stay-a", "stay-b"],
            "aki": [True, None],
        }
    )

    profile = cohort_review._selected_feature_profile(
        frame,
        {
            "id": "renal:aki",
            "module": "renal",
            "column": "aki",
            "label": "AKI",
        },
    )
    [distribution] = cohort_review._selected_feature_distributions(
        ["stay-a", "stay-b"], [profile]
    )

    assert distribution["observed"] == 1
    assert distribution["categories"][-1] == {"label": "Unknown", "count": 1}


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
                        {
                            "feature": "resp",
                            "unit": "/min",
                            "times": [0, 1],
                            "values": [18, 22],
                        }
                    ],
                }
            ],
            "patient_overview": {
                "dashboard": {
                    "trend_panels": [{"cards": [{"unit": "/min"}]}]
                }
            },
            "trajectory_review": {},
            "feature_coverage": {"modules": []},
            "quality_metrics": {},
            "eligibility_flow": {},
            "blocked_features": [],
            "provenance": {"payload_scope": "aggregate_plus_one_entity"},
            "privacy": {"direct_identifiers_returned": False},
        },
    )
    monkeypatch.setattr(
        patient_drilldown,
        "patient_review_feature",
        lambda body: {
            "feature": {"feature": body["feature"], "module": "vitals"},
            "status": "numeric_trajectory",
            "signal": {
                "feature": body["feature"],
                "unit": "mmol/L",
                "times": [0, 1],
                "values": [1.2, 2.4],
            },
        },
    )
    monkeypatch.setattr(tool_owner, "CopilotDataWorkbenchSnapshotStore", lambda: store)

    result = tool_owner._review_patient_timeline(
        _context(),
        {
            "source_id": "source-mimic",
            "entity_ordinal": 3,
            "features": ["lact"],
        },
    )

    model_receipt = json.dumps(result)
    assert result["code"] == "easyicu_patient_timeline_ready"
    assert "ent_browser_only" not in model_receipt
    assert '"times"' not in model_receipt
    assert result["details"]["loaded_trajectory_count"] == 1
    resource = result["details"]["resource"]
    snapshot = store.load(
        project_id="project-data-workbench", digest=resource["snapshot_sha256"]
    )
    assert snapshot["payload"]["selected"]["ref"] == "ent_browser_only"
    signal = snapshot["payload"]["time_lanes"][0]["signals"][0]
    assert signal["values"] == [18, 22]
    assert signal["unit"] == "1/min"
    assert snapshot["payload"]["feature_coverage"] == {"modules": []}
    assert snapshot["payload"]["loaded_feature_details"][0]["signal"]["values"] == [
        1.2,
        2.4,
    ]
    assert (
        snapshot["payload"]["patient_overview"]["dashboard"]["trend_panels"][0][
            "cards"
        ][0]["unit"]
        == "1/min"
    )


def test_patient_timeline_rejects_unknown_leading_slash_unit(
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
            "navigation": {"options": [{"ordinal": 1, "ref": "ent_browser_only"}]}
        },
    )
    monkeypatch.setattr(
        patient_drilldown,
        "patient_review_drilldown",
        lambda _body: {
            "source": {"id": "source-mimic", "label": "MIMIC-IV export"},
            "summary": {"entities": 120},
            "selected": {"label": "Entity 1", "ref": "ent_browser_only"},
            "time_lanes": [
                {
                    "lane": "vitals",
                    "signals": [
                        {
                            "feature": "resp",
                            "unit": "/private/not-a-clinical-unit",
                            "times": [0],
                            "values": [18],
                        }
                    ],
                }
            ],
            "privacy": {"direct_identifiers_returned": False},
        },
    )
    monkeypatch.setattr(tool_owner, "CopilotDataWorkbenchSnapshotStore", lambda: store)

    with pytest.raises(PiCopilotError) as blocked:
        tool_owner._review_patient_timeline(
            _context(), {"source_id": "source-mimic", "entity_ordinal": 1}
        )

    assert blocked.value.code == "copilot_data_workbench_path_forbidden"


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
            "feature_distributions": [
                {
                    "module": "vitals",
                    "features": [
                        {
                            "feature": "hr",
                            "values": [
                                {"source": "mimic", "present": True, "median": 84},
                                {"source": "eicu", "present": True, "median": 88},
                            ],
                        },
                        {
                            "feature": "resp",
                            "values": [
                                {"source": "mimic", "present": True, "median": 18},
                                {"source": "eicu", "present": True, "median": 20},
                            ],
                        },
                    ],
                }
            ],
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
        _context(),
        {"source_ids": ["mimic", "eicu"], "features": ["vitals:hr"]},
    )

    assert captured == [{"paths": [str(first), str(second)]}]
    assert result["code"] == "easyicu_crossdb_comparison_ready"
    assert str(first) not in json.dumps(result)
    resource = result["details"]["resource"]
    snapshot = store.load(
        project_id="project-data-workbench", digest=resource["snapshot_sha256"]
    )
    assert snapshot["payload"]["rows"][0]["delta"] == 20
    assert snapshot["payload"]["feature_distributions"] == [
        {
            "module": "vitals",
            "features": [
                {
                    "feature": "hr",
                    "values": [
                        {"source": "mimic", "present": True, "median": 84},
                        {"source": "eicu", "present": True, "median": 88},
                    ],
                }
            ],
        }
    ]


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
