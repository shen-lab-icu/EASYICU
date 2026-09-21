from __future__ import annotations

import ast
from pathlib import Path

from fastapi.testclient import TestClient

from easyicu.webserver import capabilities
from easyicu.webserver import settings as settings_store
from easyicu.webserver.app import app


def _settings(**patch):
    return {**settings_store.DEFAULTS, **patch}


def test_capability_status_reflects_settings_and_tool_policy(monkeypatch) -> None:
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(
            connector_pubmed_enabled=False,
            connector_zotero_enabled=False,
            mcp_tools_enabled=False,
            remote_compute_enabled=False,
        ),
    )

    response = TestClient(app).get("/api/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["settings"]["connector_pubmed_enabled"] is False
    assert body["capabilities"]["pubmed_connector"]["status"] == "disabled"
    assert body["capabilities"]["zotero_connector"]["status"] == "disabled"
    mcp = body["capabilities"]["mcp_tools"]
    assert "agent_artifact_reader" in mcp["allowed_tools"]
    assert "pubmed_metadata_search" in mcp["blocked_tools"]
    assert body["capabilities"]["remote_compute"]["status"] == "disabled"


def test_publication_skill_capabilities_default_on_and_respect_each_switch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(
            science_skills_enabled=True,
            nature_figure_skill_enabled=False,
            nature_writing_skill_enabled=True,
        ),
    )

    body = TestClient(app).get("/api/capabilities").json()
    publication = body["capabilities"]["publication_skills"]
    by_id = {row["id"]: row for row in publication["items"]}

    assert by_id["nature-figure"]["enabled"] is False
    assert by_id["nature-writing"]["enabled"] is True
    assert publication["active_skill_ids"] == ["nature-writing"]


def test_method_skill_catalog_projects_registered_method_contracts(monkeypatch) -> None:
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(science_skills_enabled=True),
    )

    body = TestClient(app).get("/api/capabilities").json()
    catalog = body["capabilities"]["method_skills"]
    by_id = {row["id"]: row for row in catalog["items"]}
    components = {row["id"]: row for row in catalog["components"]}

    assert len(by_id) == 19
    assert len(components) == 45
    assert catalog["workflow_count"] == 10
    assert catalog["analysis_module_count"] == 9
    assert catalog["builtin_skill_count"] == 19
    assert catalog["available_method_count"] == 45
    assert catalog["planned_method_count"] == 7
    assert by_id["survival-time-to-event"]["capability_id"] == (
        "survival_time_to_event_v1"
    )
    assert by_id["survival-time-to-event"]["action_ids"] == [
        "time_to_event.cox_hr",
        "time_to_event.km_logrank",
        "time_to_event.ph_check",
    ]
    assert by_id["survival-time-to-event"]["claim_ceiling"] == "reportable"
    assert by_id["target-trial-emulation"]["claim_ceiling"] == "analysis_only"
    assert by_id["cohort-characterization-table-one"]["layer"] == "analysis_module"
    assert by_id["trajectory-phenotyping"]["layer"] == "research_workflow"
    assert by_id["trajectory-phenotyping"]["included_module_ids"] == [
        "cohort-characterization-table-one",
        "missingness-measurement-audit",
    ]
    assert "phenotyping.early_subtype_assignment" in by_id[
        "trajectory-phenotyping"
    ]["action_ids"]
    assert by_id["trajectory-phenotyping"]["claim_ceiling"] == "analysis_only"
    assert by_id["adjusted-exposure-outcome-study"]["included_module_ids"] == [
        "cohort-characterization-table-one",
        "missingness-measurement-audit",
        "exposure-outcome-distribution",
    ]
    assert len(by_id["adjusted-exposure-outcome-study"]["workflow_steps"]) == 6
    assert by_id["ordinal-dose-response-study"]["claim_ceiling"] == "analysis_only"
    assert by_id["fixed-landmark-association-study"]["claim_ceiling"] == (
        "reportable"
    )
    assert by_id["time-varying-exposure-survival-study"]["capability_id"] == (
        "association_time_varying_exposure_v1"
    )
    assert components["phenotyping.early_subtype_assignment"]["kernel_modules"] == [
        "subtype_assignment"
    ]
    assert components["prediction.decision_curve"]["implementation"] == "llm_coded"
    assert components["prediction.decision_curve"]["claim_ceiling"] == "analysis_only"
    promoted = {
        "time_to_event.competing_risks_cif": ["competing_risks", "gray_test"],
        "causal_emulation.propensity_adjustment": ["propensity_weighting"],
        "causal_emulation.doubly_robust": ["survival_inputs", "doubly_robust"],
        "causal_emulation.mediation": ["mediation"],
        "association.rcs_spline": ["rcs_dose_response"],
        "prediction.reclassification": ["reclassification"],
    }
    for component_id, kernel_modules in promoted.items():
        assert components[component_id]["implementation"] == "llm_coded"
        assert components[component_id]["kernel_modules"] == kernel_modules
    assert components["time_to_event.cox_hr"]["implementation"] == "deterministic"
    assert components["time_to_event.cox_hr"]["method_family"] == "time_to_event"
    assert all("owner_module" not in row for row in by_id.values())
    assert all("runner" not in row for row in components.values())
    assert catalog["active_skill_ids"] == list(by_id)
    assert catalog["active_component_ids"] == list(components)


def test_method_skill_catalog_respects_science_skills_master_switch(monkeypatch) -> None:
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(science_skills_enabled=False),
    )

    catalog = TestClient(app).get("/api/capabilities").json()["capabilities"][
        "method_skills"
    ]
    assert catalog["enabled"] is False
    assert catalog["active_skill_ids"] == []
    assert catalog["active_component_ids"] == []
    assert all(row["enabled"] is False for row in catalog["items"])
    assert all(row["enabled"] is False for row in catalog["components"])


def test_builtin_method_skill_package_exposes_reviewed_documentation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(science_skills_enabled=True),
    )

    response = TestClient(app).get(
        "/api/capabilities/method-skills/survival-time-to-event/package"
    )

    assert response.status_code == 200
    package = response.json()
    assert package["schema_version"] == "easyicu.method-skill-package/3"
    assert package["skill_id"] == "survival-time-to-event"
    assert package["kind"] == "research_workflow"
    assert package["read_only"] is True
    assert package["execution_boundary"] == "documentation_only_host_execution_required"
    files = {row["path"]: row for row in package["files"]}
    assert set(files) == {
        "SKILL.md",
        "references/composition.md",
        "references/validation_framework.md",
        "references/workflow_contract.md",
    }
    assert "capability: `survival_time_to_event_v1`".lower() in files[
        "SKILL.md"
    ]["content"].lower()
    skill_markdown = files["SKILL.md"]["content"]
    assert "## When to use this workflow" in skill_markdown
    assert "## Required data and decisions" in skill_markdown
    assert "## Workflow" in skill_markdown
    assert "## Methods and implementation" in skill_markdown
    assert "## Validation and quality checks" in skill_markdown
    assert "## Evidence and claim boundary" in skill_markdown
    assert "## Failure behavior" in skill_markdown
    assert "## Start this workflow" in skill_markdown
    assert len(skill_markdown.encode("utf-8")) > 2500
    assert all(not path.startswith("scripts/") for path in files)
    assert all(len(row["sha256"]) == 64 for row in files.values())
    assert len(package["package_sha256"]) == 64


def test_builtin_method_component_package_and_unknown_id(monkeypatch) -> None:
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(science_skills_enabled=True),
    )
    client = TestClient(app)

    component = client.get(
        "/api/capabilities/method-skills/prediction.decision_curve/package"
    )
    missing = client.get(
        "/api/capabilities/method-skills/not-a-real-skill/package"
    )

    assert component.status_code == 200
    payload = component.json()
    assert payload["kind"] == "method_component"
    assert {row["path"] for row in payload["files"]} == {
        "SKILL.md",
        "references/method_contract.md",
        "references/validation_framework.md",
    }
    markdown = next(row for row in payload["files"] if row["path"] == "SKILL.md")
    assert "`prediction.decision_curve`" in markdown["content"]
    assert "## When to use this method" in markdown["content"]
    assert "## Method and implementation" in markdown["content"]
    assert "does not define the cohort, estimand, or complete article workflow" in (
        markdown["content"]
    )
    assert missing.status_code == 404
    assert missing.json()["detail"]["error"] == "method_skill_package_not_found"


def test_trajectory_workflow_package_separates_discovery_and_early_assignment(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(science_skills_enabled=True),
    )

    package = TestClient(app).get(
        "/api/capabilities/method-skills/trajectory-phenotyping/package"
    ).json()
    files = {row["path"]: row for row in package["files"]}

    assert package["kind"] == "research_workflow"
    assert "scripts/adapter.py" not in files
    assert "references/trajectory_assignment.md" in files
    assert "scripts/early_subtype_assignment.py" in files
    assert "freeze" in files["references/trajectory_assignment.md"]["content"].lower()
    assert "### Trajectory discovery and early assignment" in files["SKILL.md"][
        "content"
    ]
    assert "patient-disjoint development and validation sets" in files["SKILL.md"][
        "content"
    ]
    assert "fit_and_evaluate_early_subtype_assignment" in files[
        "scripts/early_subtype_assignment.py"
    ]["content"]
    ast.parse(files["scripts/early_subtype_assignment.py"]["content"])


def test_composed_association_workflow_package_exposes_ordered_project_phases(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(science_skills_enabled=True),
    )

    package = TestClient(app).get(
        "/api/capabilities/method-skills/adjusted-exposure-outcome-study/package"
    ).json()
    files = {row["path"]: row for row in package["files"]}

    assert package["kind"] == "research_workflow"
    assert set(files) == {
        "SKILL.md",
        "references/composition.md",
        "references/validation_framework.md",
        "references/workflow_contract.md",
    }
    skill = files["SKILL.md"]["content"]
    phases = [
        "Freeze the cohort, exposure, outcome, time zero",
        "Describe the cohort with Table 1",
        "Report exposure prevalence and outcome absolute risks",
        "Fit the exact typed adjusted model",
        "Run the registered diagnostics and sensitivity analyses",
        "Publish the descriptive and adjusted products",
    ]
    assert all(phase in skill for phase in phases)
    assert [skill.index(phase) for phase in phases] == sorted(
        skill.index(phase) for phase in phases
    )
    composition = files["references/composition.md"]["content"]
    assert "cohort-characterization-table-one" in composition
    assert "missingness-measurement-audit" in composition
    assert "exposure-outcome-distribution" in composition


def test_capability_tool_check_blocks_unknown_and_external_tools(monkeypatch) -> None:
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(mcp_tools_enabled=False, connector_pubmed_enabled=True),
    )
    client = TestClient(app)

    external = client.post(
        "/api/capabilities/tool-check",
        json={"tool_id": "pubmed_metadata_search"},
    ).json()
    unknown = client.post(
        "/api/capabilities/tool-check",
        json={"tool_id": "made_up_tool"},
    ).json()
    local = client.post(
        "/api/capabilities/tool-check",
        json={"tool_id": "agent_artifact_reader"},
    ).json()

    assert external["allowed"] is False
    assert external["reason"] == "mcp_tools_enabled_false"
    assert unknown["allowed"] is False
    assert unknown["reason"] == "unknown_tool"
    assert local["allowed"] is True


def test_zotero_search_fails_closed_when_connector_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(connector_zotero_enabled=False),
    )

    response = TestClient(app).post(
        "/api/capabilities/zotero/search",
        json={"query": "sepsis", "limit": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["blocked"] is True
    assert body["status"]["reason"] == "connector_zotero_enabled_false"
    assert body["items"] == []


def test_zotero_connection_test_records_audit_event(
    tmp_path: Path, monkeypatch
) -> None:
    audit_path = tmp_path / "capability_tool_audit.jsonl"
    monkeypatch.setattr(capabilities, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(capabilities, "_AUDIT_PATH", audit_path)
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(
            connector_zotero_enabled=False,
            tool_audit_enabled=True,
        ),
    )

    response = TestClient(app).post("/api/capabilities/zotero/test", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["status"]["reason"] == "connector_zotero_enabled_false"
    events = capabilities.audit_events(limit=5)["events"]
    assert events[-1]["event_type"] == "zotero_connection_test"
    assert events[-1]["detail"]["available"] is False


def test_zotero_source_maps_item_into_idea_payload(
    tmp_path: Path, monkeypatch
) -> None:
    audit_path = tmp_path / "capability_tool_audit.jsonl"
    monkeypatch.setattr(capabilities, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(capabilities, "_AUDIT_PATH", audit_path)
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(
            connector_zotero_enabled=True,
            tool_audit_enabled=True,
        ),
    )
    monkeypatch.setattr(
        capabilities,
        "zotero_status",
        lambda settings=None: {
            "enabled": True,
            "available": True,
            "status": "available",
            "reason": "local_zotero_api_ready",
        },
    )

    response = TestClient(app).post(
        "/api/capabilities/zotero/source",
        json={
            "item": {
                "key": "ABC123",
                "title": "Early Vasopressors in Septic Shock",
                "journal": "Intensive Care Medicine",
                "year": "2026",
                "doi": "10.1000/example",
                "url": "https://example.org/paper",
                "abstract": "Early vasopressors may define a measurable ICU exposure.",
                "citation_key": "smith2026vasopressors",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["blocked"] is False
    assert body["item"]["journal"] == "Intensive Care Medicine"
    suggested = body["suggested_payload"]
    assert suggested["source_type"] == "zotero"
    assert suggested["source_origin"] == "zotero_desktop"
    assert suggested["source_origin_label"] == "Zotero Desktop"
    assert suggested["title"] == "Early Vasopressors in Septic Shock"
    assert suggested["topic"] == "Early Vasopressors in Septic Shock"
    assert suggested["doi"] == "10.1000/example"
    assert suggested["citation_key"] == "smith2026vasopressors"
    assert suggested["zotero_key"] == "ABC123"
    assert "Early vasopressors" in suggested["excerpt"]
    assert body["source_adapter"]["status"] == "literature_source_ready"
    assert body["source_adapter"]["source_origin"] == "zotero_desktop"
    assert body["source_adapter"]["display_status"] == (
        "Literature source ready / 文献来源已就绪"
    )
    assert body["privacy"]["full_text_stored"] is False
    events = capabilities.audit_events(limit=5)["events"]
    assert events[-1]["event_type"] == "zotero_source_selected"


def test_zotero_paste_import_builds_source_without_connector(
    tmp_path: Path, monkeypatch
) -> None:
    audit_path = tmp_path / "capability_tool_audit.jsonl"
    monkeypatch.setattr(capabilities, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(capabilities, "_AUDIT_PATH", audit_path)
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(
            connector_zotero_enabled=False,
            tool_audit_enabled=True,
        ),
    )

    response = TestClient(app).post(
        "/api/capabilities/zotero/import",
        json={
            "text": """@article{smith2026shock,
              title={Early Vasopressors in Septic Shock},
              journal={Intensive Care Medicine},
              year={2026},
              doi={10.1000/example},
              abstract={Early vasopressors may define a measurable ICU exposure.}
            }"""
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["blocked"] is False
    assert body["source_adapter"]["status"] == "literature_source_ready"
    assert body["source_adapter"]["source_origin"] == "pasted_literature"
    assert body["source_adapter"]["display_status"] == (
        "Literature source ready / 文献来源已就绪"
    )
    assert "no Zotero setup is required" in body["source_adapter"]["display_reason"]
    suggested = body["suggested_payload"]
    assert suggested["source_type"] == "zotero"
    assert suggested["source_origin"] == "pasted_literature"
    assert suggested["source_origin_label"] == "Pasted literature metadata"
    assert suggested["title"] == "Early Vasopressors in Septic Shock"
    assert suggested["journal"] == "Intensive Care Medicine"
    assert suggested["year"] == "2026"
    assert suggested["doi"] == "10.1000/example"
    assert suggested["citation_key"] == "smith2026shock"
    assert suggested["zotero_key"] == "smith2026shock"
    assert "Early vasopressors" in suggested["excerpt"]
    assert body["privacy"]["full_text_stored"] is False
    events = capabilities.audit_events(limit=5)["events"]
    assert events[-1]["event_type"] == "zotero_paste_import"


def test_tool_audit_ledger_respects_setting(
    tmp_path: Path, monkeypatch
) -> None:
    audit_path = tmp_path / "capability_tool_audit.jsonl"
    monkeypatch.setattr(capabilities, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(capabilities, "_AUDIT_PATH", audit_path)

    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(tool_audit_enabled=True),
    )
    recorded = capabilities.record_tool_event("unit_test_event", {"value": 1})
    assert recorded["recorded"] is True
    assert audit_path.exists()
    assert capabilities.audit_events(limit=10)["count"] == 1

    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(tool_audit_enabled=False),
    )
    skipped = capabilities.record_tool_event("unit_test_event", {"value": 2})
    assert skipped["recorded"] is False
    assert capabilities.audit_events(limit=10)["count"] == 1


def test_remote_compute_policy_blocks_non_local_targets(monkeypatch) -> None:
    monkeypatch.setattr(
        settings_store,
        "load_settings",
        lambda: _settings(remote_compute_enabled=False),
    )

    blocked = capabilities.validate_compute_target({"compute_target": "hpc"})
    local = capabilities.validate_compute_target({"compute_target": "local"})

    assert blocked["ok"] is False
    assert blocked["error"] == "remote_compute_disabled"
    assert local["ok"] is True
    assert local["compute_target"] == "local"
