"""Governance contracts for Idea Mining and provider configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import provider_adapter
from easyicu.webserver import settings as settings_store
from easyicu.webserver.app import app
from easyicu.webserver.ideas import mining as idea_mining


def test_design_excerpt_preserves_late_outcome_axis_after_exposure_synonyms() -> None:
    excerpt = idea_mining._study_design_excerpt(
        (
            "SOFA-2 updates the Sequential Organ Failure Assessment score. "
            "Sepsis-3 criteria can use SOFA-2 for organ dysfunction. "
            "We conducted a retrospective multicenter ICU cohort. "
            "Adults with suspected infection were included. "
            "The primary outcome was ICU mortality; hospital mortality was a "
            "prespecified secondary outcome."
        ),
        focus_terms=(
            "SOFA-2",
            "Sepsis-3",
            "SOFA",
            "Sequential Organ Failure Assessment",
            "hospital mortality",
        ),
    )

    assert "SOFA-2" in excerpt
    assert "Sepsis-3" in excerpt
    assert "hospital mortality" in excerpt


def test_idea_mining_maps_ards_peep_pdf_excerpt_to_respiratory_concepts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(idea_mining, "_RUN_ROOT", tmp_path / "idea_runs")
    monkeypatch.setattr(idea_mining, "_HISTORY_PATH", tmp_path / "idea_history.json")
    monkeypatch.setattr(idea_mining, "_active_export", lambda: None)

    response = TestClient(app).post(
        "/api/ideas/mine",
        json={
            "source_type": "pdf",
            "title": "Balancing lung recruitment and venous congestion in ARDS: rethinking PEEP",
            "journal": "Intensive Care Medicine",
            "year": 2026,
            "excerpt": (
                "Positive end-expiratory pressure (PEEP) is a cornerstone of mechanical "
                "ventilation in acute respiratory distress syndrome. Ventilator settings "
                "and oxygenation may be associated with ICU mortality."
            ),
            "allow_network": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    idea = payload["idea_ledger"][0]
    concept_ids = {row["concept_id"] for row in idea["mapped_concepts"]}
    assert {"mech_vent", "peep", "death"} <= concept_ids
    assert idea["prior_art"]["status"] == "not_checked_external_search_required"
    assert payload["pre_experiment"]["status"] == "blocked"
    assert payload["privacy"]["external_llm_calls"] == 0
    assert payload["privacy"]["network_calls"] == 0
    assert "stay_id" not in str(payload)
    assert "subject_id" not in str(payload)


def test_provider_config_route_writes_private_env_without_returning_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / "provider.env"
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(provider_adapter, "_DEFAULT_PROVIDER_ENV_FILE", env_path)
    monkeypatch.setattr(settings_store, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(settings_store, "_CONFIG_PATH", settings_path)

    response = TestClient(app).post(
        "/api/agent-runs/provider-config",
        json={
            "provider": "openai",
            "api_key": "sk-test-provider-config",
            "base_url": "http://127.0.0.1:8787/v1",
            "model": "gpt5.4",
            "enable_ai": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["secrets_returned"] is False
    assert payload["provider_status"]["ready"] is True
    assert payload["provider_status"]["credential_present"] is True
    assert payload["provider_status"]["base_url_present"] is True
    assert payload["provider_status"]["model_present"] is True
    assert payload["settings"]["ai_enabled"] is True
    assert env_path.exists()
    assert (env_path.stat().st_mode & 0o777) == 0o600
    assert "sk-test-provider-config" in env_path.read_text(encoding="utf-8")
    assert "sk-test-provider-config" not in str(payload)
    assert "http://127.0.0.1:8787/v1" not in str(payload)


def test_provider_config_route_fails_closed_when_enable_ai_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / "provider.env"
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(provider_adapter, "_DEFAULT_PROVIDER_ENV_FILE", env_path)
    monkeypatch.setattr(settings_store, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(settings_store, "_CONFIG_PATH", settings_path)

    response = TestClient(app).post(
        "/api/agent-runs/provider-config",
        json={
            "provider": "openai",
            "api_key": "sk-test-fail-closed",
            "base_url": "http://127.0.0.1:8787/v1",
            "model": "gpt5.4",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["settings"]["ai_enabled"] is False
    # ``agent_model_mode`` was asserted here too, but it was a write-only
    # mirror of this same gate that nothing ever read; ai_enabled is the gate.
    assert "agent_model_mode" not in payload["settings"]


def test_provider_config_rejects_invalid_enable_ai_before_writing_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / "provider.env"
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(provider_adapter, "_DEFAULT_PROVIDER_ENV_FILE", env_path)
    monkeypatch.setattr(settings_store, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(settings_store, "_CONFIG_PATH", settings_path)

    response = TestClient(app).post(
        "/api/agent-runs/provider-config",
        json={
            "provider": "openai",
            "api_key": "sk-must-not-be-written",
            "enable_ai": "sometimes",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "error": "invalid_boolean",
        "field": "enable_ai",
    }
    assert not env_path.exists()
    assert not settings_path.exists()


def test_blocked_prior_art_recheck_does_not_clobber_successful_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A casual re-check without network opt-in must not re-block the seed."""
    monkeypatch.setattr(idea_mining, "_RUN_ROOT", tmp_path)
    run_id = "priorartkeep"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)
    successful = {
        "ok": True,
        "run_id": run_id,
        "idea_id": "vasopressor-timing",
        "prior_art": {
            "status": "searched_no_hits",
            "search_performed": True,
            "results": [],
        },
    }
    (run_dir / "prior_art_check.json").write_text(
        json.dumps(successful, ensure_ascii=False), encoding="utf-8"
    )

    blocked = idea_mining.check_prior_art(
        {
            "run_id": run_id,
            "idea_id": "vasopressor-timing",
            "idea_title": "vasopressor timing",
            "allow_network": False,
        }
    )

    assert blocked["prior_art"]["search_performed"] is False
    assert blocked["persisted"] is False
    assert blocked["retained_prior_art_status"] == "searched_no_hits"
    on_disk = json.loads((run_dir / "prior_art_check.json").read_text(encoding="utf-8"))
    assert on_disk["prior_art"]["search_performed"] is True
    assert on_disk["prior_art"]["status"] == "searched_no_hits"


def test_execution_gate_treats_failed_prior_art_search_as_unreviewed() -> None:
    """status=search_failed returned no reviewable metadata; the gate stays closed."""
    idea = {"go_no_go": "recommend"}
    pre_experiment = {"status": "ready", "missing_required_concepts": []}
    failed_check = {"prior_art": {"status": "search_failed", "search_performed": True}}
    gate = idea_mining._execution_gate(idea, pre_experiment, failed_check)
    assert "run prior-art review before Agent execution" in gate["blockers"]

    reviewed_check = {
        "prior_art": {"status": "searched_no_hits", "search_performed": True}
    }
    gate_ok = idea_mining._execution_gate(
        idea,
        pre_experiment,
        reviewed_check,
        {
            "prior_art_decision": "differentiated",
            "source_feasibility_status": "ready",
            "idea_definition_sha256": "a" * 64,
        },
    )
    assert gate_ok["blockers"] == []
    assert gate_ok["agent_run_ready_after_human_confirmation"] is True


def test_typed_adjudication_and_source_bound_feasibility_form_current_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(idea_mining, "_RUN_ROOT", tmp_path / "idea_runs")
    monkeypatch.setattr(idea_mining, "_HISTORY_PATH", tmp_path / "idea_history.json")
    monkeypatch.setattr(idea_mining, "_active_export", lambda: None)
    run = idea_mining.mine_ideas(
        {
            "source_type": "manual",
            "metadata_only": True,
            "topic": "early lactate and hospital mortality in ventilated adult ICU patients",
            "title": "Lactate idea",
            "excerpt": "Early lactate may identify mortality risk.",
        }
    )
    run_id = run["run_id"]
    idea_id = run["selected_idea_id"]
    fields = {
        "research_question": "Does early lactate relate to hospital mortality?",
        "population": "Mechanically ventilated adult ICU patients",
        "exposure": "Lactate during ICU hours 0-24",
        "outcome": "Hospital mortality",
        "time_zero": "ICU admission",
        "time_window": "Exposure 0-24 hours; follow-up through hospital discharge",
    }
    idea_mining.plan_idea({"run_id": run_id, "idea_id": idea_id, "plan_fields": fields})
    prior = {
        "ok": True,
        "run_id": run_id,
        "idea_id": idea_id,
        "prior_art": {
            "status": "searched",
            "search_performed": True,
            "searched_at": "2026-09-01T12:00:00+00:00",
            "results": [
                {
                    "pmid": "12345",
                    "title": "Early lactate in critical illness",
                    "year": "2024",
                    "direct_comparator_screen": {
                        "disposition": "exclude",
                        "evidence_role": "related_context",
                        "rationale": "The outcome window differs.",
                        "population_match": True,
                        "exposure_match": True,
                        "outcome_match": False,
                        "publication_type_eligible": True,
                    },
                }
            ],
        },
    }
    run_dir = idea_mining._run_dir(run_id)
    (run_dir / "prior_art_check.json").write_text(
        json.dumps(prior, ensure_ascii=False), encoding="utf-8"
    )
    adjudication = idea_mining.adjudicate_prior_art(
        {
            "run_id": run_id,
            "idea_id": idea_id,
            "decision": "differentiated",
            "rationale": "The proposed exposure and hospital-discharge estimand differ.",
        }
    )
    assert adjudication["decision"] == "differentiated"
    assert len(adjudication["comparison_axes"]) == 6

    export_dir = tmp_path / "real_export"
    export_dir.mkdir()
    (export_dir / "blood_gas.csv").write_text(
        "stay_id,charttime,lact\n1,0,2.1\n2,0,3.2\n", encoding="utf-8"
    )
    (export_dir / "outcome.csv").write_text(
        "stay_id,charttime,death\n1,48,0\n2,72,1\n", encoding="utf-8"
    )
    (export_dir / "ventilation.csv").write_text(
        "stay_id,charttime,mech_vent\n1,0,1\n2,0,1\n", encoding="utf-8"
    )
    source = {
        "id": "real-miiv-export",
        "label": "Clinical export",
        "database": "miiv",
        "path": str(export_dir),
    }
    files = [
        {
            "file": "blood_gas.csv",
            "module": "blood_gas",
            "columns": ["stay_id", "charttime", "lact"],
            "rows": 2,
        },
        {
            "file": "outcome.csv",
            "module": "outcome",
            "columns": ["stay_id", "charttime", "death"],
            "rows": 2,
        },
        {
            "file": "ventilation.csv",
            "module": "ventilation",
            "columns": ["stay_id", "charttime", "mech_vent"],
            "rows": 2,
        },
    ]
    desc = {
        "ok": True,
        "path": str(export_dir),
        "database": "miiv",
        "files": files,
        "summary": {"stays": 2, "modules": 3, "total_rows": 6},
    }
    feasibility = idea_mining.bounded_sample_feasibility(
        {
            "run_id": run_id,
            "idea_id": idea_id,
            "require_adjudication": True,
            "concept_bindings": {
                "primary_exposure": "lact",
                "outcome": "death",
                "time_zero": "mech_vent",
                "covariates": [],
            },
            "max_records": 100,
        },
        export=(source, desc),
    )
    assert feasibility["status"] == "ready"
    assert feasibility["design_answerability"] == {
        "time_zero_reconstructable": True,
        "temporal_ordering_reconstructable": True,
        "joint_observed_entities": 2,
        "repeated_measure_density": {"lact": 1.0, "death": 1.0, "mech_vent": 1.0},
    }
    readiness = idea_mining.idea_execution_readiness_binding(
        run_id, idea_id, source_path=export_dir
    )
    assert readiness["execution_ready_for_confirmation"] is True
    assert readiness["prior_art_decision"] == "differentiated"
    dumped = json.dumps(feasibility, ensure_ascii=False)
    for marker in ("stay_id", "subject_id", "hadm_id", str(export_dir)):
        assert marker not in dumped

    idea_mining.plan_idea(
        {
            "run_id": run_id,
            "idea_id": idea_id,
            "plan_fields": {**fields, "outcome": "ICU mortality"},
        }
    )
    with pytest.raises(idea_mining.IdeaMiningWebError) as stale:
        idea_mining.prior_art_adjudication_binding(run_id, idea_id)
    assert stale.value.detail["error"] == "idea_prior_art_adjudication_stale_definition"
