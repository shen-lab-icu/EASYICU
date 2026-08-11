"""Focused owner and fail-closed tests for the Copilot research workflow."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
from easyicu.research_agent.reporting.result_card import (
    build_result_interpretation_card,
)
from easyicu.research_agent.schema import UserPreferences
from easyicu.research_agent.orchestration.workflow import (
    HumanReviewPending,
    HumanReviewRequest,
)
from easyicu.webserver import agent_pipeline_runs, agent_runs, provider_adapter
from easyicu.webserver.literature_projection import (
    literature_source_resource,
    project_run_literature,
)
from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiCopilotError,
    PiSessionRecord,
    ToolExecutionContext,
)
from easyicu.webserver.pi_copilot.workflow import (
    active_export_matches_study,
    build_research_workflow_snapshot,
)


def _complete_study() -> dict[str, Any]:
    return {
        "id": "study-workflow",
        "revision": 4,
        "question": "Does an aggregate ICU feature predict mortality?",
        "data_source": {
            "path": "/private/prepared/source",
            "database": "mimiciv",
        },
        "cohort": {"preset": "adult_icu", "max_patients": 2000},
        "modules": ["vitals", "outcome"],
        "outcome": "In-hospital mortality",
        "primary_exposure": "heart_rate",
        "covariates": ["age", "sex"],
        "time_window": {"hours": 24, "anchor": "ICU admission"},
        "export_format": "parquet",
        "analysis_goal": "Descriptive prognostic association",
    }


def test_workflow_projection_advances_only_from_owner_receipts() -> None:
    empty = build_research_workflow_snapshot(
        study={"id": "study-empty"},
        active_export_present=False,
        active_job=None,
        latest_run=None,
    )
    assert empty.current_stage == "question"
    assert empty.missing_setup_fields == [
        "question",
        "data_source",
        "cohort",
        "modules",
        "outcome",
        "time_window",
        "export_format",
        "analysis_goal",
    ]
    assert next(row for row in empty.stages if row.id == "idea").status == "blocked"

    accepted = build_research_workflow_snapshot(
        study={
            **_complete_study(),
            "idea_handoff": {
                "schema_version": "easyicu.pi-idea-selection/1",
                "run_id": "idea-run-1",
                "idea_id": "idea-lactate",
                "canonical_handoff_sha256": "a" * 64,
                "status": "accepted",
                "go_no_go": "recommend",
            },
        },
        active_export_present=False,
        active_job=None,
        latest_run=None,
    )
    idea_stage = next(row for row in accepted.stages if row.id == "idea")
    assert idea_stage.status == "complete"
    assert idea_stage.reason_code == "idea_handoff_accepted"

    held = build_research_workflow_snapshot(
        study={
            **_complete_study(),
            "idea_handoff": {
                "schema_version": "easyicu.pi-idea-selection/1",
                "run_id": "idea-run-hold",
                "idea_id": "idea-lactate-hold",
                "canonical_handoff_sha256": "d" * 64,
                "status": "accepted",
                "go_no_go": "hold",
                "go_no_go_reason": "Active export feasibility must be refreshed.",
            },
        },
        active_export_present=True,
        active_job=None,
        latest_run=None,
    )
    held_by_id = {row.id: row for row in held.stages}
    assert held_by_id["idea"].status == "review_required"
    assert held_by_id["plan"].status == "blocked"
    assert held_by_id["analysis"].status == "blocked"
    assert held.next_action_code == "idea_feasibility_refresh_required"

    finished = build_research_workflow_snapshot(
        study=_complete_study(),
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "analysis_only",
            "artifact_names": [
                "agent_plan.json",
                "evidence_ledger.json",
                "result_tables.json",
                "manuscript_draft.json",
                "source_run_manifest.json",
            ],
        },
    )
    by_id = {row.id: row for row in finished.stages}
    assert by_id["analysis"].status == "complete"
    assert by_id["interpretation"].status == "review_required"
    assert by_id["manuscript"].status == "review_required"
    assert finished.next_action_code == "human_review_and_reporting"


def test_workflow_projection_keeps_plan_review_before_analysis() -> None:
    snapshot = build_research_workflow_snapshot(
        study=_complete_study(),
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "run_status": "human_review_pending",
            "pending_review_reason_codes": [
                "operator_plan_approval_required"
            ],
            "artifact_names": [
                "agent_plan.json",
                "evidence_ledger.json",
                "source_run_manifest.json",
            ],
        },
    )

    by_id = {row.id: row for row in snapshot.stages}
    assert snapshot.current_stage == "plan"
    assert snapshot.next_action_code == "operator_plan_approval_required"
    assert snapshot.completed_required_stages == 3
    assert by_id["plan"].status == "review_required"
    assert by_id["plan"].reason_code == "operator_plan_approval_required"
    assert by_id["analysis"].status == "blocked"
    assert by_id["analysis"].reason_code == "operator_plan_approval_required"


def test_active_export_must_belong_to_the_bound_study() -> None:
    study = _complete_study()
    assert active_export_matches_study(study, "/private/prepared/source") is True
    assert active_export_matches_study(study, "/private/another/export") is False
    assert active_export_matches_study({"id": "study-without-source"}, "/active") is False


def test_legacy_full_scaffold_does_not_claim_scientific_analysis_complete() -> None:
    snapshot = build_research_workflow_snapshot(
        study=_complete_study(),
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_type": "full",
            "engine": "native_summary",
            "gate_status": "analysis_only",
            "artifact_names": [
                "agent_plan.json",
                "evidence_ledger.json",
                "table1_summary.json",
                "manuscript_draft.json",
            ],
        },
    )

    by_id = {row.id: row for row in snapshot.stages}
    assert by_id["analysis"].status == "ready"
    assert by_id["analysis"].reason_code == "research_pipeline_required"
    assert by_id["interpretation"].status == "blocked"
    assert by_id["manuscript"].status == "blocked"


def test_result_interpretation_card_reuses_agent_claims_without_new_numbers() -> None:
    card = build_result_interpretation_card(
        run_id="run_safe",
        review={
            "gate": {
                "status": "analysis_only",
                "reason": "Human review remains required.",
                "checks": [{"name": "human_signoff", "passed": False}],
            },
            "readiness": {
                "status": "awaiting_human_signoff",
                "reportable": False,
            },
            "artifacts": [
                {"name": "table1_summary.json"},
                {"name": "manuscript_draft.json"},
            ],
        },
        manuscript={
            "claims": [
                {
                    "text": "The bounded Research Agent claim is analysis-only.",
                    "evidence_ids": ["ev_table1"],
                }
            ]
        },
    )
    assert card.status == "analysis_only"
    assert card.generated_numbers is False
    assert card.source == "research_agent_artifacts_only"
    assert card.claims[0].evidence_ids == ["ev_table1"]
    assert card.human_review_required is True


def test_idea_tool_never_accepts_a_host_path_from_the_model() -> None:
    context = ToolExecutionContext(
        session=PiSessionRecord(session_id="pi-idea"),
        allowed_actions={"idea"},
    )
    with pytest.raises(PiCopilotError) as rejected:
        tool_module.execute_tool(
            "easyicu_mine_ideas",
            {"topic": "Aggregate ICU question", "path": "/private/source"},
            context,
        )
    assert rejected.value.code == "pi_tool_unknown_arguments"


def test_curated_literature_projection_is_honest_and_does_not_backfill_plan_links() -> None:
    payload = project_run_literature(
        run_id="run-literature-1",
        bundle={
            "research_question": "Does an ICU exposure predict mortality?",
            "citations": [
                {
                    "key": "strobe_2007",
                    "title": "STROBE statement",
                    "year": "2007",
                    "venue": "BMJ",
                    "relevance": "Observational reporting guidance.",
                }
            ],
            "prisma": None,
            "search_provenance": {
                "curated_seed_count": 1,
                "sources_enabled": [],
                "sources_returning": [],
                "search_conducted": False,
                "note": "No retrieval source was enabled.",
            },
        },
        plan={
            "steps": [
                {
                    "step_id": "01_primary",
                    "planned_analysis_role": "primary",
                    "intent": "Fit the prespecified model.",
                }
            ]
        },
    )

    assert payload["status"] == "curated_only"
    assert payload["search"]["search_conducted"] is False
    assert payload["search"]["prisma"] is None
    assert payload["mapping_status"] == "not_bound"
    assert payload["step_citation_map"][0]["citation_keys"] == []
    assert payload["integrity"]["patient_rows_returned"] is False


def test_plan_literature_projection_keeps_only_bundle_bound_keys() -> None:
    payload = project_run_literature(
        run_id="run-literature-2",
        bundle={
            "citations": [
                {
                    "key": "method_key",
                    "title": "A real method paper",
                    "pmid": "12345",
                }
            ],
            "search_provenance": {
                "curated_seed_count": 0,
                "sources_enabled": ["pubmed"],
                "sources_returning": ["pubmed"],
                "search_conducted": True,
            },
        },
        plan={
            "steps": [
                {
                    "step_id": "primary",
                    "planned_analysis_role": "primary",
                    "intent": "Estimate the primary association.",
                    "literature_citation_keys": ["method_key", "invented_key"],
                }
            ]
        },
    )

    assert payload["status"] == "searched"
    assert payload["mapping_status"] == "complete"
    assert payload["step_citation_map"][0]["citation_keys"] == ["method_key"]
    assert payload["integrity"]["unknown_citation_keys_removed"] == [
        "invented_key"
    ]
    assert (
        payload["citations"][0]["source_url"]
        == "https://pubmed.ncbi.nlm.nih.gov/12345/"
    )


def test_literature_search_tool_uses_separate_one_turn_network_grant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: _complete_study())
    monkeypatch.setattr(
        tool_module.idea_mining,
        "discover_literature",
        lambda body: {
            "status": "searched",
            "search_performed": True,
            "queries_to_run": ["ICU mortality"],
            "network_calls": 2,
            "source_candidates": [
                {
                    "citation_key": "paper_12345",
                    "title": "A source-backed ICU study",
                    "journal": "Critical Care",
                    "year": 2025,
                    "pmid": "12345",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
                    "evidence_quote": "The abstract describes an ICU cohort.",
                }
            ],
        },
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(session_id="pi-literature"),
        allowed_actions={"literature"},
    )

    result = tool_module.execute_tool("easyicu_search_literature", {}, context)

    assert result["code"] == "easyicu_literature_search_completed"
    assert result["details"]["literature_search"]["search_performed"] is True
    assert result["details"]["resource"]["kind"] == "literature_source"
    assert result["details"]["resource"]["pmid"] == "12345"
    consumed = tool_module.execute_tool("easyicu_search_literature", {}, context)
    assert consumed["code"] == "pi_action_grant_consumed"


def test_literature_source_resource_rejects_unverified_or_unsafe_links() -> None:
    assert (
        literature_source_resource(
            {"title": "Unsafe", "url": "javascript:alert(1)"}
        )
        is None
    )
    assert literature_source_resource({"title": "No identifier"}) is None


def test_accept_idea_handoff_binds_digest_and_projects_study_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _complete_study()
    current.update({"title": "Old study", "revision": 9, "active_job_id": None})
    writes: list[tuple[dict[str, Any], dict[str, Any]]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: current)
    monkeypatch.setattr(
        tool_module.idea_mining,
        "plan_idea",
        lambda body: {
            "plan": {
                "research_question": "Does early lactate predict hospital mortality?",
                "analysis_family": "prognostic association",
                "exposure": "peak lactate",
                "comparator": "per 1 mmol/L",
                "outcome": "In-hospital mortality",
            }
        },
    )
    monkeypatch.setattr(
        tool_module.idea_mining,
        "create_handoff",
        lambda body: {
            "created_at": "2026-08-11T12:00:00Z",
            "idea_id": "idea-lactate",
            "candidate_topic": "Early lactate and mortality",
            "canonical_handoff_sha256": "b" * 64,
            "canonical_handoff": {
                "analysis_family": "association_study",
                "resolved_predictor_concept": "lact",
                "resolved_outcome_concept": "death",
                "resolved_analysis_concepts": ["lact", "age", "sex"],
                "selected_ledger_row": {
                    "mapped_concepts": [
                        {"concept_id": "lact", "module": "blood_gas"},
                        {"concept_id": "death", "module": "outcome"},
                        {"concept_id": "age", "module": "demographics"},
                        {"concept_id": "sex", "module": "demographics"},
                    ]
                },
            },
            "agent_seed": {"question": "Fallback question"},
            "go_no_go": "recommend",
            "go_no_go_reason": "Concepts are available in the selected export.",
        },
    )

    def upsert(body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        writes.append((dict(body), dict(kwargs)))
        return {**current, **body, "revision": 10}

    monkeypatch.setattr(tool_module.study_contexts, "upsert_context", upsert)
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-idea-accept",
            binding=AuthorityBinding(
                study_context_id="study-workflow",
                study_revision=9,
            ),
        ),
        allowed_actions={"idea"},
    )
    result = tool_module.execute_tool(
        "easyicu_accept_idea_handoff",
        {"run_id": "idea-run-1", "idea_id": "idea-lactate"},
        context,
    )

    assert result["code"] == "easyicu_idea_handoff_accepted"
    assert result["details"]["idea_selection"]["canonical_handoff_sha256"] == "b" * 64
    patch, options = writes[0]
    assert patch["question"] == "Does early lactate predict hospital mortality?"
    assert patch["primary_exposure"] == "lact"
    assert patch["outcome"] == "death"
    assert patch["covariates"] == ["age", "sex"]
    assert patch["modules"] == ["blood_gas", "outcome", "demographics"]
    assert patch["analysis_goal"] == "association_study"
    assert patch["idea_handoff"] == {
        "schema_version": "easyicu.pi-idea-selection/1",
        "run_id": "idea-run-1",
        "idea_id": "idea-lactate",
        "canonical_handoff_sha256": "b" * 64,
        "status": "accepted",
        "accepted_at": "2026-08-11T12:00:00Z",
        "go_no_go": "recommend",
        "go_no_go_reason": "Concepts are available in the selected export.",
    }
    assert options["expected_revision"] == 9
    with pytest.raises(PiCopilotError) as stale:
        tool_module.execute_tool("easyicu_inspect_workflow", {}, context)
    assert stale.value.code == "pi_session_authority_stale"


def test_accept_idea_handoff_requires_digest_bound_concept_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {**_complete_study(), "revision": 2, "active_job_id": None}
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: current)
    monkeypatch.setattr(tool_module.idea_mining, "plan_idea", lambda body: {"plan": {}})
    monkeypatch.setattr(
        tool_module.idea_mining,
        "create_handoff",
        lambda body: {
            "idea_id": "idea-lactate",
            "canonical_handoff_sha256": "d" * 64,
            "canonical_handoff": {
                "analysis_family": "association_study",
                "resolved_predictor_concept": "lact",
                "resolved_outcome_concept": "death",
                "resolved_analysis_concepts": ["lact"],
                "selected_ledger_row": {
                    "mapped_concepts": [
                        {"concept_id": "lact", "module": "blood_gas"}
                    ]
                },
            },
        },
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(session_id="pi-idea-incomplete"),
        allowed_actions={"idea"},
    )

    result = tool_module.execute_tool(
        "easyicu_accept_idea_handoff",
        {"run_id": "idea-run-1", "idea_id": "idea-lactate"},
        context,
    )

    assert result["code"] == "canonical_idea_execution_contract_required"
    assert result["details"]["missing_concept_modules"] == ["death"]


def test_accept_idea_handoff_clears_comparator_when_predictor_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {
        **_complete_study(),
        "revision": 2,
        "active_job_id": None,
        "primary_exposure": "lact",
        "comparator": "per 1 mmol/L",
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: current)
    monkeypatch.setattr(
        tool_module.idea_mining,
        "plan_idea",
        lambda body: {"plan": {"research_question": "Does phenotype X predict Y?"}},
    )
    monkeypatch.setattr(
        tool_module.idea_mining,
        "create_handoff",
        lambda body: {
            "idea_id": "idea-phenotype",
            "candidate_topic": "Phenotype X and mortality",
            "canonical_handoff_sha256": "e" * 64,
            "canonical_handoff": {
                "analysis_family": "association_study",
                "resolved_predictor_concept": "phenotype_x",
                "resolved_outcome_concept": "death",
                "resolved_analysis_concepts": ["phenotype_x", "age"],
                "selected_ledger_row": {
                    "mapped_concepts": [
                        {"concept_id": "phenotype_x", "module": "phenotypes"},
                        {"concept_id": "death", "module": "outcome"},
                        {"concept_id": "age", "module": "demographics"},
                    ]
                },
            },
        },
    )

    def upsert(body: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        writes.append(dict(body))
        return {**current, **body, "revision": 3}

    monkeypatch.setattr(tool_module.study_contexts, "upsert_context", upsert)
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-new-predictor",
            binding=AuthorityBinding(
                study_context_id="study-workflow",
                study_revision=2,
            ),
        ),
        allowed_actions={"idea"},
    )

    result = tool_module.execute_tool(
        "easyicu_accept_idea_handoff",
        {"run_id": "idea-run-phenotype", "idea_id": "idea-phenotype"},
        context,
    )

    assert result["code"] == "easyicu_idea_handoff_accepted"
    assert writes[0]["primary_exposure"] == "phenotype_x"
    assert writes[0]["comparator"] == ""


def test_accept_idea_handoff_fails_closed_without_canonical_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = {**_complete_study(), "revision": 2, "active_job_id": None}
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: current)
    monkeypatch.setattr(tool_module.idea_mining, "plan_idea", lambda body: {"plan": {}})
    monkeypatch.setattr(
        tool_module.idea_mining,
        "create_handoff",
        lambda body: {"idea_id": "idea-lactate", "canonical_handoff_sha256": "bad"},
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(session_id="pi-idea-invalid"),
        allowed_actions={"idea"},
    )
    result = tool_module.execute_tool(
        "easyicu_accept_idea_handoff",
        {"run_id": "idea-run-1", "idea_id": "idea-lactate"},
        context,
    )
    assert result["code"] == "canonical_idea_handoff_digest_required"


def test_extraction_uses_bound_study_source_and_returns_no_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted: list[dict[str, Any]] = []

    monkeypatch.setattr(
        tool_module,
        "_bound_context",
        lambda binding: _complete_study(),
    )
    monkeypatch.setattr(
        tool_module.sources,
        "load_registry",
        lambda: {"active_path": None, "sources": []},
    )
    from easyicu.webserver.routes import jobs as jobs_route

    def submit(body: dict[str, Any]) -> dict[str, Any]:
        submitted.append(dict(body))
        return {
            "job_id": "extract-job-1",
            "kind": "extract",
            "status": "running",
            "study_context_id": "study-workflow",
            "study_context_revision": 5,
        }

    monkeypatch.setattr(jobs_route, "jobs_extract", submit)
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-extract",
            binding=AuthorityBinding(
                study_context_id="study-workflow",
                study_revision=4,
            ),
        ),
        allowed_actions={"extract"},
    )
    result = tool_module.execute_tool("easyicu_start_extraction", {}, context)

    assert result["code"] == "easyicu_extraction_submitted"
    assert submitted[0]["path"] == "/private/prepared/source"
    assert submitted[0]["database"] == "mimiciv"
    assert "path" not in json.dumps(result)
    with pytest.raises(PiCopilotError) as stale:
        tool_module.execute_tool("easyicu_inspect_workflow", {}, context)
    assert stale.value.code == "pi_session_authority_stale"


def test_full_run_cannot_use_mock_as_scientific_output() -> None:
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-full",
            external_llm_opt_in=True,
        ),
        allowed_actions={"provider_run"},
    )
    result = tool_module.execute_tool(
        "easyicu_run",
        {"run_type": "full", "llm_provider": "mock"},
        context,
    )
    assert result["code"] == "pi_full_mock_not_scientific"


def test_full_run_uses_verified_pi_provider_not_model_selected_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        tool_module,
        "_bound_context",
        lambda binding: {
            **_complete_study(),
            "question": "Bound aggregate scientific question",
        },
    )
    from easyicu.webserver.routes import agent as agent_route

    def submit(body: dict[str, Any]) -> dict[str, Any]:
        submitted.append(dict(body))
        return {
            "job_id": "agent-job-full",
            "kind": "agent-run",
            "status": "running",
            "study_context_id": "study-workflow",
            "study_context_revision": 5,
        }

    monkeypatch.setattr(agent_route, "jobs_agent_run", submit)
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-full-owner",
            external_llm_opt_in=True,
            binding=AuthorityBinding(
                study_context_id="study-workflow",
                study_revision=4,
            ),
        ),
        allowed_actions={"provider_run"},
    )
    result = tool_module.execute_tool(
        "easyicu_run",
        {"run_type": "full", "llm_provider": "local"},
        context,
    )

    assert result["code"] == "easyicu_full_run_submitted"
    assert submitted == [
        {
            "study_context_id": "study-workflow",
            "question": "Bound aggregate scientific question",
            "run_type": "full",
            "llm_provider": "openai",
            "external_llm_opt_in": True,
            "engine": "research_agent_pipeline",
            "credential_source": "pi_verified",
        }
    ]


def _write_real_pipeline_fixture(run_dir: Path, *, manuscript: str) -> None:
    (run_dir / "evidence").mkdir(parents=True)
    (run_dir / "results").mkdir()
    readiness = {
        "execution_complete": True,
        "analysis_validated": True,
        "evidence_complete": True,
        "numeric_verified": True,
        "manuscript_ready": True,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps({"readiness": readiness}),
        encoding="utf-8",
    )
    (run_dir / "run_status.json").write_text("{}", encoding="utf-8")
    (run_dir / "analysis_plan.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "id": "model",
                        "title": "Fit specified model",
                        "literature_citation_keys": ["method_paper"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "preplan_literature_bundle.json").write_text(
        json.dumps(
            {
                "research_question": "Does an ICU exposure predict mortality?",
                "citations": [
                    {
                        "key": "method_paper",
                        "title": "A source-backed method paper",
                        "year": "2024",
                        "venue": "Statistics in Medicine",
                        "pmid": "12345",
                    }
                ],
                "prisma": None,
                "search_provenance": {
                    "curated_seed_count": 1,
                    "sources_enabled": [],
                    "sources_returning": [],
                    "search_conducted": False,
                    "note": "Curated method reference; no retrieval was run.",
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manuscript_scaffold_bound.md").write_text(
        manuscript,
        encoding="utf-8",
    )
    (run_dir / "claim_ledger.csv").write_text(
        "claim_id,claim_text,evidence_refs,status,note\n"
        "c1,The registered aggregate estimate passed validation,ev-table,analysis_only,Review required\n",
        encoding="utf-8",
    )
    (run_dir / "results" / "aggregate.csv").write_text(
        "metric,estimate,lower,upper\nassociation,1.2,1.1,1.3\n",
        encoding="utf-8",
    )
    (run_dir / "results" / "identifier_rows.csv").write_text(
        "stay_id,value\n123,8\n",
        encoding="utf-8",
    )
    (run_dir / "evidence" / "evidence_index.json").write_text(
        json.dumps(
            [
                {
                    "kind": "table",
                    "evidence_id": "ev-table",
                    "description": "Aggregate model result",
                    "relative_path": "results/aggregate.csv",
                },
                {
                    "kind": "table",
                    "evidence_id": "ev-sensitive",
                    "description": "Identifier rows",
                    "relative_path": "results/identifier_rows.csv",
                },
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "figure_gallery.json").write_text(
        json.dumps({"status": "no_figures", "figures": []}),
        encoding="utf-8",
    )


def _acquisition_receipt() -> SimpleNamespace:
    return SimpleNamespace(
        selection=SimpleNamespace(selected_concepts=["heart_rate", "mortality"]),
        materialized_concepts=["heart_rate", "mortality"],
        coverage=SimpleNamespace(sufficient=True),
        analysis_columns={"heart_rate": "heart_rate"},
        endpoint=None,
    )


def _foundation_profile() -> dict[str, Any]:
    return {
        "allowed_modules": ("demographics", "outcome"),
        "static_concepts": ("age", "sex"),
        "outcome_concepts": ("death",),
        "required_feature_concepts": (),
        "require_outcome": True,
    }


def test_pipeline_projection_uses_real_artifacts_and_withholds_identifier_table(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "real-run"
    _write_real_pipeline_fixture(
        run_dir,
        manuscript="# Results\nThe registered aggregate estimate is analysis-only.",
    )
    wrapper = tmp_path / "web-projection"

    result = agent_pipeline_runs._write_projection(
        wrapper_dir=wrapper,
        study=_complete_study(),
        provider={"provider": "openai", "model": "test-model"},
        acquisition=_acquisition_receipt(),
        run_dir=run_dir,
    )

    assert result["engine"] == "easyicu.research_agent.pipeline"
    assert result["gate"]["status"] == "analysis_only"
    tables = json.loads((wrapper / "result_tables.json").read_text(encoding="utf-8"))
    assert tables["table_count"] == 1
    assert tables["tables"][0]["evidence_id"] == "ev-table"
    assert tables["skipped_identifier_tables"] == 1
    manuscript = json.loads(
        (wrapper / "manuscript_draft.json").read_text(encoding="utf-8")
    )
    assert manuscript["claims"][0]["evidence_ids"] == ["ev-table"]
    literature = json.loads(
        (wrapper / "literature_evidence.json").read_text(encoding="utf-8")
    )
    assert literature["status"] == "curated_only"
    assert literature["mapping_status"] == "complete"
    assert literature["step_citation_map"][0]["citation_keys"] == ["method_paper"]
    ledger = json.loads((wrapper / "evidence_ledger.json").read_text(encoding="utf-8"))
    assert "literature_evidence.json" in {
        row["name"] for row in ledger["artifacts"]
    }
    assert ledger["privacy"]["projection_scan_passed"] is True
    assert ledger["privacy"]["path_values_returned"] is False


def test_pending_plan_reason_survives_projection_and_run_history(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "real-pending-run"
    _write_real_pipeline_fixture(run_dir, manuscript="")
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review the digest-bound plan before analysis.",
        authority_sha256="a" * 64,
        payload={"reason": "operator_plan_approval_required"},
    )
    pending = HumanReviewPending(
        run_id="run-pending-plan",
        thread_id="thread-pending-plan",
        run_dir=str(run_dir),
        requests=(request,),
    )
    project_root = tmp_path / "projects"
    wrapper = project_root / "study-workflow" / "run_pending_plan"

    agent_pipeline_runs._write_projection(
        wrapper_dir=wrapper,
        study=_complete_study(),
        provider={"provider": "openai", "model": "test-model"},
        acquisition=_acquisition_receipt(),
        run_dir=run_dir,
        pending=pending,
    )

    manifest = json.loads(
        (wrapper / "source_run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["pending_reviews"][0]["reason_code"] == (
        "operator_plan_approval_required"
    )
    history = agent_runs.list_run_history(
        study_id="study-workflow",
        project_root=str(project_root),
    )
    assert history["runs"][0]["run_status"] == "human_review_pending"
    assert history["runs"][0]["pending_review_reason_codes"] == [
        "operator_plan_approval_required"
    ]


def test_pipeline_projection_fails_closed_when_source_contains_a_host_path(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "unsafe-run"
    _write_real_pipeline_fixture(
        run_dir,
        manuscript="The source was loaded from /Users/example/private/source.csv.",
    )
    wrapper = tmp_path / "withheld-projection"

    result = agent_pipeline_runs._write_projection(
        wrapper_dir=wrapper,
        study=_complete_study(),
        provider={"provider": "openai", "model": "test-model"},
        acquisition=_acquisition_receipt(),
        run_dir=run_dir,
    )

    assert result["gate"]["status"] == "blocked"
    assert result["gate"]["reason"] == "research_pipeline_projection_privacy_blocked"
    assert not (wrapper / "manuscript_draft.json").exists()
    gate = json.loads((wrapper / "quality_gate.json").read_text(encoding="utf-8"))
    assert gate["privacy"]["payloads_withheld"] is True
    assert "/Users/example" not in json.dumps(result)


def test_provider_bridge_keeps_private_key_out_of_public_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_key = "test-private-provider-key"
    monkeypatch.setattr(
        provider_adapter,
        "_load_external_credentials",
        lambda *_args, **_kwargs: {
            "provider": "openai",
            "api_key": private_key,
            "base_url": "http://127.0.0.1:8317/v1/chat/completions",
            "model": "test-local-model",
            "api_key_env": "OPENAI_API_KEY",
            "base_url_env": "OPENAI_BASE_URL",
            "model_env": "OPENAI_MODEL",
        },
    )
    captured: dict[str, Any] = {}
    import easyicu.research_agent.providers as providers

    def fake_builder(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(providers, "build_provider_client", fake_builder)

    _client, public = provider_adapter.build_research_agent_provider_client(
        {"provider": "openai", "external": True},
    )

    assert captured["environment"]["OPENAI_API_KEY"] == private_key
    assert captured["environment"]["OPENAI_BASE_URL"] == "http://127.0.0.1:8317/v1"
    assert captured["environment"]["EASYICU_TRUST_LOOPBACK_PROXY_KEY"] == "1"
    assert private_key not in json.dumps(public)
    assert public["secrets_returned"] is False


def test_plan_approval_requires_fresh_provider_grant_and_forwards_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        tool_module,
        "_bound_context",
        lambda binding: _complete_study(),
    )
    from easyicu.webserver.routes import agent as agent_route

    def submit(body: dict[str, Any]) -> dict[str, Any]:
        submitted.append(dict(body))
        return {
            "job_id": "resume-job-1",
            "kind": "agent-run",
            "status": "running",
            "engine": "research_agent_pipeline",
            "review_run_id": "pipeline-run-1",
            "study_context_id": "study-workflow",
            "study_context_revision": 5,
        }

    monkeypatch.setattr(agent_route, "jobs_agent_run_review", submit)
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-review",
            external_llm_opt_in=True,
            binding=AuthorityBinding(
                study_context_id="study-workflow",
                study_revision=4,
                run_id="pipeline-run-1",
            ),
        ),
        allowed_actions={"provider_run"},
    )

    result = tool_module.execute_tool(
        "easyicu_resume",
        {"decision": "approved", "reviewer": "local reviewer"},
        context,
    )

    assert result["code"] == "research_pipeline_review_submitted"
    assert submitted == [
        {
            "study_context_id": "study-workflow",
            "run_id": "pipeline-run-1",
            "decision": "approved",
            "reviewer": "local reviewer",
            "note": "",
            "external_llm_opt_in": True,
        }
    ]
    with pytest.raises(PiCopilotError) as stale:
        context.assert_authority_fresh()
    assert stale.value.code == "pi_session_authority_stale"


def test_web_runner_delegates_to_research_agent_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_run = tmp_path / "actual-pipeline-run"
    _write_real_pipeline_fixture(
        actual_run,
        manuscript="# Results\nThe pipeline-owned aggregate result is analysis-only.",
    )
    universe = tmp_path / "universe.parquet"
    universe.write_bytes(b"typed-universe-placeholder")
    acquisition = _acquisition_receipt()
    acquisition.blocked = False
    acquisition.universe_path = universe
    acquisition.cohort_authority_path = None
    acquisition.cohort_authority_ref = None
    acquisition.trajectory_path = None
    acquisition.trajectory_authority_path = None
    acquisition.trajectory_authority_ref = None
    calls: dict[str, Any] = {}

    monkeypatch.setattr(
        provider_adapter,
        "build_research_agent_provider_client",
        lambda provider, **_kwargs: (
            object(),
            {"provider": "openai", "model": "test-model"},
        ),
    )
    import easyicu.research_agent as research_agent
    from easyicu.research_agent.acquisition import foundation

    def fake_acquire(**kwargs: Any) -> SimpleNamespace:
        calls["acquire"] = kwargs
        return acquisition

    class FakePipeline:
        def run(self, **kwargs: Any) -> SimpleNamespace:
            calls["run"] = kwargs
            kwargs["progress_callback"](
                {
                    "stage": "planning",
                    "message": "Generating plan draft 1/5.",
                    "current": 1,
                    "total": 5,
                    "status": "running",
                }
            )
            return SimpleNamespace(manifest_path=actual_run / "manifest.json")

    def fake_from_config(config: Any, *, services: Any) -> FakePipeline:
        calls["config"] = config
        calls["services"] = services
        return FakePipeline()

    monkeypatch.setattr(foundation, "acquire_universe_for_question", fake_acquire)
    monkeypatch.setattr(
        agent_pipeline_runs,
        "_data_foundation_profile",
        lambda **_kwargs: _foundation_profile(),
    )
    monkeypatch.setattr(
        research_agent.ResearchAgentPipeline,
        "from_config",
        fake_from_config,
    )

    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(tmp_path / "export"),
        study_context=_complete_study(),
        project_root=str(tmp_path / "projects"),
        provider={"provider": "openai", "external": True},
    )

    class Job:
        id = "job-real-pipeline"
        cancel_requested = False
        events: list[dict[str, Any]] = []

        def emit(self, event: dict[str, Any]) -> None:
            self.events.append(dict(event))

    result = runner(Job())

    assert calls["acquire"]["question"] == _complete_study()["question"]
    assert calls["acquire"]["allowed_modules"] == ("demographics", "outcome")
    assert calls["acquire"]["static_concepts"] == ("age", "sex")
    assert calls["run"]["cohort"] == universe
    assert calls["run"]["question"] == _complete_study()["question"]
    assert calls["acquire"]["primary_exposure_concept"] == "heart_rate"
    assert calls["run"]["primary_exposure"] == "heart_rate"
    assert calls["run"]["endpoint"] is None
    assert calls["run"]["user_preferences"]["covariates"] == ["age", "sex"]
    assert calls["config"].evidence_enforcement_mode == "strict"
    assert calls["config"].enable_reproducibility_envelope is True
    assert calls["config"].require_human_plan_review is True
    assert result["engine"] == "easyicu.research_agent.pipeline"
    assert result["gate"]["status"] == "analysis_only"
    assert any(event["step"] == "research_pipeline" for event in Job.events)
    assert any(
        event["step"] == "planning"
        and event["label"] == "Generating plan draft 1/5."
        and event["current"] == 1
        and event["total"] == 5
        for event in Job.events
    )


def test_pi_verified_provider_environment_is_full_pipeline_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.routes import agent as agent_route

    private_environment = {
        "OPENAI_API_KEY": "test-private-provider-key",
        "OPENAI_BASE_URL": "http://127.0.0.1:8317/v1",
        "OPENAI_MODEL": "test-local-model",
        "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
    }
    calls: list[bool] = []

    def project(self: object, *, external_llm_opt_in: bool) -> dict[str, str]:
        calls.append(external_llm_opt_in)
        return dict(private_environment)

    monkeypatch.setattr(
        agent_route.PiProviderConfigStore,
        "research_agent_environment",
        project,
    )

    resolved = agent_route._provider_environment_for_agent_run(
        credential_source="pi_verified",
        engine="research_agent_pipeline",
        run_type="full",
        external_llm_opt_in=True,
    )

    assert resolved == private_environment
    assert calls == [True]

    with pytest.raises(Exception) as wrong_engine:
        agent_route._provider_environment_for_agent_run(
            credential_source="pi_verified",
            engine="native_summary",
            run_type="full",
            external_llm_opt_in=True,
        )
    assert getattr(wrong_engine.value, "detail") == {
        "error": "pi_provider_research_pipeline_only"
    }


def test_research_pipeline_runner_uses_in_memory_provider_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_run = tmp_path / "actual-provider-run"
    _write_real_pipeline_fixture(
        actual_run,
        manuscript="# Results\nThe provider-bound result is analysis-only.",
    )
    universe = tmp_path / "universe.parquet"
    universe.write_bytes(b"typed-universe-placeholder")
    acquisition = _acquisition_receipt()
    acquisition.blocked = False
    acquisition.universe_path = universe
    acquisition.cohort_authority_path = None
    acquisition.cohort_authority_ref = None
    acquisition.trajectory_path = None
    acquisition.trajectory_authority_path = None
    acquisition.trajectory_authority_ref = None
    expected_environment = {
        "OPENAI_API_KEY": "test-private-provider-key",
        "OPENAI_BASE_URL": "http://127.0.0.1:8317/v1",
        "OPENAI_MODEL": "test-local-model",
        "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
    }
    captured: dict[str, Any] = {}

    def build_client(
        provider: dict[str, Any],
        *,
        environ: dict[str, str] | None = None,
    ) -> tuple[object, dict[str, Any]]:
        captured["provider"] = dict(provider)
        captured["environment"] = dict(environ or {})
        return object(), {"provider": "openai", "model": "test-local-model"}

    monkeypatch.setattr(
        provider_adapter,
        "build_research_agent_provider_client",
        build_client,
    )
    import easyicu.research_agent as research_agent
    from easyicu.research_agent.acquisition import foundation

    monkeypatch.setattr(
        foundation,
        "acquire_universe_for_question",
        lambda **_kwargs: acquisition,
    )
    monkeypatch.setattr(
        agent_pipeline_runs,
        "_data_foundation_profile",
        lambda **_kwargs: _foundation_profile(),
    )

    class FakePipeline:
        def run(self, **_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(manifest_path=actual_run / "manifest.json")

    monkeypatch.setattr(
        research_agent.ResearchAgentPipeline,
        "from_config",
        lambda _config, *, services: FakePipeline(),
    )

    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(tmp_path / "export"),
        study_context=_complete_study(),
        project_root=str(tmp_path / "projects"),
        provider={"provider": "openai", "external": True},
        provider_environment=expected_environment,
    )

    class Job:
        id = "job-provider-authority"
        cancel_requested = False
        events: list[dict[str, Any]] = []

        def emit(self, event: dict[str, Any]) -> None:
            self.events.append(dict(event))

    result = runner(Job())

    assert captured["environment"] == expected_environment
    assert result["provider"]["model"] == "test-local-model"
    assert "test-private-provider-key" not in json.dumps(result)


def test_web_data_foundation_profile_keeps_continuous_outcome_static(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="typed-demo",
            concepts=[
                CatalogConcept(
                    concept_id="age",
                    file_name="demographics.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
                CatalogConcept(
                    concept_id="sex",
                    file_name="demographics.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
                CatalogConcept(
                    concept_id="los_icu",
                    file_name="outcome.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
                CatalogConcept(
                    concept_id="death",
                    file_name="outcome.parquet",
                    typed_metadata=True,
                    column_role="event_status",
                ),
            ],
        ),
    )

    profile = agent_pipeline_runs._data_foundation_profile(
        export_path="/typed/demo",
        study={"modules": ["demographics", "outcome"]},
        target="los_icu",
    )

    assert profile == {
        "allowed_modules": ("demographics", "outcome"),
        "static_concepts": ("age", "sex", "los_icu"),
        "outcome_concepts": (),
        "required_feature_concepts": (),
        "require_outcome": False,
    }


def test_web_data_foundation_profile_keeps_event_outcome_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="typed-demo",
            concepts=[
                CatalogConcept(
                    concept_id="age",
                    file_name="demographics.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
                CatalogConcept(
                    concept_id="death",
                    file_name="outcome.parquet",
                    typed_metadata=True,
                    column_role="event_status",
                ),
            ],
        ),
    )

    profile = agent_pipeline_runs._data_foundation_profile(
        export_path="/typed/demo",
        study={"modules": ["demographics", "outcome"]},
        target="death",
    )

    assert profile["static_concepts"] == ("age",)
    assert profile["outcome_concepts"] == ("death",)
    assert profile["require_outcome"] is True


def test_web_data_foundation_profile_keeps_legacy_owner_declared_event_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="legacy-demo",
            concepts=[
                CatalogConcept(
                    concept_id="age",
                    file_name="demographics.parquet",
                    typed_metadata=False,
                ),
                CatalogConcept(
                    concept_id="death",
                    file_name="outcome.parquet",
                    typed_metadata=False,
                    column_role="event_status",
                ),
            ],
        ),
    )

    profile = agent_pipeline_runs._data_foundation_profile(
        export_path="/legacy/demo",
        study={"modules": ["demographics", "outcome"]},
        target="death",
    )

    assert profile["static_concepts"] == ("age",)
    assert profile["outcome_concepts"] == ("death",)
    assert profile["require_outcome"] is True


def test_web_data_foundation_materializes_typed_exposure_and_covariates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="typed-demo",
            concepts=[
                CatalogConcept(
                    concept_id="age",
                    file_name="demographics.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
                CatalogConcept(
                    concept_id="sex",
                    file_name="demographics.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
                CatalogConcept(
                    concept_id="death",
                    file_name="outcome.parquet",
                    typed_metadata=True,
                    column_role="event_status",
                ),
                CatalogConcept(
                    concept_id="sep3_sofa2",
                    file_name="sepsis3_sofa2.parquet",
                    typed_metadata=True,
                    column_role="event_status",
                ),
            ],
        ),
    )

    profile = agent_pipeline_runs._data_foundation_profile(
        export_path="/typed/demo",
        study={
            "modules": ["demographics", "outcome", "sepsis3_sofa2"],
        },
        target="death",
        primary_exposure="sep3_sofa2_max",
        covariates=("age", "sex"),
    )

    assert profile == {
        "allowed_modules": ("demographics", "outcome", "sepsis3_sofa2"),
        "static_concepts": ("age", "sex"),
        "outcome_concepts": ("death",),
        "required_feature_concepts": ("sep3_sofa2",),
        "require_outcome": True,
    }


def test_web_data_foundation_rejects_unmaterialized_primary_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="typed-demo",
            concepts=[
                CatalogConcept(
                    concept_id="age",
                    file_name="demographics.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
            ],
        ),
    )

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs._data_foundation_profile(
            export_path="/typed/demo",
            study={"modules": ["demographics"]},
            target=None,
            primary_exposure="missing_exposure",
        )

    assert exc.value.code == (
        "research_pipeline_primary_exposure_outside_configured_modules"
    )


def test_web_study_context_compiles_to_strict_user_preferences() -> None:
    study = {
        **_complete_study(),
        "purpose": "Demo-only product validation.",
        "comparator": "Compare aggregate summaries by sex.",
        "confirmations": {
            "demo_only": True,
            "non_causal": True,
            "not_for_manuscript": True,
        },
    }

    compiled = agent_pipeline_runs._research_user_preferences(study)
    validated = UserPreferences.model_validate(compiled)

    assert set(compiled) == {
        "extra_notes",
        "must_have_outputs",
        "subgroup_sensitivity",
        "timing_and_design",
        "data_constraints",
        "covariates",
    }
    assert validated.extra_notes == "Demo-only product validation."
    assert "not_for_manuscript" in str(validated.data_constraints)
    assert validated.covariates == ["age", "sex"]
