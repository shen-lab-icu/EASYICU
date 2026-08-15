"""Focused owner and fail-closed tests for the Copilot research workflow."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pandas as pd

from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
from easyicu.research_agent.reporting.result_card import (
    build_result_interpretation_card,
)
from easyicu.research_agent.planning.scientific_review import (
    PlanScientificFinding,
    PlanScientificReview,
)
from easyicu.research_agent.schema import UserPreferences
from easyicu.research_agent.orchestration.workflow import (
    HumanReviewPending,
    HumanReviewRequest,
)
from easyicu.webserver import (
    agent_pipeline_runs,
    agent_runs,
    dataio,
    literature_authority,
    provider_adapter,
)
from easyicu.webserver import study_contexts as study_context_owner
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
    registered_export_matches_study,
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
        "covariate_selection": "exact",
        "covariate_rationales": {
            "age": "Age is a baseline demographic confounder selected before analysis.",
            "sex": "Sex is a baseline demographic confounder selected before analysis.",
        },
        "covariate_temporal_roles": {
            "age": "baseline_static",
            "sex": "baseline_static",
        },
        "execution_concepts": {
            "outcome": "death",
            "primary_exposure": "heart_rate",
            "covariates": ["age", "sex"],
        },
        "analysis_design": {
            "analysis_unit": "icu_stay",
            "variance_estimator": "model_based",
        },
        "time_window": {"hours": 24, "anchor": "ICU admission"},
        "export_format": "parquet",
        "analysis_goal": "Descriptive prognostic association",
    }


def _write_pipeline_export(root: Path, *, database: str = "miiv") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"stay_id": [1], "age": [65]}).to_parquet(
        root / "demographics.parquet", index=False
    )
    (root / "_manifest.json").write_text(
        json.dumps(
            {
                "database": database,
                "format": "parquet",
                "concept_selection": {
                    "mode": "explicit",
                    "modules": {"demographics": ["age"]},
                },
                "feature_definitions": {"included": False},
                "files": [
                    {
                        "file": "demographics.parquet",
                        "module": "demographics",
                        "concepts": 1,
                        "concept_ids": ["age"],
                        "rows": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def _study_with_package_binding(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    export = _write_pipeline_export(root)
    study = {
        **_complete_study(),
        "data_source": {"path": str(export), "database": "miiv"},
    }
    receipt = dataio.validate_research_pipeline_source(
        str(export),
        database="miiv",
    )
    return study, dict(receipt["binding"])


_PI_PROVIDER_ENVIRONMENT = {
    "OPENAI_API_KEY": "test-private-provider-key",
    "OPENAI_BASE_URL": "http://127.0.0.1:8317/v1",
    "OPENAI_MODEL": "test-local-model",
    "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
}


def test_web_cancellation_is_a_typed_progress_control_signal() -> None:
    from easyicu.research_agent.orchestration.progress import ProgressControlSignal

    assert issubclass(agent_pipeline_runs.ResearchPipelineRunError, ProgressControlSignal)
    job = SimpleNamespace(cancel_requested=True, emit=lambda _event: None)

    with pytest.raises(ProgressControlSignal) as raised:
        agent_pipeline_runs._progress(job, step="planning", label="Planning")

    assert raised.value.code == "research_pipeline_cancelled"


def _nonapprovable_review_payload(*, finding_code: str) -> dict[str, Any]:
    return PlanScientificReview(
        status="changes_required",
        approval_allowed=False,
        top_journal_candidate=False,
        score=70,
        dimension_scores={"study_design": 70},
        findings=[
            PlanScientificFinding(
                code=finding_code,
                severity="major",
                dimension="study_design",
                message="The exact reviewed plan needs bounded follow-up.",
                remediation="Address the finding without changing the study authority.",
            )
        ],
        context_sha256="a" * 64,
        plan_sha256="b" * 64,
        literature_sha256="c" * 64,
        figure_strategy_sha256="d" * 64,
        generated_at="2026-08-14T00:00:00Z",
    ).model_dump(mode="json")


@pytest.mark.parametrize(
    ("finding_code", "expected_fragment"),
    [
        ("DIRECT_COMPARATOR_NOT_ESTABLISHED", ""),
        (
            "CONTINUOUS_COVARIATE_FUNCTIONAL_FORM_UNCHECKED",
            "CONTINUOUS_COVARIATE_FUNCTIONAL_FORM_UNCHECKED",
        ),
    ],
)
def test_plan_revision_bridge_falls_back_to_fresh_plan_without_agent_findings(
    finding_code: str,
    expected_fragment: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _complete_study()
    source_run_id = "run-reviewed-plan"
    monkeypatch.setattr(
        agent_runs,
        "list_run_history",
        lambda **_kwargs: {
            "runs": [
                {
                    "run_id": source_run_id,
                    "project_dir": "/private/reviewed-run",
                    "scientific_configuration_sha256": (
                        study_context_owner.scientific_configuration_sha256(study)
                    ),
                }
            ]
        },
    )
    monkeypatch.setattr(
        agent_runs,
        "read_run_review",
        lambda _project_dir: {
            "ok": True,
            "artifact_payloads": {
                "scientific_plan_review.json": _nonapprovable_review_payload(
                    finding_code=finding_code
                )
            },
        },
    )

    contract = agent_pipeline_runs._compile_plan_revision_contract(
        study=study,
        project_root="/private/projects",
        source_run_id=source_run_id,
    )

    if expected_fragment:
        assert expected_fragment in contract
        assert "generate a fresh plan" in contract
    else:
        assert contract == ""


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
    assert finished.current_stage == "interpretation"
    assert finished.completed_required_stages == 5
    assert finished.next_action_code == "evidence_bound_interpretation_ready"


def test_workflow_projects_exact_path_free_study_setup_receipt() -> None:
    study = {
        **_complete_study(),
        "purpose": "Publication-quality ICU methods study",
        "confirmations": {"study_design_reviewed": True},
    }

    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run=None,
    )

    receipt = snapshot.study_setup_receipt
    assert receipt.study_context_id == "study-workflow"
    assert receipt.revision == 4
    assert receipt.configuration["question"] == study["question"]
    assert receipt.configuration["cohort"] == study["cohort"]
    assert receipt.configuration["modules"] == study["modules"]
    assert receipt.configuration["execution_concepts"] == study["execution_concepts"]
    assert receipt.configuration["analysis_design"] == study["analysis_design"]
    assert (
        receipt.configuration["covariate_rationales"] == study["covariate_rationales"]
    )
    assert (
        receipt.configuration["covariate_temporal_roles"]
        == study["covariate_temporal_roles"]
    )
    assert receipt.configuration["confirmations"] == study["confirmations"]
    assert receipt.configuration["data_source"] == {
        "database": "mimiciv",
        "path_hash": "58809605ee2154d6",
    }
    serialized = snapshot.model_dump_json()
    assert "/private/prepared/source" not in serialized
    assert '"path"' not in serialized


def test_workflow_keeps_typed_execution_decisions_in_setup_gate() -> None:
    missing_design = {**_complete_study(), "analysis_design": {}}
    design_snapshot = build_research_workflow_snapshot(
        study=missing_design,
        active_export_present=True,
        active_job=None,
        latest_run=None,
    )
    assert design_snapshot.missing_setup_fields == ["analysis_design"]
    assert (
        next(row for row in design_snapshot.stages if row.id == "setup").status
        == "ready"
    )

    labelled_window = {
        **_complete_study(),
        "time_window": {
            "preset": "full_available_stay",
            "label": "Whole stay",
            "anchor": "ICU admission",
        },
    }
    window_snapshot = build_research_workflow_snapshot(
        study=labelled_window,
        active_export_present=True,
        active_job=None,
        latest_run=None,
    )
    assert window_snapshot.missing_setup_fields == ["time_window.hours"]

    conflated_window = {
        **_complete_study(),
        "time_window": {
            "hours": 24,
            "anchor": "suspected infection onset",
        },
    }
    conflated_snapshot = build_research_workflow_snapshot(
        study=conflated_window,
        active_export_present=True,
        active_job=None,
        latest_run=None,
    )
    assert conflated_snapshot.missing_setup_fields == ["time_window.anchor_supported"]

    unaddressed_repeats = {
        **_complete_study(),
        "cohort": {
            **_complete_study()["cohort"],
            "exclude_readmissions": False,
        },
    }
    dependence_snapshot = build_research_workflow_snapshot(
        study=unaddressed_repeats,
        active_export_present=True,
        active_job=None,
        latest_run=None,
    )
    assert dependence_snapshot.missing_setup_fields == ["analysis_design.dependence"]


def test_pipeline_factory_rejects_missing_typed_analysis_design_before_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation_called = False

    def foundation(**_kwargs: Any) -> dict[str, Any]:
        nonlocal foundation_called
        foundation_called = True
        return _foundation_profile()

    monkeypatch.setattr(agent_pipeline_runs, "_data_foundation_profile", foundation)
    study = {**_complete_study(), "analysis_design": {}}

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.make_research_pipeline_run_runner(
            export_path="/typed/demo",
            study_context=study,
            project_root=None,
            provider={"provider": "openai", "external": True},
        )

    assert exc.value.code == "research_pipeline_analysis_design_required"
    assert exc.value.details["required_fields"] == [
        "analysis_unit",
        "variance_estimator",
    ]
    assert foundation_called is False


def test_pipeline_factory_rejects_non_executable_time_window_label_before_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation_called = False

    def foundation(**_kwargs: Any) -> dict[str, Any]:
        nonlocal foundation_called
        foundation_called = True
        return _foundation_profile()

    monkeypatch.setattr(agent_pipeline_runs, "_data_foundation_profile", foundation)
    study = {
        **_complete_study(),
        "time_window": {
            "preset": "full_available_stay",
            "label": "Whole stay",
            "anchor": "ICU admission",
        },
    }

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.make_research_pipeline_run_runner(
            export_path="/typed/demo",
            study_context=study,
            project_root=None,
            provider={"provider": "openai", "external": True},
        )

    assert exc.value.code == "research_pipeline_time_window_hours_required"
    assert exc.value.details == {"field": "time_window.hours"}
    assert foundation_called is False


@pytest.mark.parametrize("hours", ["not-a-number", True, float("nan")])
def test_pipeline_factory_rejects_invalid_time_window_without_24h_fallback(
    hours: Any,
) -> None:
    study = {
        **_complete_study(),
        "time_window": {"hours": hours, "anchor": "ICU admission"},
    }

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.make_research_pipeline_run_runner(
            export_path="/typed/demo",
            study_context=study,
            project_root=None,
            provider={"provider": "openai", "external": True},
        )

    assert exc.value.code == "research_pipeline_time_window_invalid"


@pytest.mark.parametrize("database", [None, "unknown-icu-database"])
def test_pipeline_factory_rejects_missing_or_unknown_database(database: Any) -> None:
    source = dict(_complete_study()["data_source"])
    source["database"] = database
    study = {**_complete_study(), "data_source": source}

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.make_research_pipeline_run_runner(
            export_path="/typed/demo",
            study_context=study,
            project_root=None,
            provider={"provider": "openai", "external": True},
        )

    assert exc.value.code == (
        "research_pipeline_database_required"
        if database is None
        else "research_pipeline_database_unknown"
    )


def test_pipeline_factory_rejects_clinical_anchor_as_materialization_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation_called = False

    def foundation(**_kwargs: Any) -> dict[str, Any]:
        nonlocal foundation_called
        foundation_called = True
        return _foundation_profile()

    monkeypatch.setattr(agent_pipeline_runs, "_data_foundation_profile", foundation)
    study = {
        **_complete_study(),
        "time_window": {
            "hours": 24,
            "anchor": "suspected infection onset",
        },
    }

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.make_research_pipeline_run_runner(
            export_path="/typed/demo",
            study_context=study,
            project_root=None,
            provider={"provider": "openai", "external": True},
        )

    assert exc.value.code == (
        "research_pipeline_materialization_window_anchor_unsupported"
    )
    assert exc.value.details["supported_anchor"] == "icu_admission"
    assert exc.value.details["window_role"] == "outer_observation_window"
    assert foundation_called is False


def test_pipeline_factory_rejects_unaddressed_repeat_stay_dependence_before_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation_called = False

    def foundation(**_kwargs: Any) -> dict[str, Any]:
        nonlocal foundation_called
        foundation_called = True
        return _foundation_profile()

    monkeypatch.setattr(agent_pipeline_runs, "_data_foundation_profile", foundation)
    study = {
        **_complete_study(),
        "cohort": {
            **_complete_study()["cohort"],
            "exclude_readmissions": False,
        },
    }

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.make_research_pipeline_run_runner(
            export_path="/typed/demo",
            study_context=study,
            project_root=None,
            provider={"provider": "openai", "external": True},
        )

    assert exc.value.code == ("research_pipeline_repeated_stay_dependence_unaddressed")
    assert exc.value.details["required_design"] == {
        "variance_estimator": "cluster_robust",
        "cluster_unit": "patient",
    }
    assert foundation_called is False


def test_pipeline_design_gate_accepts_counts_only_repeat_stays() -> None:
    study = {
        **_complete_study(),
        "cohort": {
            **_complete_study()["cohort"],
            "exclude_readmissions": False,
        },
        "analysis_design": {
            "analysis_family": "descriptive_epidemiology",
            "analysis_unit": "icu_stay",
            "variance_estimator": "none_counts_only",
        },
    }

    assert agent_pipeline_runs.validate_analysis_design_for_execution(study) == {
        "analysis_unit": "icu_stay",
        "variance_estimator": "none_counts_only",
    }


def test_workflow_projection_keeps_plan_review_before_analysis() -> None:
    study = _complete_study()
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_type": "full",
            "run_id": "run-plan-review",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "run_status": "human_review_pending",
            "pending_review_reason_codes": ["operator_plan_approval_required"],
            "artifact_names": [
                "agent_plan.json",
                "evidence_ledger.json",
                "source_run_manifest.json",
            ],
        },
        plan_review_authority={
            "run_id": "run-plan-review",
            "resumable_here": True,
            "scientific_configuration_sha256": (
                study_context_owner.scientific_configuration_sha256(study)
            ),
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


def test_nonapprovable_plan_projects_bounded_score_and_authorization_questions() -> (
    None
):
    study = _complete_study()
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_type": "full",
            "run_id": "run-science-review",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "run_status": "human_review_pending",
            "pending_review_reason_codes": ["plan_scientific_changes_required"],
            "artifact_names": [
                "agent_plan.json",
                "scientific_plan_review.json",
                "evidence_ledger.json",
                "source_run_manifest.json",
            ],
        },
        plan_review_authority={
            "run_id": "run-science-review",
            "resumable_here": True,
            "scientific_configuration_sha256": (
                study_context_owner.scientific_configuration_sha256(study)
            ),
            "scientific_plan_review": {
                "status": "changes_required",
                "score": 58,
                "top_journal_candidate": False,
                "dimension_scores": {
                    "icu_clinical_design": 0,
                    "figures": 70,
                },
                "findings": [
                    {
                        "code": "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED",
                        "requires_user_authorization": True,
                        "authorization_question": (
                            "Use a new landmark version or keep this study descriptive?"
                        ),
                    }
                ],
                "facts": {
                    "remediation_buckets": {
                        "agent_plan_revision": [
                            "CONTINUOUS_COVARIATE_FUNCTIONAL_FORM_UNCHECKED"
                        ],
                        "study_authority_change": [
                            "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"
                        ],
                        "external_evidence": ["DIRECT_COMPARATOR_NOT_ESTABLISHED"],
                        "independent_review": [],
                    }
                },
            },
        },
    )

    assert snapshot.next_action_code == "plan_scientific_changes_required"
    assert snapshot.plan_review_summary == {
        "status": "changes_required",
        "score": 58,
        "top_journal_candidate": False,
        "review_scope": "pre_execution_plan",
        "rendered_outputs_assessed": False,
        "dimension_scores": {"icu_clinical_design": 0, "figures": 70},
        "finding_codes": ["POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"],
        "authorization_questions": [
            {
                "code": "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED",
                "question": "Use a new landmark version or keep this study descriptive?",
            }
        ],
        "remediation_buckets": {
            "agent_plan_revision": ["CONTINUOUS_COVARIATE_FUNCTIONAL_FORM_UNCHECKED"],
            "study_authority_change": ["POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"],
            "external_evidence": ["DIRECT_COMPARATOR_NOT_ESTABLISHED"],
            "independent_review": [],
        },
    }


@pytest.mark.parametrize(
    ("plan_review_authority", "stored_digest", "expected_reason"),
    [
        (None, "", "plan_review_not_resumable"),
        (
            {
                "run_id": "run-old-plan",
                "resumable_here": True,
                "scientific_configuration_sha256": "a" * 64,
            },
            "a" * 64,
            "plan_configuration_superseded",
        ),
    ],
)
def test_workflow_never_offers_approval_for_stale_or_unresumable_plan(
    plan_review_authority: dict[str, Any] | None,
    stored_digest: str,
    expected_reason: str,
) -> None:
    snapshot = build_research_workflow_snapshot(
        study=_complete_study(),
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_id": "run-old-plan",
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "run_status": "human_review_pending",
            "pending_review_reason_codes": ["operator_plan_approval_required"],
            "scientific_configuration_sha256": stored_digest,
            "artifact_names": [
                "agent_plan.json",
                "evidence_ledger.json",
                "source_run_manifest.json",
            ],
        },
        plan_review_authority=plan_review_authority,
    )

    by_id = {row.id: row for row in snapshot.stages}
    assert snapshot.current_stage == "plan"
    assert snapshot.next_action_code == expected_reason
    assert by_id["plan"].status == "ready"
    assert by_id["plan"].reason_code == expected_reason
    assert by_id["analysis"].status == "blocked"
    assert by_id["analysis"].reason_code == expected_reason


def test_fresh_planning_job_takes_precedence_over_superseded_plan() -> None:
    snapshot = build_research_workflow_snapshot(
        study=_complete_study(),
        active_export_present=True,
        active_job={"kind": "agent-run", "status": "running"},
        latest_run={
            "run_id": "run-old-plan",
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "run_status": "human_review_pending",
            "pending_review_reason_codes": ["operator_plan_approval_required"],
            "scientific_configuration_sha256": "a" * 64,
            "artifact_names": [
                "agent_plan.json",
                "evidence_ledger.json",
                "source_run_manifest.json",
            ],
        },
        plan_review_authority=None,
    )

    by_id = {row.id: row for row in snapshot.stages}
    assert snapshot.next_action_code == "analysis_running"
    assert by_id["plan"].status == "running"
    assert by_id["analysis"].status == "running"


def test_terminal_failed_pipeline_returns_to_fresh_plan_confirmation() -> None:
    snapshot = build_research_workflow_snapshot(
        study=_complete_study(),
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_id": "run-failed-history",
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "run_status": "completed",
            "artifact_names": [
                "agent_plan.json",
                "evidence_ledger.json",
                "source_run_manifest.json",
            ],
        },
    )

    by_id = {row.id: row for row in snapshot.stages}
    assert snapshot.current_stage == "plan"
    assert snapshot.next_action_code == "failed_pipeline_requires_fresh_plan"
    assert by_id["plan"].status == "ready"
    assert by_id["plan"].reason_code == "failed_pipeline_requires_fresh_plan"
    assert by_id["analysis"].status == "blocked"
    assert by_id["analysis"].reason_code == "failed_pipeline_requires_fresh_plan"


def test_completed_preflight_advances_to_provider_plan_confirmation() -> None:
    snapshot = build_research_workflow_snapshot(
        study=_complete_study(),
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_type": "preflight",
            "gate_status": "analysis_only",
            "readiness_status": "awaiting_human_signoff",
            "artifact_names": [
                "cohort_summary.json",
                "quality_gate.json",
                "evidence_ledger.json",
            ],
        },
    )

    assert snapshot.current_stage == "plan"
    assert snapshot.next_action_code == "provider_plan_ready"
    plan = next(row for row in snapshot.stages if row.id == "plan")
    assert plan.status == "ready"
    assert plan.reason_code == "provider_plan_ready"


def test_active_export_must_belong_to_the_bound_study() -> None:
    study = _complete_study()
    assert active_export_matches_study(study, "/private/prepared/source") is True
    assert active_export_matches_study(study, "/private/another/export") is False
    assert (
        active_export_matches_study({"id": "study-without-source"}, "/active") is False
    )


def test_project_bound_registered_export_does_not_depend_on_global_active_source() -> (
    None
):
    study = _complete_study()
    registry = {
        "active_path": "/private/another/export",
        "sources": [
            {
                "id": "src_project",
                "path": "/private/prepared/source",
                "ok": True,
            },
            {
                "id": "src_global",
                "path": "/private/another/export",
                "ok": True,
            },
        ],
    }

    assert registered_export_matches_study(study, registry) is True
    registry["sources"][0]["ok"] = False
    assert registered_export_matches_study(study, registry) is False


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
        result_tables={
            "tables": [
                {
                    "name": "primary_estimate.csv",
                    "label": "Primary estimate",
                    "evidence_id": "ev-primary",
                    "headers": [
                        "estimate",
                        "ci_low",
                        "ci_high",
                        "effect_scale",
                    ],
                    "rows": [["1.25", "1.10", "1.42", "odds_ratio"]],
                }
            ]
        },
    )
    assert card.status == "analysis_only"
    assert card.generated_numbers is False
    assert card.source == "research_agent_artifacts_only"
    assert card.claims[0].evidence_ids == ["ev_table1"]
    assert card.result_tables[0].evidence_id == "ev-primary"
    assert card.result_tables[0].entries == [["1.25", "1.10", "1.42", "odds_ratio"]]
    assert card.human_review_required is True


def test_interpretation_and_manuscript_tools_bound_large_agent_drafts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ToolExecutionContext(
        session=PiSessionRecord(session_id="pi-result-review")
    )
    manuscript = {
        "status": "locked_pending_human_review",
        "question": "What is the bounded aggregate association?",
        "source": "research_agent_manuscript_scaffold_bound",
        "claims": [
            {
                "text": f"Evidence-bound claim {index}: " + ("bounded text " * 180),
                "evidence_ids": [f"ev-{index}"],
                "status": "bound",
            }
            for index in range(28)
        ],
        "markdown_preview": "full manuscript " * 10_000,
    }
    review = {
        "gate": {
            "status": "analysis_only",
            "reason": "research_agent_pipeline_complete_human_interpretation_required",
            "checks": [
                {"id": "numeric_verified", "passed": True},
                {"id": "paper_authorized", "passed": False},
            ],
        },
        "readiness": {
            "status": "blocked",
            "reportable": False,
            "signable": False,
        },
        "artifacts": [
            {"name": "result_tables.json", "sha256": "a" * 64, "bytes": 100},
            {"name": "manuscript_draft.json", "sha256": "b" * 64, "bytes": 100},
        ],
        "artifact_payloads": {
            "manuscript_draft.json": manuscript,
            "result_tables.json": {
                "tables": [
                    {
                        "name": "primary_estimate.csv",
                        "label": "Primary aggregate estimate",
                        "evidence_id": "ev-primary",
                        "headers": [
                            "estimate",
                            "ci_low",
                            "ci_high",
                            "effect_scale",
                        ],
                        "rows": [["1.25", "1.10", "1.42", "odds_ratio"]],
                    }
                ]
            },
            "scientific_readiness.json": {
                "findings": [
                    {
                        "code": "PAPER_AUTHORITY_NOT_GRANTED",
                        "message": "Publication authority remains withheld.",
                    }
                ]
            },
        },
        "signed": False,
        "signoff_stale": False,
        "signoff": None,
    }
    monkeypatch.setattr(
        tool_module,
        "_select_run",
        lambda _context, _requested_run_id=None: {
            "run_id": "run-result-review",
            "project_dir": "/host-only/run",
        },
    )
    monkeypatch.setattr(tool_module, "_run_review", lambda _row: review)

    interpretation = tool_module.execute_tool(
        "easyicu_inspect_interpretation",
        {"run_id": "run-result-review"},
        context,
    )
    draft = tool_module.execute_tool(
        "easyicu_inspect_manuscript",
        {"run_id": "run-result-review"},
        context,
    )

    assert interpretation["code"] == "easyicu_result_interpretation_projected"
    assert interpretation["details"]["interpretation"]["result_tables"][0][
        "entries"
    ] == [["1.25", "1.10", "1.42", "odds_ratio"]]
    assert len(interpretation["details"]["interpretation"]["claims"]) == 12
    assert draft["code"] == "easyicu_manuscript_projected"
    assert len(draft["details"]["manuscript"]["review_claims"]) == 12
    assert "markdown_preview" not in draft["details"]["manuscript"]
    assert len(json.dumps(interpretation).encode("utf-8")) < 32_768
    assert len(json.dumps(draft).encode("utf-8")) < 32_768


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


def test_curated_literature_projection_is_honest_and_does_not_backfill_plan_links() -> (
    None
):
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
                "searched_at": "2026-08-11T12:00:00+00:00",
                "search_queries": {"pubmed": ["ICU AND exposure AND outcome"]},
            },
            "screening_decisions": [
                {
                    "citation_key": "method_key",
                    "source": "pubmed",
                    "disposition": "include",
                    "evidence_role": "direct_comparator",
                    "rationale": "P/E/O matched in the retained abstract.",
                    "population_match": True,
                    "exposure_match": True,
                    "outcome_match": True,
                    "design_excerpt_available": True,
                }
            ],
        },
        plan={
            "steps": [
                {
                    "step_id": "primary",
                    "planned_analysis_role": "primary",
                    "intent": "Estimate the primary association.",
                    "literature_citation_keys": ["method_key", "invented_key"],
                    "literature_design_bindings": [
                        {
                            "citation_key": "method_key",
                            "design_elements": ["estimand"],
                            "application": "Use the article to prespecify the estimand.",
                        }
                    ],
                },
                {
                    "step_id": "render",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Render the already-bound estimate.",
                },
            ]
        },
    )

    assert payload["status"] == "searched"
    assert payload["mapping_status"] == "partial"
    assert payload["scientific_mapping_status"] == "complete"
    assert payload["scientific_plan_step_count"] == 1
    assert payload["scientific_mapped_step_count"] == 1
    assert payload["search"]["searched_at"] == "2026-08-11T12:00:00+00:00"
    assert payload["search"]["queries"]["pubmed"] == ["ICU AND exposure AND outcome"]
    assert payload["direct_comparator_keys"] == ["method_key"]
    assert payload["citations"][0]["screening"]["population_match"] is True
    assert payload["citation_year_range"] == {"oldest": None, "newest": None}
    assert payload["step_citation_map"][0]["citation_keys"] == ["method_key"]
    assert (
        payload["step_citation_map"][0]["citation_bindings"][0]["evidence_role"]
        == "direct_comparator"
    )
    assert payload["integrity"]["unknown_citation_keys_removed"] == ["invented_key"]
    assert (
        payload["citations"][0]["source_url"]
        == "https://pubmed.ncbi.nlm.nih.gov/12345/"
    )


def test_web_projection_refuses_ineligible_publication_type_as_comparator() -> None:
    payload = project_run_literature(
        run_id="run-literature-review",
        bundle={
            "citations": [
                {
                    "key": "review_key",
                    "title": "Systematic review of the same ICU question",
                    "year": "2025",
                    "pmid": "12346",
                    "publication_types": ["Systematic Review", "Review"],
                }
            ],
            "search_provenance": {
                "curated_seed_count": 0,
                "sources_enabled": ["pubmed"],
                "sources_returning": ["pubmed"],
                "search_conducted": True,
                "search_queries": {"pubmed": ["ICU question"]},
            },
            "screening_decisions": [
                {
                    "citation_key": "review_key",
                    "source": "pubmed",
                    "disposition": "include",
                    "evidence_role": "direct_comparator",
                    "rationale": "Legacy decision before publication-type gate.",
                    "population_match": True,
                    "exposure_match": True,
                    "outcome_match": True,
                    "design_excerpt_available": True,
                    "publication_type_eligible": False,
                }
            ],
        },
        plan={"steps": []},
    )

    assert payload["direct_comparator_count"] == 0
    assert payload["citations"][0]["publication_types"] == [
        "Systematic Review",
        "Review",
    ]
    assert payload["citations"][0]["screening"]["publication_type_eligible"] is False


def test_loaded_literature_projection_uses_digest_verified_final_plan(
    tmp_path: Path,
) -> None:
    import hashlib

    from easyicu.webserver.literature_projection import load_run_literature_projection

    bundle = {
        "citations": [{"key": "method_key", "title": "A method paper"}],
        "search_provenance": {"search_conducted": False},
    }
    (tmp_path / "preplan_literature_bundle.json").write_text(
        json.dumps(bundle), encoding="utf-8"
    )
    initial = {
        "steps": [
            {
                "step_id": "primary",
                "planned_analysis_role": "primary",
                "literature_citation_keys": ["method_key"],
            }
        ]
    }
    (tmp_path / "analysis_plan.json").write_text(json.dumps(initial), encoding="utf-8")
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    final_path = evidence_dir / "analysis_plan_revision_2.json"
    final = {
        "steps": [
            {
                "step_id": "primary",
                "planned_analysis_role": "primary",
                "literature_citation_keys": [],
            }
        ]
    }
    final_raw = json.dumps(final).encode("utf-8")
    final_path.write_bytes(final_raw)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "current_plan_authority": {
                    "relative_path": "evidence/analysis_plan_revision_2.json",
                    "sha256": hashlib.sha256(final_raw).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )

    payload = load_run_literature_projection(
        run_dir=tmp_path,
        run_id="run-final-plan",
    )

    assert payload["scientific_mapping_status"] == "not_bound"
    assert payload["integrity"]["current_plan_authority_verified"] is True


def test_literature_search_tool_uses_separate_one_turn_network_grant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_module, "_bound_context", lambda binding: _complete_study()
    )
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
    methodology = result["details"]["literature_search"]["methodology"]
    assert methodology["schema_version"].startswith("easyicu.method_literature_pack/")
    assert len(methodology["sha256"]) == 64
    assert {
        "reporting_standard",
        "time_alignment",
        "dependence",
        "functional_form",
        "missing_data",
        "interpretation",
    } <= {row["layer"] for row in methodology["cards"]}
    assert any(row.get("pmid") == "17938396" for row in methodology["sources"])
    consumed = tool_module.execute_tool("easyicu_search_literature", {}, context)
    assert consumed["code"] == "pi_action_grant_consumed"


def test_direct_study_literature_search_requires_typed_exposure_and_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _complete_study()
    study.update(
        {
            "question": "Study an ICU exposure and outcome.",
            "primary_exposure": "",
            "outcome": "",
            "execution_concepts": {},
        }
    )
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: study)
    search_called = False

    def discover(body: dict[str, Any]) -> dict[str, Any]:
        nonlocal search_called
        search_called = True
        return {}

    monkeypatch.setattr(tool_module.idea_mining, "discover_literature", discover)
    context = ToolExecutionContext(
        session=PiSessionRecord(session_id="pi-incomplete-literature"),
        allowed_actions={"literature"},
    )

    result = tool_module.execute_tool("easyicu_search_literature", {}, context)

    assert result["status"] == "blocked"
    assert result["code"] == "literature_study_scope_incomplete"
    assert search_called is False
    assert "literature" in context.allowed_actions


def test_literature_search_compiles_query_from_typed_execution_concepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _complete_study()
    study.update(
        {
            "question": "Estimate a governed ICU association.",
            "primary_exposure": (
                "Canonical EasyICU Sepsis-3: suspected infection plus "
                "traditional SOFA >=2 point increase, anchored to onset"
            ),
            "outcome": "In-hospital mortality",
            "execution_concepts": {
                "primary_exposure": "sep3_sofa1",
                "outcome": "death",
                "covariates": [],
            },
        }
    )
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: study)
    captured: dict[str, Any] = {}

    def discover(body: dict[str, Any]) -> dict[str, Any]:
        captured.update(body)
        return {
            "status": "searched_no_hits",
            "search_performed": True,
            "queries_to_run": [
                '("Sepsis-3"[Title/Abstract] AND "mortality"[Title/Abstract])'
            ],
            "network_calls": 1,
            "source_candidates": [],
        }

    monkeypatch.setattr(tool_module.idea_mining, "discover_literature", discover)

    result = tool_module.execute_tool(
        "easyicu_search_literature",
        {},
        ToolExecutionContext(
            session=PiSessionRecord(session_id="pi-typed-literature-query"),
            allowed_actions={"literature"},
        ),
    )

    assert result["status"] == "ok"
    assert captured["exposure_concept"] == "sep3_sofa1"
    assert captured["outcome_concept"] == "death"


def test_literature_search_binds_exact_receipt_to_real_study_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _complete_study()
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: study)
    monkeypatch.setattr(
        tool_module.study_contexts,
        "get_context",
        lambda study_id: study if study_id == study["id"] else None,
    )
    persisted = {
        "schema_version": "easyicu.web-literature-authority/2",
        "receipt_id": "lit_" + "a" * 24,
        "receipt_sha256": "b" * 64,
        "status": "searched",
        "result_count": 1,
        "searched_at": "2026-08-12T12:00:00+00:00",
        "study_configuration_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        literature_authority,
        "persist_literature_authority",
        lambda **kwargs: persisted,
    )
    writes: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def bind(
        study_id: str,
        authority: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        writes.append(({"id": study_id, "literature_authority": authority}, kwargs))
        return {**study, "literature_authority": authority, "revision": 5}

    monkeypatch.setattr(
        tool_module.study_contexts,
        "bind_literature_authority",
        bind,
    )
    monkeypatch.setattr(
        tool_module.idea_mining,
        "discover_literature",
        lambda body: {
            "status": "searched",
            "search_performed": True,
            "searched_at": "2026-08-12T12:00:00+00:00",
            "queries_to_run": ["ICU feature AND mortality"],
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
        session=PiSessionRecord(
            session_id="pi-bound-generic-literature",
            binding=AuthorityBinding(
                study_context_id=study["id"],
                study_revision=study["revision"],
            ),
        ),
        allowed_actions={"literature"},
    )

    result = tool_module.execute_tool("easyicu_search_literature", {}, context)

    assert result["code"] == "easyicu_literature_search_completed"
    assert (
        result["details"]["literature_search"]["study_literature_authority"]
        == persisted
    )
    assert result["details"]["rebind_required"] is True
    assert writes == [
        (
            {"id": study["id"], "literature_authority": persisted},
            {
                "expected_revision": study["revision"],
            },
        )
    ]
    with pytest.raises(PiCopilotError) as stale:
        context.assert_authority_fresh()
    assert stale.value.code == "pi_session_authority_stale"


def test_generic_literature_authority_returns_compact_rebind_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _complete_study()
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: study)
    monkeypatch.setattr(
        tool_module.study_contexts,
        "get_context",
        lambda study_id: study if study_id == study["id"] else None,
    )
    persisted = {
        "schema_version": "easyicu.web-literature-authority/2",
        "receipt_id": "lit_" + "a" * 24,
        "receipt_sha256": "b" * 64,
        "status": "searched",
        "result_count": 8,
        "searched_at": "2026-08-12T12:00:00+00:00",
        "study_configuration_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        literature_authority,
        "persist_literature_authority",
        lambda **kwargs: persisted,
    )
    monkeypatch.setattr(
        tool_module.study_contexts,
        "bind_literature_authority",
        lambda study_id, authority, **kwargs: {
            **study,
            "literature_authority": authority,
            "revision": 5,
        },
    )
    long_query = "Sepsis-3[Title/Abstract] AND mortality[Title/Abstract] " * 30
    monkeypatch.setattr(
        tool_module.idea_mining,
        "discover_literature",
        lambda body: {
            "status": "searched",
            "search_performed": True,
            "searched_at": "2026-08-12T12:00:00+00:00",
            "queries_to_run": [f"{index} {long_query}" for index in range(4)],
            "query_strata": [
                {
                    "id": f"stratum_{index}",
                    "query": f"{index} {long_query}",
                    "returned_count": 20,
                    "retained_count": 2,
                }
                for index in range(4)
            ],
            "network_calls": 8,
            "source_candidates": [
                {
                    "citation_key": f"paper_{index}",
                    "title": f"Adult ICU Sepsis-3 study {index}",
                    "journal": "Critical Care",
                    "year": 2025,
                    "pmid": str(10_000 + index),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{10_000 + index}/",
                    "design_excerpt": "adult ICU observational excerpt " * 80,
                    "publication_types": ["Observational Study"],
                    "matched_query_strata": [f"stratum_{index % 4}"],
                    "matched_queries": [f"{query} {long_query}" for query in range(4)],
                }
                for index in range(8)
            ],
        },
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-compact-generic-literature",
            binding=AuthorityBinding(
                study_context_id=study["id"],
                study_revision=study["revision"],
            ),
        ),
        allowed_actions={"literature"},
    )

    result = tool_module.execute_tool(
        "easyicu_search_literature", {"limit": 8}, context
    )

    assert result["status"] == "ok"
    assert result["details"]["host_rebind_after_turn"] is True
    assert result["details"]["authority_update"] == {
        "study_context_id": study["id"],
        "study_revision": 5,
        "reason": "study_literature_authority_updated",
    }
    literature = result["details"]["literature_search"]
    assert literature["study_literature_authority"] == persisted
    assert literature["exact_queries_bound_in_host_receipt"] is True
    assert literature["query_previews_truncated"] is True
    assert len(literature["articles"]) == 8
    assert all("matched_queries" not in row for row in literature["articles"])
    assert len(result["details"]["resources"]) <= 8
    assert "study" not in result["details"]
    assert len(json.dumps(result, ensure_ascii=False).encode("utf-8")) < 20_000
    with pytest.raises(PiCopilotError) as stale:
        context.assert_authority_fresh()
    assert stale.value.code == "pi_session_authority_stale"


def test_literature_search_binds_prior_art_to_an_accepted_idea(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _complete_study()
    study["idea_handoff"] = {
        "run_id": "idea-run-1",
        "idea_id": "idea-1",
        "status": "accepted",
    }
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: study)

    def fail_discovery(body: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("accepted ideas must use their persisted prior-art owner")

    monkeypatch.setattr(
        tool_module.idea_mining,
        "discover_literature",
        fail_discovery,
    )
    monkeypatch.setattr(
        tool_module.idea_mining,
        "check_prior_art",
        lambda body: {
            "prior_art": {
                "status": "searched",
                "search_performed": True,
                "network_calls": 2,
                "queries_to_run": ["sepsis AND mortality"],
                "results": [
                    {
                        "pmid": "26903338",
                        "title": "Sepsis-3 consensus definitions",
                        "source": "JAMA",
                        "pubdate": "2016",
                        "query": "sepsis AND mortality",
                        "abstract_excerpt": (
                            "Sepsis is life-threatening organ dysfunction caused by "
                            "a dysregulated host response to infection."
                        ),
                        "evidence_sentence": (
                            "The consensus defined Sepsis-3 using organ dysfunction."
                        ),
                    }
                ],
            }
        },
    )
    monkeypatch.setattr(
        tool_module.idea_mining,
        "prior_art_receipt_binding",
        lambda run_id: {
            "prior_art_binding_schema_version": "easyicu.idea-prior-art-binding/2",
            "prior_art_sha256": "a" * 64,
            "prior_art_status": "searched",
            "prior_art_result_count": 1,
        },
    )

    result = tool_module.execute_tool(
        "easyicu_search_literature",
        {},
        ToolExecutionContext(
            session=PiSessionRecord(session_id="pi-bound-literature"),
            allowed_actions={"literature"},
        ),
    )

    literature = result["details"]["literature_search"]
    assert result["code"] == "easyicu_literature_search_completed"
    assert literature["idea_handoff_refresh_required"] is True
    assert literature["bound_idea_run_id"] == "idea-run-1"
    assert result["details"]["resource"]["pmid"] == "26903338"
    assert (
        result["details"]["resource"]["relevance"]
        == "Sepsis is life-threatening organ dysfunction caused by a dysregulated host response to infection."
    )
    assert result["details"]["literature_search"]["articles"] == [
        {
            "citation_key": "idea_pubmed_26903338",
            "title": "Sepsis-3 consensus definitions",
            "journal": "JAMA",
            "year": "2016",
            "pmid": "26903338",
            "matched_query_strata": ["accepted_idea_prior_art"],
            "publication_types": [],
            "screening_status": "retrieval_candidate_unreviewed",
            "evidence_role": "retrieval_candidate",
            "evidence_excerpt": (
                "Sepsis is life-threatening organ dysfunction caused by a "
                "dysregulated host response to infection."
            ),
        }
    ]
    assert "re-accepted" in result["summary"]


def test_bound_literature_projection_stays_bounded_with_long_abstracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _complete_study()
    study["idea_handoff"] = {
        "run_id": "idea-run-bounded",
        "idea_id": "idea-bounded",
        "status": "accepted",
    }
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: study)
    monkeypatch.setattr(
        tool_module.idea_mining,
        "check_prior_art",
        lambda body: {
            "prior_art": {
                "status": "searched",
                "search_performed": True,
                "network_calls": 2,
                "queries_to_run": ["sepsis AND mortality"],
                "results": [
                    {
                        "pmid": str(10000 + index),
                        "title": f"Sepsis paper {index}",
                        "journal": "Critical Care",
                        "year": 2025,
                        "abstract_excerpt": "organ dysfunction " * 180,
                    }
                    for index in range(12)
                ],
            }
        },
    )
    monkeypatch.setattr(
        tool_module.idea_mining,
        "prior_art_receipt_binding",
        lambda run_id: {
            "prior_art_binding_schema_version": "easyicu.idea-prior-art-binding/2",
            "prior_art_sha256": "a" * 64,
            "prior_art_status": "searched",
            "prior_art_result_count": 12,
        },
    )

    result = tool_module.execute_tool(
        "easyicu_search_literature",
        {},
        ToolExecutionContext(
            session=PiSessionRecord(session_id="pi-bounded-literature"),
            allowed_actions={"literature"},
        ),
    )

    assert result["status"] == "ok"
    assert 5 < len(result["details"]["resources"]) <= 8
    assert any(row.get("pmid") == "17938396" for row in result["details"]["resources"])
    assert len(result["details"]["literature_search"]["articles"]) == 5
    assert all(
        len(row["evidence_excerpt"]) <= 360
        for row in result["details"]["literature_search"]["articles"]
    )
    assert len(json.dumps(result)) < 20_000


def test_literature_source_resource_rejects_unverified_or_unsafe_links() -> None:
    assert (
        literature_source_resource({"title": "Unsafe", "url": "javascript:alert(1)"})
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
                    "requested_adjustment_concepts": ["age", "sex"],
                    "mapped_concepts": [
                        {"concept_id": "lact", "module": "blood_gas"},
                        {"concept_id": "death", "module": "outcome"},
                        {"concept_id": "age", "module": "demographics"},
                        {"concept_id": "sex", "module": "demographics"},
                    ],
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


def test_accept_idea_handoff_does_not_turn_every_mentioned_feature_into_a_covariate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _complete_study()
    current.update({"revision": 3, "active_job_id": None})
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: current)
    monkeypatch.setattr(
        tool_module.idea_mining,
        "plan_idea",
        lambda body: {"plan": {"research_question": "Question"}},
    )
    monkeypatch.setattr(
        tool_module.idea_mining,
        "create_handoff",
        lambda body: {
            "idea_id": "idea-sepsis",
            "canonical_handoff_sha256": "c" * 64,
            "canonical_handoff": {
                "analysis_family": "association_study",
                "resolved_predictor_concept": "sep3_sofa1",
                "resolved_outcome_concept": "death",
                "resolved_analysis_concepts": [
                    "sep3_sofa1",
                    "lact",
                    "age",
                    "sex",
                    "urine",
                ],
                "selected_ledger_row": {
                    "requested_adjustment_concepts": ["age", "sex"],
                    "mapped_concepts": [
                        {"concept_id": "sep3_sofa1", "module": "sepsis3_sofa1"},
                        {"concept_id": "death", "module": "outcome"},
                        {"concept_id": "lact", "module": "blood_gas"},
                        {"concept_id": "age", "module": "demographics"},
                        {"concept_id": "sex", "module": "demographics"},
                        {"concept_id": "urine", "module": "renal"},
                    ],
                },
            },
            "go_no_go": "recommend",
        },
    )

    def upsert(body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        writes.append(dict(body))
        return {**current, **body, "revision": 4}

    monkeypatch.setattr(tool_module.study_contexts, "upsert_context", upsert)
    result = tool_module.execute_tool(
        "easyicu_accept_idea_handoff",
        {"run_id": "idea-run", "idea_id": "idea-sepsis"},
        ToolExecutionContext(
            session=PiSessionRecord(session_id="pi-adjustment"),
            allowed_actions={"idea"},
        ),
    )

    assert result["code"] == "easyicu_idea_handoff_accepted"
    assert writes[0]["covariates"] == ["age", "sex"]
    assert writes[0]["modules"] == [
        "sepsis3_sofa1",
        "outcome",
        "blood_gas",
        "demographics",
        "renal",
    ]


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
                    "mapped_concepts": [{"concept_id": "lact", "module": "blood_gas"}]
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


def test_extraction_reuses_project_bound_registered_export_even_when_not_global_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_module, "_bound_context", lambda binding: _complete_study()
    )
    monkeypatch.setattr(
        tool_module.sources,
        "load_registry",
        lambda: {
            "active_path": "/private/another/export",
            "sources": [
                {
                    "id": "src_project",
                    "path": "/private/prepared/source",
                    "database": "mimiciv",
                    "ok": True,
                }
            ],
        },
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(session_id="pi-extract-reuse"),
        allowed_actions={"extract"},
    )

    result = tool_module.execute_tool("easyicu_start_extraction", {}, context)

    assert result["code"] == "easyicu_registered_export_reused"
    assert result["details"]["active_export"]["source_id"] == "src_project"
    assert "/private/" not in json.dumps(result)


def test_data_package_tool_returns_digest_bound_path_free_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _complete_study()
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: current)
    from easyicu.webserver import data_package_review as review_owner

    sealed: list[dict] = []
    monkeypatch.setattr(
        review_owner,
        "DataPackageReviewSnapshotStore",
        lambda: SimpleNamespace(persist=lambda payload: sealed.append(dict(payload))),
    )

    monkeypatch.setattr(
        review_owner,
        "build_registered_data_package_review",
        lambda study: {
            "schema_version": "easyicu.data-package-review/1",
            "status": "ready_for_plan",
            "code": "easyicu_data_package_review_ready",
            "study_context_id": study["id"],
            "study_context_revision": study["revision"],
            "review_sha256": "d" * 64,
            "denominator": {"analysis_unit": "icu_stay", "count": 2000},
            "concepts": [],
            "analysis_results_withheld": True,
            "privacy": {"host_paths_returned": False},
        },
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-data-package",
            binding=AuthorityBinding(
                study_context_id=current["id"],
                study_revision=current["revision"],
            ),
        )
    )

    result = tool_module.execute_tool("easyicu_inspect_data_package", {}, context)

    assert result["code"] == "easyicu_data_package_review_ready"
    assert sealed and sealed[0]["review_sha256"] == "d" * 64
    assert result["details"]["resource"] == {
        "kind": "data_package_review",
        "study_context_id": "study-workflow",
        "study_revision": 4,
        "review_sha256": "d" * 64,
        "label": "Data package review",
        "media_type": "application/json",
    }
    assert "/private/" not in json.dumps(result)


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
            "path": "/private/prepared/source",
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
        "manuscript_ready": False,
    }
    (run_dir / "run_status.json").write_text("{}", encoding="utf-8")
    plan_payload = {
        "steps": [
            {
                "id": "model",
                "title": "Fit specified model",
                "literature_citation_keys": ["method_paper"],
            }
        ]
    }
    plan_bytes = json.dumps(plan_payload).encode("utf-8")
    (run_dir / "analysis_plan.json").write_bytes(plan_bytes)
    import hashlib

    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "readiness": readiness,
                "current_plan_authority": {
                    "relative_path": "analysis_plan.json",
                    "sha256": hashlib.sha256(plan_bytes).hexdigest(),
                },
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
        "row_role,exposure_level_index,exposure_level,n_rows,exposure_denominator,exposure_pct,exposure_ci_low_pct,exposure_ci_high_pct,exposure_standard_error_pct,exposure_interval_covariance,exposure_interval_cluster_count,outcome_observed_n,outcome_missing_n,outcome_events,outcome_denominator,outcome_rate_pct,interval_method\n"
        "exposure_level,0,0,60,100,60.0,,,,none_counts_only,,60,0,5,60,8.3,none_counts_only\n",
        encoding="utf-8",
    )
    (run_dir / "results" / "identifier_rows.csv").write_text(
        "metric,a,b,c,d,e,f,g,h,i,j,k,stay_id\n"
        "sensitive,1,2,3,4,5,6,7,8,9,10,11,123\n",
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
        "primary_exposure_source_concept": "heart_rate",
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
    assert "outcome_events" in tables["tables"][0]["headers"]
    assert "outcome_rate_pct" in tables["tables"][0]["headers"]
    event_index = tables["tables"][0]["headers"].index("outcome_events")
    assert tables["tables"][0]["rows"][0][event_index] == "5"
    assert tables["tables"][0]["preview_columns_truncated"] is True
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
    artifact_names = {row["name"] for row in ledger["artifacts"]}
    assert "literature_evidence.json" in artifact_names
    assert "system_validation_report.json" in artifact_names
    assert "system_validation_report_receipt.json" in artifact_names
    assert "system_validation_report.html" in artifact_names
    assert ledger["privacy"]["projection_scan_passed"] is True
    assert ledger["privacy"]["path_values_returned"] is False
    report = json.loads(
        (wrapper / "system_validation_report.json").read_text(encoding="utf-8")
    )
    assert report["authority_class"] == "engineering_validation_only"
    assert report["publication_authorized"] is False
    assert (wrapper / "system_validation_report.html").is_file()


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
    assert history["runs"][0]["scientific_configuration_sha256"] == (
        study_context_owner.scientific_configuration_sha256(_complete_study())
    )


def test_pending_plan_cannot_resume_after_scientific_setup_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "superseded-plan"
    run_dir.mkdir()
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review the digest-bound plan before analysis.",
        authority_sha256="a" * 64,
        payload={"reason": "operator_plan_approval_required"},
    )
    pending = HumanReviewPending(
        run_id="run-superseded-plan",
        thread_id="thread-superseded-plan",
        run_dir=str(run_dir),
        requests=(request,),
    )
    original = _complete_study()
    entry = agent_pipeline_runs._PendingRun(
        pipeline=SimpleNamespace(),
        pending=pending,
        wrapper_dir=tmp_path / "wrapper",
        study=original,
        provider={},
        acquisition=SimpleNamespace(),
        created_at=1.0,
    )
    monkeypatch.setitem(
        agent_pipeline_runs._PENDING,
        "run-superseded-plan",
        entry,
    )
    changed = {
        **original,
        "analysis_design": {
            "analysis_unit": "icu_stay",
            "variance_estimator": "cluster_robust",
            "cluster_unit": "hospital_admission",
        },
    }

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.resume_research_pipeline(
            run_id="run-superseded-plan",
            study_context_id="study-workflow",
            decision="approved",
            reviewer="local reviewer",
            note="",
            job=SimpleNamespace(emit=lambda _event: None, cancel_requested=False),
            current_study_context=changed,
        )

    assert exc.value.code == "research_pipeline_review_configuration_superseded"
    assert (
        exc.value.details["planned_scientific_configuration_sha256"]
        != exc.value.details["current_scientific_configuration_sha256"]
    )


def test_superseded_plan_can_still_be_rejected_without_execution_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.orchestration.workflow import HumanReviewRejected

    run_dir = tmp_path / "superseded-rejection"
    run_dir.mkdir()
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review the digest-bound plan before analysis.",
        authority_sha256="a" * 64,
        payload={"reason": "operator_plan_approval_required"},
    )
    pending = HumanReviewPending(
        run_id="run-superseded-rejection",
        thread_id="thread-superseded-rejection",
        run_dir=str(run_dir),
        requests=(request,),
    )
    original = _complete_study()

    class RejectingPipeline:
        def resume_human_review(self, decisions, **_kwargs):
            assert {row["decision"] for row in decisions} == {"rejected"}
            raise HumanReviewRejected([request.review_id])

    entry = agent_pipeline_runs._PendingRun(
        pipeline=RejectingPipeline(),
        pending=pending,
        wrapper_dir=tmp_path / "wrapper-rejection",
        study=original,
        provider={},
        acquisition=SimpleNamespace(),
        created_at=1.0,
    )
    monkeypatch.setitem(
        agent_pipeline_runs._PENDING,
        "run-superseded-rejection",
        entry,
    )
    monkeypatch.setattr(
        agent_pipeline_runs,
        "_write_projection",
        lambda **_kwargs: {"gate": {"status": "blocked"}},
    )
    monkeypatch.setattr(
        agent_pipeline_runs, "remove_review_recovery_record", lambda _run_id: None
    )
    monkeypatch.setattr(
        agent_pipeline_runs, "_remove_local_recovery", lambda _wrapper: None
    )

    changed = {
        **original,
        "analysis_design": {
            "analysis_unit": "icu_stay",
            "variance_estimator": "cluster_robust",
            "cluster_unit": "hospital_admission",
        },
    }
    result = agent_pipeline_runs.resume_research_pipeline(
        run_id="run-superseded-rejection",
        study_context_id="study-workflow",
        decision="rejected",
        reviewer="local reviewer",
        note="Superseded by a revised setup.",
        job=SimpleNamespace(emit=lambda _event: None, cancel_requested=False),
        current_study_context=changed,
    )

    assert result["gate"]["status"] == "blocked"


def test_pending_review_projects_from_the_typed_pause_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "pending-plan"
    run_dir.mkdir()
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review the digest-bound plan before analysis.",
        authority_sha256="c" * 64,
        payload={"reason": "operator_plan_approval_required"},
    )
    pending = HumanReviewPending(
        run_id="run-pending-plan",
        thread_id="thread-pending-plan",
        run_dir=str(run_dir),
        requests=(request,),
    )
    study = _complete_study()
    monkeypatch.setitem(
        agent_pipeline_runs._PENDING,
        pending.run_id,
        agent_pipeline_runs._PendingRun(
            pipeline=SimpleNamespace(),
            pending=pending,
            wrapper_dir=tmp_path / "wrapper",
            study=study,
            provider={},
            acquisition=SimpleNamespace(),
            created_at=1.0,
        ),
    )

    projected = agent_pipeline_runs.pending_review(pending.run_id)

    assert projected is not None
    assert projected["run_id"] == pending.run_id
    assert projected["scientific_plan_review"] == {}


def test_pending_plan_resume_routes_pipeline_events_to_the_resume_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "resumed-plan"
    run_dir.mkdir()
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review the digest-bound plan before analysis.",
        authority_sha256="b" * 64,
        payload={"reason": "operator_plan_approval_required"},
    )
    pending = HumanReviewPending(
        run_id="run-resumed-plan",
        thread_id="thread-resumed-plan",
        run_dir=str(run_dir),
        requests=(request,),
    )
    calls: dict[str, Any] = {}

    class _Pipeline:
        def resume_human_review(self, decisions, *, run_id, progress_callback=None):
            # Simulate the pipeline's run-locked Provider transition. The Web
            # bridge is forbidden from calling TaskProviderHardStop.resume.
            hard_stop.ledger.resume_task(hard_stop.task_id)
            calls["decisions"] = decisions
            calls["run_id"] = run_id
            progress_callback(
                {"stage": "coder", "message": "Generating analysis code."}
            )
            return SimpleNamespace(manifest_path=run_dir / "manifest.json")

    study, package_binding = _study_with_package_binding(
        tmp_path / "resume-package"
    )
    hard_stop = agent_pipeline_runs._start_web_provider_hard_stop(
        wrapper_dir=tmp_path / "wrapper",
        job_id="resume-contract",
        declaration_sha256=(
            study_context_owner.scientific_configuration_sha256(study)
        ),
    )
    hard_stop.pause()

    def web_resume_forbidden(_self: Any) -> None:
        raise AssertionError("Web must not resume Provider state before Pipeline lock")

    monkeypatch.setattr(type(hard_stop), "resume", web_resume_forbidden)
    entry = agent_pipeline_runs._PendingRun(
        pipeline=_Pipeline(),
        pending=pending,
        wrapper_dir=tmp_path / "wrapper",
        study=study,
        provider={},
        acquisition=_acquisition_receipt(),
        created_at=1.0,
        prepared_package_binding=package_binding,
        provider_hard_stop=hard_stop,
    )
    monkeypatch.setitem(agent_pipeline_runs._PENDING, pending.run_id, entry)
    monkeypatch.setattr(
        agent_pipeline_runs,
        "_write_projection",
        lambda **_kwargs: {"status": "done"},
    )

    class _Job:
        cancel_requested = False

        def __init__(self) -> None:
            self.events: list[dict] = []

        def emit(self, event: dict) -> None:
            self.events.append(event)

    job = _Job()
    assert hard_stop.ledger.snapshot()["tasks"][0]["status"] == "paused"
    result = agent_pipeline_runs.resume_research_pipeline(
        run_id=pending.run_id,
        study_context_id=study["id"],
        decision="approved",
        reviewer="local reviewer",
        note="",
        job=job,
        current_study_context=study,
    )

    assert result == {"status": "done"}
    assert calls["run_id"] == pending.run_id
    assert [event["step"] for event in job.events] == ["human_review", "coder"]
    assert job.events[-1]["label"] == "Generating analysis code."
    ledger = json.loads(hard_stop.ledger.path.read_text(encoding="utf-8"))
    assert ledger["tasks"][0]["task_id"] == hard_stop.task_id
    assert ledger["tasks"][0]["status"] == "completed"


def test_recoverable_review_resume_failure_keeps_pending_run_and_pauses_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "recoverable-review"
    run_dir.mkdir()
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review the digest-bound plan before analysis.",
        authority_sha256="d" * 64,
        payload={"reason": "operator_plan_approval_required"},
    )
    pending = HumanReviewPending(
        run_id="run-recoverable-review",
        thread_id="thread-recoverable-review",
        run_dir=str(run_dir),
        requests=(request,),
    )

    class _Pipeline:
        has_resumable_human_review = True

        def resume_human_review(self, decisions, *, run_id, progress_callback=None):
            raise ValueError("review decision evidence could not be persisted")

    study, package_binding = _study_with_package_binding(
        tmp_path / "recoverable-package"
    )
    hard_stop = agent_pipeline_runs._start_web_provider_hard_stop(
        wrapper_dir=tmp_path / "wrapper",
        job_id="recoverable-review",
        declaration_sha256=(
            study_context_owner.scientific_configuration_sha256(study)
        ),
    )
    hard_stop.pause()

    def web_pause_forbidden(_self: Any) -> None:
        raise AssertionError("Web must not pause Provider state after Pipeline returns")

    monkeypatch.setattr(type(hard_stop), "pause", web_pause_forbidden)
    entry = agent_pipeline_runs._PendingRun(
        pipeline=_Pipeline(),
        pending=pending,
        wrapper_dir=tmp_path / "wrapper",
        study=study,
        provider={},
        acquisition=_acquisition_receipt(),
        created_at=1.0,
        prepared_package_binding=package_binding,
        provider_hard_stop=hard_stop,
    )
    monkeypatch.setitem(agent_pipeline_runs._PENDING, pending.run_id, entry)

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.resume_research_pipeline(
            run_id=pending.run_id,
            study_context_id=study["id"],
            decision="approved",
            reviewer="local reviewer",
            note="",
            job=SimpleNamespace(emit=lambda _event: None, cancel_requested=False),
            current_study_context=study,
        )

    assert exc.value.code == "research_pipeline_review_resume_failed"
    assert exc.value.details["review_resumable"] is True
    assert agent_pipeline_runs._PENDING[pending.run_id] is entry
    task_row = hard_stop.ledger.snapshot()["tasks"][0]
    assert task_row["status"] == "paused"
    assert hard_stop.ledger.snapshot()["terminal"] is False


def test_review_resume_claim_is_exclusive_before_touching_provider_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "already-claimed-review"
    run_dir.mkdir()
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review the digest-bound plan before analysis.",
        authority_sha256="e" * 64,
        payload={"reason": "operator_plan_approval_required"},
    )
    pending = HumanReviewPending(
        run_id="run-already-claimed-review",
        thread_id="thread-already-claimed-review",
        run_dir=str(run_dir),
        requests=(request,),
    )

    class _Pipeline:
        def resume_human_review(self, *args, **kwargs):
            raise AssertionError("a second job must not touch the live workflow")

    study, package_binding = _study_with_package_binding(
        tmp_path / "contended-package"
    )
    hard_stop = agent_pipeline_runs._start_web_provider_hard_stop(
        wrapper_dir=tmp_path / "wrapper",
        job_id="already-claimed-review",
        declaration_sha256=(
            study_context_owner.scientific_configuration_sha256(study)
        ),
    )
    hard_stop.pause()
    entry = agent_pipeline_runs._PendingRun(
        pipeline=_Pipeline(),
        pending=pending,
        wrapper_dir=tmp_path / "wrapper",
        study=study,
        provider={},
        acquisition=_acquisition_receipt(),
        created_at=1.0,
        prepared_package_binding=package_binding,
        provider_hard_stop=hard_stop,
    )
    monkeypatch.setitem(agent_pipeline_runs._PENDING, pending.run_id, entry)

    class _ContendedLock:
        def __init__(self) -> None:
            self.entries = 0

        def __enter__(self):
            self.entries += 1
            if self.entries == 2:
                # Simulate another resume job claiming the entry after this
                # caller read it but before this caller obtains the lease.
                agent_pipeline_runs._PENDING.pop(pending.run_id, None)

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(agent_pipeline_runs, "_PENDING_LOCK", _ContendedLock())
    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.resume_research_pipeline(
            run_id=pending.run_id,
            study_context_id=study["id"],
            decision="approved",
            reviewer="local reviewer",
            note="",
            job=SimpleNamespace(emit=lambda _event: None, cancel_requested=False),
            current_study_context=study,
        )

    assert exc.value.code == "research_pipeline_review_resume_in_progress"
    assert hard_stop.ledger.snapshot()["tasks"][0]["status"] == "paused"


def test_guided_project_rail_projects_real_mode_from_bound_study_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver import guided_sessions

    rows = [
        {
            "id": "draft-real-project",
            "title": "Real MIMIC-IV project",
            "data_mode": "demo",
            "surface_visibility": "product",
            "updated_at": "2026-08-12T10:00:00Z",
        }
    ]
    monkeypatch.setattr(
        guided_sessions,
        "_read_raw",
        lambda: {"drafts": rows},
    )
    monkeypatch.setattr(
        guided_sessions,
        "read_project_study_setup",
        lambda project_id: SimpleNamespace(
            data_source={"database": "miiv", "label": "MIMIC-IV full"}
        ),
    )

    payload = guided_sessions.list_guided_drafts(limit=20)

    assert payload["drafts"][0]["data_mode"] == "real"
    assert rows[0]["data_mode"] == "demo", "projection mutated registry metadata"


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
            "auth_header": "x-api-key",
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
    assert captured["environment"]["EASYICU_OPENAI_AUTH_HEADER"] == "x-api-key"
    assert captured["request_timeout"] == 480.0
    assert captured["max_retries"] == 1
    assert captured["retryable_http_status_codes"] == (500, 502, 503, 504)
    assert captured["allow_environment_overrides"] is False
    assert private_key not in json.dumps(public)
    assert public["request_timeout_seconds"] == 480.0
    assert public["transport_max_attempts"] == 2
    assert public["retryable_http_status_codes"] == [500, 502, 503, 504]
    assert public["secrets_returned"] is False


def test_web_runner_timeout_is_typed_and_records_bounded_retry_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            exc = TimeoutError("provider request timed out")
            exc.add_note(
                "structured-retry history: validator rejected /Users/example/run "
                "api_key=test-secret-value"
            )
            raise exc

    monkeypatch.setattr(
        research_agent.ResearchAgentPipeline,
        "from_config",
        lambda _config, *, services: FakePipeline(),
    )
    project_root = tmp_path / "projects"
    export_path = _write_pipeline_export(tmp_path / "export")
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(export_path),
        study_context=_complete_study(),
        project_root=str(project_root),
        provider={"provider": "openai", "external": True},
        provider_environment=_PI_PROVIDER_ENVIRONMENT,
    )

    class Job:
        id = "job-timeout"
        cancel_requested = False
        events: list[dict[str, Any]] = []

        def emit(self, event: dict[str, Any]) -> None:
            self.events.append(dict(event))

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as raised:
        runner(Job())

    assert raised.value.code == "research_pipeline_provider_timeout"
    assert "No analysis was run" in str(raised.value)
    diagnostic = (
        project_root
        / "study-workflow"
        / "run_job-timeout"
        / "diagnostics"
        / "research_pipeline_failure.json"
    )
    payload = json.loads(diagnostic.read_text(encoding="utf-8"))
    rendered = json.dumps(payload)
    assert payload["raw_model_output_recorded"] is False
    assert payload["code"] == "research_pipeline_provider_timeout"
    assert "/Users/example" not in rendered
    assert "test-secret-value" not in rendered
    assert any(
        event.get("label", "").startswith("The model provider timed out")
        for event in Job.events
    )


def test_planner_failure_artifact_persists_only_safe_attempt_metadata(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.providers.structured_retry import (
        StructuredAttempt,
        StructuredResponseFailure,
    )

    secret = "sk-provider-secret-in-model-output"
    failure = StructuredResponseFailure(
        [
            StructuredAttempt(
                attempt=0,
                raw_head=f'{{"answer": "{secret}"}}',
                raw_chars=42,
                error_class="ValidationError",
                error_message=f"raw input included {secret}",
                finish_reason="length",
                usage_summary={
                    "prompt_tokens": 100,
                    "completion_tokens": 55,
                    "total_tokens": 155,
                },
                transport_attempts=2,
            )
        ],
        role="planner",
    )

    relative = agent_pipeline_runs._write_pipeline_failure_diagnostic(
        wrapper_dir=tmp_path,
        exc=failure,
        code="research_pipeline_plan_contract_exhausted",
    )

    assert relative == "diagnostics/research_pipeline_failure.json"
    payload = json.loads((tmp_path / relative).read_text(encoding="utf-8"))
    assert payload["schema_version"] == "easyicu.web-research-pipeline-failure/3"
    assert payload["structured_attempts"] == [
        {
            "attempt": 1,
            "raw_chars": 42,
            "error_class": "validation",
            "finish_reason": "length",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 55,
                "total_tokens": 155,
            },
            "transport_attempts": 2,
        }
    ]
    rendered = json.dumps(payload)
    assert secret not in rendered
    assert payload["raw_model_output_recorded"] is False
    assert payload["prompt_recorded"] is False
    assert payload["secrets_recorded"] is False
    assert payload["failure_type"] == "structured_response"


def test_pipeline_failure_type_is_a_closed_nonsecret_category(tmp_path: Path) -> None:
    secret_error_type = type("sk_secret_shaped_failure_type", (RuntimeError,), {})

    relative = agent_pipeline_runs._write_pipeline_failure_diagnostic(
        wrapper_dir=tmp_path,
        exc=secret_error_type("ordinary failure"),
        code="research_pipeline_execution_failed",
    )

    payload = json.loads((tmp_path / relative).read_text(encoding="utf-8"))
    assert payload["failure_type"] == "error"
    assert "sk_secret" not in json.dumps(payload)


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


@pytest.mark.parametrize(
    (
        "budget_mode",
        "runner_image",
        "expected_profile_name",
        "expected_profile_version",
    ),
    [
        (None, None, "npj_dm_e1_canary_dev", "20260814"),
        (
            "full_reviewed",
            "easyicu-research-agent:e1-demo-local",
            "npj_dm_e1_demo_dev",
            "20260815",
        ),
    ],
)
def test_web_runner_delegates_to_research_agent_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget_mode: str | None,
    runner_image: str | None,
    expected_profile_name: str,
    expected_profile_version: str,
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

    export_path = _write_pipeline_export(tmp_path / "export")
    runner_kwargs: dict[str, Any] = {}
    if budget_mode is not None:
        runner_kwargs["budget_mode"] = budget_mode
    if runner_image is not None:
        runner_kwargs["runner_image"] = runner_image
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(export_path),
        study_context=_complete_study(),
        project_root=str(tmp_path / "projects"),
        provider={"provider": "openai", "external": True},
        provider_environment=_PI_PROVIDER_ENVIRONMENT,
        **runner_kwargs,
    )

    class Job:
        id = "job-real-pipeline"
        cancel_requested = False
        events: list[dict[str, Any]] = []

        def emit(self, event: dict[str, Any]) -> None:
            self.events.append(dict(event))

    result = runner(Job())

    assert calls["acquire"]["question"] == _complete_study()["question"]
    from easyicu.research_agent.providers.hard_stop import HardStopClient

    assert isinstance(calls["acquire"]["llm"], HardStopClient)
    assert calls["acquire"]["llm"]._role == "acquisition"
    assert calls["services"].provider_hard_stop is not None
    assert (
        calls["services"].human_review_gate.reviewer_identity_resolver()
        == "easyicu_local_web_operator"
    )
    assert (
        calls["acquire"]["llm"]._task
        is calls["services"].provider_hard_stop
    )
    assert calls["acquire"]["allowed_modules"] == ("demographics", "outcome")
    assert calls["acquire"]["static_concepts"] == ("age", "sex")
    assert calls["run"]["cohort"] == universe
    assert calls["run"]["question"] == _complete_study()["question"]
    assert calls["acquire"]["primary_exposure_concept"] == "heart_rate"
    assert calls["run"]["primary_exposure"] == "heart_rate"
    assert calls["run"]["endpoint"] is None
    assert calls["run"]["user_preferences"]["covariates"] == ["age", "sex"]
    assert calls["config"].evidence_enforcement_mode == "strict"
    assert calls["config"].writer_digest_widened is True
    assert calls["config"].enable_reproducibility_envelope is True
    assert calls["config"].require_human_plan_review is True
    assert calls["config"].require_reportable_scientific_capability is True
    assert calls["config"].required_primary_cohort_selection_mode == "all_input_rows"
    assert calls["config"].enable_pubmed is False
    expected_limits = provider_adapter.web_research_agent_hard_stop_limits(
        budget_mode or "planner_canary"
    )
    assert (
        calls["config"].max_provider_attempts_per_run
        == expected_limits.max_provider_attempts_per_run
    )
    assert (
        calls["config"].max_total_tokens_per_run
        == expected_limits.max_total_tokens_per_run
    )
    assert (
        calls["config"].max_estimated_cost_usd_per_batch
        == expected_limits.max_estimated_cost_usd_per_batch
    )
    assert calls["config"].runner_kind == "docker"
    assert calls["config"].runner_network == "none"
    assert calls["config"].runner_image == (
        runner_image or "easyicu-research-agent:1.0.0"
    )
    assert calls["config"].submission_profile_name == expected_profile_name
    assert calls["config"].submission_profile_version == expected_profile_version
    assert calls["config"].enable_memory is False
    assert calls["config"].enable_experience_bank is False
    assert calls["config"].enable_reviewed_memory is False
    assert (
        calls["services"].provider_hard_stop.ledger.limits
        == expected_limits
    )
    ledger = json.loads(
        (
            tmp_path
            / "projects"
            / "study-workflow"
            / "run_job-real-pipeline"
            / ".runtime"
            / "provider_hard_stop_ledger.json"
        ).read_text(encoding="utf-8")
    )
    assert ledger["tasks"][0]["status"] == "completed"
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


def test_web_runner_enables_live_pubmed_only_with_host_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    captured: dict[str, Any] = {}

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
            actual = tmp_path / "run"
            actual.mkdir(exist_ok=True)
            (actual / "manifest.json").write_text("{}", encoding="utf-8")
            return SimpleNamespace(manifest_path=actual / "manifest.json")

    def fake_from_config(config: Any, *, services: Any) -> FakePipeline:
        captured["config"] = config
        return FakePipeline()

    monkeypatch.setattr(
        research_agent.ResearchAgentPipeline,
        "from_config",
        fake_from_config,
    )
    export_path = _write_pipeline_export(tmp_path / "export")
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(export_path),
        study_context=_complete_study(),
        project_root=str(tmp_path / "projects"),
        provider={"provider": "openai", "external": True},
        provider_environment=_PI_PROVIDER_ENVIRONMENT,
        literature_search_authorized=True,
    )

    class Job:
        id = "job-live-literature"
        cancel_requested = False

        def emit(self, _event: dict[str, Any]) -> None:
            return None

    runner(Job())

    assert captured["config"].enable_pubmed is True


def test_web_runner_reuses_digest_bound_web_literature_without_second_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    captured: dict[str, Any] = {}
    seed = {
        "research_question": _complete_study()["question"],
        "citations": [],
        "prisma": {
            "identified": 0,
            "duplicates_removed": 0,
            "screened": 0,
            "eligible": 0,
            "included": 0,
        },
        "search_provenance": {
            "schema_version": "easyicu.literature_search_provenance/1",
            "curated_seed_count": 0,
            "sources_enabled": ["web_pubmed"],
            "sources_returning": [],
            "search_queries": {"web_pubmed": ["ICU feature AND mortality"]},
            "search_conducted": True,
            "searched_at": "2026-08-12T12:00:00+00:00",
            "note": "Bound Web search",
        },
        "screening_decisions": [],
    }
    study = {**_complete_study(), "literature_authority": {"status": "searched"}}
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
    monkeypatch.setattr(
        literature_authority,
        "load_bound_literature",
        lambda **kwargs: seed,
    )

    class FakePipeline:
        def run(self, **_kwargs: Any) -> SimpleNamespace:
            actual = tmp_path / "run"
            actual.mkdir(exist_ok=True)
            (actual / "manifest.json").write_text("{}", encoding="utf-8")
            return SimpleNamespace(manifest_path=actual / "manifest.json")

    def fake_from_config(config: Any, *, services: Any) -> FakePipeline:
        captured["config"] = config
        return FakePipeline()

    monkeypatch.setattr(
        research_agent.ResearchAgentPipeline,
        "from_config",
        fake_from_config,
    )
    export_path = _write_pipeline_export(tmp_path / "export")
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(export_path),
        study_context=study,
        project_root=str(tmp_path / "projects"),
        provider={"provider": "openai", "external": True},
        provider_environment=_PI_PROVIDER_ENVIRONMENT,
        literature_search_authorized=True,
    )

    class Job:
        id = "job-bound-web-literature"
        cancel_requested = False

        def emit(self, _event: dict[str, Any]) -> None:
            return None

    runner(Job())

    assert (
        dict(captured["config"].bound_preplan_literature)["research_question"]
        == seed["research_question"]
    )
    assert captured["config"].bound_preplan_literature["search_provenance"][
        "search_queries"
    ]["web_pubmed"] == ("ICU feature AND mortality",)
    assert captured["config"].enable_pubmed is False


def test_web_runner_binds_only_structured_cohort_filters() -> None:
    descriptive_only = {
        "cohort": {
            "label": "All stays with configured exposure and outcome fields",
            "review": "All ICU stays with available configured data",
            "exclude_readmissions": False,
        }
    }
    explicitly_filtered = {
        "cohort": {
            "label": "Adults only",
            "age_min": 18,
            "exclude_readmissions": False,
        }
    }

    assert (
        agent_pipeline_runs._primary_cohort_selection_mode(descriptive_only)
        == "all_input_rows"
    )
    assert (
        agent_pipeline_runs._primary_cohort_selection_mode(explicitly_filtered)
        == "predicate_filtered"
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

    with pytest.raises(Exception) as direct_fallback:
        agent_route._provider_environment_for_agent_run(
            credential_source="scientific_provider",
            engine="research_agent_pipeline",
            run_type="full",
            external_llm_opt_in=True,
        )
    assert getattr(direct_fallback.value, "detail") == {
        "error": "research_pipeline_pi_verified_credentials_required"
    }


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_pipeline_route_rejects_raw_tabular_files_before_provider_resolution(
    suffix: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.routes import agent as agent_route

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / f"patients{suffix}").write_text("stay_id\n1\n", encoding="utf-8")
    study = {
        **_complete_study(),
        "data_source": {"path": str(raw), "database": "miiv"},
    }
    monkeypatch.setattr(agent_route.context_store, "get_context", lambda _id: study)
    provider_called = False

    def provider_environment(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        nonlocal provider_called
        provider_called = True
        return dict(_PI_PROVIDER_ENVIRONMENT)

    monkeypatch.setattr(
        agent_route.PiProviderConfigStore,
        "research_agent_environment",
        provider_environment,
    )

    with pytest.raises(Exception) as raised:
        agent_route.jobs_agent_run(
            {
                "path": str(raw),
                "study_context_id": study["id"],
                "engine": "research_agent_pipeline",
                "run_type": "full",
                "credential_source": "pi_verified",
                "external_llm_opt_in": True,
            }
        )

    assert getattr(raised.value, "detail")["error"] in {
        "research_pipeline_manifest_required",
        "no_export_files",
    }
    assert provider_called is False


def test_pipeline_route_ignores_client_project_root_and_uses_pi_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.pi_copilot.workspace import ProjectWorkspace
    from easyicu.webserver.routes import agent as agent_route

    export = _write_pipeline_export(tmp_path / "export")
    study = {
        **_complete_study(),
        "data_source": {"path": str(export), "database": "miiv"},
    }
    workspace = ProjectWorkspace(tmp_path / "pi-workspace")
    captured: dict[str, Any] = {}
    monkeypatch.setattr(agent_route.context_store, "get_context", lambda _id: study)
    monkeypatch.setattr(
        agent_route,
        "_research_pipeline_workspace",
        lambda: workspace,
    )
    monkeypatch.setattr(
        agent_route.PiProviderConfigStore,
        "research_agent_environment",
        lambda self, **_kwargs: dict(_PI_PROVIDER_ENVIRONMENT),
    )
    monkeypatch.setattr(agent_route.settings_store, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.setattr(
        agent_route.capabilities,
        "validate_compute_target",
        lambda _body: {"ok": True, "compute_target": "local"},
    )
    monkeypatch.setattr(
        agent_route.agent_runs,
        "resolve_agent_provider_config",
        lambda **_kwargs: {"provider": "openai", "external": True},
    )
    monkeypatch.setattr(
        agent_route.context_store,
        "build_agent_context_binding",
        lambda *_args, **_kwargs: {},
    )

    def make_runner(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return lambda _job: {"gate": {"status": "blocked"}}

    monkeypatch.setattr(
        agent_route.agent_pipeline_runs,
        "make_research_pipeline_run_runner",
        make_runner,
    )
    monkeypatch.setattr(
        agent_route,
        "submit_job",
        lambda _kind, _runner: SimpleNamespace(id="job-workspace", kind="agent-run", status="queued"),
    )
    monkeypatch.setattr(
        agent_route.context_store,
        "handoff_context",
        lambda *_args, **_kwargs: {"revision": 5},
    )
    monkeypatch.setattr(
        agent_route.capabilities,
        "record_tool_event",
        lambda *_args, **_kwargs: None,
    )

    result = agent_route.jobs_agent_run(
        {
            "path": str(export),
            "study_context_id": study["id"],
            "engine": "research_agent_pipeline",
            "run_type": "full",
            "credential_source": "pi_verified",
            "external_llm_opt_in": True,
            "project_root": str(tmp_path / "client-controlled"),
        }
    )

    assert result["job_id"] == "job-workspace"
    assert Path(captured["project_root"]) == workspace.project_root(study["id"])
    assert Path(captured["project_root"]) != tmp_path / "client-controlled"
    assert captured["budget_mode"] == "planner_canary"


def test_pipeline_route_rejects_client_selected_full_reviewed_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.routes import agent as agent_route

    export = _write_pipeline_export(tmp_path / "export")
    study = {
        **_complete_study(),
        "data_source": {"path": str(export), "database": "miiv"},
    }
    monkeypatch.setattr(agent_route.context_store, "get_context", lambda _id: study)

    with pytest.raises(Exception) as raised:
        agent_route.jobs_agent_run(
            {
                "path": str(export),
                "study_context_id": study["id"],
                "engine": "research_agent_pipeline",
                "run_type": "full",
                "budget_mode": "full_reviewed",
            }
        )

    assert getattr(raised.value, "detail") == {
        "error": "research_pipeline_budget_mode_server_owned"
    }


def test_planner_canary_cannot_be_approved_into_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.routes import agent as agent_route

    monkeypatch.setattr(agent_route.settings_store, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.setattr(
        agent_route.agent_pipeline_runs,
        "pending_review",
        lambda _run_id: {
            "study_id": "study-workflow",
            "resumable_here": True,
            "budget_mode": "planner_canary",
        },
    )

    with pytest.raises(Exception) as raised:
        agent_route.jobs_agent_run_review(
            {
                "run_id": "run-canary",
                "study_context_id": "study-workflow",
                "decision": "approved",
                "external_llm_opt_in": True,
            }
        )

    assert getattr(raised.value, "detail") == {
        "error": "research_pipeline_planner_canary_execution_blocked"
    }


def test_pipeline_bridge_cannot_approve_canary_when_route_is_bypassed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review canary plan.",
        authority_sha256="a" * 64,
        payload={"reason": "operator_plan_approval_required"},
    )
    pending = HumanReviewPending(
        run_id="run-canary-bypass",
        thread_id="run-canary-bypass",
        run_dir=str(tmp_path / "run-canary-bypass"),
        requests=(request,),
    )
    pipeline_called = False

    class _Pipeline:
        def resume_human_review(self, *_args: Any, **_kwargs: Any) -> Any:
            nonlocal pipeline_called
            pipeline_called = True
            raise AssertionError("canary must not reach execution")

    monkeypatch.setitem(
        agent_pipeline_runs._PENDING,
        pending.run_id,
        agent_pipeline_runs._PendingRun(
            pipeline=_Pipeline(),
            pending=pending,
            wrapper_dir=tmp_path,
            study={"id": "study-canary"},
            provider={},
            acquisition=SimpleNamespace(),
            created_at=1.0,
            budget_mode="planner_canary",
        ),
    )

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.resume_research_pipeline(
            run_id=pending.run_id,
            study_context_id="study-canary",
            decision="approved",
            reviewer="server reviewer",
            note="",
            job=SimpleNamespace(emit=lambda _event: None, cancel_requested=False),
        )

    assert exc.value.code == "research_pipeline_planner_canary_execution_blocked"
    assert pipeline_called is False


def test_signoff_ignores_client_reviewer_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.routes import agent as agent_route

    captured: dict[str, Any] = {}

    def create_signoff(_project_dir: str, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(agent_route.agent_runs, "create_human_signoff", create_signoff)

    agent_route.post_agent_run_signoff(
        {"project_dir": "/server/run", "reviewer": "client-claims-to-be-PI"}
    )

    assert captured["reviewer"] == "easyicu_local_web_operator"


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

    export_path = _write_pipeline_export(tmp_path / "export")
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(export_path),
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


def test_pipeline_bridge_rejects_direct_scientific_provider_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = _write_pipeline_export(tmp_path / "export")
    monkeypatch.setattr(
        agent_pipeline_runs,
        "_data_foundation_profile",
        lambda **_kwargs: _foundation_profile(),
    )

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.make_research_pipeline_run_runner(
            export_path=str(export),
            study_context=_complete_study(),
            project_root=str(tmp_path / "projects"),
            provider={"provider": "openai", "external": True},
            provider_environment=None,
        )

    assert exc.value.code == "research_pipeline_pi_verified_credentials_required"


def test_pipeline_revalidates_package_before_provider_or_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = _write_pipeline_export(tmp_path / "export")
    study = {
        **_complete_study(),
        "data_source": {"path": str(export), "database": "miiv"},
    }
    monkeypatch.setattr(
        agent_pipeline_runs,
        "_data_foundation_profile",
        lambda **_kwargs: _foundation_profile(),
    )
    provider_called = False

    def provider_client(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal provider_called
        provider_called = True
        raise AssertionError("provider must not be reached after package drift")

    monkeypatch.setattr(
        provider_adapter,
        "build_research_agent_provider_client",
        provider_client,
    )
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(export),
        study_context=study,
        project_root=str(tmp_path / "projects"),
        provider={"provider": "openai", "external": True},
        provider_environment=_PI_PROVIDER_ENVIRONMENT,
    )
    (export / "demographics.parquet").write_bytes(b"changed-after-submit")

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        runner(SimpleNamespace(id="job-drift", emit=lambda _event: None))

    assert exc.value.code == "research_pipeline_package_binding_changed"
    assert provider_called is False


def test_plan_approval_revalidates_the_exact_prepared_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study, package_binding = _study_with_package_binding(tmp_path / "package")
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review package-bound plan.",
        authority_sha256="f" * 64,
        payload={"reason": "operator_plan_approval_required"},
    )
    pending = HumanReviewPending(
        run_id="run-package-drift",
        thread_id="run-package-drift",
        run_dir=str(tmp_path / "run-package-drift"),
        requests=(request,),
    )
    pipeline_called = False

    class _Pipeline:
        def resume_human_review(self, *_args: Any, **_kwargs: Any) -> Any:
            nonlocal pipeline_called
            pipeline_called = True
            raise AssertionError("drifted package must not reach Pipeline")

    entry = agent_pipeline_runs._PendingRun(
        pipeline=_Pipeline(),
        pending=pending,
        wrapper_dir=tmp_path,
        study=study,
        provider={},
        acquisition=SimpleNamespace(),
        created_at=1.0,
        prepared_package_binding=package_binding,
    )
    monkeypatch.setitem(agent_pipeline_runs._PENDING, pending.run_id, entry)
    export = Path(study["data_source"]["path"])
    (export / "demographics.parquet").write_bytes(b"changed-before-approval")

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.resume_research_pipeline(
            run_id=pending.run_id,
            study_context_id=study["id"],
            decision="approved",
            reviewer="server reviewer",
            note="",
            job=SimpleNamespace(emit=lambda _event: None, cancel_requested=False),
            current_study_context=study,
        )

    assert exc.value.code == "research_pipeline_package_binding_changed"
    assert pipeline_called is False


def test_provider_public_identity_binds_endpoint_without_disclosing_it() -> None:
    common = {
        "provider": "openai",
        "api_key": "test-key",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "model": "test-model",
        "model_env": "OPENAI_MODEL",
        "auth_header": "authorization",
    }

    first = provider_adapter._credential_public_metadata(
        {**common, "base_url": "https://one.example/v1/chat/completions"}
    )
    second = provider_adapter._credential_public_metadata(
        {**common, "base_url": "https://two.example/v1/chat/completions"}
    )

    assert first["endpoint_fingerprint"] != second["endpoint_fingerprint"]
    assert "one.example" not in json.dumps(first)


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
        "primary_exposure_source_concept": None,
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
        "primary_exposure_source_concept": "sep3_sofa2",
    }


def test_web_data_foundation_materializes_sensitivity_support_without_adjustment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module
    from easyicu.research_agent.planning.sensitivity_authority import (
        normalize_prespecified_sensitivities,
    )

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
                CatalogConcept(
                    concept_id="icu_readmission",
                    file_name="outcome.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
            ],
        ),
    )
    specs = normalize_prespecified_sensitivities(
        [
            {
                "spec_id": "non_readmission_only",
                "axis": "repeated_stays",
                "strategy": "non_readmission_restriction",
                "execution_variables": ["icu_readmission"],
            }
        ]
    )

    profile = agent_pipeline_runs._data_foundation_profile(
        export_path="/typed/demo",
        study={"modules": ["demographics", "outcome"]},
        target="death",
        covariates=(),
        sensitivity_specs=specs,
    )

    assert profile["static_concepts"] == ("age", "icu_readmission")
    assert profile["required_feature_concepts"] == ()


def test_web_data_foundation_materializes_owner_readmission_indicator_for_first_stay(
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
                CatalogConcept(
                    concept_id="icu_readmission",
                    file_name="outcome.parquet",
                    typed_metadata=False,
                    column_role="event_status",
                ),
            ],
        ),
    )

    profile = agent_pipeline_runs._data_foundation_profile(
        export_path="/typed/demo",
        study={
            "modules": ["demographics", "outcome"],
            "cohort": {"exclude_readmissions": True},
        },
        target="death",
    )

    assert profile["static_concepts"] == ("age", "icu_readmission")
    assert profile["required_feature_concepts"] == ()


def test_web_data_foundation_rejects_first_stay_without_owner_indicator(
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

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs._data_foundation_profile(
            export_path="/typed/demo",
            study={
                "modules": ["demographics", "outcome"],
                "cohort": {"exclude_readmissions": True},
            },
            target="death",
        )

    assert exc.value.code == "research_pipeline_readmission_indicator_unavailable"


def test_web_study_context_compiles_typed_sensitivity_authority() -> None:
    study = {
        **_complete_study(),
        "sensitivity_specs": [
            {
                "spec_id": "landmark_24h",
                "axis": "timing",
                "strategy": "landmark",
                "landmark_hours": 24,
                "require_alive_at_landmark": True,
                "exclude_negative_event_times": True,
            },
            {
                "spec_id": "non_readmission_only",
                "axis": "repeated_stays",
                "strategy": "non_readmission_restriction",
                "execution_variables": ["icu_readmission"],
            },
        ],
    }

    compiled = agent_pipeline_runs._research_user_preferences(study)
    validated = UserPreferences.model_validate(compiled)

    assert compiled["landmark_hours"] == 24
    assert [spec.spec_id for spec in validated.sensitivity_specs] == [
        "landmark_24h",
        "non_readmission_only",
    ]


def test_web_data_foundation_resolves_only_issued_operational_exposure() -> None:
    acquisition = SimpleNamespace(
        analysis_columns={"sep3_sofa2": "sep3_sofa2_max"},
        materialized_columns=("stay_id", "sep3_sofa2_max", "death"),
    )

    assert (
        agent_pipeline_runs._resolve_materialized_primary_exposure(
            configured="sep3_sofa2_max",
            source_concept="sep3_sofa2",
            acquisition=acquisition,
        )
        == "sep3_sofa2_max"
    )
    assert (
        agent_pipeline_runs._resolve_materialized_primary_exposure(
            configured="sep3_sofa2",
            source_concept="sep3_sofa2",
            acquisition=acquisition,
        )
        == "sep3_sofa2_max"
    )
    assert (
        agent_pipeline_runs._resolve_materialized_primary_exposure(
            configured="sep3_sofa2_mean",
            source_concept="sep3_sofa2",
            acquisition=acquisition,
        )
        is None
    )


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


def test_pipeline_factory_validates_execution_concepts_before_job_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def reject(**kwargs: Any) -> dict[str, Any]:
        calls.append(dict(kwargs))
        raise agent_pipeline_runs.ResearchPipelineRunError(
            "research_pipeline_target_outside_configured_modules",
            "The configured outcome is not available in the selected feature modules.",
            details={
                "field": "execution_concepts.outcome",
                "concept_id": kwargs.get("target"),
            },
        )

    monkeypatch.setattr(agent_pipeline_runs, "_data_foundation_profile", reject)
    study = {
        **_complete_study(),
        "outcome": "In-hospital mortality from the outcome module",
        "execution_concepts": {
            "outcome": "death",
            "primary_exposure": "heart_rate",
            "covariates": ["age", "sex"],
        },
    }

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.make_research_pipeline_run_runner(
            export_path="/typed/demo",
            study_context=study,
            project_root=None,
            provider={"provider": "openai", "external": True},
        )

    assert exc.value.code == "research_pipeline_target_outside_configured_modules"
    assert exc.value.details == {
        "field": "execution_concepts.outcome",
        "concept_id": "death",
    }
    assert calls[0]["target"] == "death"
    assert calls[0]["primary_exposure"] == "heart_rate"
    assert calls[0]["covariates"] == ("age", "sex")


def test_pipeline_factory_rejects_generic_sepsis_sofa2_before_job_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation_called = False

    def unexpected_foundation(**_kwargs: Any) -> dict[str, Any]:
        nonlocal foundation_called
        foundation_called = True
        return _foundation_profile()

    monkeypatch.setattr(
        agent_pipeline_runs,
        "_data_foundation_profile",
        unexpected_foundation,
    )
    study = {
        **_complete_study(),
        "question": "What is standard Sepsis-3 prevalence and mortality association?",
        "primary_exposure": "Sepsis-3 using SOFA-2",
        "modules": ["outcome", "sepsis3_sofa2"],
        "execution_concepts": {
            "outcome": "death",
            "primary_exposure": "sep3_sofa2",
            "covariates": [],
        },
    }

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.make_research_pipeline_run_runner(
            export_path="/typed/demo",
            study_context=study,
            project_root=None,
            provider={"provider": "openai", "external": True},
        )

    assert exc.value.code == "concept_explicit_selection_required"
    assert exc.value.details["canonical_alternative"] == "sep3_sofa1"
    assert foundation_called is False


def test_pipeline_factory_rejects_unimplemented_cluster_variance_before_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation_called = False

    def foundation(**_kwargs: Any) -> dict[str, Any]:
        nonlocal foundation_called
        foundation_called = True
        return _foundation_profile()

    monkeypatch.setattr(agent_pipeline_runs, "_data_foundation_profile", foundation)
    study = {
        **_complete_study(),
        "analysis_design": {
            "analysis_unit": "icu_stay",
            "variance_estimator": "cluster_robust",
            "cluster_unit": "hospital_admission",
        },
    }

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.make_research_pipeline_run_runner(
            export_path="/typed/demo",
            study_context=study,
            project_root=None,
            provider={"provider": "openai", "external": True},
        )

    assert exc.value.code == "research_pipeline_cluster_unit_unsupported"
    assert exc.value.details == {
        "cluster_unit": "hospital_admission",
        "supported_cluster_units": ["patient"],
    }
    assert foundation_called is False


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
        "data_constraints",
        "covariates",
        "covariate_selection",
        "covariate_rationales",
        "covariate_temporal_roles",
    }
    assert validated.extra_notes == "Demo-only product validation."
    assert "not_for_manuscript" in str(validated.data_constraints)
    assert validated.timing_and_design is None
    constraints = json.loads(str(validated.data_constraints))
    assert constraints["materialization_window"] == {
        "role": "outer_observation_window",
        "hours": 24,
        "anchor": "ICU admission",
    }
    assert validated.covariates == ["age", "sex"]
    assert validated.covariate_selection == "exact"
    assert validated.covariate_temporal_roles == {
        "age": "baseline_static",
        "sex": "baseline_static",
    }


def test_web_typed_descriptive_family_overrides_free_text_risk_contrast_routing() -> None:
    study = {
        **_complete_study(),
        "question": (
            "What are the observed outcome risks and their risk difference "
            "between exposure groups?"
        ),
        "analysis_goal": (
            "Descriptive, unadjusted, noncausal absolute risks and risk difference."
        ),
        "analysis_design": {
            "analysis_family": "descriptive_epidemiology",
            "analysis_unit": "icu_stay",
            "variance_estimator": "model_based",
        },
    }

    compiled = agent_pipeline_runs._research_user_preferences(study)
    validated = UserPreferences.model_validate(compiled)

    assert validated.inferred_analysis_family == "descriptive_epidemiology"
    constraints = json.loads(str(validated.data_constraints))
    assert constraints["analysis_design"]["analysis_family"] == (
        "descriptive_epidemiology"
    )


def test_web_materialization_window_never_declares_clinical_time_zero() -> None:
    study = {
        **_complete_study(),
        "question": (
            "Classify Sepsis-3 at suspected-infection onset while materializing "
            "features over the first 24 hours after ICU admission."
        ),
        "primary_exposure": "sep3_sofa1",
        "execution_concepts": {
            "outcome": "death",
            "primary_exposure": "sep3_sofa1",
            "covariates": ["age", "sex"],
        },
    }

    compiled = agent_pipeline_runs._research_user_preferences(study)
    validated = UserPreferences.model_validate(compiled)

    assert validated.timing_and_design is None
    constraints = json.loads(str(validated.data_constraints))
    assert constraints["materialization_window"]["anchor"] == "ICU admission"
    assert constraints["materialization_window"]["role"] == "outer_observation_window"


def _large_diagnosis_study(count: int) -> dict[str, Any]:
    study = _complete_study()
    study["cohort"] = {
        "preset": "adult_icu",
        "label": "Adult ICU stays with suspected infection",
        "comparison": (
            "Sepsis-3 positive versus Sepsis-3 negative at suspected-infection onset"
        ),
        "icd_include": (
            "Sepsis and septic shock diagnoses recorded during the index admission"
        ),
        "age_min": 18,
        "age_max": 89,
        "max_patients": 20000,
        "icd_enabled": True,
        "include_diagnoses": [
            f"A41.{index} Sepsis due to unspecified organism, variant {index}"
            for index in range(count)
        ],
        "exclude_diagnoses": [
            f"Z51.{index} Encounter for palliative care variant {index}"
            for index in range(8)
        ],
    }
    # The only place the repeated-stay signal lives for this study.
    study["confirmations"] = {"repeated_icu_stays_retained": True}
    return study


@pytest.mark.parametrize("count", [0, 12, 18, 28, 64])
def test_web_data_constraints_never_silently_drop_a_constraint_key(
    count: int,
) -> None:
    """A long cohort must not delete confirmations or the executed window.

    ``data_constraints`` is transported as one JSON string and is read
    downstream both as prompt text and by token-scanning scientific gates.
    Cutting the serialized text at a character offset used to remove whole
    trailing keys -- ``sort_keys`` sorts ``confirmations`` and
    ``materialization_window`` last -- so a study with enough ICD codes lost
    its repeated-stay signal and its executed window at the same time.
    """

    compiled = agent_pipeline_runs._research_user_preferences(
        _large_diagnosis_study(count)
    )
    validated = UserPreferences.model_validate(compiled)

    constraints = json.loads(str(validated.data_constraints))
    assert set(constraints) == {
        "analysis_design",
        "cohort",
        "confirmations",
        "materialization_window",
    }
    assert constraints["confirmations"] == {"repeated_icu_stays_retained": True}
    assert constraints["materialization_window"]["role"] == "outer_observation_window"
    assert "repeat" in str(validated.data_constraints).casefold()


def test_web_data_constraints_elide_list_items_visibly() -> None:
    """Anything actually dropped is dropped inside the structure, in the open."""

    compiled = agent_pipeline_runs._research_user_preferences(
        _large_diagnosis_study(64)
    )
    constraints = json.loads(str(compiled["data_constraints"]))
    included = constraints["cohort"]["include_diagnoses"]

    assert len(included) < 64
    assert included[-1] == f"[{64 - (len(included) - 1)} omitted]"
    # The marker must not be able to satisfy a gate's text scan on its own.
    assert "readmission" not in included[-1].casefold()


def test_web_oversized_data_constraints_fail_closed_instead_of_truncating() -> None:
    study = _complete_study()
    study["cohort"] = {
        name: "L" * 500
        for name in (
            "preset",
            "label",
            "review",
            "review_scope",
            "comparison",
            "source_type",
            "comparison_mode",
            "icd_include",
            "icd_exclude",
        )
    }

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as excinfo:
        agent_pipeline_runs._research_user_preferences(study)

    assert excinfo.value.code == "research_pipeline_data_constraints_too_large"
    assert excinfo.value.details["section_chars"]["cohort"] > 2_400


def test_web_study_context_preserves_an_explicit_empty_adjustment_set() -> None:
    study = _complete_study()
    study["covariates"] = []
    study["execution_concepts"] = {
        **study.get("execution_concepts", {}),
        "covariates": [],
    }
    study["covariate_rationales"] = {}
    study["covariate_temporal_roles"] = {}

    compiled = agent_pipeline_runs._research_user_preferences(study)

    assert compiled["covariates"] == []
    assert compiled["covariate_selection"] == "exact"
    assert UserPreferences.model_validate(compiled).covariates == []


def test_available_covariates_do_not_silently_become_an_exact_adjustment_set() -> None:
    study = _complete_study()
    study.pop("covariate_selection")

    compiled = agent_pipeline_runs._research_user_preferences(study)
    validated = UserPreferences.model_validate(compiled)

    assert compiled["covariates"] == ["age", "sex"]
    assert "covariate_selection" not in compiled
    assert validated.covariates == ["age", "sex"]
    assert validated.covariate_selection == "planner_selectable"


def test_invalid_web_adjustment_authority_fails_closed() -> None:
    study = _complete_study()
    study["covariate_selection"] = "suggested"

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs._research_user_preferences(study)

    assert exc.value.code == "research_pipeline_covariate_selection_invalid"
