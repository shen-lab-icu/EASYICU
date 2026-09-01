"""Focused owner and fail-closed tests for the Copilot research workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pandas as pd
from starlette.requests import Request

from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
from easyicu.research_agent.acquisition.patient_grouping import PatientGroupingBinding
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
    research_launch_resume,
    research_launch_runtime,
    research_launch_scientific,
    research_pipeline_run_preparation,
    research_run_submission,
)
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.literature_projection import (
    literature_source_resource,
    project_run_literature,
)
from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot import cohort_eligibility
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiCopilotError,
    ResearchProviderBinding,
    PiSessionRecord,
    ToolExecutionContext,
)
from easyicu.webserver.pi_copilot.workflow import (
    active_export_matches_study,
    build_research_workflow_snapshot,
    registered_export_matches_study,
)
from easyicu.webserver.pi_copilot.run_authority import (
    resumable_planner_checkpoint_job_id,
)


def _install_pending_review(monkeypatch, entry):
    registry = agent_pipeline_runs.PendingReviewRegistry()
    monkeypatch.setattr(agent_pipeline_runs, "_PENDING_REVIEWS", registry)
    registry.register(entry)
    return registry


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/jobs/agent-run",
            "raw_path": b"/api/jobs/agent-run",
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
        }
    )


def _record_pipeline_submission(
    submitted: list[Any],
    request: research_run_submission.ResearchRunSubmissionRequest,
    *,
    job_id: str,
    authorize: Any = None,
    account_environment: Any = None,
) -> research_run_submission.ResearchRunSubmissionReceipt:
    if authorize is not None:
        authorize()
    submitted.append(
        {"request": request, "account_environment": account_environment}
    )
    return research_run_submission.ResearchRunSubmissionReceipt(
        job_id=job_id,
        kind="agent-run",
        status="running",
        study_context_id=request.study_context_id,
        study_context_revision=5,
        budget_mode="full_reviewed",
        planner_start_mode=request.planner_start_mode,
    )


def _confirmed_cohort_decision(
    option_id: str,
    *,
    study_context_id: str,
    study_context_revision: int,
    current_cohort: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = dict(current_cohort or {})
    target = cohort_eligibility.selection_cohort_for_option(
        {"cohort": base}, option_id
    )
    scope = study_context_owner.normalize_primary_cohort_scope(
        {"cohort": target}
    )
    session_id = f"pi-{study_context_id}"
    seed = (
        f"{session_id}:{study_context_id}:{study_context_revision - 1}:"
        f"{option_id}:{scope.sha256}"
    )
    event = cohort_eligibility.build_selection_event(
        option_id=option_id,
        study_context_id=study_context_id,
        expected_revision=study_context_revision - 1,
        session_id=session_id,
        user_turn_id=f"turn-{session_id}",
        event_id=hashlib.sha256(f"event:{seed}".encode()).hexdigest(),
        one_use_grant_id=hashlib.sha256(f"grant:{seed}".encode()).hexdigest(),
        primary_cohort_contract_sha256=scope.sha256,
        actor_id_sha256=hashlib.sha256(f"actor:{session_id}".encode()).hexdigest(),
        selected_at="2026-08-29T12:00:00Z",
    )
    authority = cohort_eligibility.confirmation_authority_for_option(
        option_id,
        study_context_id=study_context_id,
        study_context_revision=study_context_revision,
        current_cohort=base,
        selection_event=event,
        confirmed_at="2026-08-29T12:00:00Z",
    )
    return target, authority


def test_typed_selected_design_requires_complete_reviewable_recommendation() -> None:
    legacy = {
        "design_selection": {
            "candidates": [
                {"disposition": "selected", "reviewable_plan": None},
                {"disposition": "rejected", "reviewable_plan": None},
            ]
        }
    }
    complete = {
        "design_selection": {
            "candidates": [
                {
                    "disposition": "selected",
                    "reviewable_plan": [
                        "population and unit",
                        "exposure timing and aggregation",
                        "outcome and follow-up",
                        "adjustment and model",
                        "missing-data strategy",
                        "sensitivity and feasibility",
                    ],
                }
            ]
        }
    }

    assert not agent_pipeline_runs._plan_has_complete_reviewable_recommendation(
        legacy
    )
    assert agent_pipeline_runs._plan_has_complete_reviewable_recommendation(complete)
    assert agent_pipeline_runs._plan_has_complete_reviewable_recommendation({})


def _complete_study() -> dict[str, Any]:
    cohort, authority = _confirmed_cohort_decision(
        "no_eligibility_filter",
        study_context_id="study-workflow",
        study_context_revision=4,
        current_cohort={"max_patients": 2000},
    )
    return {
        "id": "study-workflow",
        "revision": 4,
        "question": "Does an aggregate ICU feature predict mortality?",
        "data_source": {
            "path": "/private/prepared/source",
            "database": "mimiciv",
        },
        "cohort": cohort,
        "cohort_eligibility_authority": authority,
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
        "confirmations": {
            "feature_time_window": True,
            "export_format": True,
            "extraction_completed": True,
        },
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


def _assume_execution_runtime_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """An executing launch now probes the container runtime before it starts.

    That gate has its own contract tests in
    ``test_web_execution_runtime_preflight.py``; the launches here are about
    scope and resume authority and must not depend on the host's daemon.
    """

    from easyicu.research_agent.execution import runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "probe_runner_availability",
        lambda kind, **_kwargs: runner_module.RunnerAvailability(
            kind=kind, available=True, image="easyicu-research-agent:test"
        ),
    )


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


def _allow_current_scientific_review(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep non-review resume tests focused on their owner boundary."""

    monkeypatch.setattr(
        agent_pipeline_runs,
        "_load_pending_scientific_review",
        lambda *_args, **_kwargs: {
            "schema_version": "easyicu.plan_scientific_review/2",
            "approval_allowed": True,
        },
    )


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


def test_identified_local_database_can_generate_a_candidate_plan() -> None:
    """Eligibility is proposed in the candidate plan, not asked beforehand."""

    raw_bound = build_research_workflow_snapshot(
        study={
            "id": "study-raw-bound",
            "revision": 3,
            "question": "在 MIMIC-IV 成人 ICU 人群中，Sepsis-3 的患病率是多少？",
            # Identified by the local source picker, not yet extracted.
            "data_source": {
                "path": "/private/demo_sources/mimic_iv/raw",
                "label": "MIMIC-IV",
                "database": "miiv",
            },
        },
        active_export_present=False,
        active_job=None,
        latest_run=None,
    )

    assert raw_bound.planning_prerequisites_missing == []
    plan_stage = next(row for row in raw_bound.stages if row.id == "plan")
    assert plan_stage.status == "ready"
    assert plan_stage.reason_code == "provider_ready_to_generate_plan"
    assert raw_bound.next_action_code == "provider_ready_to_generate_plan"
    assert "outcome" in raw_bound.missing_setup_fields
    assert "modules" in raw_bound.missing_setup_fields


def test_plan_still_blocked_without_a_data_source() -> None:
    """Which data to use remains the user's decision and still gates planning."""

    unbound = build_research_workflow_snapshot(
        study={
            "id": "study-unbound",
            "revision": 1,
            "question": "在 MIMIC-IV 成人 ICU 人群中，Sepsis-3 的患病率是多少？",
        },
        active_export_present=False,
        active_job=None,
        latest_run=None,
    )

    assert unbound.planning_prerequisites_missing == ["data_source"]
    plan_stage = next(row for row in unbound.stages if row.id == "plan")
    assert plan_stage.status == "blocked"


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
        "cohort_eligibility",
        "outcome",
        "analysis_goal",
        "time_window",
        "export_format",
        "modules",
    ]
    assert next(row for row in empty.stages if row.id == "idea").status == "blocked"

    default_only_export = {
        **_complete_study(),
        "confirmations": {"feature_time_window": True},
    }
    default_only_snapshot = build_research_workflow_snapshot(
        study=default_only_export,
        active_export_present=True,
        active_job=None,
        latest_run=None,
    )
    assert "export_format" in default_only_snapshot.missing_setup_fields

    delegated_adjustment = {
        **default_only_export,
        "covariates": [],
        "covariate_selection": "planner_selectable",
    }
    delegated_adjustment_snapshot = build_research_workflow_snapshot(
        study=delegated_adjustment,
        active_export_present=True,
        active_job=None,
        latest_run=None,
    )
    assert "covariates" not in delegated_adjustment_snapshot.missing_setup_fields
    assert "export_format" in delegated_adjustment_snapshot.missing_setup_fields

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
        "path_digest": "58809605ee2154d663851cf4776d7954",
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

    repeated_cohort, repeated_authority = _confirmed_cohort_decision(
        "adults_all_admissions",
        study_context_id="study-workflow",
        study_context_revision=4,
        current_cohort={"max_patients": 2000},
    )
    unaddressed_repeats = {
        **_complete_study(),
        "cohort": repeated_cohort,
        "cohort_eligibility_authority": repeated_authority,
    }
    dependence_snapshot = build_research_workflow_snapshot(
        study=unaddressed_repeats,
        active_export_present=True,
        active_job=None,
        latest_run=None,
    )
    assert dependence_snapshot.missing_setup_fields == ["analysis_design.dependence"]


def test_workflow_requires_named_clinical_definition_confirmation() -> None:
    sepsis_study = {
        **_complete_study(),
        "question": "Estimate Sepsis-3 prevalence and ICU mortality association.",
        "primary_exposure": "",
        "analysis_goal": "",
        "execution_concepts": {"outcome": "death"},
        "analysis_design": {},
        "cohort": {
            **_complete_study()["cohort"],
            "sepsis_definition": {"runtime_profile": "locked-v1"},
        },
    }

    unconfirmed = build_research_workflow_snapshot(
        study=sepsis_study,
        active_export_present=True,
        active_job=None,
        latest_run=None,
    )
    assert "confirmations.clinical_definition_sepsis" in (
        unconfirmed.missing_setup_fields
    )
    assert "primary_exposure" in unconfirmed.missing_setup_fields
    assert unconfirmed.missing_setup_fields.index("primary_exposure") < (
        unconfirmed.missing_setup_fields.index("analysis_goal")
    )
    assert "covariates" not in unconfirmed.missing_setup_fields

    confirmed = build_research_workflow_snapshot(
        study={
            **sepsis_study,
            "confirmations": {"clinical_definition_sepsis": True},
        },
        active_export_present=True,
        active_job=None,
        latest_run=None,
    )
    assert "confirmations.clinical_definition_sepsis" not in (
        confirmed.missing_setup_fields
    )


def test_workflow_accepts_owner_locked_canonical_clinical_definition() -> None:
    locked_study = {
        **_complete_study(),
        "question": "Estimate Sepsis-3 prevalence.",
        "cohort": {
            **_complete_study()["cohort"],
            "sepsis_definition": {
                "runtime_profile": "easyicu-locked-v1",
                "implementation_profile": "standard-profile",
                "definition_locked": True,
                "locked_core": {
                    "suspected_infection_windows": "owner validated",
                    "sofa_window": "owner validated",
                    "delta_rule": "owner validated",
                    "sofa_threshold": "owner validated",
                },
            },
        },
        "confirmations": {"feature_time_window": True},
    }

    snapshot = build_research_workflow_snapshot(
        study=locked_study,
        active_export_present=True,
        active_job=None,
        latest_run=None,
    )

    assert not any(
        field.startswith("confirmations.clinical_definition_")
        for field in snapshot.missing_setup_fields
    )


def test_workflow_requires_explicit_feature_time_window_confirmation() -> None:
    unconfirmed = {
        **_complete_study(),
        "confirmations": {"export_format": True},
    }

    snapshot = build_research_workflow_snapshot(
        study=unconfirmed,
        active_export_present=True,
        active_job=None,
        latest_run=None,
    )

    assert snapshot.missing_setup_fields == [
        "confirmations.feature_time_window"
    ]

    confirmed = build_research_workflow_snapshot(
        study={
            **unconfirmed,
            "confirmations": {
                "feature_time_window": True,
                "export_format": True,
            },
        },
        active_export_present=True,
        active_job=None,
        latest_run=None,
    )
    assert "confirmations.feature_time_window" not in confirmed.missing_setup_fields


def test_pipeline_factory_rejects_missing_typed_analysis_design_before_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation_called = False

    def foundation(**_kwargs: Any) -> dict[str, Any]:
        nonlocal foundation_called
        foundation_called = True
        return _foundation_profile()

    monkeypatch.setattr(
        research_pipeline_run_preparation,
        "_data_foundation_profile",
        foundation,
    )
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


def _design_free_study(export: Path) -> dict[str, Any]:
    """Only what the user owns: a question and one bound data source."""

    return {
        "id": "study-design-free",
        "revision": 1,
        "question": "How common is Sepsis-3 in adult ICU stays, and is it "
        "associated with ICU mortality?",
        "data_source": {"path": str(export), "database": "miiv"},
    }


def test_planner_only_run_supplies_neutral_materialization_scope(
    tmp_path: Path,
) -> None:
    """A plan-only run must not require the choices the plan exists to make."""

    export = _write_pipeline_export(tmp_path / "export")
    study = _design_free_study(export)

    scoped = research_launch_scientific._neutral_materialization_scope(
        study, export_path=str(export)
    )

    assert scoped["modules"] == ["demographics"]
    assert scoped["time_window"] == {"hours": 24.0, "anchor": "icu_admission"}
    assert scoped["materialization_scope_source"] == {
        "owner": "easyicu.webserver.agent_pipeline_runs",
        "kind": "easyicu_neutral_default",
        "applied_fields": ["modules", "time_window"],
    }
    # The Planner -- not Copilot, and not this default -- owns the design.
    for planner_owned in (
        "outcome",
        "primary_exposure",
        "cohort",
        "covariates",
        "analysis_goal",
        "analysis_design",
    ):
        assert planner_owned not in scoped


def test_planner_only_launch_reaches_the_planner_without_a_configured_design(
    tmp_path: Path,
) -> None:
    export = _write_pipeline_export(tmp_path / "export")

    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(export),
        study_context=_design_free_study(export),
        project_root=str(tmp_path / "workspace"),
        provider={"provider": "openai", "external": True},
        provider_environment={"OPENAI_API_KEY": "test-key"},
        credential_source="pi_verified",
        budget_mode="planner_canary",
    )

    assert callable(runner)


def test_planner_only_launch_does_not_require_a_prepared_package(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw-mimiciv"
    raw.mkdir()

    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(raw),
        study_context=_design_free_study(raw),
        project_root=str(tmp_path / "workspace"),
        provider={"provider": "openai", "external": True},
        provider_environment={"OPENAI_API_KEY": "test-key"},
        credential_source="pi_verified",
        budget_mode="planner_canary",
    )

    assert callable(runner)


def test_metadata_only_planning_acquisition_writes_zero_patient_rows(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    llm = ScriptedMockLLMClient(
        [
            json.dumps(
                {
                    "selected_concepts": ["lact", "sep3", "death"],
                    "inclusion_exclusion": ["Adult ICU stays"],
                    "rationale": "Exposure, phenotype, and mortality outcome.",
                }
            )
        ]
    )

    acquisition = agent_pipeline_runs._metadata_only_planning_acquisition(
        database="miiv",
        question="Is lactate associated with mortality in Sepsis-3?",
        llm=llm,
        output_dir=tmp_path / "planning",
    )

    assert acquisition.blocked is False
    assert acquisition.universe_path is not None
    cohort = pd.read_parquet(acquisition.universe_path)
    assert cohort.empty
    assert list(cohort.columns) == ["stay_id", "lact", "sep3", "death"]
    receipt = json.loads(acquisition.provenance_path.read_text(encoding="utf-8"))
    assert receipt["row_identity_column"] == "stay_id"
    assert receipt["patient_rows_read"] is False
    assert receipt["patient_rows_written"] is False
    assert receipt["observed_feasibility_claims"] is False
    assert receipt["execution_authorized"] is False


def test_metadata_only_planning_acquisition_projects_verified_patient_schema(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    llm = ScriptedMockLLMClient(
        [
            json.dumps(
                {
                    "selected_concepts": ["lact", "death"],
                    "inclusion_exclusion": ["Adult ICU stays"],
                    "rationale": "Exposure and mortality outcome.",
                }
            )
        ]
    )

    patient_grouping = PatientGroupingBinding(
        mapping_path=tmp_path / "private-patient-map.parquet",
        mapping_sha256="a" * 64,
        mapping_stay_column="stay_id",
        mapping_patient_column="patient_key",
        authority_coordinates={
            "schema_version": "easyicu.patient_grouping_runtime_authority/1",
            "authority_ref": "test/identity-bridge/v1",
            "database": "miiv",
            "mapping_sha256": "a" * 64,
            "grouping_derivation": "prefix_before_:s",
            "provider_visible_values": False,
        },
    )
    acquisition = agent_pipeline_runs._metadata_only_planning_acquisition(
        database="miiv",
        question="Is lactate associated with mortality?",
        llm=llm,
        output_dir=tmp_path / "planning",
        patient_grouping=patient_grouping,
        operationalized_columns=("lact_max",),
    )

    cohort = pd.read_parquet(acquisition.universe_path)
    assert cohort.empty
    assert list(cohort.columns) == [
        "stay_id",
        "patient_stay_id",
        "lact_max",
        "lact",
        "death",
    ]
    receipt = json.loads(acquisition.provenance_path.read_text(encoding="utf-8"))
    assert receipt["patient_identity_column"] == "patient_stay_id"
    assert receipt["operationalized_columns"] == ["lact_max"]
    assert receipt["replacement_row_identity"] == {
        "output_identity_column": "patient_stay_id",
        "mapping_file_sha256": "a" * 64,
        "mapped_cohort_rows": 0,
        "patient_group_derivation": {
            "algorithm": "prefix_before_:s",
            "delimiter": ":s",
        },
        "authority_coordinates": {
            "schema_version": "easyicu.patient_grouping_runtime_authority/1",
            "authority_ref": "test/identity-bridge/v1",
            "database": "miiv",
            "mapping_sha256": "a" * 64,
            "grouping_derivation": "prefix_before_:s",
            "provider_visible_values": False,
        },
    }
    assert receipt["patient_rows_read"] is False
    assert receipt["patient_rows_written"] is False


def test_metadata_planning_schema_projects_exact_operational_covariates() -> None:
    columns = research_launch_scientific._metadata_planning_operationalized_columns(
        primary_exposure_source="lact",
        primary_exposure_aggregation="max",
        covariates=("age", "sex", "charlson"),
        covariate_selection="exact",
        covariate_operationalizations={
            "age": "age",
            "sex": "sex",
            "charlson": "charlson_first",
        },
        sensitivity_specs=(
            SimpleNamespace(
                source_materialization_variables=(
                    "death_time",
                    "los_icu",
                    "charlson_first",
                )
            ),
        ),
    )

    assert columns == (
        "lact_max",
        "age",
        "sex",
        "charlson_first",
        "death_time",
        "los_icu",
    )


def test_metadata_planning_schema_projects_derived_landmark_event_time() -> None:
    from easyicu.research_agent.planning.sensitivity_authority import (
        PrespecifiedSensitivitySpec,
    )

    columns = research_launch_scientific._metadata_planning_operationalized_columns(
        primary_exposure_source="lact",
        primary_exposure_aggregation="max",
        covariates=("age", "sex", "charlson"),
        covariate_selection="exact",
        covariate_operationalizations={"charlson": "charlson_first"},
        sensitivity_specs=(
            PrespecifiedSensitivitySpec(
                spec_id="landmark_24h_primary",
                axis="timing",
                strategy="landmark",
                landmark_hours=24,
                require_alive_at_landmark=True,
                exclude_negative_event_times=True,
                event_time_variable="death_time",
                observation_duration_variable="los_icu",
                observation_duration_unit="days",
            ),
        ),
    )

    assert columns == (
        "lact_max",
        "age",
        "sex",
        "charlson_first",
        "los_icu",
        "death_time",
    )


def test_metadata_only_planning_coordinates_keep_explicit_lactate_and_mortality() -> None:
    coordinates = research_launch_scientific._metadata_only_planning_coordinates(
        database="miiv",
        question="我想研究 ICU 患者的乳酸水平和院内死亡有没有关系。",
    )

    assert coordinates["target_outcome"] == "death"
    assert coordinates["primary_exposure"] == "lact"
    assert coordinates["endpoint"].model_dump(mode="json") == {
        "name": "death",
        "kind": "binary",
        "absence_semantics": "no_absent_rows",
        "levels": [0, 1],
        "event_column": None,
        "time_column": None,
        "time_origin": None,
        "censoring_rule": None,
    }
    assert coordinates["execution_authorized"] is False


def test_planner_proposal_uses_icu_policy_and_available_materialization() -> None:
    acquisition = SimpleNamespace(
        analysis_columns={},
        materialized_columns=(
            "lact_max",
            "lact_min",
            "lact_mean",
            "lact_first",
        ),
    )

    proposed = agent_pipeline_runs._resolve_planner_proposed_primary_exposure(
        source_concept="lact",
        acquisition=acquisition,
    )

    # Lactate's preferred median representation is not emitted by the current
    # sealed materializer. The next case-neutral allowed representation is max;
    # it remains a candidate Plan choice, not persisted user configuration.
    assert proposed == "lact_max"


def test_planner_proposal_prefers_owner_issued_event_status_coordinate() -> None:
    acquisition = SimpleNamespace(
        analysis_columns={"death": "death_max"},
        materialized_columns=("death_max",),
    )

    assert agent_pipeline_runs._resolve_planner_proposed_primary_exposure(
        source_concept="death",
        acquisition=acquisition,
    ) == "death_max"


def test_materialized_target_outcome_uses_owner_issued_event_status_coordinate() -> None:
    acquisition = SimpleNamespace(
        analysis_columns={"death": "death_max"},
        materialized_columns=("death_max", "death_first"),
    )

    assert agent_pipeline_runs._resolve_materialized_target_outcome(
        source_concept="death",
        acquisition=acquisition,
    ) == "death_max"


def test_materialized_target_outcome_rejects_unmaterialized_projection() -> None:
    acquisition = SimpleNamespace(
        analysis_columns={"death": "death_max"},
        materialized_columns=("lact_max",),
    )

    assert agent_pipeline_runs._resolve_materialized_target_outcome(
        source_concept="death",
        acquisition=acquisition,
    ) is None


def test_plan_first_web_preferences_do_not_promote_planner_choices_to_user_authority() -> None:
    study = {
        "question": "Does a continuous ICU measurement relate to death?",
        "covariate_selection": "planner_selectable",
    }

    preferences = agent_pipeline_runs._research_user_preferences(study)

    assert preferences.get("covariate_selection", "planner_selectable") == (
        "planner_selectable"
    )
    assert "covariates" not in preferences
    assert "sensitivity_specs" not in preferences


def test_plan_first_package_does_not_override_user_scientific_authority(
) -> None:
    preferences = agent_pipeline_runs._research_user_preferences(
        {
            "covariate_selection": "exact",
            "covariates": ["age"],
            "covariate_rationales": {
                "age": "Baseline age is a prespecified confounding factor."
            },
            "covariate_temporal_roles": {"age": "baseline_static"},
        }
    )

    assert preferences["covariate_selection"] == "exact"
    assert preferences["covariates"] == ["age"]


def test_metadata_only_planning_coordinates_do_not_invent_unnamed_slots() -> None:
    coordinates = research_launch_scientific._metadata_only_planning_coordinates(
        database="miiv",
        question="我想先看看这批 ICU 数据能做什么。",
    )

    assert coordinates["target_outcome"] is None
    assert coordinates["primary_exposure"] is None
    assert coordinates["endpoint"] is None


def test_metadata_only_planning_ignores_unmapped_display_labels(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    llm = ScriptedMockLLMClient(
        [
            json.dumps(
                {
                    "selected_concepts": ["lact", "sep3", "death"],
                    "inclusion_exclusion": ["Adult ICU stays"],
                    "rationale": "Plan the requested exposure and outcome.",
                }
            )
        ]
    )

    acquisition = agent_pipeline_runs._metadata_only_planning_acquisition(
        database="miiv",
        question="Is lactate associated with ICU mortality in Sepsis-3?",
        llm=llm,
        output_dir=tmp_path / "planning",
        required_concepts=(
            "death",
            "ICU 住院期间死亡（death during the same ICU stay）",
            "入 ICU 后 24 小时内乳酸水平",
        ),
    )

    assert acquisition.blocked is False
    assert acquisition.coverage.missing == []
    assert acquisition.coverage.requested == ["lact", "sep3", "death"]
    assert list(pd.read_parquet(acquisition.universe_path).columns) == [
        "stay_id",
        "lact",
        "sep3",
        "death",
    ]


def test_planner_only_runner_reaches_pipeline_with_metadata_not_patient_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.research_agent as research_agent
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    actual_run = tmp_path / "actual-planner-run"
    _write_real_pipeline_fixture(
        actual_run,
        manuscript="# Plan\nMetadata-only planning remains review-bound.",
    )
    llm = ScriptedMockLLMClient(
        [
            json.dumps(
                {
                    "selected_concepts": ["lact", "sep3", "death"],
                    "inclusion_exclusion": ["Adult ICU stays"],
                    "rationale": "Plan the requested exposure and outcome.",
                }
            )
        ]
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        provider_adapter,
        "build_research_agent_provider_client",
        lambda *_args, **_kwargs: (
            llm,
            {"provider": "mock", "model": "metadata-only-test"},
        ),
    )

    class FakePipeline:
        def run(self, **kwargs: Any) -> SimpleNamespace:
            cohort = pd.read_parquet(kwargs["cohort"])
            captured["cohort_rows"] = len(cohort)
            captured["cohort_columns"] = list(cohort.columns)
            captured["planning_authority"] = cohort.attrs.get(
                "easyicu_planning_authority"
            )
            captured["cohort_authority_path"] = kwargs["cohort_authority_path"]
            captured["id_columns"] = kwargs["id_columns"]
            captured["target_outcome"] = kwargs["target_outcome"]
            captured["primary_exposure"] = kwargs["primary_exposure"]
            captured["endpoint"] = (
                kwargs["endpoint"].model_dump(mode="json")
                if kwargs["endpoint"] is not None
                else None
            )
            return SimpleNamespace(manifest_path=actual_run / "manifest.json")

    monkeypatch.setattr(
        research_agent.ResearchAgentPipeline,
        "from_config",
        lambda _config, *, services: FakePipeline(),
    )
    def patient_grouping(_study: Any) -> PatientGroupingBinding:
        return PatientGroupingBinding(
            mapping_path=tmp_path / "private-patient-map.parquet",
            mapping_sha256="a" * 64,
            mapping_stay_column="stay_id",
            mapping_patient_column="patient_key",
            authority_coordinates={
                "schema_version": "easyicu.patient_grouping_runtime_authority/1",
                "authority_ref": "test/identity-bridge/v1",
                "database": "miiv",
                "mapping_sha256": "a" * 64,
                "grouping_derivation": "prefix_before_:s",
                "provider_visible_values": False,
            },
        )

    monkeypatch.setattr(
        research_launch_scientific,
        "_patient_grouping_for_analysis_design",
        patient_grouping,
    )
    monkeypatch.setattr(
        research_pipeline_run_preparation,
        "_patient_grouping_for_analysis_design",
        patient_grouping,
    )
    prepared = _write_pipeline_export(tmp_path / "prepared-mimiciv")
    study = {
        **_design_free_study(prepared),
        "outcome": "ICU 住院期间死亡（death during the same ICU stay）",
        "primary_exposure": "入 ICU 后 24 小时内乳酸水平",
        "analysis_design": {
            "analysis_family": "association_study",
            "analysis_unit": "icu_stay",
            "variance_estimator": "cluster_robust",
            "cluster_unit": "patient",
        },
    }
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(prepared),
        study_context=study,
        project_root=str(tmp_path / "projects"),
        provider={"provider": "openai", "external": True},
        provider_environment={"OPENAI_API_KEY": "test-key"},
        credential_source="pi_verified",
        budget_mode="planner_canary",
    )

    class Job:
        id = "job-metadata-only-plan"
        cancel_requested = False
        events: list[dict[str, Any]] = []

        def emit(self, event: dict[str, Any]) -> None:
            self.events.append(dict(event))

    result = runner(Job())

    assert captured == {
        "cohort_rows": 0,
        "cohort_columns": [
            "stay_id",
            "patient_stay_id",
            "lact",
            "sep3",
            "death",
        ],
        "planning_authority": {
            "kind": "metadata_only_planning_catalog",
            "patient_rows_read": False,
            "replacement_row_identity": {
                "output_identity_column": "patient_stay_id",
                "mapping_file_sha256": "a" * 64,
                "mapped_cohort_rows": 0,
                "patient_group_derivation": {
                    "algorithm": "prefix_before_:s",
                    "delimiter": ":s",
                },
                "authority_coordinates": {
                    "schema_version": (
                        "easyicu.patient_grouping_runtime_authority/1"
                    ),
                    "authority_ref": "test/identity-bridge/v1",
                    "database": "miiv",
                    "mapping_sha256": "a" * 64,
                    "grouping_derivation": "prefix_before_:s",
                    "provider_visible_values": False,
                },
            },
        },
        "cohort_authority_path": None,
        "id_columns": ["patient_stay_id"],
        "target_outcome": "death",
        "primary_exposure": "sep3",
        "endpoint": {
            "name": "death",
            "kind": "binary",
            "absence_semantics": "no_absent_rows",
            "levels": [0, 1],
            "event_column": None,
            "time_column": None,
            "time_origin": None,
            "censoring_rule": None,
        },
    }, result
    assert result["provider"]["model"] == "metadata-only-test"


def test_full_reviewed_launch_uses_a_neutral_scope_until_plan_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prepared package can reach Planner before the user designs its plan."""

    export = _write_pipeline_export(tmp_path / "export")
    _assume_execution_runtime_ready(monkeypatch)

    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(export),
        study_context=_design_free_study(export),
        project_root=str(tmp_path / "workspace"),
        provider={"provider": "openai", "external": True},
        provider_environment={"OPENAI_API_KEY": "test-key"},
        credential_source="pi_verified",
        budget_mode="full_reviewed",
    )

    assert callable(runner)


@pytest.mark.parametrize(
    ("committed", "expected_code"),
    (
        (
            {"preset": "full_available_stay", "label": "Whole stay",
             "anchor": "ICU admission"},
            "research_pipeline_time_window_hours_required",
        ),
        (
            {"hours": 24},
            "research_pipeline_time_window_anchor_required",
        ),
    ),
)
def test_neutral_scope_never_completes_a_partially_committed_window(
    tmp_path: Path,
    committed: dict[str, Any],
    expected_code: str,
) -> None:
    """Silently finishing a half-specified window would execute other science.

    A prose window label carries no executable hours.  Filling them in from the
    neutral default would run "first 24 hours" while the conversation agreed to
    "whole stay", so a committed-but-incomplete scope must still fail at its
    owner rather than be completed here.
    """

    export = _write_pipeline_export(tmp_path / "export")
    study = {**_design_free_study(export), "time_window": committed}

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.make_research_pipeline_run_runner(
            export_path=str(export),
            study_context=study,
            project_root=str(tmp_path / "workspace"),
            provider={"provider": "openai", "external": True},
            provider_environment={"OPENAI_API_KEY": "test-key"},
            credential_source="pi_verified",
            budget_mode="planner_canary",
        )

    assert exc.value.code == expected_code


def test_neutral_scope_preserves_a_configured_scope(tmp_path: Path) -> None:
    export = _write_pipeline_export(tmp_path / "export")
    study = {
        **_design_free_study(export),
        "modules": ["vitals"],
        "time_window": {"hours": 72, "anchor": "icu_admission"},
    }

    scoped = research_launch_scientific._neutral_materialization_scope(
        study, export_path=str(export)
    )

    assert scoped["modules"] == ["vitals"]
    assert scoped["time_window"] == {"hours": 72, "anchor": "icu_admission"}
    assert "materialization_scope_source" not in scoped


def test_pipeline_factory_rejects_non_executable_time_window_label_before_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation_called = False

    def foundation(**_kwargs: Any) -> dict[str, Any]:
        nonlocal foundation_called
        foundation_called = True
        return _foundation_profile()

    monkeypatch.setattr(
        research_pipeline_run_preparation,
        "_data_foundation_profile",
        foundation,
    )
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

    monkeypatch.setattr(
        research_pipeline_run_preparation,
        "_data_foundation_profile",
        foundation,
    )
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

    monkeypatch.setattr(
        research_pipeline_run_preparation,
        "_data_foundation_profile",
        foundation,
    )
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


def test_counts_only_design_does_not_request_unneeded_longitudinal_trajectory() -> None:
    study = {
        **_complete_study(),
        "analysis_design": {
            "analysis_family": "descriptive_epidemiology",
            "analysis_unit": "icu_stay",
            "variance_estimator": "none_counts_only",
        },
    }

    assert agent_pipeline_runs._analysis_requires_longitudinal_trajectory(
        study,
        validated_design={
            "analysis_unit": "icu_stay",
            "variance_estimator": "none_counts_only",
        },
    ) is False


def test_landmark_sensitivity_keeps_trajectory_for_counts_only_design() -> None:
    study = {
        **_complete_study(),
        "analysis_design": {
            "analysis_family": "descriptive_epidemiology",
            "analysis_unit": "icu_stay",
            "variance_estimator": "none_counts_only",
        },
        "sensitivity_specs": [
            {
                "spec_id": "landmark_24h",
                "axis": "timing",
                "strategy": "landmark",
                "landmark_hours": 24,
                "require_alive_at_landmark": True,
            }
        ],
    }

    assert agent_pipeline_runs._analysis_requires_longitudinal_trajectory(
        study,
        validated_design={
            "analysis_unit": "icu_stay",
            "variance_estimator": "none_counts_only",
        },
    ) is True


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
            "plan_approval_allowed": True,
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


def test_live_review_authority_overrides_stale_approvable_run_history() -> None:
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
            "artifact_names": ["agent_plan.json", "source_run_manifest.json"],
        },
        plan_review_authority={
            "run_id": "run-plan-review",
            "resumable_here": True,
            "scientific_configuration_sha256": (
                study_context_owner.scientific_configuration_sha256(study)
            ),
            "requests": [
                {
                    "reason_code": "plan_scientific_changes_required",
                    "approval_allowed": False,
                }
            ],
        },
    )

    by_id = {row.id: row for row in snapshot.stages}
    assert snapshot.next_action_code == "plan_scientific_changes_required"
    assert by_id["plan"].reason_code == "plan_scientific_changes_required"
    assert by_id["analysis"].reason_code == "plan_scientific_changes_required"


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
                        "message": "Exposure timing is not closed by an executable design.",
                        "evidence_refs": [
                            "research_context.json",
                            "analysis_plan.json",
                        ],
                        "remediation": (
                            "Create a new landmark version or keep the current "
                            "study descriptive."
                        ),
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
                "question": (
                    "Use a new landmark version or keep this study descriptive?"
                ),
                "evidence": "Exposure timing is not closed by an executable design.",
                "evidence_refs": [
                    "research_context.json",
                    "analysis_plan.json",
                ],
                "remediation": (
                    "Create a new landmark version or keep the current study descriptive."
                ),
            }
        ],
        "remediation_buckets": {
            "agent_plan_revision": [
                "CONTINUOUS_COVARIATE_FUNCTIONAL_FORM_UNCHECKED",
            ],
            "study_authority_change": [
                "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"
            ],
            "external_evidence": ["DIRECT_COMPARATOR_NOT_ESTABLISHED"],
            "independent_review": [],
        },
    }


def test_plan_review_separates_system_proposals_from_user_authorization() -> None:
    study = _complete_study()
    review = {
        "status": "changes_required",
        "findings": [
            {
                "code": "OUTCOME_DEFINITION_UNRESOLVED",
                "requires_user_authorization": True,
                "authorization_question": "Which endpoint should be used?",
            },
            {
                "code": "ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED",
                "requires_user_authorization": True,
                "authorization_question": "Which sensitivity analyses?",
            },
            {
                "code": "ADJUSTMENT_SET_NOT_USER_CONFIRMED",
                "requires_user_authorization": True,
                "authorization_question": "Which adjustment variables?",
            },
        ],
        "facts": {
            "remediation_buckets": {
                "agent_plan_revision": [],
                "study_authority_change": [
                    "OUTCOME_DEFINITION_UNRESOLVED",
                    "ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED",
                    "ADJUSTMENT_SET_NOT_USER_CONFIRMED",
                ],
                "external_evidence": [],
                "independent_review": [],
            }
        },
    }
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_type": "full",
            "run_id": "run-planner-only-review",
            "budget_mode": "full_reviewed",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "run_status": "human_review_pending",
            "pending_review_reason_codes": ["plan_scientific_changes_required"],
            "artifact_names": ["agent_plan.json", "scientific_plan_review.json"],
        },
        plan_review_authority={
            "run_id": "run-planner-only-review",
            "resumable_here": True,
            "scientific_configuration_sha256": (
                study_context_owner.scientific_configuration_sha256(study)
            ),
            "scientific_plan_review": review,
        },
    )

    assert snapshot.next_action_code == "plan_scientific_changes_required"
    assert snapshot.plan_review_summary is not None
    assert snapshot.plan_review_summary["authorization_questions"] == [
        {
            "code": "ADJUSTMENT_SET_NOT_USER_CONFIRMED",
            "question": "Which adjustment variables?",
        }
    ]
    assert snapshot.plan_review_summary["remediation_buckets"] == {
        "agent_plan_revision": [
            "OUTCOME_DEFINITION_UNRESOLVED",
            "ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED",
        ],
        "study_authority_change": ["ADJUSTMENT_SET_NOT_USER_CONFIRMED"],
        "external_evidence": [],
        "independent_review": [],
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


def test_failed_approved_execution_retries_exact_plan_when_study_is_unchanged() -> None:
    study = _complete_study()
    digest = study_context_owner.scientific_configuration_sha256(study)
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_id": "run-execution-failed",
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "gate_reason": "research_agent_pipeline_failed_closed",
            "run_status": "blocked",
            "scientific_configuration_sha256": digest,
            "artifact_names": [
                "agent_plan.json",
                "evidence_ledger.json",
                "source_run_manifest.json",
            ],
        },
    )

    by_id = {row.id: row for row in snapshot.stages}
    assert snapshot.next_action_code == "failed_pipeline_execution_retry_available"
    assert by_id["plan"].reason_code == "failed_pipeline_execution_retry_available"
    assert by_id["analysis"].reason_code == (
        "failed_pipeline_execution_retry_available"
    )


def test_durable_web_execution_failure_also_retries_exact_approved_plan() -> None:
    study = _complete_study()
    digest = study_context_owner.scientific_configuration_sha256(study)
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_id": "run-web-wrapper",
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "gate_reason": "research_pipeline_execution_failed",
            "run_status": "failed",
            "scientific_configuration_sha256": digest,
            "artifact_names": [
                "agent_plan.json",
                "evidence_ledger.json",
                "source_run_manifest.json",
            ],
        },
    )

    assert snapshot.next_action_code == "failed_pipeline_execution_retry_available"


def test_retry_bridge_failure_preserves_exact_approved_plan_retry() -> None:
    study = _complete_study()
    digest = study_context_owner.scientific_configuration_sha256(study)
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_id": "run-web-wrapper",
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "gate_reason": (
                "research_pipeline_execution_retry_unexpected_plan_review"
            ),
            "run_status": "failed",
            "scientific_configuration_sha256": digest,
            "artifact_names": [
                "agent_plan.json",
                "evidence_ledger.json",
                "source_run_manifest.json",
            ],
        },
    )

    assert snapshot.next_action_code == "failed_pipeline_execution_retry_available"


def test_efficiency_budget_failure_offers_owned_checkpoint_resume() -> None:
    study = _complete_study()
    digest = study_context_owner.scientific_configuration_sha256(study)
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_id": "run-budgeted-plan",
            "study_id": study["id"],
            "scientific_configuration_sha256": digest,
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "gate_reason": "research_pipeline_planner_efficiency_budget_exhausted",
            "run_status": "failed",
            "development_planner_checkpoint_available": True,
            "artifact_names": [
                "evidence_ledger.json",
                "source_run_manifest.json",
            ],
        },
    )

    by_id = {row.id: row for row in snapshot.stages}
    assert snapshot.next_action_code == "planner_checkpoint_resume_available"
    assert by_id["plan"].status == "ready"
    assert by_id["plan"].reason_code == "planner_checkpoint_resume_available"
    assert by_id["analysis"].status == "blocked"


def test_contract_exhaustion_with_checkpoint_requires_a_fresh_plan() -> None:
    study = _complete_study()
    digest = study_context_owner.scientific_configuration_sha256(study)
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_id": "run-contract-repair",
            "study_id": study["id"],
            "scientific_configuration_sha256": digest,
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "gate_reason": "research_pipeline_plan_contract_exhausted",
            "run_status": "failed",
            "development_planner_checkpoint_available": True,
            "artifact_names": [
                "evidence_ledger.json",
                "source_run_manifest.json",
            ],
        },
    )

    assert snapshot.next_action_code == "failed_pipeline_requires_fresh_plan"


def test_compile_gate_failure_with_checkpoint_requires_a_fresh_plan() -> None:
    study = _complete_study()
    digest = study_context_owner.scientific_configuration_sha256(study)
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_id": "run-compile-revalidation",
            "study_id": study["id"],
            "scientific_configuration_sha256": digest,
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "gate_reason": "research_pipeline_progressive_compile_failed",
            "run_status": "failed",
            "development_planner_checkpoint_available": True,
            "artifact_names": [
                "evidence_ledger.json",
                "source_run_manifest.json",
            ],
        },
    )

    assert snapshot.next_action_code == "failed_pipeline_requires_fresh_plan"


def test_submission_owner_does_not_resume_contract_failure_without_checkpoint(
    tmp_path: Path,
) -> None:
    study = _complete_study()
    root = tmp_path / "projects" / study["id"]
    project_dir = root / study["id"] / "run_failed-outline"
    project_dir.mkdir(parents=True)
    row = {
            "run_id": "run_failed-outline",
            "study_id": study["id"],
            "run_status": "failed",
            "gate_reason": "research_pipeline_plan_contract_exhausted",
            "scientific_configuration_sha256": (
                study_context_owner.scientific_configuration_sha256(study)
            ),
            "project_dir": str(project_dir),
            "development_planner_checkpoint_available": False,
        }

    assert resumable_planner_checkpoint_job_id(
        study=study, rows=[row], project_root=root
    ) == ""


def test_submission_owner_resumes_compile_failed_checkpoint(
    tmp_path: Path,
) -> None:
    study = _complete_study()
    root = tmp_path / "projects" / study["id"]
    project_dir = root / study["id"] / "run_compile-revalidation"
    project_dir.mkdir(parents=True)
    row = {
            "run_id": "run_compile-revalidation",
            "study_id": study["id"],
            "run_status": "failed",
            "gate_reason": "research_pipeline_progressive_compile_failed",
            "scientific_configuration_sha256": (
                study_context_owner.scientific_configuration_sha256(study)
            ),
            "project_dir": str(project_dir),
            "development_planner_checkpoint_available": True,
        }

    assert (
        resumable_planner_checkpoint_job_id(
            study=study, rows=[row], project_root=root
        )
        == "compile-revalidation"
    )


def test_submission_owner_recovers_checkpoint_hidden_by_empty_foundation_projection(
    tmp_path: Path,
) -> None:
    study = _complete_study()
    root = tmp_path / "projects" / study["id"]
    project_dir = root / study["id"] / "run_compile-revalidation"
    project_dir.mkdir(parents=True)
    digest = study_context_owner.scientific_configuration_sha256(study)
    empty_projection = {
        "run_id": "run_empty-foundation",
        "study_id": study["id"],
        "run_status": "blocked",
        "gate_reason": "data_foundation_blocked",
        "scientific_configuration_sha256": digest,
        "development_planner_checkpoint_available": False,
    }
    checkpoint = {
        "run_id": "run_compile-revalidation",
        "study_id": study["id"],
        "run_status": "failed",
        "gate_reason": "research_pipeline_progressive_compile_failed",
        "scientific_configuration_sha256": digest,
        "project_dir": str(project_dir),
        "development_planner_checkpoint_available": True,
    }
    assert (
        resumable_planner_checkpoint_job_id(
            study=study,
            rows=[empty_projection, checkpoint],
            project_root=root,
        )
        == "compile-revalidation"
    )


def test_validated_analysis_advances_even_when_publication_gate_stays_closed() -> None:
    snapshot = build_research_workflow_snapshot(
        study=_complete_study(),
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_id": "run-analysis-only-withheld-paper",
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "gate_reason": "research_agent_pipeline_failed_closed",
            "gate_checks": {
                "execution_complete": True,
                "analysis_validated": True,
                "evidence_complete": False,
                "numeric_verified": True,
                "manuscript_ready": False,
                "publication_ready": False,
            },
            "run_status": "blocked",
            "artifact_names": [
                "agent_plan.json",
                "evidence_ledger.json",
                "result_tables.json",
                "figure_gallery.json",
                "manuscript_draft.json",
                "source_run_manifest.json",
            ],
        },
    )

    by_id = {row.id: row for row in snapshot.stages}
    assert snapshot.current_stage == "interpretation"
    assert snapshot.next_action_code == "evidence_bound_interpretation_ready"
    assert by_id["plan"].status == "complete"
    assert by_id["analysis"].status == "complete"
    assert by_id["analysis"].reason_code == "validated_analysis_ready"
    assert by_id["interpretation"].status == "review_required"
    assert by_id["manuscript"].status == "review_required"


def test_completed_numeric_outputs_do_not_force_a_fresh_plan_during_validation_repair() -> None:
    study = _complete_study()
    digest = study_context_owner.scientific_configuration_sha256(study)
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_id": "run-analysis-validation-open",
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "gate_reason": "research_agent_pipeline_failed_closed",
            "gate_checks": {
                "execution_complete": True,
                "analysis_validated": False,
                "evidence_complete": True,
                "numeric_verified": True,
            },
            "run_status": "analysis_only",
            "scientific_configuration_sha256": digest,
            "artifact_names": [
                "agent_plan.json",
                "evidence_ledger.json",
                "result_tables.json",
                "figure_gallery.json",
                "source_run_manifest.json",
            ],
        },
    )

    by_id = {row.id: row for row in snapshot.stages}
    assert by_id["setup"].status == "complete"
    assert by_id["extraction"].status == "complete"
    assert by_id["plan"].status == "complete"
    assert by_id["analysis"].status == "review_required"
    assert by_id["analysis"].reason_code == "analysis_outputs_require_validation"
    assert by_id["interpretation"].status == "review_required"
    assert snapshot.next_action_code == "analysis_outputs_require_validation"
    assert snapshot.analysis_validation_retry_available is True


def test_validated_analysis_receipt_does_not_fall_back_to_legacy_blank_setup() -> None:
    study = {
        "id": "study-legacy-blank-setup",
        "revision": 7,
        "question": "Is lactate associated with in-hospital mortality?",
        "data_source": {"path": "/private/prepared/source", "database": "miiv"},
        "cohort": {"label": "ICU patients"},
        "modules": [],
        "outcome": "",
        "analysis_goal": "",
        "time_window": {},
        "export_format": "",
        "confirmations": {"extraction_completed": True},
    }
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=False,
        active_job=None,
        latest_run={
            "run_id": "run-complete-analysis",
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "gate_checks": {
                "execution_complete": True,
                "analysis_validated": True,
                "numeric_verified": True,
            },
            "artifact_names": [
                "agent_plan.json",
                "evidence_ledger.json",
                "result_tables.json",
                "figure_gallery.json",
                "manuscript_draft.json",
                "source_run_manifest.json",
            ],
        },
    )

    by_id = {row.id: row for row in snapshot.stages}
    assert by_id["setup"].status == "complete"
    assert by_id["setup"].reason_code == "approved_plan_setup_receipt"
    assert by_id["extraction"].status == "complete"
    assert by_id["extraction"].reason_code == "approved_analysis_input_receipt"
    assert snapshot.current_stage == "interpretation"
    assert snapshot.next_action_code == "evidence_bound_interpretation_ready"


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
    assert snapshot.next_action_code == "provider_ready_to_generate_plan"
    plan = next(row for row in snapshot.stages if row.id == "plan")
    assert plan.status == "ready"
    assert plan.reason_code == "provider_ready_to_generate_plan"


def test_question_and_confirmed_data_package_offer_planner_before_setup_questionnaire() -> None:
    cohort, authority = _confirmed_cohort_decision(
        "no_eligibility_filter",
        study_context_id="study-planner-first",
        study_context_revision=2,
    )
    study = {
        "id": "study-planner-first",
        "revision": 2,
        "question": "Is early peak lactate associated with hospital mortality?",
        "purpose": "Generate an evidence-bound research plan.",
        "data_source": {
            "path": "/private/prepared/source",
            "database": "miiv",
        },
        "cohort": cohort,
        "cohort_eligibility_authority": authority,
        "modules": [],
        "outcome": "",
        "primary_exposure": "",
        "analysis_goal": "",
        "time_window": {},
        "export_format": "",
        "confirmations": {"extraction_completed": True},
    }

    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run=None,
    )

    by_id = {row.id: row for row in snapshot.stages}
    assert snapshot.current_stage == "plan"
    assert snapshot.next_action_code == "provider_ready_to_generate_plan"
    assert "cohort" not in snapshot.missing_setup_fields
    assert "cohort_eligibility" not in snapshot.missing_setup_fields
    assert "outcome" in snapshot.missing_setup_fields
    assert by_id["setup"].status == "ready"
    assert by_id["plan"].status == "ready"
    assert by_id["analysis"].status == "blocked"


def test_completed_preflight_receipt_does_not_fall_back_to_extraction() -> None:
    study = _complete_study()
    study["confirmations"] = {
        key: value
        for key, value in study["confirmations"].items()
        if key != "extraction_completed"
    }
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=False,
        active_job=None,
        latest_run={
            "run_type": "preflight",
            "gate_status": "analysis_only",
            "readiness_status": "awaiting_human_signoff",
            "artifact_names": ["evidence_ledger.json"],
        },
    )

    by_id = {row.id: row for row in snapshot.stages}
    assert by_id["extraction"].status == "complete"
    assert snapshot.current_stage == "plan"
    assert snapshot.next_action_code == "provider_ready_to_generate_plan"


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


def test_bound_database_source_is_not_a_completed_feature_extraction() -> None:
    study = _complete_study()
    study["confirmations"] = {
        key: value
        for key, value in study["confirmations"].items()
        if key != "extraction_completed"
    }
    registry = {
        "sources": [
            {"path": "/private/prepared/source", "ok": True, "modules": ["vitals"]}
        ]
    }

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


def test_result_interpretation_card_separates_validated_analysis_from_publication_gate() -> None:
    card = build_result_interpretation_card(
        run_id="run_analysis_only",
        review={
            "gate": {
                "status": "blocked",
                "reason": "publication_requirements_incomplete",
                "checks": [
                    {"id": "analysis_validated", "passed": True},
                    {"id": "numeric_verified", "passed": True},
                    {"id": "paper_authorized", "passed": False},
                ],
            },
            "readiness": {"status": "blocked", "reportable": False},
            "artifacts": [{"name": "result_tables.json"}],
        },
        manuscript=None,
        result_tables={
            "tables": [
                {
                    "name": "distribution.csv",
                    "label": "Aggregate distribution",
                    "evidence_id": "ev-distribution",
                    "headers": [
                        "n_rows",
                        "exposure_denominator",
                        "exposure_pct",
                        "outcome_events",
                        "outcome_denominator",
                        "outcome_rate_pct",
                    ],
                    "rows": [["100", "100", "40", "8", "40", "20"]],
                }
            ]
        },
        scientific_readiness={
            "status": "analysis_only",
            "claim_ceiling": "analysis_only",
            "facts": {"analysis": {"analysis_validated": True}},
        },
    )

    assert card.status == "analysis_only"
    assert card.claim_ceiling == "analysis_only"
    assert card.result_tables[0].entries == [["100", "100", "40", "8", "40", "20"]]


def test_result_interpretation_card_keeps_unverified_numbers_blocked() -> None:
    card = build_result_interpretation_card(
        run_id="run_unverified",
        review={
            "gate": {
                "status": "blocked",
                "checks": [{"id": "numeric_verified", "passed": False}],
            },
            "readiness": {"status": "blocked", "reportable": False},
        },
        manuscript=None,
        result_tables=None,
        scientific_readiness={
            "status": "analysis_only",
            "claim_ceiling": "analysis_only",
            "facts": {"analysis": {"analysis_validated": True}},
        },
    )

    assert card.status == "blocked"
    assert card.claim_ceiling == "unsupported"


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
    manuscript["claims"][0]["text"] = (
        "Host-only grouping uses stay_id and must not enter the model projection."
    )
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
    assert len(interpretation["details"]["interpretation"]["claims"]) == 11
    assert interpretation["details"]["withheld_claim_count"] == 1
    assert draft["code"] == "easyicu_manuscript_projected"
    assert len(draft["details"]["manuscript"]["review_claims"]) == 11
    assert draft["details"]["manuscript"]["withheld_review_claim_count"] == 1
    assert "markdown_preview" not in draft["details"]["manuscript"]
    assert len(json.dumps(interpretation).encode("utf-8")) < 32_768
    assert len(json.dumps(draft).encode("utf-8")) < 32_768


def test_validation_projects_analysis_status_and_owner_operational_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ToolExecutionContext(
        session=PiSessionRecord(session_id="pi-validation-mapping")
    )
    review = {
        "gate": {
            "status": "blocked",
            "reason": "publication_requirements_incomplete",
            "reportable": False,
            "draft_unlocked": False,
            "checks": [
                {"id": "analysis_validated", "passed": True},
                {"id": "numeric_verified", "passed": True},
                {"id": "publication_ready", "passed": False},
            ],
        },
        "readiness": {"status": "blocked", "reportable": False},
        "artifact_payloads": {
            "scientific_readiness.json": {
                "status": "analysis_only",
                "claim_ceiling": "analysis_only",
                "publication_ready": False,
                "facts": {"analysis": {"analysis_validated": True}},
            },
            "result_tables.json": {
                "table_count": 1,
                "tables": [
                    {
                        "headers": [
                            "concept",
                            "indicator_semantics",
                            "value_column",
                        ],
                        "rows": [
                            [
                                "sep3_sofa2",
                                "binary_event_presence",
                                "sep3_sofa2_max",
                            ]
                        ],
                    }
                ],
            },
        },
        "signed": False,
        "signoff_stale": False,
    }
    monkeypatch.setattr(
        tool_module,
        "_select_run",
        lambda _context, _requested_run_id=None: {"run_id": "run-mapping"},
    )
    monkeypatch.setattr(tool_module, "_run_review", lambda _row: review)

    result = tool_module.execute_tool(
        "easyicu_inspect_validation", {"run_id": "run-mapping"}, context
    )

    execution = result["details"]["analysis_execution"]
    assert execution["analysis_validated"] is True
    assert execution["numeric_verified"] is True
    assert execution["publication_gate_separate"] is True
    assert execution["operational_mappings"] == [
        {
            "semantic_concept": "sep3_sofa2",
            "operational_value_column": "sep3_sofa2_max",
            "authority": "validated_result_measurement_audit",
            "indicator_semantics": "binary_event_presence",
        }
    ]


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


def test_blocked_literature_routes_an_unplanned_study_to_the_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal here must name the owning next step, not dead-end the turn.

    Before a plan exists the exposure and outcome are empty *by design* -- the
    Planner chooses them.  A bare block made Pi conclude the plan itself was
    impossible and stop without ever calling the Planner, so the receipt has to
    say where the authority actually lives.
    """

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
    monkeypatch.setattr(
        tool_module.idea_mining,
        "discover_literature",
        lambda body: pytest.fail("literature must not run before a plan"),
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(session_id="pi-literature-routes-to-plan"),
        allowed_actions={"literature"},
    )

    result = tool_module.execute_tool("easyicu_search_literature", {}, context)

    assert result["status"] == "blocked"
    assert result["code"] == "literature_study_scope_incomplete"
    assert result["details"]["plan_generation_ready"] is True
    assert result["details"]["next_action_code"] == "provider_ready_to_generate_plan"
    # The summary is what the model reads; it must point at the plan and must
    # not read as a data or permission failure.
    assert "Planner" in result["summary"]
    assert "not a data or permission failure" in result["summary"]
    # The one-turn grant is not spent on a refusal.
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

    def submit(
        body: dict[str, Any],
        *,
        account_environment: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        assert account_environment is None
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
    monkeypatch.setattr(
        tool_module.extraction_handoff,
        "compile_registered_export_handoff",
        lambda _study, _source: SimpleNamespace(
            reusable=True,
            public_receipt=lambda: {
                "schema_version": "easyicu.pi-extraction-handoff/1",
                "source_id": "src_project",
                "reusable": True,
                "mismatch_codes": [],
            },
        ),
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(session_id="pi-extract-reuse"),
        allowed_actions={"extract"},
    )

    result = tool_module.execute_tool("easyicu_start_extraction", {}, context)

    assert result["code"] == "easyicu_registered_export_reused"
    assert result["details"]["active_export"]["source_id"] == "src_project"
    assert "/private/" not in json.dumps(result)


def test_extraction_rebuilds_registered_export_when_requested_contract_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = {
        **_complete_study(),
        "data_source": {
            "path": "/private/registered/export",
            "database": "miiv",
        },
        "cohort": {
            "preset": "icd",
            "age_min": 18,
            "include_diagnoses": ["A41"],
        },
        "modules": ["demographics", "blood_gas", "outcome"],
        "time_window": {"observation_hours": 24, "anchor": "ICU admission"},
        "export_format": "csv",
    }
    submitted: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: study)
    monkeypatch.setattr(
        tool_module.sources,
        "load_registry",
        lambda: {
            "sources": [
                {
                    "id": "src_project",
                    "path": "/private/registered/export",
                    "database": "miiv",
                    "ok": True,
                }
            ]
        },
    )
    handoff_cohort = {
        "preset": "icd",
        "age_min": 18,
        "age_max": 100,
        "exclude_readmissions": False,
        "icd_enabled": True,
        "icd_include": ["A41"],
        "icd_exclude": [],
        "observation_window_hours": 24,
    }
    monkeypatch.setattr(
        tool_module.extraction_handoff,
        "compile_registered_export_handoff",
        lambda _study, _source: SimpleNamespace(
            reusable=False,
            source_data_path="/private/demo/raw",
            database="miiv",
            modules=("demographics", "blood_gas", "outcome"),
            export_format="csv",
            cohort=handoff_cohort,
            public_receipt=lambda: {
                "schema_version": "easyicu.pi-extraction-handoff/1",
                "reusable": False,
                "mismatch_codes": [
                    "registered_export_cohort_mismatch",
                    "registered_export_format_mismatch",
                ],
            },
        ),
    )
    from easyicu.webserver.routes import jobs as jobs_route

    monkeypatch.setattr(
        jobs_route,
        "jobs_extract",
        lambda body: submitted.append(dict(body))
        or {
            "job_id": "extract-a41",
            "kind": "extract",
            "status": "running",
            "study_context_id": study["id"],
            "study_context_revision": study["revision"],
        },
    )

    result = tool_module.execute_tool(
        "easyicu_start_extraction",
        {},
        ToolExecutionContext(
            session=PiSessionRecord(session_id="pi-extract-contract-mismatch"),
            allowed_actions={"extract"},
        ),
    )

    assert result["code"] == "easyicu_extraction_submitted"
    assert submitted[0]["path"] == "/private/demo/raw"
    assert submitted[0]["registered_export_path"] == "/private/registered/export"
    assert submitted[0]["format"] == "csv"
    assert submitted[0]["cohort"] == handoff_cohort
    assert result["details"]["extraction_contract"]["reusable"] is False
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


def test_full_run_reports_setup_owner_before_provider_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete = {
        "id": "study-incomplete",
        "revision": 3,
        "question": "What is the aggregate Sepsis-3 prevalence?",
        "data_source": {"path": "/private/raw/miiv", "database": "mimiciv"},
        "cohort": {"preset": "adult_icu"},
    }
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: incomplete)
    def reject(*_args: Any, **_kwargs: Any) -> Any:
        raise research_run_submission.ResearchRunSubmissionError(
            {
                "error": "study_setup_incomplete",
                "message": "Bind the research question and data source before generating the candidate plan.",
                "owner": "easyicu.webserver.pi_copilot.workflow",
                "next_action_code": "study_setup_incomplete",
                "planning_prerequisites_missing": ["data_source"],
                "missing_setup_fields": ["data_source", "outcome", "modules"],
            }
        )

    monkeypatch.setattr(
        research_run_submission,
        "submit_research_run",
        reject,
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-incomplete-plan",
            external_llm_opt_in=True,
            binding=AuthorityBinding(
                study_context_id="study-incomplete",
                study_revision=3,
            ),
        )
    )

    result = tool_module.execute_tool(
        "easyicu_run", {"run_type": "full"}, context
    )

    assert result["code"] == "study_setup_incomplete"
    assert result["owner"] == "easyicu.webserver.pi_copilot.workflow"
    assert result["details"]["next_action_code"] == "study_setup_incomplete"
    assert result["details"]["planning_prerequisites_missing"] == ["data_source"]
    assert "provider_run" not in json.dumps(result)


def test_full_run_requires_server_receipted_eligibility_before_provider_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = {
        "id": "study-unconfirmed-eligibility",
        "revision": 3,
        "question": "What is the aggregate Sepsis-3 prevalence?",
        "data_source": {"path": "/private/raw/miiv", "database": "mimiciv"},
        # A legacy preset is a value, not proof that the researcher chose it.
        "cohort": {"preset": "adult_icu"},
    }
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: study)
    def reject(*_args: Any, **_kwargs: Any) -> Any:
        raise research_run_submission.ResearchRunSubmissionError(
            {
                "error": "cohort_eligibility_confirmation_required",
                "message": "Confirm one cohort eligibility option before generating the candidate plan.",
                "owner": "easyicu.webserver.pi_copilot.cohort_eligibility",
                "next_action_code": "cohort_eligibility_confirmation_required",
                "planning_prerequisites_missing": ["cohort_eligibility"],
                "missing_setup_fields": ["cohort_eligibility", "outcome", "modules"],
            }
        )

    monkeypatch.setattr(research_run_submission, "submit_research_run", reject)
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-unconfirmed-eligibility",
            external_llm_opt_in=True,
            binding=AuthorityBinding(
                study_context_id="study-unconfirmed-eligibility",
                study_revision=3,
            ),
        )
    )

    result = tool_module.execute_tool(
        "easyicu_run", {"run_type": "full"}, context
    )

    assert result["code"] == "cohort_eligibility_confirmation_required"
    assert result["owner"] == (
        "easyicu.webserver.pi_copilot.cohort_eligibility"
    )
    assert result["details"]["eligibility"]["selection_state"] == (
        "legacy_unconfirmed"
    )
    assert "provider_run" not in json.dumps(result)


def test_planner_owned_design_choices_do_not_block_plan_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plan proposes outcome/exposure details after eligibility is settled.

    Regression: the full-run gate required every setup slot, so Copilot had to
    interrogate the user for outcome, exposure, window and modules before a
    plan could be generated. Eligibility is the narrow pre-plan exception
    because the Planner is forbidden to invent the study denominator.
    """

    cohort, authority = _confirmed_cohort_decision(
        "no_eligibility_filter",
        study_context_id="study-planner-owned",
        study_context_revision=3,
    )
    monkeypatch.setattr(
        tool_module,
        "_bound_context",
        lambda _binding: {
            "id": "study-planner-owned",
            "revision": 3,
            "question": "What is the aggregate Sepsis-3 prevalence?",
            "data_source": {"path": "/private/raw/miiv", "database": "mimiciv"},
            "cohort": cohort,
            "cohort_eligibility_authority": authority,
        },
    )
    monkeypatch.setattr(
        tool_module,
        "_workflow_snapshot",
        lambda _context, *, study_override=None: {
            "next_action_code": "provider_ready_to_generate_plan",
            "planning_prerequisites_missing": [],
            "missing_setup_fields": [
                "outcome",
                "primary_exposure",
                "analysis_goal",
                "time_window",
                "export_format",
                "modules",
            ],
        },
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-planner-owned",
            external_llm_opt_in=True,
            binding=AuthorityBinding(
                study_context_id="study-planner-owned",
                study_revision=3,
            ),
        )
    )

    result = tool_module.execute_tool(
        "easyicu_run", {"run_type": "full"}, context
    )

    # It must get past the setup gate; the remaining stop is the ordinary
    # one-turn provider authorization, not a setup questionnaire.
    assert result["code"] != "study_setup_incomplete"


def test_full_run_uses_verified_pi_provider_and_prepared_source_authority(
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
    monkeypatch.setattr(
        research_run_submission,
        "submit_research_run",
        lambda request, **kwargs: _record_pipeline_submission(
            submitted, request, job_id="agent-job-full", **kwargs
        ),
    )
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
    request = submitted[0]["request"]
    assert request.study_context_id == "study-workflow"
    assert request.provider == "openai"
    assert request.credential_source == "pi_verified"
    assert request.planner_start_mode == "auto"
    assert submitted[0]["account_environment"] is None


def test_candidate_plan_approval_starts_package_bound_run_instead_of_resuming_canary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _complete_study()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: study)
    monkeypatch.setattr(
        tool_module,
        "_workflow_snapshot",
        lambda _context, *, study_override=None: {
            "next_action_code": "plan_execution_upgrade_required",
            "planning_prerequisites_missing": [],
        },
    )

    def replacement_run(context, params, *, planner_start_mode="auto"):
        captured.update(
            context=context,
            params=dict(params),
            planner_start_mode=planner_start_mode,
        )
        return {"status": "ok", "code": "package_bound_run_submitted"}

    monkeypatch.setattr(tool_module, "_run", replacement_run)
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-upgrade-canary",
            external_llm_opt_in=True,
            binding=AuthorityBinding(
                study_context_id=study["id"],
                study_revision=study["revision"],
                run_id="run-preview-only",
            ),
        ),
        allowed_actions={"provider_run", "configure", "extract"},
    )

    result = tool_module.execute_tool(
        "easyicu_resume",
        {"run_id": "run-preview-only", "decision": "approved"},
        context,
    )

    assert result["code"] == "package_bound_run_submitted"
    assert captured["params"] == {"run_type": "full"}
    assert captured["planner_start_mode"] == "fresh"


@pytest.mark.parametrize(
    "workflow_action_code",
    [
        "failed_pipeline_execution_retry_available",
        "failed_pipeline_requires_fresh_plan",
    ],
)
def test_failed_package_bound_run_retry_does_not_return_to_preview_canary(
    monkeypatch: pytest.MonkeyPatch,
    workflow_action_code: str,
) -> None:
    study = _complete_study()
    submitted: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: study)
    monkeypatch.setattr(
        tool_module,
        "_workflow_snapshot",
        lambda _context, *, study_override=None: {
            "next_action_code": workflow_action_code,
            "planning_prerequisites_missing": [],
        },
    )
    monkeypatch.setattr(
        research_run_submission,
        "submit_research_run",
        lambda request, **kwargs: _record_pipeline_submission(
            submitted, request, job_id="agent-job-package-retry", **kwargs
        ),
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-package-retry",
            external_llm_opt_in=True,
            binding=AuthorityBinding(
                study_context_id=study["id"],
                study_revision=study["revision"],
            ),
        ),
        allowed_actions={"provider_run"},
    )

    result = tool_module.execute_tool(
        "easyicu_run", {"run_type": "full"}, context
    )

    assert result["code"] == "easyicu_full_run_submitted"
    assert submitted[0]["request"].planner_start_mode == "auto"


def _run_submission_rejection(
    monkeypatch: pytest.MonkeyPatch, detail: dict[str, Any]
) -> dict[str, Any]:
    """Drive easyicu_run to one owner rejection and return Pi's receipt."""

    monkeypatch.setattr(
        tool_module,
        "_bound_context",
        lambda _binding: {
            **_complete_study(),
            "question": "Bound aggregate scientific question",
        },
    )

    def submit(*_args: Any, **_kwargs: Any) -> Any:
        raise research_run_submission.ResearchRunSubmissionError(detail)

    monkeypatch.setattr(research_run_submission, "submit_research_run", submit)
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-run-rejection",
            external_llm_opt_in=True,
            research_provider=ResearchProviderBinding(
                provider="openai",
                credential_source="pi_verified",
                model="gpt-5.6",
            ),
            binding=AuthorityBinding(
                study_context_id="study-workflow",
                study_revision=4,
            ),
        ),
        allowed_actions={"provider_run"},
    )
    return tool_module.execute_tool("easyicu_run", {}, context)


def test_unprepared_source_rejection_names_the_preparation_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing *step* must not be reported as an unusable data source.

    The raw reason code alone made Pi tell the user their MIMIC-IV folder had
    failed validation and stop, when the real state is simply that no export
    package has been prepared from it yet.
    """

    result = _run_submission_rejection(
        monkeypatch,
        {
            "error": "research_pipeline_manifest_required",
            "raw_file_types": [],
        },
    )

    assert result["status"] == "blocked"
    assert result["code"] == "research_pipeline_manifest_required"
    assert "easyicu_start_extraction" in result["summary"]
    assert "not a permission problem" in result["summary"]
    assert "re-pick" in result["summary"]


def test_other_run_rejections_carry_the_owning_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not replace a precise lower-layer diagnosis with a generic sentence."""

    result = _run_submission_rejection(
        monkeypatch,
        {
            "error": "research_pipeline_time_window_anchor_required",
            "message": "The typed study time window requires an explicit "
            "scientific anchor.",
        },
    )

    assert result["code"] == "research_pipeline_time_window_anchor_required"
    assert result["summary"] == (
        "The typed study time window requires an explicit scientific anchor."
    )


def test_run_rejection_without_a_message_keeps_the_boundary_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_submission_rejection(
        monkeypatch, {"error": "research_pipeline_budget_mode_invalid"}
    )

    assert result["code"] == "research_pipeline_budget_mode_invalid"
    assert result["summary"] == (
        "The existing EasyICU run submission boundary rejected the request."
    )


def test_full_run_uses_the_codex_account_frozen_into_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver import codex_account_sessions

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        tool_module,
        "_bound_context",
        lambda _binding: {
            **_complete_study(),
            "question": "Bound aggregate scientific question",
        },
    )

    def account_environment(binding: str, *, model: str) -> dict[str, str]:
        assert binding == "a" * 64
        assert model == "gpt-5.6-luna"
        return {
            "EASYICU_CODEX_SESSION_SHA256": binding,
            "EASYICU_CODEX_MODEL": model,
        }

    def submit(request: Any, **kwargs: Any) -> Any:
        kwargs["authorize"]()
        captured["request"] = request
        account_environment = kwargs.get("account_environment")
        captured["environment"] = dict(account_environment or {})
        return research_run_submission.ResearchRunSubmissionReceipt(
            job_id="agent-job-codex",
            kind="agent-run",
            status="running",
            study_context_id=request.study_context_id,
            study_context_revision=5,
            budget_mode="full_reviewed",
            planner_start_mode=request.planner_start_mode,
        )

    monkeypatch.setattr(
        codex_account_sessions,
        "environment_for_binding",
        account_environment,
    )
    monkeypatch.setattr(research_run_submission, "submit_research_run", submit)
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-codex-owner",
            external_llm_opt_in=True,
            research_provider=ResearchProviderBinding(
                provider="codex",
                credential_source="codex_user_auth",
                authentication_mode="chatgpt_account",
                model="gpt-5.6-luna",
                account_session_sha256="a" * 64,
            ),
            binding=AuthorityBinding(
                study_context_id="study-workflow",
                study_revision=4,
            ),
        ),
        allowed_actions={"provider_run"},
    )

    result = tool_module.execute_tool("easyicu_run", {}, context)

    assert result["code"] == "easyicu_full_run_submitted"
    assert captured["request"].provider == "codex"
    assert captured["request"].credential_source == "codex_user_auth"
    assert captured["environment"]["EASYICU_CODEX_MODEL"] == "gpt-5.6-luna"


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
        materialized_columns=("heart_rate", "mortality"),
        coverage=SimpleNamespace(sufficient=True),
        analysis_columns={
            "heart_rate": "heart_rate",
            "death": "mortality",
        },
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


def _write_development_resume_acquisition(
    run_dir: Path,
    *,
    feature_concepts: tuple[str, ...] = (),
) -> None:
    pipeline_input = run_dir.parent.parent / "pipeline_input"
    pipeline_input.mkdir(parents=True, exist_ok=True)
    universe = pipeline_input / "web_research_universe.parquet"
    universe.write_bytes(b"typed-development-resume-universe")
    (pipeline_input / "web_research_universe_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu.cohort_materializer/1",
                "database": "miiv",
                "cohort_window_hours": [0.0, 24.0],
                "feature_concepts": list(feature_concepts),
                "outcome_concepts": ["death"],
                "static_concepts": ["age", "sex"],
                "cohort_file_sha256": hashlib.sha256(
                    universe.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def _write_development_resume_literature(
    run_dir: Path,
    *,
    research_question: str,
) -> None:
    (run_dir / "preplan_literature_bundle.json").write_text(
        json.dumps(
            {
                "research_question": research_question,
                "citations": [],
                "screening_decisions": [],
                "design_evidence_cards": [],
            }
        ),
        encoding="utf-8",
    )


def _write_development_resume_planner_catalog(
    run_dir: Path,
    *,
    selected_concepts: tuple[str, ...] = ("lact", "death"),
    patient_identity_column: str | None = None,
    operationalized_columns: tuple[str, ...] = (),
) -> None:
    pipeline_input = run_dir.parent.parent / "pipeline_input"
    pipeline_input.mkdir(parents=True, exist_ok=True)
    universe = pipeline_input / "planner_catalog.parquet"
    columns: dict[str, pd.Series] = {"stay_id": pd.Series(dtype="int64")}
    if patient_identity_column:
        columns[patient_identity_column] = pd.Series(dtype="string")
    columns.update(
        {
            column: pd.Series(dtype="float64")
            for column in operationalized_columns
        }
    )
    columns.update(
        {
            concept: pd.Series(dtype="float64")
            for concept in selected_concepts
        }
    )
    frame = pd.DataFrame(columns)
    replacement_row_identity = (
        {
            "output_identity_column": patient_identity_column,
            "mapping_file_sha256": "a" * 64,
            "mapped_cohort_rows": 0,
            "patient_group_derivation": {
                "algorithm": "prefix_before_:s",
                "delimiter": ":s",
            },
            "authority_coordinates": {
                "schema_version": "easyicu.patient_grouping_runtime_authority/1",
                "authority_ref": "test/identity-bridge/v1",
                "database": "miiv",
                "mapping_sha256": "a" * 64,
                "grouping_derivation": "prefix_before_:s",
                "provider_visible_values": False,
            },
        }
        if patient_identity_column
        else None
    )
    frame.attrs["easyicu_planning_authority"] = {
        "kind": "metadata_only_planning_catalog",
        "patient_rows_read": False,
        **(
            {"replacement_row_identity": replacement_row_identity}
            if replacement_row_identity is not None
            else {}
        ),
    }
    frame.to_parquet(universe, index=False)
    selected_sha256 = hashlib.sha256(
        json.dumps(
            list(selected_concepts),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    (pipeline_input / "planner_catalog_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu.metadata-only-planning-catalog/1",
                "database": "miiv",
                "catalog_source": "easyicu-database-capability:miiv",
                "row_identity_column": "stay_id",
                "patient_identity_column": patient_identity_column,
                "operationalized_columns": list(operationalized_columns),
                "replacement_row_identity": replacement_row_identity,
                "selected_concepts": list(selected_concepts),
                "selected_concepts_sha256": selected_sha256,
                "patient_rows_read": False,
                "patient_rows_written": False,
                "observed_feasibility_claims": False,
                "execution_authorized": False,
                "planning_target_outcome": "death",
                "planning_endpoint": {
                    "name": "death",
                    "kind": "binary",
                    "absence_semantics": "no_absent_rows",
                    "levels": [0, 1],
                    "event_column": None,
                    "time_column": None,
                    "time_origin": None,
                    "censoring_rule": None,
                },
            }
        ),
        encoding="utf-8",
    )


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


def test_pipeline_projection_does_not_build_engineering_report_for_manuscript_run(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "manuscript-run"
    _write_real_pipeline_fixture(
        run_dir,
        manuscript="# Results\nThe evidence-bound manuscript is ready for review.",
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["readiness"]["manuscript_ready"] = True
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    wrapper = tmp_path / "web-projection"

    agent_pipeline_runs._write_projection(
        wrapper_dir=wrapper,
        study=_complete_study(),
        provider={"provider": "openai", "model": "test-model"},
        acquisition=_acquisition_receipt(),
        run_dir=run_dir,
    )

    source_manifest = json.loads(
        (wrapper / "source_run_manifest.json").read_text(encoding="utf-8")
    )
    assert source_manifest["readiness"]["manuscript_ready"] is True
    assert source_manifest["system_validation_report_available"] is False
    assert source_manifest["system_validation_document_count"] == 0
    assert not (wrapper / "system_validation_report.json").exists()
    assert not (wrapper / "system_validation_report.html").exists()
    assert (wrapper / "manuscript_draft.json").is_file()


def test_pending_plan_without_current_review_projects_stale_policy_reason(
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
        "scientific_plan_review_policy_stale"
    )
    assert manifest["plan_approval_allowed"] is False
    history = agent_runs.list_run_history(
        study_id="study-workflow",
        project_root=str(project_root),
    )
    assert history["runs"][0]["run_status"] == "human_review_pending"
    assert history["runs"][0]["pending_review_reason_codes"] == [
        "scientific_plan_review_policy_stale"
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
    _install_pending_review(monkeypatch, entry)
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
    _install_pending_review(monkeypatch, entry)
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
    _install_pending_review(
        monkeypatch,
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
    assert projected["plan_approval_allowed"] is False
    assert projected["requests"][0]["reason_code"] == (
        "scientific_plan_review_policy_stale"
    )


def test_pending_plan_resume_routes_pipeline_events_to_the_resume_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_current_scientific_review(monkeypatch)
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
    _install_pending_review(monkeypatch, entry)
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
    _allow_current_scientific_review(monkeypatch)
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
    registry = _install_pending_review(monkeypatch, entry)

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
    assert registry.get(pending.run_id) is entry
    task_row = hard_stop.ledger.snapshot()["tasks"][0]
    assert task_row["status"] == "paused"
    assert hard_stop.ledger.snapshot()["terminal"] is False

    diagnostic = json.loads(
        (entry.wrapper_dir / exc.value.details["diagnostic"]).read_text(
            encoding="utf-8"
        )
    )
    assert diagnostic["exception_type"] == "builtins.ValueError"
    assert diagnostic["review_resumable"] is True
    assert diagnostic["raw_exception_recorded"] is False
    assert "review decision evidence could not be persisted" not in json.dumps(
        diagnostic
    )


def test_review_resume_claim_is_exclusive_before_touching_provider_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_current_scientific_review(monkeypatch)
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
    registry = _install_pending_review(monkeypatch, entry)

    monkeypatch.setattr(registry, "lease", lambda *_args, **_kwargs: False)
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


def test_guided_project_rail_projects_real_mode_from_authoritative_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver import guided_sessions
    from easyicu.webserver.pi_copilot import project_authority

    rows = [
        {
            "id": "draft-context-source",
            "title": "Bound source project",
            "data_mode": "unbound",
            "surface_visibility": "product",
            "updated_at": "2026-08-25T10:00:00Z",
        }
    ]
    monkeypatch.setattr(guided_sessions, "_read_raw", lambda: {"drafts": rows})
    monkeypatch.setattr(guided_sessions, "read_project_study_setup", lambda _id: None)
    monkeypatch.setattr(
        project_authority,
        "ProjectAuthorityStore",
        lambda: SimpleNamespace(resolve=lambda _project_id: "study-context-source"),
    )
    context_batch_calls: list[list[str]] = []
    monkeypatch.setattr(
        study_context_owner,
        "get_contexts",
        lambda study_ids: context_batch_calls.append(study_ids)
        or {
            study_id: {
                "id": study_id,
                "data_source": {"database": "miiv", "label": "MIMIC-IV"},
            }
            for study_id in study_ids
        },
    )
    monkeypatch.setattr(agent_runs, "list_run_history", lambda **_kwargs: {"runs": []})

    payload = guided_sessions.list_guided_drafts(limit=20)

    assert payload["drafts"][0]["data_mode"] == "real"
    assert payload["drafts"][0]["workflow_status"] == "configured"
    assert rows[0]["data_mode"] == "unbound", "projection mutated registry metadata"
    assert context_batch_calls == [["study-context-source"]]


def test_guided_project_rail_projects_only_the_visible_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver import guided_sessions
    from easyicu.webserver.pi_copilot import project_authority

    rows = [
        {
            "id": f"draft-{index}",
            "title": f"Project {index}",
            "surface_visibility": "product",
            "updated_at": f"2026-08-25T10:0{index}:00Z",
        }
        for index in range(5)
    ]
    setup_calls: list[str] = []
    resolve_calls: list[str] = []
    monkeypatch.setattr(guided_sessions, "_read_raw", lambda: {"drafts": rows})
    monkeypatch.setattr(
        guided_sessions,
        "read_project_study_setup",
        lambda project_id: setup_calls.append(project_id) or None,
    )
    monkeypatch.setattr(
        project_authority,
        "ProjectAuthorityStore",
        lambda: SimpleNamespace(
            resolve=lambda project_id: resolve_calls.append(project_id) or None
        ),
    )

    payload = guided_sessions.list_guided_drafts(limit=2)

    assert payload["count"] == 5
    assert [row["id"] for row in payload["drafts"]] == ["draft-4", "draft-3"]
    assert setup_calls == ["draft-4", "draft-3"]
    assert resolve_calls == ["draft-4", "draft-3"]


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
    _assume_execution_runtime_ready(monkeypatch)
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
        research_pipeline_run_preparation,
        "_data_foundation_profile",
        lambda **_kwargs: _foundation_profile(),
    )

    class FakePipeline:
        def run(self, **_kwargs: Any) -> SimpleNamespace:
            exc = TimeoutError("provider request timed out")
            note = (
                "structured-retry history: validator rejected /Users/example/run "
                "api_key=test-secret-value"
            )
            add_note = getattr(exc, "add_note", None)
            if add_note is not None:
                add_note(note)
            else:
                exc.__notes__ = [note]
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
        budget_mode="full_reviewed",
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
    history = agent_runs.list_run_history(
        study_id="study-workflow",
        project_root=str(project_root),
    )
    assert history["count"] == 1
    assert history["runs"][0]["run_id"] == "run_job-timeout"
    assert history["runs"][0]["run_status"] == "failed"
    assert history["runs"][0]["gate_status"] == "blocked"
    review = agent_runs.read_run_review(history["runs"][0]["project_dir"])
    assert review["readiness"]["status"] == "blocked"
    assert review["gate"]["reason"] == "research_pipeline_provider_timeout"


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
    assert payload["schema_version"] == "easyicu.web-research-pipeline-failure/4"
    assert payload["typed_failure"] == {}
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
    assert payload["exception_types"] == ["StructuredResponseFailure"]
    assert all(
        set(frame) == {"file", "function", "line"}
        for frame in payload["traceback_frames"]
    )


def test_pipeline_failure_type_is_a_closed_nonsecret_category(tmp_path: Path) -> None:
    secret_error_type = type("sk_secret_shaped_failure_type", (RuntimeError,), {})

    relative = agent_pipeline_runs._write_pipeline_failure_diagnostic(
        wrapper_dir=tmp_path,
        exc=secret_error_type("ordinary failure"),
        code="research_pipeline_execution_failed",
    )

    payload = json.loads((tmp_path / relative).read_text(encoding="utf-8"))
    assert payload["failure_type"] == "error"
    assert payload["typed_failure"] == {}
    assert "sk_secret" not in json.dumps(payload)


def test_pipeline_failure_projects_only_safe_progressive_compiler_coordinates(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.planning.progressive_contract import (
        ProgressivePlanCompileError,
    )

    secret = "sk-provider-secret-in-compiler-message"
    failure = ProgressivePlanCompileError(
        "progressive_unknown_variable",
        f"model candidate echoed {secret}",
        step_id="05_primary",
        step_index=4,
        path="model_terms[1]",
    )

    code = agent_pipeline_runs._pipeline_failure_code(failure)
    relative = agent_pipeline_runs._write_pipeline_failure_diagnostic(
        wrapper_dir=tmp_path,
        exc=failure,
        code=code,
    )

    payload = json.loads((tmp_path / relative).read_text(encoding="utf-8"))
    assert code == "research_pipeline_progressive_compile_failed"
    assert payload["typed_failure"] == {
        "owner": "easyicu.planning.progressive_compiler_v1",
        "reason_code": "progressive_unknown_variable",
        "step_id": "05_primary",
        "step_index": 4,
        "path": "model_terms[1]",
    }
    assert secret not in json.dumps(payload)


def test_pipeline_failure_projects_only_safe_pydantic_coordinates(
    tmp_path: Path,
) -> None:
    from pydantic import BaseModel, Field, ValidationError

    class _NestedContract(BaseModel):
        panels: list[int] = Field(min_length=2)

    secret = "sk-provider-secret-in-schema-input"
    try:
        _NestedContract.model_validate({"panels": [secret]})
    except ValidationError as failure:
        code = agent_pipeline_runs._pipeline_failure_code(failure)
        relative = agent_pipeline_runs._write_pipeline_failure_diagnostic(
            wrapper_dir=tmp_path,
            exc=failure,
            code=code,
        )
    else:  # pragma: no cover - the invalid fixture must stay invalid
        raise AssertionError("fixture unexpectedly passed schema validation")

    payload = json.loads((tmp_path / relative).read_text(encoding="utf-8"))
    assert code == "research_pipeline_schema_validation_failed"
    assert payload["typed_failure"] == {
        "owner": "easyicu.schema_validation_v1",
        "reason_code": "pydantic_contract_validation_failed",
        "error_count": 1,
        "coordinates": [
            {
                "location": ["panels", 0],
                "error_type": "int_parsing",
            }
        ],
    }
    assert secret not in json.dumps(payload)


def test_pipeline_failure_projects_closed_planner_efficiency_receipt(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.providers.efficiency_budget import (
        PlannerEfficiencyBudgetExhausted,
    )

    failure = PlannerEfficiencyBudgetExhausted(
        reason="reported_token_limit",
        snapshot={
            "calls": 5,
            "reported_tokens": 101_000,
            "elapsed_seconds": 123.4567891,
            "limits": {
                "max_calls": 6,
                "max_reported_tokens": 100_000,
                "max_wall_seconds": 600.0,
            },
        },
    )

    code = agent_pipeline_runs._pipeline_failure_code(failure)
    relative = agent_pipeline_runs._write_pipeline_failure_diagnostic(
        wrapper_dir=tmp_path,
        exc=failure,
        code=code,
    )

    payload = json.loads((tmp_path / relative).read_text(encoding="utf-8"))
    assert code == "research_pipeline_planner_efficiency_budget_exhausted"
    assert payload["failure_type"] == "provider_budget"
    assert payload["typed_failure"] == {
        "owner": "easyicu.providers.planner_efficiency_budget_v1",
        "reason_code": "planner_efficiency_budget_exhausted",
        "reason": "reported_token_limit",
        "calls": 5,
        "reported_tokens": 101_000,
        "elapsed_seconds": 123.456789,
        "limits": {
            "max_calls": 6,
            "max_reported_tokens": 100_000,
            "max_wall_seconds": 600.0,
        },
    }


def test_pipeline_failure_projects_codex_hard_timeout_as_provider_timeout(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.providers.codex_app_server import (
        CodexAppServerError,
    )

    failure = CodexAppServerError(
        "codex_auth_notification_hard_timeout",
        "provider detail that must not be projected",
    )

    code = agent_pipeline_runs._pipeline_failure_code(failure)
    relative = agent_pipeline_runs._write_pipeline_failure_diagnostic(
        wrapper_dir=tmp_path,
        exc=failure,
        code=code,
    )

    payload = json.loads((tmp_path / relative).read_text(encoding="utf-8"))
    assert code == "research_pipeline_provider_timeout"
    assert payload["failure_type"] == "timeout"
    assert payload["typed_failure"] == {
        "owner": "easyicu.providers.codex_app_server_v1",
        "reason_code": "codex_auth_notification_hard_timeout",
    }
    assert "provider detail" not in json.dumps(payload)


def test_plan_approval_requires_fresh_provider_grant_and_forwards_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted: list[dict[str, Any]] = []
    account_environments: list[dict[str, str] | None] = []
    monkeypatch.setattr(
        tool_module,
        "_bound_context",
        lambda binding: _complete_study(),
    )
    from easyicu.webserver.routes import agent as agent_route
    from easyicu.webserver import codex_account_sessions

    def submit(
        body: dict[str, Any],
        *,
        account_environment: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        submitted.append(dict(body))
        account_environments.append(account_environment)
        return {
            "job_id": "resume-job-1",
            "kind": "agent-run",
            "status": "running",
            "engine": "research_agent_pipeline",
            "review_run_id": "pipeline-run-1",
            "study_context_id": "study-workflow",
            "study_context_revision": 5,
        }

    monkeypatch.setattr(agent_route, "submit_agent_run_review", submit)
    monkeypatch.setattr(
        codex_account_sessions,
        "environment_for_binding",
        lambda binding, *, model: {
            "CODEX_HOME": "/isolated/account",
            "EASYICU_CODEX_MODEL": model,
            "EASYICU_CODEX_ACCOUNT_SESSION_SHA256": binding,
        },
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-review",
            external_llm_opt_in=True,
            research_provider=ResearchProviderBinding(
                provider="codex",
                credential_source="codex_user_auth",
                authentication_mode="chatgpt_account",
                model="gpt-5.6-luna",
                account_session_sha256="a" * 64,
            ),
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
    assert account_environments == [
        {
            "CODEX_HOME": "/isolated/account",
            "EASYICU_CODEX_MODEL": "gpt-5.6-luna",
            "EASYICU_CODEX_ACCOUNT_SESSION_SHA256": "a" * 64,
        }
    ]
    with pytest.raises(PiCopilotError) as stale:
        context.assert_authority_fresh()
    assert stale.value.code == "pi_session_authority_stale"


@pytest.mark.parametrize(
    (
        "budget_mode",
        "runner_image",
        "runner_image_environment",
        "expected_runner_image",
        "expected_profile_name",
        "expected_profile_version",
    ),
    [
        (
            "full_reviewed",
            None,
            None,
            "easyicu-research-agent:1.0.0",
            "npj_dm_e1_demo_dev",
            "20260819",
        ),
        (
            "full_reviewed",
            None,
            "easyicu-research-agent:isolated-exact-head",
            "easyicu-research-agent:isolated-exact-head",
            "npj_dm_e1_demo_dev",
            "20260819",
        ),
        (
            "full_reviewed",
            "easyicu-research-agent:e1-demo-local",
            " \n",
            "easyicu-research-agent:e1-demo-local",
            "npj_dm_e1_demo_dev",
            "20260819",
        ),
    ],
)
def test_web_runner_delegates_to_research_agent_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget_mode: str | None,
    runner_image: str | None,
    runner_image_environment: str | None,
    expected_runner_image: str,
    expected_profile_name: str,
    expected_profile_version: str,
) -> None:
    if runner_image_environment is not None:
        monkeypatch.setenv("EASYICU_RUNNER_IMAGE", runner_image_environment)
    _assume_execution_runtime_ready(monkeypatch)
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

    def fake_build_provider(provider: dict[str, Any], **kwargs: Any):
        calls["provider_build"] = kwargs
        return (
            object(),
            {"provider": "openai", "model": "test-model"},
        )

    monkeypatch.setattr(
        provider_adapter,
        "build_research_agent_provider_client",
        fake_build_provider,
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
        research_pipeline_run_preparation,
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
    expected_resume_path = None
    if budget_mode is None and runner_image_environment is None:
        expected_resume_path = (
            tmp_path
            / "projects"
            / "study-workflow"
            / "run_prior-canary"
            / "pipeline"
            / "run_prior"
            / "progressive_planner_checkpoint_000.json"
        )
        expected_resume_path.parent.mkdir(parents=True)
        expected_resume_path.write_text("{}", encoding="utf-8")
        _write_development_resume_acquisition(expected_resume_path.parent)
        _write_development_resume_literature(
            expected_resume_path.parent,
            research_question=_complete_study()["question"],
        )
        monkeypatch.setattr(
            research_launch_resume,
            "load_progressive_planner_checkpoint_chain",
            lambda **_kwargs: [object()],
        )
        monkeypatch.setenv(
            "EASYICU_DEVELOPMENT_PROGRESSIVE_RESUME_SOURCE_JOB_ID",
            "prior-canary",
        )
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
    expected_provider_timeout = 120.0 if budget_mode != "full_reviewed" else None
    assert calls["provider_build"]["request_timeout"] == expected_provider_timeout
    assert (
        calls["provider_build"]["request_hard_timeout"]
        == expected_provider_timeout
    )
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
    assert calls["acquire"]["concept_selection_authority"] == "host_exact"
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
    assert calls["config"].enable_replanning is False
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
    assert calls["config"].runner_image == expected_runner_image
    assert calls["config"].submission_profile_name == expected_profile_name
    assert calls["config"].submission_profile_version == expected_profile_version
    assert calls["config"].planner_strategy == (
        "progressive_v2"
        if budget_mode in {None, "full_reviewed"}
        else "monolithic_v1"
    )
    assert calls["config"].development_progressive_resume_checkpoint_path == (
        expected_resume_path
    )
    assert calls[
        "config"
    ].development_progressive_resume_checkpoint_sha256 == (
        hashlib.sha256(b"{}").hexdigest() if expected_resume_path else None
    )
    assert (
        calls["config"].development_progressive_resume_reuse_bound_literature
        is bool(expected_resume_path)
    )
    # User-facing Web planning relies on the provider hard stop above. The
    # smaller routine-E1 iteration envelope must not interrupt a valid
    # progressive plan after an arbitrary number of calls.
    assert calls["config"].development_planner_efficiency_max_calls is None
    assert (
        calls["config"].development_planner_efficiency_max_reported_tokens is None
    )
    assert calls["config"].development_planner_efficiency_max_wall_seconds is None
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


def test_web_runner_rejects_invalid_server_owned_image_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EASYICU_RUNNER_IMAGE", " \n")
    monkeypatch.setattr(
        research_pipeline_run_preparation,
        "_data_foundation_profile",
        lambda **_kwargs: _foundation_profile(),
    )
    export_path = _write_pipeline_export(tmp_path / "export")

    with pytest.raises(
        agent_pipeline_runs.ResearchPipelineRunError
    ) as exc_info:
        agent_pipeline_runs.make_research_pipeline_run_runner(
            export_path=str(export_path),
            study_context=_complete_study(),
            project_root=str(tmp_path / "projects"),
            provider={"provider": "openai", "external": True},
            provider_environment=_PI_PROVIDER_ENVIRONMENT,
        )

    assert exc_info.value.code == "research_pipeline_runner_image_invalid"


def test_web_runner_allows_server_owned_resume_for_full_reviewed_development(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        research_pipeline_run_preparation,
        "_data_foundation_profile",
        lambda **_kwargs: _foundation_profile(),
    )
    export_path = _write_pipeline_export(tmp_path / "export")
    checkpoint = (
        tmp_path
        / "projects"
        / "study-workflow"
        / "run_prior-canary"
        / "pipeline"
        / "run_prior"
        / "progressive_planner_checkpoint_000.json"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}", encoding="utf-8")
    _write_development_resume_acquisition(checkpoint.parent)
    # A resume replays the exact literature authority hashed into the
    # checkpoint rather than repeating a live search, so the bundle has to be
    # staged beside it like every other resumable receipt.
    _write_development_resume_literature(
        checkpoint.parent,
        research_question=_complete_study()["question"],
    )
    monkeypatch.setattr(
        research_launch_resume,
        "load_progressive_planner_checkpoint_chain",
        lambda **_kwargs: [object()],
    )
    _assume_execution_runtime_ready(monkeypatch)

    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(export_path),
        study_context=_complete_study(),
        project_root=str(tmp_path / "projects"),
        provider={"provider": "openai", "external": True},
        provider_environment=_PI_PROVIDER_ENVIRONMENT,
        development_resume_source_job_id="prior-canary",
        budget_mode="full_reviewed",
    )

    assert callable(runner)


def test_development_resume_selects_one_server_owned_checkpoint_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = (
        tmp_path
        / "projects"
        / "study-workflow"
        / "run_prior-canary"
        / "pipeline"
        / "run_prior"
    )
    run_dir.mkdir(parents=True)
    for sequence in range(3):
        (run_dir / f"progressive_planner_checkpoint_{sequence:03d}.json").write_text(
            f'{{"sequence":{sequence}}}',
            encoding="utf-8",
        )
    monkeypatch.setattr(
        research_launch_resume,
        "load_progressive_planner_checkpoint_chain",
        lambda **_kwargs: [object()],
    )

    path, digest = research_launch_resume._development_progressive_resume_binding(
        project_root=str(tmp_path / "projects"),
        study_id="study-workflow",
        source_job_id="prior-canary",
        budget_mode="planner_canary",
        checkpoint_sequence="1",
    )

    assert path.name == "progressive_planner_checkpoint_001.json"
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_development_resume_rejects_missing_server_checkpoint_sequence(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / "projects"
        / "study-workflow"
        / "run_prior-canary"
        / "pipeline"
        / "run_prior"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "progressive_planner_checkpoint_000.json").write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc_info:
        research_launch_resume._development_progressive_resume_binding(
            project_root=str(tmp_path / "projects"),
            study_id="study-workflow",
            source_job_id="prior-canary",
            budget_mode="planner_canary",
            checkpoint_sequence="2",
        )

    assert exc_info.value.code == (
        "research_pipeline_development_resume_sequence_missing"
    )


def test_development_resume_restores_exact_typed_acquisition_roster(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_prior" / "pipeline" / "run_pipeline"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "progressive_planner_checkpoint_004.json"
    checkpoint.write_text("{}", encoding="utf-8")
    _write_development_resume_acquisition(
        run_dir,
        feature_concepts=("admission_type", "sepsis3"),
    )

    profile = research_launch_resume._development_resume_acquisition_profile(
        checkpoint_path=checkpoint,
        database="miiv",
        cohort_window=(0.0, 24.0),
        outcome_concepts=("death",),
        static_concepts=("age", "sex"),
        required_feature_concepts=("sepsis3",),
    )

    assert profile.kind == "materialized_patient_universe"
    assert profile.feature_concepts == ("admission_type", "sepsis3")
    assert profile.outcome_concepts == ("death",)
    assert profile.static_concepts == ("age", "sex")


def test_development_resume_restores_exact_literature_authority(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_prior" / "pipeline" / "run_pipeline"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "progressive_planner_checkpoint_004.json"
    checkpoint.write_text("{}", encoding="utf-8")
    question = "Is lactate associated with hospital mortality?"
    _write_development_resume_literature(
        run_dir,
        research_question=question,
    )

    bundle = research_launch_resume._development_resume_literature_bundle(
        checkpoint_path=checkpoint
    )

    assert bundle["research_question"] == question
    assert bundle["citations"] == []


def test_development_resume_restores_metadata_only_planner_catalog_without_rows(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_prior" / "pipeline" / "run_pipeline"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "progressive_planner_checkpoint_004.json"
    checkpoint.write_text("{}", encoding="utf-8")
    _write_development_resume_planner_catalog(
        run_dir,
        patient_identity_column="patient_stay_id",
        operationalized_columns=("lact_max",),
    )
    patient_grouping = PatientGroupingBinding(
        mapping_path=tmp_path / "private-patient-map.parquet",
        mapping_sha256="a" * 64,
        mapping_stay_column="stay_id",
        mapping_patient_column="patient_key",
        authority_coordinates={
            "schema_version": "easyicu.patient_grouping_runtime_authority/1",
            "authority_ref": "test/identity-bridge/v1",
            "database": "miiv",
            "mapping_sha256": "a" * 64,
            "grouping_derivation": "prefix_before_:s",
            "provider_visible_values": False,
        },
    )
    coordinates = research_launch_scientific._metadata_only_planning_coordinates(
        database="miiv",
        question="我想研究 ICU 患者的乳酸水平和院内死亡有没有关系。",
    )

    profile = research_launch_resume._development_resume_acquisition_profile(
        checkpoint_path=checkpoint,
        database="miiv",
        cohort_window=(0.0, 24.0),
        outcome_concepts=(),
        static_concepts=(),
        required_feature_concepts=(),
        planning_target_outcome=coordinates["target_outcome"],
        planning_endpoint=coordinates["endpoint"],
        planning_operationalized_columns=("lact_max",),
    )
    acquisition = agent_pipeline_runs._restore_metadata_only_planning_acquisition(
        database="miiv",
        profile=profile,
        output_dir=tmp_path / "run_current" / "pipeline_input",
        endpoint=coordinates["endpoint"],
        patient_grouping=patient_grouping,
        operationalized_columns=("lact_max",),
    )

    assert profile.kind == "metadata_only_planning_catalog"
    assert profile.selected_concepts == ("lact", "death")
    assert acquisition.selection.selection_authority == "host_exact"
    assert acquisition.selection.selected_concepts == ["lact", "death"]
    restored = pd.read_parquet(acquisition.universe_path)
    assert restored.empty
    assert list(restored.columns) == [
        "stay_id",
        "patient_stay_id",
        "lact_max",
        "lact",
        "death",
    ]
    assert acquisition.universe_path == (
        tmp_path / "run_current" / "pipeline_input" / "planner_catalog.parquet"
    )
    assert acquisition.provenance_path == (
        tmp_path
        / "run_current"
        / "pipeline_input"
        / "planner_catalog_receipt.json"
    )
    assert acquisition.note.startswith("Verified metadata-only Planner replay")

    chained_run_dir = (
        tmp_path / "run_current" / "pipeline" / "run_pipeline_continuation"
    )
    chained_run_dir.mkdir(parents=True)
    chained_checkpoint = (
        chained_run_dir / "progressive_planner_checkpoint_005.json"
    )
    chained_checkpoint.write_text("{}", encoding="utf-8")
    chained_profile = research_launch_resume._development_resume_acquisition_profile(
        checkpoint_path=chained_checkpoint,
        database="miiv",
        cohort_window=(0.0, 24.0),
        outcome_concepts=(),
        static_concepts=(),
        required_feature_concepts=(),
        planning_target_outcome=coordinates["target_outcome"],
        planning_endpoint=coordinates["endpoint"],
        planning_operationalized_columns=("lact_max",),
    )
    assert chained_profile.selected_concepts == ("lact", "death")


def test_development_resume_rejects_metadata_catalog_with_patient_rows(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_prior" / "pipeline" / "run_pipeline"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "progressive_planner_checkpoint_004.json"
    checkpoint.write_text("{}", encoding="utf-8")
    _write_development_resume_planner_catalog(run_dir)
    universe = run_dir.parent.parent / "pipeline_input" / "planner_catalog.parquet"
    pd.DataFrame({"stay_id": [1], "lact": [2.0], "death": [0]}).to_parquet(
        universe,
        index=False,
    )
    coordinates = research_launch_scientific._metadata_only_planning_coordinates(
        database="miiv",
        question="我想研究 ICU 患者的乳酸水平和院内死亡有没有关系。",
    )

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc_info:
        research_launch_resume._development_resume_acquisition_profile(
            checkpoint_path=checkpoint,
            database="miiv",
            cohort_window=(0.0, 24.0),
            outcome_concepts=(),
            static_concepts=(),
            required_feature_concepts=(),
            planning_target_outcome=coordinates["target_outcome"],
            planning_endpoint=coordinates["endpoint"],
        )

    assert exc_info.value.code == (
        "research_pipeline_development_resume_acquisition_invalid"
    )


def test_development_resume_recovers_legacy_chained_metadata_catalog(
    tmp_path: Path,
) -> None:
    study_root = tmp_path / "projects" / "study-workflow"
    source_run = study_root / "run_source" / "pipeline" / "run_pipeline_source"
    source_run.mkdir(parents=True)
    source_checkpoint = source_run / "progressive_planner_checkpoint_004.json"
    source_checkpoint.write_text('{"validated":"shared"}', encoding="utf-8")
    _write_development_resume_planner_catalog(source_run)

    legacy_run = study_root / "run_legacy" / "pipeline" / "run_pipeline_legacy"
    legacy_run.mkdir(parents=True)
    legacy_checkpoint = legacy_run / "progressive_planner_checkpoint_004.json"
    legacy_checkpoint.write_bytes(source_checkpoint.read_bytes())
    coordinates = research_launch_scientific._metadata_only_planning_coordinates(
        database="miiv",
        question="我想研究 ICU 患者的乳酸水平和院内死亡有没有关系。",
    )

    profile = research_launch_resume._development_resume_acquisition_profile(
        checkpoint_path=legacy_checkpoint,
        database="miiv",
        cohort_window=(0.0, 24.0),
        outcome_concepts=(),
        static_concepts=(),
        required_feature_concepts=(),
        planning_target_outcome=coordinates["target_outcome"],
        planning_endpoint=coordinates["endpoint"],
    )

    assert profile.kind == "metadata_only_planning_catalog"
    assert profile.selected_concepts == ("lact", "death")
    assert profile.universe_path == (
        study_root / "run_source" / "pipeline_input" / "planner_catalog.parquet"
    )


def test_development_resume_rejects_changed_acquisition_bytes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_prior" / "pipeline" / "run_pipeline"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "progressive_planner_checkpoint_004.json"
    checkpoint.write_text("{}", encoding="utf-8")
    _write_development_resume_acquisition(run_dir)
    universe = (
        run_dir.parent.parent
        / "pipeline_input"
        / "web_research_universe.parquet"
    )
    universe.write_bytes(b"changed-after-receipt")

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc_info:
        research_launch_resume._development_resume_acquisition_profile(
            checkpoint_path=checkpoint,
            database="miiv",
            cohort_window=(0.0, 24.0),
            outcome_concepts=("death",),
            static_concepts=("age", "sex"),
            required_feature_concepts=(),
        )

    assert exc_info.value.code == (
        "research_pipeline_development_resume_acquisition_digest_mismatch"
    )


def test_web_runner_enables_live_pubmed_only_with_host_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assume_execution_runtime_ready(monkeypatch)
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
        research_pipeline_run_preparation,
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
        budget_mode="full_reviewed",
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
    _assume_execution_runtime_ready(monkeypatch)
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
        research_pipeline_run_preparation,
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
        budget_mode="full_reviewed",
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
    monkeypatch.setenv("EASYICU_DEVELOPMENT_REVIEWED_EXECUTION", "1")
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
            },
            request=_request(),
        )

    assert getattr(raised.value, "detail")["error"] in {
        "research_pipeline_manifest_required",
        "no_export_files",
    }
    assert provider_called is False


@pytest.mark.parametrize(
    ("planner_start_mode", "resume_source_job_id"),
    [
        ("fresh", ""),
        ("resume_checkpoint", "prior-canary"),
    ],
)
def test_pipeline_route_ignores_client_project_root_and_uses_pi_workspace(
    planner_start_mode: str,
    resume_source_job_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.pi_copilot.workspace import ProjectWorkspace
    from easyicu.webserver import research_run_submission
    from easyicu.webserver.routes import agent as agent_route

    export = tmp_path / "raw-mimiciv"
    export.mkdir()
    (export / "patients.csv").write_text("stay_id\n1\n", encoding="utf-8")
    study = {
        **_complete_study(),
        "data_source": {"path": str(export), "database": "miiv"},
    }
    workspace = ProjectWorkspace(tmp_path / "pi-workspace")
    captured: dict[str, Any] = {}
    monkeypatch.setattr(agent_route.context_store, "get_context", lambda _id: study)
    monkeypatch.setattr(
        research_run_submission,
        "research_pipeline_workspace",
        lambda: workspace,
    )
    monkeypatch.setattr(
        research_run_submission,
        "resumable_planner_checkpoint_job_id",
        lambda **_kwargs: resume_source_job_id,
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
        research_run_submission,
        "_submit_job",
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

    payload = {
        "path": str(export),
        "study_context_id": study["id"],
        "engine": "research_agent_pipeline",
        "run_type": "full",
        "credential_source": "pi_verified",
        "external_llm_opt_in": True,
        "project_root": str(tmp_path / "client-controlled"),
        "planner_start_mode": planner_start_mode,
    }
    if resume_source_job_id:
        payload["development_resume_source_job_id"] = "client-forged-checkpoint"
    result = agent_route.jobs_agent_run(
        payload,
        request=_request(),
    )

    assert result["job_id"] == "job-workspace"
    assert Path(captured["project_root"]) == workspace.project_root(study["id"])
    assert Path(captured["project_root"]) != tmp_path / "client-controlled"
    assert captured["budget_mode"] == "planner_canary"
    assert result["planner_start_mode"] == planner_start_mode
    if resume_source_job_id:
        assert captured["development_resume_source_job_id"] == resume_source_job_id
        assert result["resume_source_job_id"] == resume_source_job_id
    else:
        assert "development_resume_source_job_id" not in captured


def test_monitor_history_merges_default_and_copilot_pipeline_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.routes import agent as agent_route

    pipeline_root = tmp_path / "pi-project"
    pipeline_root.mkdir()
    calls: list[str | None] = []

    class Workspace:
        def existing_project_root(self, project_id: str) -> Path:
            assert project_id == "study-workflow"
            return pipeline_root

    def history(*, study_id: str, project_root: str | None = None, limit: int) -> dict[str, Any]:
        calls.append(project_root)
        if project_root is None:
            rows = [
                {
                    "run_id": "run_preflight",
                    "project_dir": str(tmp_path / "default" / "run_preflight"),
                    "updated_at_epoch": 10,
                }
            ]
        else:
            rows = [
                {
                    "run_id": "run_pipeline",
                    "project_dir": str(pipeline_root / "study-workflow" / "run_pipeline"),
                    "updated_at_epoch": 20,
                }
            ]
        return {"ok": True, "project_root": project_root or "default", "runs": rows, "count": len(rows)}

    monkeypatch.setattr(agent_route, "research_pipeline_workspace", lambda: Workspace())
    monkeypatch.setattr(agent_route.agent_runs, "list_run_history", history)
    result = agent_route.post_agent_run_history(
        {"study_id": "study-workflow", "limit": 50}
    )

    assert calls == [None, str(pipeline_root)]
    assert result["count"] == 2
    assert [row["run_id"] for row in result["runs"]] == [
        "run_pipeline",
        "run_preflight",
    ]


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
            },
            request=_request(),
        )

    assert getattr(raised.value, "detail") == {
        "error": "research_pipeline_budget_mode_server_owned"
    }


def test_pipeline_development_execution_mode_is_server_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.routes import agent as agent_route

    monkeypatch.delenv("EASYICU_DEVELOPMENT_REVIEWED_EXECUTION", raising=False)
    assert agent_route._server_research_pipeline_budget_mode() == "planner_canary"

    monkeypatch.setenv("EASYICU_DEVELOPMENT_REVIEWED_EXECUTION", "1")
    assert agent_route._server_research_pipeline_budget_mode() == "full_reviewed"

    monkeypatch.setenv("EASYICU_DEVELOPMENT_REVIEWED_EXECUTION", "true")
    with pytest.raises(Exception) as raised:
        agent_route._server_research_pipeline_budget_mode()
    assert getattr(raised.value, "detail") == {
        "error": "research_pipeline_development_mode_invalid"
    }


def test_candidate_plan_click_stays_planner_only_with_or_without_prepared_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.routes import agent as agent_route

    monkeypatch.setenv("EASYICU_DEVELOPMENT_REVIEWED_EXECUTION", "1")
    assert agent_route._research_pipeline_budget_mode_for_source(
        prepared_manifest=None,
        metadata_only_planning_authorized=True,
    ) == "planner_canary"
    assert agent_route._research_pipeline_budget_mode_for_source(
        prepared_manifest=tmp_path / "manifest.json",
        metadata_only_planning_authorized=True,
    ) == "planner_canary"
    monkeypatch.delenv("EASYICU_DEVELOPMENT_REVIEWED_EXECUTION", raising=False)
    assert agent_route._research_pipeline_budget_mode_for_source(
        prepared_manifest=tmp_path / "manifest.json",
        metadata_only_planning_authorized=True,
    ) == "planner_canary"
    monkeypatch.setenv("EASYICU_DEVELOPMENT_REVIEWED_EXECUTION", "1")
    assert agent_route._research_pipeline_budget_mode_for_source(
        prepared_manifest=None,
        metadata_only_planning_authorized=False,
    ) == "full_reviewed"


def test_planner_only_plan_requests_package_bound_regeneration() -> None:
    study = _complete_study()
    digest = study_context_owner.scientific_configuration_sha256(study)
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_type": "full",
            "run_id": "run-preview-only",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "run_status": "human_review_pending",
            "pending_review_reason_codes": ["operator_plan_approval_required"],
            "artifact_names": ["agent_plan.json", "source_run_manifest.json"],
        },
        plan_review_authority={
            "run_id": "run-preview-only",
            "resumable_here": True,
            "scientific_configuration_sha256": digest,
            "budget_mode": "planner_canary",
        },
    )

    by_id = {row.id: row for row in snapshot.stages}
    assert snapshot.plan_execution_ready is False
    assert snapshot.next_action_code == "plan_execution_upgrade_required"
    assert by_id["plan"].reason_code == "plan_execution_upgrade_required"
    assert by_id["analysis"].reason_code == "plan_execution_upgrade_required"


def test_legacy_scientific_review_requests_fresh_plan_without_reextracting() -> None:
    study = _complete_study()
    digest = study_context_owner.scientific_configuration_sha256(study)
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_type": "full",
            "run_id": "run-stale-science-policy",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "run_status": "human_review_pending",
            "pending_review_reason_codes": [
                "scientific_plan_review_policy_stale"
            ],
            "artifact_names": ["agent_plan.json", "source_run_manifest.json"],
        },
        plan_review_authority={
            "run_id": "run-stale-science-policy",
            "resumable_here": True,
            "scientific_configuration_sha256": digest,
            "budget_mode": "full_reviewed",
            "plan_approval_allowed": False,
            "requests": [
                {
                    "reason_code": "scientific_plan_review_policy_stale",
                    "approval_allowed": False,
                }
            ],
        },
    )

    by_id = {row.id: row for row in snapshot.stages}
    assert snapshot.plan_execution_ready is False
    assert snapshot.next_action_code == "scientific_plan_review_policy_stale"
    assert by_id["plan"].reason_code == "scientific_plan_review_policy_stale"
    assert by_id["analysis"].reason_code == "scientific_plan_review_policy_stale"


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
            },
            request=_request(),
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

    _install_pending_review(
        monkeypatch,
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
    _assume_execution_runtime_ready(monkeypatch)
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
        request_timeout: float | None = None,
        request_hard_timeout: float | None = None,
        environ: dict[str, str] | None = None,
    ) -> tuple[object, dict[str, Any]]:
        captured["provider"] = dict(provider)
        captured["environment"] = dict(environ or {})
        captured["request_timeout"] = request_timeout
        captured["request_hard_timeout"] = request_hard_timeout
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
        research_pipeline_run_preparation,
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
        budget_mode="full_reviewed",
    )

    class Job:
        id = "job-provider-authority"
        cancel_requested = False
        events: list[dict[str, Any]] = []

        def emit(self, event: dict[str, Any]) -> None:
            self.events.append(dict(event))

    result = runner(Job())

    assert captured["environment"] == expected_environment
    assert captured["request_timeout"] is None
    assert captured["request_hard_timeout"] is None
    assert result["provider"]["model"] == "test-local-model"
    assert "test-private-provider-key" not in json.dumps(result)


def test_pipeline_bridge_rejects_direct_scientific_provider_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = _write_pipeline_export(tmp_path / "export")
    monkeypatch.setattr(
        research_pipeline_run_preparation,
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
    _assume_execution_runtime_ready(monkeypatch)
    export = _write_pipeline_export(tmp_path / "export")
    study = {
        **_complete_study(),
        "data_source": {"path": str(export), "database": "miiv"},
    }
    monkeypatch.setattr(
        research_pipeline_run_preparation,
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
        budget_mode="full_reviewed",
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
    _allow_current_scientific_review(monkeypatch)
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
    _install_pending_review(monkeypatch, entry)
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

    profile = research_launch_scientific._data_foundation_profile(
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

    profile = research_launch_scientific._data_foundation_profile(
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

    profile = research_launch_scientific._data_foundation_profile(
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

    profile = research_launch_scientific._data_foundation_profile(
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

    profile = research_launch_scientific._data_foundation_profile(
        export_path="/typed/demo",
        study={"modules": ["demographics", "outcome"]},
        target="death",
        covariates=(),
        sensitivity_specs=specs,
    )

    assert profile["static_concepts"] == ("age", "icu_readmission")
    assert profile["required_feature_concepts"] == ()


def test_web_data_foundation_keeps_available_readmission_safety_coordinate(
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
                    typed_metadata=True,
                    column_role="value",
                ),
            ],
        ),
    )

    profile = research_launch_scientific._data_foundation_profile(
        export_path="/typed/demo",
        study={
            "modules": ["demographics", "outcome"],
            "covariate_selection": "planner_selectable",
        },
        target="death",
    )

    assert profile["static_concepts"] == ("age", "icu_readmission")


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

    profile = research_launch_scientific._data_foundation_profile(
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
        research_launch_scientific._data_foundation_profile(
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
    continuous = SimpleNamespace(
        analysis_columns={},
        materialized_columns=("stay_id", "lact_max", "death"),
    )
    assert (
        agent_pipeline_runs._resolve_materialized_primary_exposure(
            configured="lact",
            source_concept="lact",
            aggregation="max",
            acquisition=continuous,
        )
        == "lact_max"
    )
    assert (
        agent_pipeline_runs._resolve_materialized_primary_exposure(
            configured="lact",
            source_concept="lact",
            aggregation="mean",
            acquisition=continuous,
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
        research_launch_scientific._data_foundation_profile(
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

    monkeypatch.setattr(
        research_pipeline_run_preparation,
        "_data_foundation_profile",
        reject,
    )
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
            budget_mode="full_reviewed",
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
        research_pipeline_run_preparation,
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


def test_pipeline_factory_accepts_owner_confirmed_explicit_sepsis_concept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = {
        **_complete_study(),
        "question": "What is standard Sepsis-3 prevalence and mortality?",
        "primary_exposure": "Sepsis-3 using SOFA-2",
        "modules": ["outcome", "sepsis3_sofa2"],
        "execution_concepts": {
            "outcome": "death",
            "primary_exposure": "sep3_sofa2",
            "covariates": [],
        },
        "confirmations": {
            "concept_selection_sep3_sofa2_authorized": True,
        },
    }
    monkeypatch.setattr(
        research_pipeline_run_preparation,
        "_validate_analysis_design",
        lambda _study: {},
    )

    research_launch_scientific._validate_primary_concept_selection(
        study,
        "sep3_sofa2",
    )


def test_pipeline_factory_rejects_unimplemented_cluster_variance_before_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation_called = False

    def foundation(**_kwargs: Any) -> dict[str, Any]:
        nonlocal foundation_called
        foundation_called = True
        return _foundation_profile()

    monkeypatch.setattr(
        research_pipeline_run_preparation,
        "_data_foundation_profile",
        foundation,
    )
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
        "data_constraints",
        "covariates",
        "covariate_selection",
        "covariate_rationales",
        "covariate_temporal_roles",
        "covariate_operationalizations",
    }
    # A stated comparator is the estimand's reference group, so it travels as a
    # declaration the Planner may honour. Filed as `subgroup_sensitivity` it
    # reached the Planner as "Include subgroup/sensitivity requests: ...",
    # turning a reference group into extra analyses nobody requested; the
    # contrast itself belongs to PlannedModelRequirement.
    assert validated.subgroup_sensitivity is None
    assert validated.extra_notes == (
        "Demo-only product validation.\n"
        "Comparator stated by the researcher: Compare aggregate summaries by sex."
    )
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
    # An exact covariate roster with no reviewed column bindings compiles the
    # empty map, not a missing key: the Research Agent must be able to tell
    # "nothing was bound" from "this study never reached that decision".
    assert validated.covariate_operationalizations == {}


def test_reviewed_covariate_column_bindings_reach_the_research_agent() -> None:
    study = {
        **_complete_study(),
        "covariate_operationalizations": {"age": "age_years", "sex": "sex_female"},
    }

    compiled = agent_pipeline_runs._research_user_preferences(study)
    validated = UserPreferences.model_validate(compiled)

    assert validated.covariate_operationalizations == {
        "age": "age_years",
        "sex": "sex_female",
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
