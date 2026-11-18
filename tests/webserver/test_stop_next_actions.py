"""Stops offer explicit choices without granting analysis or changing a plan."""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pytest
from pydantic import ValidationError

from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
)
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver import agent_runs, stop_next_actions as owner
from easyicu.webserver.pi_copilot.projections import (
    project_archived_job_stop,
    project_job,
    project_run_outcome,
    project_run_row,
)
from easyicu.webserver.pi_copilot.workflow import (
    build_research_workflow_snapshot,
    project_run_history,
)
from easyicu.webserver.research_pipeline_run_errors import (
    ResearchPipelineRunError,
    WINDOWED_SUMMARY_STOP_MESSAGES,
)
from easyicu.webserver.research_run_submission import ResearchRunSubmissionRequest
from easyicu.webserver.run_record import RunGate


def test_no_stop_asks_for_a_data_handling_declaration() -> None:
    # A declaration is recorded, not required (owner decision 2026-10-27), so
    # no stop, including an old saved one, sends the user to declare it.
    reason = "provider_not_cleared_for_credentialed_data"
    assert reason not in owner.STOP_POLICIES
    assert "provider_settings" not in owner.STOP_POLICIES.values()
    gate = {"status": "blocked", "reason": reason}
    for projected in (owner.project_stop_next_actions(gate),
                      project_job({"id": "fixture", "status": "failed", "error": reason})["next_actions"]):
        assert all(action["kind"] != "open_provider_settings" for action in projected)


def test_an_old_saved_provider_settings_choice_is_no_longer_a_choice() -> None:
    # The declaration step is gone, so a choice saved before it went is not
    # offered again: it no longer validates, and projection derives afresh.
    saved = {
        "action_id": "open_provider_settings_0123456789abcdef",
        "kind": "open_provider_settings",
        "label_key": "stop_next_actions.open_provider_settings",
        "params": {"route": "provider_settings"},
    }
    with pytest.raises(ValidationError):
        owner.StopNextAction.model_validate(saved)
    gate = {"status": "blocked", "reason": "provider_not_cleared_for_credentialed_data", "next_actions": [saved]}
    assert all(action["kind"] != "open_provider_settings" for action in owner.project_stop_next_actions(gate))


def test_a_provider_stop_offers_retry_in_the_run_and_conversation(tmp_path) -> None:
    assert agent_pipeline_runs._write_pipeline_failure_projection(
        wrapper_dir=tmp_path,
        study={"id": "synthetic_study"},
        provider={},
        code="research_pipeline_planner_provider_unavailable",
        failure_type="provider_http",
        diagnostic=None,
    )
    gate = json.loads((tmp_path / "quality_gate.json").read_text())["gate"]
    assert [action["kind"] for action in gate["next_actions"]] == ["retry"]
    outcome = project_run_outcome(
        {
            "ok": True,
            "run_id": "synthetic_run",
            "gate": gate,
        }
    )
    assert outcome["next_actions"] == gate["next_actions"]


@pytest.mark.parametrize("reason", [
    "research_pipeline_execution_failed",
    "research_agent_pipeline_failed_closed",
    "research_pipeline_execution_retry_recovery_seed_missing",
    "research_pipeline_execution_retry_unexpected_plan_review",
])
def test_only_the_workflow_retry_owner_can_offer_retry_of_failed_execution(reason):
    from easyicu.webserver import study_contexts
    from tests.webserver.copilot.research_workflow_fixtures import complete_study

    gate = {"status": "blocked", "reason": reason}
    assert [row["kind"] for row in owner.project_stop_next_actions(gate)] == ["edit_plan"]
    study = {**complete_study(), "question": "How does the admission lactate vary by season?"}
    snapshot = build_research_workflow_snapshot(
        study=study, active_export_present=True, active_job=None,
        latest_run={
            "run_id": "seasonal_lactate", "run_type": "full", "run_status": "failed",
            "engine": "easyicu.research_agent.pipeline", "gate_status": "blocked",
            "gate_reason": reason,
            "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(study),
            "artifact_names": ["agent_plan.json", "evidence_ledger.json", "source_run_manifest.json"],
        },
    )
    assert snapshot.next_action_code == "failed_pipeline_execution_retry_available"
    assert [action.kind for action in snapshot.next_actions] == ["retry", "edit_plan"]
    assert snapshot.next_actions[0].params.stage == "execution"
    assert [row["kind"] for row in owner.project_stop_next_actions(
        {**gate, "reason": "data_foundation_blocked"}, execution_retry_available=True,
    )] == ["edit_plan"]


def test_wrapped_execution_step_failure_offers_the_workflow_retry():
    reason = "research_agent_pipeline_failed_closed"
    detail = {
        "reason_code": "research_pipeline_execution_failed",
        "step_id": "normalize_handover_temperatures",
        "message": "The handover temperature series could not be normalized.",
    }
    base_actions = owner.next_actions_for_stop(reason, detail)
    assert [(choice.kind, choice.params.change_question) for choice in base_actions] == [
        ("edit_plan", None),
    ]
    choices = owner.project_stop_next_actions(
        {"status": "blocked", "reason": reason, "detail": detail},
        execution_retry_available=True,
    )
    assert [(choice["kind"], choice["params"].get("stage")) for choice in choices] == [
        ("retry", "execution"), ("edit_plan", None),
    ]


def test_a_required_design_gap_offers_plan_edit_without_dropping_it(tmp_path) -> None:
    stop = ProgressivePlanCompileError(
        "progressive_capability_gap",
        "Synthetic provider text is not an action.",
        cause_code="design_element_unsupported",
        concepts=("synthetic_covariate",),
    )
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path,
        study={"id": "synthetic_study"},
        provider={},
        exc=stop,
        code=agent_pipeline_runs._pipeline_failure_code(stop),
        execution_retry_id=None,
    )
    gate = json.loads((tmp_path / "quality_gate.json").read_text())["gate"]
    assert [action["kind"] for action in gate["next_actions"]] == ["edit_plan"]
    assert gate["next_actions"][0]["params"]["requires_plan_approval"] is True


def test_a_desktop_plan_without_an_owner_offers_edit_with_its_step(tmp_path):
    from easyicu.research_agent.execution.owner_declaration import require_host_rendered_plan
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(research_question="Describe synthetic values", steps=[
        AnalysisStep(step_id="custom", intent="Compute values", method="custom"),
    ])
    with pytest.raises(ValueError) as error:
        require_host_rendered_plan(plan=plan)
    code = agent_pipeline_runs._pipeline_failure_code(error.value)
    assert code == "research_pipeline_plan_step_needs_sandbox"
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path, study={"id": "synthetic_study"}, provider={},
        exc=error.value, code=code, execution_retry_id=None,
    )
    gate = json.loads((tmp_path / "quality_gate.json").read_text())["gate"]
    assert [action["kind"] for action in gate["next_actions"]] == ["edit_plan"]
    assert gate["next_actions"][0]["params"]["step_id"] == "custom"
    assert gate["next_actions"][0]["params"]["requires_plan_approval"] is True


@pytest.mark.parametrize("step_id", ["safe_step", "../private/path", "x" * 81, None])
def test_desktop_plan_stop_projects_only_safe_step_coordinates(step_id):
    exc = ValueError("untrusted diagnostic text")
    exc.easyicu_safe_diagnostic = {
        "owner": "easyicu.execution.host_plan_v1",
        "reason_code": "research_pipeline_plan_step_needs_sandbox",
        "step_id": step_id,
        "message": "must not project",
    }
    projected = agent_pipeline_runs._safe_pipeline_typed_failure(exc)
    assert projected == {
        "owner": "easyicu.execution.host_plan_v1",
        "reason_code": "research_pipeline_plan_step_needs_sandbox",
        **({"step_id": step_id} if step_id == "safe_step" else {}),
    }


@pytest.mark.parametrize("cause", [RuntimeError, TimeoutError])
def test_an_unevaluated_desktop_owner_is_still_a_plan_stop(monkeypatch, cause):
    from easyicu.research_agent.execution import owner_declaration
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    def unavailable(*args, **kwargs):
        raise cause("synthetic ownership check unavailable")

    monkeypatch.setattr(owner_declaration, "resolve_standard_executor", unavailable)
    plan = AnalysisPlan(research_question="Describe synthetic values", steps=[
        AnalysisStep(step_id="custom", intent="Compute values", method="custom"),
    ])
    with pytest.raises(ValueError) as error:
        owner_declaration.require_host_rendered_plan(plan=plan)
    assert agent_pipeline_runs._pipeline_failure_code(error.value) == "research_pipeline_plan_step_needs_sandbox"
    assert agent_pipeline_runs._safe_pipeline_typed_failure(error.value)["step_id"] == "custom"


@pytest.mark.parametrize(
    "reason, kind",
    [
        ("research_pipeline_planner_provider_unavailable", "retry"),
        ("study_family_database_out_of_scope", "rephrase_in_scope"),
    ],
)
def test_a_job_that_stops_before_a_run_receipt_still_offers_a_choice(
    reason, kind
) -> None:
    job = project_job({"id": "synthetic_job", "status": "failed", "error": reason})
    assert [action["kind"] for action in job["next_actions"]] == [kind]


def _owned_terminal_codes() -> set[str]:
    """Read the actual raisers/return owners, not the action registry itself."""

    root = Path(__file__).resolve().parents[2]
    paths = subprocess.run(
        [
            "git", "grep", "-lz", "-e", "ResearchPipelineRunError", "--",
            ":(glob)src/easyicu/webserver/**/*.py",
            ":!*_restricted*", ":!*D3.3*", ":!*D3.4*", ":!*S166b*",
            ":!*meta_generalization*",
        ],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.split("\0")
    codes = set()

    def literals(expression):
        if isinstance(expression, ast.IfExp):
            return [*literals(expression.body), *literals(expression.orelse)]
        return [expression]

    for path in paths:
        if not path.endswith(".py") or not path.startswith("src/easyicu/webserver/"):
            continue
        for node in ast.walk(ast.parse((root / path).read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ResearchPipelineRunError"
                and node.args
            ):
                codes.update(
                    value.value for value in literals(node.args[0])
                    if isinstance(value, ast.Constant) and isinstance(value.value, str)
                )
    for function in (
        agent_pipeline_runs._pipeline_failure_code,
        agent_runs._gate_reason,
        agent_pipeline_runs._pending_review_reason_code,
    ):
        for node in ast.walk(ast.parse(inspect.getsource(function))):
            if (
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                codes.add(node.value.value)
    # These gates store their terminal reason in a local before returning it.
    for function in (
        agent_pipeline_runs._gate_from_axes,
        agent_runs._cancelled_agent_result,
    ):
        for node in ast.walk(ast.parse(inspect.getsource(function))):
            values = []
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "reason"
                for target in node.targets
            ):
                values = literals(node.value)
            if isinstance(node, ast.Dict):
                values = [
                    value
                    for key, value in zip(node.keys, node.values, strict=True)
                    if isinstance(key, ast.Constant) and key.value == "reason"
                ]
            codes.update(
                value.value
                for value in values
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            )
    from easyicu.research_agent.planning.approval_stops import PLAN_APPROVAL_STOPS
    from easyicu.webserver.study_family_scope import STUDY_FAMILY_DATABASE_OUT_OF_SCOPE
    from easyicu.webserver.target_trial_setup import TARGET_TRIAL_DATABASE_OUT_OF_SCOPE

    codes.update(
        (
            *PLAN_APPROVAL_STOPS,
            *WINDOWED_SUMMARY_STOP_MESSAGES,
            STUDY_FAMILY_DATABASE_OUT_OF_SCOPE,
            TARGET_TRIAL_DATABASE_OUT_OF_SCOPE,
        )
    )
    return codes - {
        "full_agent_complete_human_signoff_required",
        "preflight_complete_human_signoff_required",
        "research_agent_pipeline_complete_human_interpretation_required",
    }


# These causes have no currently authorized remedy. Keep exact cases here so
# assigning a new stop policy to "none" cannot silently exempt it.
_DESIGN_STOPS = frozenset({
    "case_duration_plan_approval_required",
    "controlled_candidate_provider_identity_changed",
    "extraction_database_filter_unsupported",
    "planning_menu_missing_supported_concept",
    "research_pipeline_deferred_review_stage_mismatch",
    "research_pipeline_deferred_review_unverified",
    "research_pipeline_first_observed_input_binding_unavailable",
    "research_pipeline_metadata_provider_capacity_unavailable",
    "research_pipeline_metadata_shared_scope_unavailable",
    "research_pipeline_planning_native_metadata_invalid",
})


def _wrapped_stop_routes(reason, cause):
    codes = {reason} if owner.STOP_POLICIES.get(reason) == "retry" else {reason, cause}
    policies = {owner.STOP_POLICIES.get(code) for code in codes}
    if codes & _DESIGN_STOPS:
        return []
    if "data_extraction" in policies or "export_time_window_unbounded" in codes:
        return ["data"]
    if "model_connection" in policies:
        return ["connect"]
    if policies & {"scope", "study_edit", "data"} or codes & {
        *WINDOWED_SUMMARY_STOP_MESSAGES,
        "question_named_level_undistinguished",
        "question_treatment_use_unrecorded",
        "question_mean_survival_unplanned",
        "study_stated_variable_unrecorded",
        "study_coordinate_unrecognized",
        "research_pipeline_hospital_first_stay_restriction_unverified",
    }:
        return ["draft"]
    return ["confirm"]


def _stop_card_cases():
    from easyicu.databases.research_approval import CAPABILITY_APPROVAL_CODES
    from easyicu.research_agent.contracts.execution_result import STANDARD_EXECUTOR_FAILURE_REASONS
    from easyicu.research_agent.contracts.executor_stop import EXECUTOR_STOP_REASONS
    from easyicu.research_agent.planning.approval_stops import PLAN_APPROVAL_STOPS
    from easyicu.research_agent.providers.transport_retry import RETRY_STOP_REASONS
    from easyicu.research_agent.reporting.completion import AnalysisOutcome, ManuscriptOutcome
    from easyicu.research_agent.reporting.writer_stop import WRITER_PROVIDER_TRANSPORT_UNAVAILABLE
    from easyicu.webserver.pi_copilot.survival_plan_decision import DECISION_CODE
    from easyicu.webserver.pi_copilot.workflow_attempts import PreservedPlanFailure

    from easyicu.webserver.target_trial_setup import (
        TARGET_TRIAL_COMPILE_FAILED,
        TARGET_TRIAL_COMPILE_INTERRUPTED,
        TARGET_TRIAL_COMPILE_STOPS,
    )

    terminal = _owned_terminal_codes() | CAPABILITY_APPROVAL_CODES | {"caller_bound_population_mode_conflict"}
    assert terminal <= owner.STOP_POLICIES.keys(), sorted(terminal - owner.STOP_POLICIES.keys())
    from tests.webserver.copilot.research_workflow_fixtures import complete_study

    study = {**complete_study(), "question": "Can collection shifts explain the evening assay distribution?"}
    job_workflow = build_research_workflow_snapshot(
        study=study, active_export_present=True, active_job=None, latest_run=None,
    ).model_dump(mode="json")
    assert job_workflow["next_action_code"] == "provider_ready_to_generate_plan"
    cases = []
    for code, policy in sorted(owner.STOP_POLICIES.items()):
        if code == "human_plan_review_required":
            # This gate's actual stops are its pending reason codes.
            continue
        run = {
            "run_id": "synthetic", "run_status": "failed",
            "gate_status": "blocked", "gate_reason": code,
        }
        actions = project_run_row(run)["next_actions"]
        job = project_job({"id": "synthetic_job", "status": "failed", "error": code})
        assert actions == job["next_actions"], code
        assert bool(actions) == (code not in _DESIGN_STOPS), code
        workflow_code = "failed_pipeline_requires_fresh_plan" if policy == "retry" else code
        cases.append({
            "code": code, "surface": "job",
            "job": {**job, "kind": "extract" if code == "no_export_files" else "agent-run"},
            "design": code in _DESIGN_STOPS,
            "workflow": job_workflow,
        })
        if code in PLAN_APPROVAL_STOPS or code == "operator_plan_approval_required":
            # An approval pause renders in the confirmation owner, not as a failed run.
            cases.append({"code": code, "surface": "review", "workflow": {"next_action_code": code}})
        else:
            cases.append({
                "code": code, "surface": "run", "design": code in _DESIGN_STOPS,
                "run": {**run, "next_actions": actions}, "workflow": {"next_action_code": workflow_code},
            })
    for kind in ("agent-run", "extract"):
        job = project_job({"id": "evening_collection_cancelled", "kind": kind, "status": "cancelled"})
        legacy = project_archived_job_stop({**job, "next_actions": []})
        assert legacy["next_actions"] == job["next_actions"]
        cases.append({"code": kind, "surface": "job", "job": legacy, "workflow": job_workflow})
    analysis_reasons = set(get_args(get_args(AnalysisOutcome.model_fields["reason_code"].annotation)[0]))
    manuscript_reasons = set(get_args(get_args(ManuscriptOutcome.model_fields["reason_code"].annotation)[0]))
    details = {
        **{reason: spec.cause_codes for reason, spec in EXECUTOR_STOP_REASONS.items()},
        **dict.fromkeys((analysis_reasons | manuscript_reasons) - {"human_plan_review_required"}, ()),
        "deterministic_standard_executor_failed_closed": STANDARD_EXECUTOR_FAILURE_REASONS,
        WRITER_PROVIDER_TRANSPORT_UNAVAILABLE: RETRY_STOP_REASONS,
    }
    wrapped_pairs = dict.fromkeys(
        (reason, cause)
        for reason, causes in sorted(details.items())
        for cause in sorted(causes) or [None]
    )
    for code in sorted(owner.STOP_POLICIES):
        wrapped_pairs.setdefault((code, None))
        wrapped_pairs.setdefault(("deterministic_standard_executor_failed_closed", code))
    for reason, cause in wrapped_pairs:
        detail = {
            "reason_code": reason, "cause_code": cause,
            "message": "Can the nightly sample distinguish the two collection protocols?",
        }
        gate = {"status": "blocked", "reason": "research_agent_pipeline_failed_closed", "detail": detail}
        actions = owner.project_stop_next_actions(gate)
        routes = _wrapped_stop_routes(reason, cause)
        cases.append({
            "code": reason, "cause": cause, "surface": "run",
            "design": not routes,
            "routes": routes,
            "run": {
                "run_id": "synthetic", "run_status": "failed",
                "gate_reason": gate["reason"], "gate_detail": detail,
                "next_actions": actions,
            },
            "workflow": {"next_action_code": "failed_pipeline_execution_retry_futile", "next_actions": actions},
        })
    for code in sorted(manuscript_reasons - analysis_reasons):
        cases.append({
            "code": code, "surface": "outcome",
            "run": {
                "run_id": "synthetic", "present": True, "analysis_results_available": True,
                "analysis_validated": True, "numeric_verified": True,
                "manuscript_outcome": {"status": "not_passed", "reason_code": code},
            },
            "workflow": {
                "report_rewrite_available": code == WRITER_PROVIDER_TRANSPORT_UNAVAILABLE,
                "analysis_validation_retry_available": code != WRITER_PROVIDER_TRANSPORT_UNAVAILABLE,
            },
        })
    for code, detail, workflow in (
        ("progressive_capability_gap", {"element": "synthetic_step", "question_required": False}, "provider_ready_to_generate_plan"),
        ("data_foundation_blocked", {"missing_concepts": ["synthetic_marker", "synthetic_second"]}, "provider_ready_to_generate_plan"),
        ("data_foundation_blocked", {"reason_code": "study_coordinate_unavailable", "concept": "synthetic_marker"}, "study_coordinate_unavailable"),
        ("data_foundation_blocked", {"reason_code": "study_coordinate_unavailable", "concept": "synthetic_source", "database": "miiv"}, "study_coordinate_unavailable"),
        ("research_pipeline_provider_timeout", {}, "planner_checkpoint_resume_available"),
        ("research_pipeline_provider_timeout", {}, "plan_execution_upgrade_required"),
        ("research_pipeline_execution_failed", {}, "failed_pipeline_execution_retry_available"),
    ):
        gate = {"status": "blocked", "reason": code, "detail": detail}
        actions = owner.project_stop_next_actions(
            gate, execution_retry_available=workflow == "failed_pipeline_execution_retry_available",
        )
        cases.append({
            "code": code, "surface": "run",
            "run": {"run_id": "synthetic", "run_status": "failed", "gate_reason": code, "next_actions": actions},
            "workflow": {"next_action_code": workflow},
        })
    for missing, expected in (
        (["synthetic_source"], ["synthetic_alternative"]),
        (["synthetic_source", "overnight_pulse"], ["synthetic_alternative", "overnight_pulse"]),
    ):
        cases.append({
            "code": "study_coordinate_unavailable", "surface": "review", "drafts": expected,
            "workflow": {
                "next_action_code": "study_coordinate_unavailable",
                "next_actions": owner.project_stop_next_actions({
                    "status": "blocked", "reason": "study_coordinate_unavailable",
                    "detail": {"missing_concepts": missing, "database": "miiv"},
                }),
            },
        })
    for workflow, label in (
        ("provider_ready_to_generate_plan", ("Retry plan generation", "重试生成计划")),
        ("planner_checkpoint_resume_available", ("Continue plan generation", "继续生成计划")),
        ("plan_execution_upgrade_required", ("Retry data preparation", "重试数据准备")),
    ):
        code = "research_pipeline_cancelled"
        cases.append({
            "code": code, "surface": "run", "routes": ["confirm"], "label": label,
            "run": {
                "run_id": "synthetic", "run_status": "cancelled", "gate_reason": code,
                "gate_detail": {"message": "Pause the evening candidate while I verify its sampling periods."},
                "next_actions": owner.project_stop_next_actions({"status": "blocked", "reason": code}),
            },
            "workflow": {"next_action_code": workflow},
        })
    for context in (
        {},
        {"landmark_hours": 24, "horizon_days": 28, "exposure": "synthetic", "card_sha256": "a" * 64},
    ):
        cases.append({
            "code": DECISION_CODE, "surface": "review", "workflow": {
                "next_action_code": "plan_scientific_changes_required",
                "plan_review_summary": {"authorization_questions": [{
                    "code": DECISION_CODE, "decision_context": context,
                }]},
            },
        })
    for code in get_args(PreservedPlanFailure.model_fields["reason"].annotation):
        cases.append({
            "code": code, "surface": "review",
            "workflow": {
                "next_action_code": "plan_execution_upgrade_required",
                "plan_review_summary": {"run_id": "synthetic_candidate"},
                "latest_attempt_failure": {
                    "run_id": "synthetic", "candidate_run_id": "synthetic_candidate",
                    "reason": code, "checkpoint_resume_available": False,
                },
            },
        })
    for code in (*TARGET_TRIAL_COMPILE_STOPS, TARGET_TRIAL_COMPILE_FAILED,
                 TARGET_TRIAL_COMPILE_INTERRUPTED, "target_trial_record_missing"):
        cases.append({
            "code": code, "surface": "review",
            "workflow": {"next_action_code": "target_trial_review", "target_trial_card": {
                "state": "stopped", "reason_code": code,
            }},
        })
    for code in (
        "agent_plan_configuration_required", "agent_plan_revision_nonconvergent",
        "bound_export_cohort_invalid", "bound_export_cohort_mismatch",
        "failed_pipeline_execution_retry_available", "failed_pipeline_execution_retry_futile",
        "failed_pipeline_requires_fresh_plan", "plan_configuration_superseded",
        "plan_execution_upgrade_required", "plan_review_not_resumable",
        "plan_scientific_changes_required", "planner_checkpoint_resume_available",
        "scientific_plan_review_policy_stale", "study_family_database_out_of_scope",
        "study_family_reading_disagreement", "study_primary_outcome_confirmation_required",
        "study_coordinate_unavailable", "study_coordinate_unrecognized", "plan_icd_source_unverified",
    ):
        cases.append({
            "code": code, "surface": "review", "workflow": {
                "next_action_code": code,
                "next_actions": [action.model_dump(mode="json") for action in owner.next_actions_for_stop(code)],
                "plan_review_summary": {
                    "run_id": "synthetic_candidate",
                    "authorization_questions": [],
                    "automatic_revision_blockers": [],
                    "remediation_buckets": {"agent_plan_revision": ["SYNTHETIC"]},
                },
            },
        })
    return cases


def _recovery_contract_cases(monkeypatch):
    from easyicu.research_agent import concept_availability
    from easyicu.webserver import study_contexts
    from tests.webserver.copilot.research_workflow_fixtures import complete_study

    # Cross the renderer's inputs independently, including receipt shapes
    # without a plan. The source cases below also exercise the real workflow.
    codes = (
        "failed_pipeline_requires_fresh_plan",
        "failed_pipeline_execution_retry_available",
        "failed_pipeline_execution_retry_futile",
        "provider_ready_to_generate_plan",
        "planner_checkpoint_resume_available",
        "plan_execution_upgrade_required",
    )
    stops = (
        ("research_pipeline_execution_failed", ({}, {"step_id": "align_dusk_assays"})),
        ("design_element_unsupported", (
            {}, {"element": "dusk_calibration"}, {"element": "dusk_calibration", "concept": "dusk_assay",
                                                 "question_required": False},
        )),
        ("research_pipeline_required_concept_without_values", (
            {}, {"concept": "dusk_assay"}, {"missing_concepts": ["dusk_assay", "synthetic_source"]},
        )),
        ("research_pipeline_provider_timeout", ({}, {"step_id": "align_dusk_assays"})),
    )
    for (reason, details), code, wrapped, has_plan in product(stops, codes, (False, True), (False, True)):
        for coordinates in details:
            detail = {"message": "The dusk assays were logged against different reference windows.",
                      **coordinates}
            gate_reason = reason
            if wrapped:
                gate_reason = "research_agent_pipeline_failed_closed"
                detail["reason_code"] = reason
            if reason == "research_pipeline_required_concept_without_values":
                detail["database"] = "miiv"
            actions = owner.project_stop_next_actions(
                {"status": "blocked", "reason": gate_reason, "detail": detail},
                execution_retry_available=code == "failed_pipeline_execution_retry_available",
            )
            yield {
                "code": code, "surface": "coordinate-recovery",
                "cause": f"{reason}/{list(coordinates)}/{wrapped=}/{has_plan=}",
                "run": {
                    "run_id": "dusk_assays", "run_status": "failed", "gate_status": "blocked",
                    "gate_reason": gate_reason, "gate_detail": detail,
                    "artifact_names": ["agent_plan.json"] if has_plan else [],
                    "next_actions": actions,
                },
                "workflow": {"next_action_code": code, "next_actions": actions},
            }


    retry_code = "failed_pipeline_execution_retry_available"
    execution = owner.project_stop_next_actions(
        {"status": "blocked", "reason": "research_pipeline_execution_failed"},
        execution_retry_available=True,
    )[:1]
    planning = owner.next_actions_for_stop("research_pipeline_provider_timeout")[0].model_dump(mode="json")
    for label, recovery in (("execution only", []), ("planning retry", [planning]), ("legacy edit", [{"kind": "edit_plan"}])):
        yield {
            "code": retry_code, "surface": "confirmation-edit",
            "cause": f"Resume the temperature normalization after handover/{label}",
            "workflow": {"next_action_code": retry_code, "next_actions": [*recovery, *execution]},
            "expect_edit": bool(recovery),
        }

    study = {**complete_study(), "question": "Compare the evening reading across sampling schedules."}
    unresolved = ["midnight_platelet_index", "handover_temperature_band"]
    recovered = "shift_fluid_index"
    for wrapped, has_plan, available, recognized in product((False, True), repeat=4):
        if available and not recognized:
            continue
        gate_reason = "research_agent_pipeline_failed_closed" if wrapped else "study_coordinate_unavailable"
        detail = {
            **({"reason_code": "study_coordinate_unavailable"} if wrapped else {}),
            "missing_concepts": [*unresolved, recovered],
            "message": "The night register may use a different name for this measurement.",
        }
        run = {
            "run_id": "night_register", "run_type": "full", "run_status": "failed",
            "engine": "easyicu.research_agent.pipeline", "gate_status": "blocked",
            "gate_reason": gate_reason, "gate_detail": detail,
            "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(study),
            "artifact_names": [
                "source_run_manifest.json",
                *(["agent_plan.json", "evidence_ledger.json"] if has_plan else []),
            ],
        }
        with monkeypatch.context() as patch:
            patch.setattr(
                concept_availability, "explain_concept_availability",
                lambda *, concept, database: SimpleNamespace(available=available or concept == recovered, reason=(
                    "recorded" if available else "absent_from_source" if recognized else "concept_not_found"
                )),
            )
            workflow = build_research_workflow_snapshot(
                study=study, active_export_present=True, active_job=None, latest_run=run,
            )
        yield {
            "code": workflow.next_action_code, "surface": "coordinate-recovery",
            "cause": f"shift inventory/{wrapped=}/{has_plan=}/{available=}/{recognized=}",
            "run": {**run, "artifact_names": [name for name in run["artifact_names"] if name != "source_run_manifest.json"]},
            "workflow": workflow.model_dump(mode="json"),
            **({"missing": unresolved, "recovered": recovered} if not available else {}),
        }


def _archived_job_supersession_cases():
    newer = "2026-01-05T19:00:00Z"
    for change, updates, created in (
        ("run", {"runs": [{"run_id": "repeat_handover_audit", "updated_at": newer}]}, None),
        ("study", {"study_setup_receipt": {"updated_at": newer}}, None),
        ("export", {"bound_export_updated_at": newer}, None),
        ("export after a recorded start", {"bound_export_updated_at": newer}, 1767636000),
    ):
        job = project_archived_job_stop({
            "job_id": "handover_export", "kind": "extract", "status": "failed",
            "error_code": "no_export_files", "created_at_epoch": created,
        })
        yield {
            "code": job["error_code"], "surface": "job-supersession", "cause": f"handover inventory/{change}",
            "job": job, "workflow": {"next_action_code": "provider_ready_to_generate_plan", "runs": []},
            "superseded_workflow": updates,
        }
    for provider, reason, checked in (
        ("codex", "research_pipeline_codex_user_auth_required", {"codexAuth": {"authentication_verified": True}}),
        ("openai", "research_pipeline_pi_verified_credentials_required",
         {"runtime": {"configuration": {"verified_at": newer}}}),
    ):
        job = project_archived_job_stop({
            "job_id": "handover_model_check", "kind": "agent-run", "status": "failed", "error_code": reason,
        })
        yield {
            "code": reason, "surface": "job-supersession", "cause": "Connect the model for the handover audit.",
            "job": job, "workflow": {"next_action_code": reason, "runs": []}, "provider": provider,
            "superseded_workflow": {"next_action_code": "provider_ready_to_generate_plan"},
            "verified_provider": checked,
        }


def test_every_stop_card_has_a_forward_action_with_an_owner(monkeypatch) -> None:
    monkeypatch.setattr(
        owner, "concept_selection_policy",
        lambda concept: SimpleNamespace(canonical_alternative="synthetic_alternative")
        if concept == "synthetic_source" else None,
    )
    monkeypatch.setattr(owner, "explain_concept_availability", lambda **_kw: SimpleNamespace(available=True))
    cases = [*_stop_card_cases(), *_recovery_contract_cases(monkeypatch), *_archived_job_supersession_cases()]
    root = Path(__file__).resolve().parents[2]
    js = root / "src/easyicu/webserver/static/js"
    result = subprocess.run(
        [
            "node", str(root / "tests/js/guided_pi_stop_buttons.test.js"), "--catalog",
            *(str(js / name) for name in (
                "html-escape.js", "product-labels.js",
                "screens-guided-pi-confirmation-data.js", "screens-guided-pi-confirmation.js",
                "screens-guided-pi-target-trial.js", "screens-guided-pi-run-stop.js",
                "screens-guided-pi-events.js", "screens-guided-pi-run-outcome.js",
                "screens-guided-pi-plan-actions.js", "screens-guided-pi-error-text.js",
            )),
        ],
        cwd=root, input=json.dumps(cases), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout, end="")


def test_outcome_confirmation_offers_an_explicit_edit_instead_of_plan_replacement():
    code = "study_primary_outcome_confirmation_required"
    actions = owner.next_actions_for_stop(code)
    assert [action.kind for action in actions] == ["edit_study"]
    assert actions[0].params.field == "outcome"


@pytest.mark.parametrize("code", [
    "case_duration_plan_approval_required",
    "research_pipeline_first_observed_input_binding_unavailable",
    "controlled_candidate_provider_identity_changed",
    "research_pipeline_deferred_review_stage_mismatch",
    "research_pipeline_deferred_review_unverified",
    "research_pipeline_planning_native_metadata_invalid",
])
def test_stale_source_or_first_icu_binding_never_offers_retry_or_population_removal(code):
    detail = {"element": "population", "question_required": False,
              "reason_code": "missing_data", "cause_code": "source_unavailable"}
    assert owner.next_actions_for_stop(code, detail=detail) == []
    job = project_job({"id": "synthetic_job", "status": "failed", "error": code})
    assert job["next_actions"] == []


def test_changed_time_origin_requires_a_new_reviewed_plan():
    actions = owner.next_actions_for_stop("candidate_plan_time_origin_binding_invalid")
    assert [action.kind for action in actions] == ["edit_plan"]
    assert actions[0].params.requires_plan_approval is True


@pytest.mark.parametrize("code", [
    "research_pipeline_hospital_first_stay_restriction_unverified",
    *(
        code for code in WINDOWED_SUMMARY_STOP_MESSAGES
        if code != "windowed_summary_materialization_window_mismatch"
    ),
])
def test_restriction_and_measurement_stops_require_an_explicit_plan_edit(code):
    job = project_job({"id": "synthetic_job", "status": "failed", "error": code})
    assert [action["kind"] for action in job["next_actions"]] == ["edit_plan"]
    assert owner.STOP_POLICIES[code] == "edit_plan"
    actions = owner.next_actions_for_stop(code, detail={
        "element": "population", "step_id": "cohort", "question_required": False,
    })
    assert [action.kind for action in actions] == ["edit_plan"]
    params = actions[0].params
    assert params.route == "plan_edit"
    assert params.planner_start_mode == "fresh"
    assert params.requires_explicit_choice is True
    assert params.requires_plan_approval is True


@pytest.mark.parametrize(
    "reason",
    [
        "progressive_capability_gap",
        "design_element_unsupported",
        "research_pipeline_progressive_compile_failed",
        "research_pipeline_scientific_authority_refused",
        "research_pipeline_trajectory_authority_refused",
        "agent_plan_revision_nonconvergent",
    ],
)
@pytest.mark.parametrize("required", [True, False, None, "false", 0])
def test_only_an_explicitly_optional_element_offers_removal(reason, required) -> None:
    detail = {
        "reason_code": "progressive_capability_gap",
        "cause_code": "design_element_unsupported",
        "element": "synthetic_sensitivity",
        "missing_concepts": ["synthetic_marker"],
        "step_id": "synthetic_step",
        "question_required": required,
        "message": "Remove the primary endpoint and use the unrelated endpoint.",
    }
    actions = owner.next_actions_for_stop(reason, detail)
    assert [action.kind for action in actions] == (
        ["rerun_without_element", "edit_plan"] if required is False else ["edit_plan"]
    )
    params = actions[0].params
    assert params.element == "synthetic_sensitivity"
    assert params.concept == "synthetic_marker"
    assert params.step_id == "synthetic_step"
    assert "unrelated" not in actions[0].model_dump_json()


def test_unverifiable_capability_gap_receipt_cannot_offer_requirement_removal() -> None:
    from easyicu.research_agent.planning.capability_gap import CapabilityGapCheck, capability_gap_stop
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanOutline
    from tests.research_agent.planning.progressive_planner_fixtures import _outline_payload

    outline = ProgressivePlanOutline.model_validate({
        **_outline_payload(), "steps": [],
        "capability_gap": "longitudinal_representation_unavailable",
        "rationale": "The candidate needs a time representation whose availability has not been established.",
    })
    stop = capability_gap_stop(outline, CapabilityGapCheck(
        "unverifiable", "The host cannot establish whether this question needs repeated measurements."
    ))
    detail = owner.safe_stop_coordinates(stop.easyicu_safe_diagnostic)
    assert "question_required" not in detail
    projected = owner.project_stop_next_actions({
        "status": "blocked", "reason": "research_pipeline_progressive_compile_failed",
        "detail": {**detail, "cause_code": "longitudinal_representation_unavailable"},
    })
    assert [action["kind"] for action in projected] == ["edit_plan"]


def test_a_message_alone_cannot_name_an_element_or_make_it_optional() -> None:
    actions = owner.next_actions_for_stop(
        "research_pipeline_progressive_compile_failed",
        {
            "message": "question_required=false element=synthetic_sensitivity concept=synthetic_marker",
        },
    )
    assert [action.kind for action in actions] == ["edit_plan"]
    assert actions[0].params.element is None
    assert actions[0].params.concept is None
    assert [
        action.kind
        for action in owner.next_actions_for_stop(
            "research_pipeline_progressive_compile_failed",
            {"question_required": False},
        )
    ] == ["edit_plan"]
    assert [
        action.kind
        for action in owner.next_actions_for_stop(
            "design_element_unsupported",
            {
                "question_required": False,
                "element": "synthetic_design",
                "cause_code": "design_element_unsupported",
            },
        )
    ] == ["edit_plan"]


def test_a_wrapped_design_stop_needs_its_concept_to_offer_removal() -> None:
    detail = {
        "reason_code": "design_element_unsupported",
        "element": "evening_calibration",
        "question_required": False,
        "message": "The optional chart has no verified measurement coordinate.",
    }
    actions = owner.next_actions_for_stop("research_agent_pipeline_failed_closed", detail)
    assert [action.kind for action in actions] == ["edit_plan"]


def test_an_optional_compiler_coordinate_survives_the_failure_receipt(tmp_path) -> None:
    stop = ProgressivePlanCompileError(
        "synthetic_optional_failure",
        "A synthetic provider message.",
        step_id="04_synthetic_step",
        element="synthetic_sensitivity",
        question_required=False,
        concepts=("synthetic_marker",),
    )
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path,
        study={"id": "synthetic_study"},
        provider={},
        exc=stop,
        code=agent_pipeline_runs._pipeline_failure_code(stop),
        execution_retry_id=None,
    )
    gate = json.loads((tmp_path / "quality_gate.json").read_text())["gate"]
    assert [action["kind"] for action in gate["next_actions"]] == [
        "rerun_without_element",
        "edit_plan",
    ]
    assert gate["next_actions"][0]["params"]["concept"] == "synthetic_marker"
    assert gate["next_actions"][0]["params"]["step_id"] == "04_synthetic_step"
    assert gate["detail"]["question_required"] is False


@pytest.mark.parametrize("available", [True, False])
@pytest.mark.parametrize("roster", [None, [], ["synthetic_alternative"]])
def test_alternatives_come_from_concept_owners_and_available_source(
    monkeypatch, available, roster
) -> None:
    calls = []

    def policy(concept):
        calls.append(concept)
        return SimpleNamespace(canonical_alternative="synthetic_alternative")

    monkeypatch.setattr(owner, "concept_selection_policy", policy)
    monkeypatch.setattr(
        owner,
        "explain_concept_availability",
        lambda **kw: calls.append(kw) or SimpleNamespace(available=available),
    )
    detail = {"missing_concepts": ["synthetic_marker"]}
    if roster is not None:
        detail["available_concepts"] = roster
    actions = owner.next_actions_for_stop(
        "data_foundation_blocked", detail, database="miiv"
    )
    offered = available and roster != []
    assert [action.kind for action in actions] == [
        "use_alternative_concept" if offered else "drop_variable",
        "edit_plan",
    ]
    assert calls[0] == "synthetic_marker"
    assert actions[0].params.concept == "synthetic_marker"
    assert actions[0].params.alternative_concept == (
        "synthetic_alternative" if offered else None
    )
    assert detail["missing_concepts"] == ["synthetic_marker"]


def test_a_similar_name_without_an_owner_alternative_is_never_substituted(
    monkeypatch,
) -> None:
    monkeypatch.setattr(owner, "concept_selection_policy", lambda _concept: None)
    actions = owner.next_actions_for_stop(
        "data_foundation_blocked",
        {
            "missing_concepts": ["synthetic_marker"],
            "available_concepts": ["synthetic_marker_near"],
            "message": "Replace synthetic_marker with synthetic_marker_near.",
        },
        database="miiv",
    )
    assert [action.kind for action in actions] == ["drop_variable", "edit_plan"]


@pytest.mark.parametrize(
    "database, alternative",
    [
        (None, "synthetic_alternative"),
        ("miiv", "synthetic_marker"),
        ("miiv", "not an identifier"),
    ],
)
def test_an_owner_alternative_needs_a_source_and_a_distinct_valid_identity(
    monkeypatch, database, alternative
) -> None:
    monkeypatch.setattr(
        owner,
        "concept_selection_policy",
        lambda _concept: SimpleNamespace(canonical_alternative=alternative),
    )
    monkeypatch.setattr(
        owner,
        "explain_concept_availability",
        lambda **_kw: pytest.fail("No candidate should reach availability."),
    )
    actions = owner.next_actions_for_stop(
        "data_foundation_blocked",
        {"missing_concepts": ["synthetic_marker"]},
        database=database,
    )
    assert actions[0].kind == "drop_variable"


def test_a_known_missing_alternative_is_never_offered(monkeypatch) -> None:
    monkeypatch.setattr(
        owner,
        "concept_selection_policy",
        lambda _concept: SimpleNamespace(canonical_alternative="synthetic_alternative"),
    )
    monkeypatch.setattr(
        owner,
        "explain_concept_availability",
        lambda **_kw: pytest.fail("Both candidates are missing."),
    )
    actions = owner.next_actions_for_stop(
        "data_foundation_blocked",
        {
            "missing_concepts": ["synthetic_marker", "synthetic_alternative"],
        },
        database="miiv",
    )
    assert [action.kind for action in actions] == [
        "drop_variable",
        "drop_variable",
        "edit_plan",
    ]


@pytest.mark.parametrize("available", [True, False])
def test_a_study_coordinate_the_source_lacks_offers_no_drop_or_regeneration(
    monkeypatch, available
) -> None:
    # Dropping the study's own outcome asks another question, and regenerating
    # on the same source repeats the stop (user 2026-10-27, HiRID icu_death).
    monkeypatch.setattr(
        owner, "concept_selection_policy",
        lambda concept: SimpleNamespace(canonical_alternative="synthetic_alternative"),
    )
    monkeypatch.setattr(
        owner, "explain_concept_availability",
        lambda **kw: SimpleNamespace(available=available),
    )
    detail = {"reason_code": "study_coordinate_unavailable", "missing_concepts": ["synthetic_outcome"]}
    for reason in ("data_foundation_blocked", "study_coordinate_unavailable"):
        actions = owner.next_actions_for_stop(reason, detail, database="miiv")
        assert [action.kind for action in actions] == (
            ["use_alternative_concept"] if available else ["edit_study"]
        )


def test_study_coordinates_without_alternatives_keep_their_own_edit_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        owner, "concept_selection_policy",
        lambda concept: SimpleNamespace(canonical_alternative="evening_pressure")
        if concept == "night_pressure" else None,
    )
    monkeypatch.setattr(owner, "explain_concept_availability", lambda **_kw: SimpleNamespace(available=True))
    choices = owner.next_actions_for_stop(
        "study_coordinate_unavailable",
        {
            "missing_concepts": ["night_pressure", "overnight_pulse", "morning_weight"],
            "message": "The overnight measurements are recorded on a different form.",
        },
        database="miiv",
    )
    assert [(action.kind, action.params.concept) for action in choices] == [
        ("use_alternative_concept", "night_pressure"),
        ("edit_study", "overnight_pulse"),
        ("edit_study", "morning_weight"),
    ]


def test_a_planning_menu_gap_offers_no_plan_change() -> None:
    # The source provides the concept; EasyICU's menu lacks it. No edit, drop
    # or alternative fixes that on this build.
    detail = {"reason_code": "planning_menu_missing_supported_concept", "missing_concepts": ["icu_death"]}
    assert owner.next_actions_for_stop("data_foundation_blocked", detail, database="hirid") == []


def test_a_typed_single_missing_concept_can_be_dropped() -> None:
    actions = owner.next_actions_for_stop(
        "concept_unavailable", {"concept": "synthetic_marker"}
    )
    assert actions[0].kind == "drop_variable"
    assert actions[0].params.concept == "synthetic_marker"


def test_all_named_missing_variables_get_a_choice_and_missing_coordinates_get_edit() -> (
    None
):
    actions = owner.next_actions_for_stop(
        "data_foundation_blocked",
        {
            "missing_concepts": [
                "synthetic_one",
                "synthetic_two",
                "synthetic_one",
                "Not a code!",
            ],
        },
    )
    assert [action.kind for action in actions] == [
        "drop_variable",
        "drop_variable",
        "edit_plan",
    ]
    assert [action.params.concept for action in actions[:2]] == [
        "synthetic_one",
        "synthetic_two",
    ]
    assert [
        action.kind for action in owner.next_actions_for_stop("data_foundation_blocked")
    ] == ["edit_plan"]


def test_a_changed_question_supersedes_the_failed_runs_stop() -> None:
    from easyicu.webserver import study_contexts
    from tests.webserver.copilot.research_workflow_fixtures import complete_study

    study = {**complete_study(), "question": "Does evening sampling distinguish the two assays?"}
    run = {
        "run_id": "evening_assay", "run_type": "full", "run_status": "failed",
        "engine": "easyicu.research_agent.pipeline", "gate_status": "blocked",
        "gate_reason": "research_agent_pipeline_failed_closed",
        "gate_detail": {"reason_code": "study_stated_variable_unrecorded"},
        "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(study),
        "artifact_names": ["source_run_manifest.json"],
    }
    for question, change_required in (
        (study["question"], True),
        ("How does the recorded assay vary across sampling shifts?", None),
    ):
        snapshot = build_research_workflow_snapshot(
            study={**study, "question": question}, active_export_present=True,
            active_job=None, latest_run=run,
        )
        assert snapshot.next_action_code == "failed_pipeline_requires_fresh_plan"
        assert [action.kind for action in snapshot.next_actions] == ["edit_plan"]
        assert snapshot.next_actions[0].params.change_question is change_required


def test_one_recovered_coordinate_leaves_the_other_coordinate_blocked(monkeypatch) -> None:
    from easyicu.research_agent import concept_availability
    from easyicu.webserver import study_contexts
    from tests.webserver.copilot.research_workflow_fixtures import complete_study

    study = {**complete_study(), "question": "Compare overnight pulse and morning weight by shift."}
    recorded = {"overnight_pulse"}
    monkeypatch.setattr(
        concept_availability, "explain_concept_availability",
        lambda *, concept, database: SimpleNamespace(available=concept in recorded, reason="test_roster"),
    )
    monkeypatch.setattr(owner, "concept_selection_policy", lambda _concept: None)
    run = {
        "run_id": "shift_check", "run_type": "full", "run_status": "failed",
        "engine": "easyicu.research_agent.pipeline", "gate_status": "blocked",
        "gate_reason": "research_agent_pipeline_failed_closed",
        "gate_detail": {
            "reason_code": "deterministic_standard_executor_failed_closed",
            "cause_code": "study_coordinate_unavailable",
            "missing_concepts": ["overnight_pulse", "morning_weight"],
        },
        "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(study),
        "artifact_names": ["source_run_manifest.json"],
    }
    inputs = dict(study=study, active_export_present=True, active_job=None, latest_run=run)
    snapshot = build_research_workflow_snapshot(**inputs)
    assert snapshot.next_action_code == "study_coordinate_unavailable"
    assert "morning_weight" in {action.params.concept for action in snapshot.next_actions}

    recorded.add("morning_weight")
    snapshot = build_research_workflow_snapshot(**inputs)
    assert snapshot.next_action_code == "failed_pipeline_requires_fresh_plan"
    assert [action.kind for action in snapshot.next_actions] == ["edit_plan"]
    assert snapshot.next_actions[0].params.change_question is None


@pytest.mark.parametrize("wrapped", [False, True])
def test_an_unrecognized_coordinate_is_rechecked_after_the_dictionary_changes(monkeypatch, wrapped):
    from easyicu.research_agent import concept_availability
    from easyicu.webserver import study_contexts
    from tests.webserver.copilot.research_workflow_fixtures import complete_study

    study = {**complete_study(), "question": "Does the morning humidity index vary by ward?"}
    coordinate = "morning_humidity_index"
    stop = "study_coordinate_unrecognized"
    available = False
    monkeypatch.setattr(
        concept_availability, "explain_concept_availability",
        lambda **_kw: SimpleNamespace(available=available, reason="recorded" if available else "concept_not_found"),
    )
    run = {
        "run_id": "morning_humidity", "run_type": "full", "run_status": "failed",
        "engine": "easyicu.research_agent.pipeline", "gate_status": "blocked",
        "gate_reason": "data_foundation_blocked" if wrapped else stop,
        "gate_detail": {
            "missing_concepts": [coordinate],
            **({"reason_code": stop} if wrapped else {}),
        },
        "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(study),
        "artifact_names": ["source_run_manifest.json"],
    }
    inputs = dict(study=study, active_export_present=True, active_job=None, latest_run=run)
    blocked = build_research_workflow_snapshot(**inputs)
    assert blocked.next_action_code == stop
    assert [(choice.kind, choice.params.change_question) for choice in blocked.next_actions] == [
        ("edit_plan", True),
    ]
    available = True
    ready = build_research_workflow_snapshot(**inputs)
    assert ready.next_action_code == "failed_pipeline_requires_fresh_plan"
    assert [(choice.kind, choice.params.change_question) for choice in ready.next_actions] == [
        ("edit_plan", None),
    ]


@pytest.mark.parametrize("shape", ["wrapped", "gate_reason", "gate_reason_code"])
@pytest.mark.parametrize("state", ["available", "futile", "concept_not_found", "superseded"])
def test_a_source_stop_is_recomputed_for_the_current_study(monkeypatch, shape, state):
    from easyicu.research_agent import concept_availability
    from easyicu.webserver import study_contexts
    from tests.webserver.copilot.research_workflow_fixtures import complete_study

    study = {**complete_study(), "question": "Do dusk assay readings move with the shift schedule?"}
    run = {
        "run_id": "dusk_assay", "run_type": "full", "run_status": "failed",
        "engine": "easyicu.research_agent.pipeline", "gate_status": "blocked",
        "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(study),
        "artifact_names": ["agent_plan.json", "evidence_ledger.json", "source_run_manifest.json"],
        "gate_detail": {
            "missing_concepts": ["dusk_assay"],
            "message": "The assay recorded after handover uses a different codebook.",
        },
    }
    if shape == "wrapped":
        run["gate_reason"] = "research_agent_pipeline_failed_closed"
        run["gate_detail"].update(
            reason_code="deterministic_standard_executor_failed_closed",
            cause_code="study_coordinate_unavailable",
        )
    else:
        run[shape] = "study_coordinate_unavailable"
    available = False
    reason = "absent_from_source"
    monkeypatch.setattr(
        concept_availability, "explain_concept_availability",
        lambda **_kw: SimpleNamespace(available=available, reason=reason),
    )
    monkeypatch.setattr(owner, "concept_selection_policy", lambda _concept: None)
    inputs = dict(study=study, active_export_present=True, active_job=None, latest_run=run,
                  execution_retry={"state": state})
    assert build_research_workflow_snapshot(**inputs).next_action_code == "study_coordinate_unavailable"

    if state == "superseded":
        inputs["study"] = {**study, "question": "Which assay has the most complete evening records?"}
    else:
        available = state != "concept_not_found"
        reason = "recorded" if available else "concept_not_found"
    snapshot = build_research_workflow_snapshot(**inputs)
    if state == "concept_not_found":
        assert snapshot.next_action_code == "study_coordinate_unrecognized"
        assert [(action.kind, action.params.change_question) for action in snapshot.next_actions] == [
            ("edit_plan", True),
        ]
    elif shape == "wrapped" and state in {"available", "futile"}:
        retry = state == "available"
        assert snapshot.next_action_code == (
            "failed_pipeline_execution_retry_available" if retry else "failed_pipeline_execution_retry_futile"
        )
        assert [action.kind for action in snapshot.next_actions] == (["retry", "edit_plan"] if retry else ["edit_plan"])
    else:
        assert snapshot.next_action_code == "failed_pipeline_requires_fresh_plan"
        assert [action.kind for action in snapshot.next_actions] == ["edit_plan"]


def test_a_wrapped_retry_owns_the_stop_even_with_unfinished_work_detail():
    gate = {
        "status": "blocked", "reason": "research_agent_pipeline_failed_closed",
        "detail": {
            "reason_code": "research_pipeline_provider_timeout", "cause_code": "study_coordinate_unavailable",
            "intake_error_code": "export_time_window_unbounded",
            "step_id": "match_dusk_assays", "element": "dusk_calibration",
            "missing_concepts": ["dusk_assay"],
            "message": "The model timed out while reconciling the two evening sampling windows.",
        },
    }
    assert [choice["kind"] for choice in owner.project_stop_next_actions(gate)] == ["retry"]
    choices = owner.project_stop_next_actions(gate, execution_retry_available=True)
    assert [(choice["kind"], choice["params"].get("stage")) for choice in choices] == [
        ("retry", "execution"), ("edit_plan", None),
    ]


def test_typed_missing_data_survives_the_exception_receipt(tmp_path) -> None:
    stop = ResearchPipelineRunError(
        "research_pipeline_required_concept_without_values",
        "Synthetic text.",
        details={"required_concepts": ["synthetic_marker"], "database": "miiv"},
    )
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path,
        study={"id": "synthetic_study"},
        provider={},
        exc=stop,
        code=stop.code,
        execution_retry_id=None,
    )
    gate = json.loads((tmp_path / "quality_gate.json").read_text())["gate"]
    assert gate["next_actions"][0]["kind"] == "drop_variable"
    assert gate["next_actions"][0]["params"]["concept"] == "synthetic_marker"


@pytest.mark.parametrize(
    "code, payload",
    [
        (
            "research_pipeline_plan_outcome_materialization_unavailable",
            {"missing_source_concepts": ["synthetic_marker"]},
        ),
        (
            "research_pipeline_cohort_concept_unavailable",
            {"concept_id": "synthetic_marker"},
        ),
        (
            "research_pipeline_target_outcome_materialization_unavailable",
            {"source_concept": "synthetic_marker"},
        ),
    ],
)
def test_source_missing_coordinates_reach_the_next_action(
    tmp_path, code, payload
) -> None:
    stop = ResearchPipelineRunError(
        code, "Synthetic text names no replacement.", details=payload
    )
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path,
        study={"id": "synthetic_study"},
        provider={},
        exc=stop,
        code=stop.code,
        execution_retry_id=None,
    )
    gate = json.loads((tmp_path / "quality_gate.json").read_text())["gate"]
    assert gate["next_actions"][0]["kind"] == "drop_variable"
    assert gate["next_actions"][0]["params"]["concept"] == "synthetic_marker"


def test_source_missing_coordinates_come_from_the_actual_catalog_owner(
    tmp_path,
) -> None:
    with pytest.raises(ResearchPipelineRunError) as caught:
        agent_pipeline_runs._resolve_planning_outcome_columns(
            source_concepts=["synthetic_endpoint"],
            acquisition=SimpleNamespace(
                selection=SimpleNamespace(selected_concepts=["synthetic_other"])
            ),
        )
    stop = caught.value
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path,
        study={"id": "synthetic_study"},
        provider={},
        exc=stop,
        code=stop.code,
        execution_retry_id=None,
    )
    gate = json.loads((tmp_path / "quality_gate.json").read_text())["gate"]
    assert gate["next_actions"][0]["kind"] == "drop_variable"
    assert gate["next_actions"][0]["params"]["concept"] == "synthetic_endpoint"


def test_provider_stops_keep_retry_even_when_other_details_suggest_removal() -> None:
    actions = owner.next_actions_for_stop(
        "research_pipeline_planner_provider_unavailable",
        {
            "reason_code": "missing_data",
            "question_required": False,
            "element": "synthetic_sensitivity",
            "missing_concepts": ["synthetic_marker"],
        },
    )
    assert [action.kind for action in actions] == ["retry"]
    assert actions[0].params.route == "retry"


@pytest.mark.parametrize(
    "reason, family, key",
    [
        ("study_family_database_out_of_scope", "survival", "v1_scope.study_family"),
        (
            "target_trial_database_out_of_scope",
            "causal_inference",
            "v1_scope.target_trial",
        ),
    ],
)
def test_scope_choices_use_the_scope_owner(reason, family, key) -> None:
    from easyicu.webserver.research_capability_scope import eligible_family_databases

    actions = owner.next_actions_for_stop(reason, {"family": family}, database="eicu")
    assert [action.kind for action in actions] == ["rephrase_in_scope"]
    assert actions[0].params.scope_owner == "easyicu.webserver.study_family_scope"
    assert actions[0].params.scope_label_key == key
    assert actions[0].params.database == "eicu"
    assert actions[0].params.supported_databases == eligible_family_databases(family)


def test_a_scope_refusal_before_any_run_still_offers_a_conversation_choice() -> None:
    snapshot = build_research_workflow_snapshot(
        study={
            "id": "synthetic_study",
            "question": "Study the synthetic marker.",
            "data_source": {"path": "/prepared/synthetic", "database": "eicu"},
            "analysis_design": {
                "analysis_family": "survival",
                "analysis_unit": "icu_stay",
                "variance_estimator": "model_based",
            },
        },
        active_export_present=True,
        active_job=None,
        latest_run=None,
    )
    assert snapshot.next_action_code == "study_family_database_out_of_scope"
    assert [action.kind for action in snapshot.next_actions] == ["rephrase_in_scope"]
    assert snapshot.next_actions[0].params.family == "survival"


def test_choices_reach_run_status_and_both_conversation_history_readers() -> None:
    gate = {
        "status": "blocked",
        "reason": "research_pipeline_planner_provider_unavailable",
    }
    actions = owner.project_stop_next_actions(gate)
    row = {
        "run_id": "synthetic_run",
        "gate_status": "blocked",
        "gate_reason": gate["reason"],
    }
    assert RunGate.from_payload(gate).to_dict()["next_actions"] == actions
    assert project_run_row(row)["next_actions"] == actions
    assert project_run_history([row])[0]["next_actions"] == actions


def test_an_alternative_is_rechecked_from_its_source_when_the_record_is_read(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        owner,
        "concept_selection_policy",
        lambda _concept: SimpleNamespace(canonical_alternative="synthetic_alternative"),
    )
    calls = []

    def availability(**kw):
        calls.append(kw)
        return SimpleNamespace(available=kw["database"] == "miiv")

    monkeypatch.setattr(owner, "explain_concept_availability", availability)
    gate = {
        "status": "blocked",
        "reason": "data_foundation_blocked",
        "detail": {"missing_concepts": ["synthetic_marker"]},
    }
    recorded = owner.project_stop_next_actions(gate, database="miiv")
    assert recorded[0]["kind"] == "use_alternative_concept"
    assert (
        owner.project_stop_next_actions({**gate, "next_actions": recorded}) == recorded
    )
    assert calls[-1]["database"] == "miiv"
    changed = {
        **gate,
        "detail": {**gate["detail"], "database": "eicu"},
        "next_actions": recorded,
    }
    assert owner.project_stop_next_actions(changed)[0]["kind"] == "drop_variable"


def test_recorded_parameters_do_not_replace_the_stops_typed_coordinates() -> None:
    recorded = owner.project_stop_next_actions(
        {
            "status": "blocked",
            "reason": "data_foundation_blocked",
            "detail": {"missing_concepts": ["synthetic_old_marker"]},
        }
    )
    actions = owner.project_stop_next_actions(
        {
            "status": "blocked",
            "reason": "data_foundation_blocked",
            "detail": {"missing_concepts": ["synthetic_current_marker"]},
            "next_actions": recorded,
        }
    )
    assert actions[0]["params"]["concept"] == "synthetic_current_marker"


def test_normal_approval_and_success_offer_no_stop_choices() -> None:
    for status in ("analysis_only", "running", "complete"):
        assert (
            owner.project_stop_next_actions(
                {"status": status, "reason": "synthetic_failure"}
            )
            == []
        )
    gate = {"status": "blocked", "reason": "human_plan_review_required"}
    assert (
        owner.project_stop_next_actions(
            gate, pending_reason_codes=["operator_plan_approval_required"]
        )
        == []
    )
    actions = owner.project_stop_next_actions(
        gate, pending_reason_codes=["agent_plan_revision_nonconvergent"]
    )
    assert [action["kind"] for action in actions] == ["edit_plan"]


def _approval_snapshot(reason="operator_plan_approval_required", detail=None):
    from easyicu.webserver import study_contexts
    from tests.webserver.copilot.research_workflow_fixtures import complete_study

    study = complete_study()
    return build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_type": "full",
            "run_id": "synthetic_run",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "gate_reason": "human_plan_review_required",
            "run_status": "human_review_pending",
            "pending_review_reason_codes": [reason],
            "gate_detail": detail,
            "artifact_names": [
                "agent_plan.json",
                "evidence_ledger.json",
                "source_run_manifest.json",
            ],
        },
        plan_review_authority={
            "run_id": "synthetic_run",
            "resumable_here": True,
            "plan_approval_allowed": True,
            "research_input_state": "prepared",
            "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(
                study
            ),
        },
    )


def test_an_ordinary_approval_snapshot_has_no_stop_choices() -> None:
    snapshot = _approval_snapshot()
    assert snapshot.next_action_code == "operator_plan_approval_required"
    assert snapshot.next_actions == []


@pytest.mark.parametrize("required", [True, False])
def test_a_nonconvergent_snapshot_keeps_the_stops_typed_optionality(required) -> None:
    snapshot = _approval_snapshot(
        "agent_plan_revision_nonconvergent",
        {
            "element": "synthetic_sensitivity",
            "concept": "synthetic_marker",
            "question_required": required,
        },
    )
    assert snapshot.next_action_code == "agent_plan_revision_nonconvergent"
    assert [action.kind for action in snapshot.next_actions] == (
        ["edit_plan"] if required else ["rerun_without_element", "edit_plan"]
    )
    assert snapshot.next_actions[0].params.concept == "synthetic_marker"


def test_a_recorded_choice_cannot_remove_a_required_element_or_replace_retry() -> None:
    optional = {
        "element": "synthetic_sensitivity",
        "concept": "synthetic_marker",
        "question_required": False,
    }
    gate = {
        "status": "blocked",
        "reason": "research_pipeline_progressive_compile_failed",
        "detail": optional,
    }
    recorded = owner.project_stop_next_actions(gate)
    gate["next_actions"] = recorded
    gate["detail"] = {**optional, "question_required": True}
    assert [action["kind"] for action in owner.project_stop_next_actions(gate)] == [
        "edit_plan"
    ]
    gate["reason"] = "research_pipeline_planner_provider_unavailable"
    assert [action["kind"] for action in owner.project_stop_next_actions(gate)] == [
        "retry"
    ]


@pytest.mark.parametrize(
    "kind", ["rerun_without_element", "drop_variable", "use_alternative_concept"]
)
def test_plan_changing_choices_require_explicit_edit_and_fresh_approval(kind) -> None:
    action = owner._action(
        kind,
        owner.PlanEditParams(
            element="synthetic_sensitivity",
            concept="synthetic_marker",
            alternative_concept="synthetic_alternative",
            database="miiv",
        ),
    )
    params = action.params
    assert params.route == "plan_edit"
    assert params.requires_explicit_choice is True
    assert params.requires_plan_approval is True
    assert params.intent == "candidate_plan"
    assert params.planner_start_mode == "fresh"
    for field, value in [
        ("intent", "reviewed_analysis"),
        ("requires_plan_approval", False),
        ("route", "execute"),
    ]:
        with pytest.raises(ValidationError):
            owner.PlanEditParams.model_validate({**params.model_dump(), field: value})
    # The pre-existing submission owner rejects sending a plan amendment into
    # reviewed analysis even when it came from a typed next-action button.
    from easyicu.webserver.plan_change_request import PlanChangeRequest

    amendment = PlanChangeRequest(
        source_run_id="synthetic_run", user_message="Synthetic user edit."
    )
    candidate = ResearchRunSubmissionRequest(
        study_context_id="synthetic_study",
        provider="mock",
        credential_source="scientific_provider",
        external_llm_opt_in=False,
        intent=params.intent,
        planner_start_mode=params.planner_start_mode,
        plan_change_request=amendment,
    )
    assert candidate.intent == "candidate_plan"
    assert candidate.plan_change_request is amendment
    with pytest.raises(ValidationError, match="plan_changes_require_fresh_candidate"):
        ResearchRunSubmissionRequest(
            study_context_id="synthetic_study",
            provider="mock",
            credential_source="scientific_provider",
            external_llm_opt_in=False,
            intent="reviewed_analysis",
            plan_change_request=amendment,
        )


def test_choices_are_stable_bounded_and_do_not_forward_arbitrary_payloads() -> None:
    detail = {
        "element": "synthetic_sensitivity",
        "question_required": False,
        "concept": "synthetic_marker",
        "message": "A synthetic message.",
        "secret": "synthetic_do_not_forward",
        "step_id": "../synthetic_path",
    }
    first = owner.next_actions_for_stop(
        "research_pipeline_progressive_compile_failed", detail
    )
    second = owner.next_actions_for_stop(
        "research_pipeline_progressive_compile_failed", detail
    )
    assert first == second
    assert len({action.action_id for action in first}) == len(first)
    assert "synthetic_do_not_forward" not in first[0].model_dump_json()
    assert first[0].params.step_id is None
    assert (
        owner.safe_stop_coordinates({"question_required": 0, "element": "Not a code!"})
        == {}
    )
    assert owner.next_actions_for_stop("not a code") == []
    assert [
        action.kind
        for action in owner.next_actions_for_stop("synthetic_historical_stop")
    ] == ["edit_plan"]
