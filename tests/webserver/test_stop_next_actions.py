"""Stops offer explicit choices without granting analysis or changing a plan."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
)
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver import agent_runs, stop_next_actions as owner
from easyicu.webserver.pi_copilot.projections import (
    project_job,
    project_run_outcome,
    project_run_row,
)
from easyicu.webserver.pi_copilot.workflow import (
    build_research_workflow_snapshot,
    project_run_history,
)
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from easyicu.webserver.research_run_submission import ResearchRunSubmissionRequest
from easyicu.webserver.run_record import RunGate


def test_uncleared_provider_offers_only_settings_without_execution_authority() -> None:
    reason = "provider_not_cleared_for_credentialed_data"
    gate = {
        "status": "blocked", "reason": reason,
        "detail": {"reason_code": "concept_unavailable", "missing_concepts": ["age"]},
    }
    projected = owner.project_stop_next_actions(gate)
    assert len(projected) == 1
    action = owner.StopNextAction.model_validate(projected[0])
    assert action.kind == "open_provider_settings"
    assert action.label_key == "stop_next_actions.open_provider_settings"
    assert action.params.model_dump() == {"route": "provider_settings"}
    assert reason in owner.STOP_POLICIES
    assert owner.project_stop_next_actions(gate) == projected
    assert project_job({"id": "fixture", "status": "failed", "error": reason})["next_actions"] == projected
    with pytest.raises(ValidationError):
        action.params.route = "retry"


@pytest.mark.parametrize("params", [{"route": "retry"}, {"route": "plan_edit"},
                                    {"route": "provider_settings", "execute": True}])
def test_provider_settings_choice_rejects_execution_and_mismatched_params(params) -> None:
    with pytest.raises(ValidationError):
        owner.StopNextAction.model_validate({
            "action_id": "open_provider_settings_0123456789abcdef",
            "kind": "open_provider_settings",
            "label_key": "stop_next_actions.open_provider_settings",
            "params": params,
        })


def test_provider_settings_params_cannot_masquerade_as_retry() -> None:
    with pytest.raises(ValidationError):
        owner.StopNextAction.model_validate({
            "action_id": "retry_0123456789abcdef", "kind": "retry",
            "label_key": "stop_next_actions.retry", "params": {"route": "provider_settings"},
        })


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

    root = Path(agent_pipeline_runs.__file__).parent
    codes = set()

    def literals(expression):
        if isinstance(expression, ast.IfExp):
            return [*literals(expression.body), *literals(expression.orelse)]
        return [expression]

    for path in root.rglob("*.py"):
        if "meta_generalization" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ResearchPipelineRunError"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                codes.add(node.args[0].value)
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
            STUDY_FAMILY_DATABASE_OUT_OF_SCOPE,
            TARGET_TRIAL_DATABASE_OUT_OF_SCOPE,
        )
    )
    return codes - {
        "full_agent_complete_human_signoff_required",
        "preflight_complete_human_signoff_required",
        "research_agent_pipeline_complete_human_interpretation_required",
    }


def test_each_terminal_code_from_the_owners_has_an_explicit_disposition() -> None:
    codes = _owned_terminal_codes()
    assert len(codes) > 100
    assert codes <= owner.STOP_POLICIES.keys(), sorted(
        codes - owner.STOP_POLICIES.keys()
    )
    for code in sorted(codes):
        actions = owner.next_actions_for_stop(code)
        assert actions, code
        assert all(action.action_id and action.label_key for action in actions)


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
    actions = owner.next_actions_for_stop(reason, {"family": family}, database="eicu")
    assert [action.kind for action in actions] == ["rephrase_in_scope"]
    assert actions[0].params.scope_owner == "easyicu.webserver.study_family_scope"
    assert actions[0].params.scope_label_key == key
    assert actions[0].params.database == "eicu"
    assert actions[0].params.supported_databases == ("miiv",)


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
