"""Prepared plan repair composes existing authorities; it grants no approval."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.run_input import RunInputIdentityError
from easyicu.research_agent.planning.scientific_review import (
    PlanScientificFinding,
    PlanScientificReview,
)
from easyicu.webserver import research_plan_revision as owner


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    study = {
        "id": "study-1",
        "question": "Describe sepsis",
        "data_source": {"path": "/prepared", "database": "miiv"},
    }
    digest = owner.study_contexts.scientific_configuration_sha256(study)
    wrapper = tmp_path / "study-1" / "run_source"
    run_dir = wrapper / "pipeline" / "run_inner"
    run_dir.mkdir(parents=True)
    capsule = run_dir / "run_input_capsule.json"
    capsule.write_text(json.dumps({"scientific_identity": {"target_outcome": "death"}}))
    review = PlanScientificReview(
        status="changes_required",
        approval_allowed=False,
        top_journal_candidate=False,
        score=74,
        dimension_scores={"content_completeness": 45},
        findings=[
            PlanScientificFinding(
                code="ACCEPTED_BASELINE_CONTENT_MISSING",
                severity="blocker",
                dimension="content_completeness",
                message="Accepted baseline is missing.",
                remediation="Restore the bound content.",
                remediation_route="agent_plan_revision",
            )
        ],
        context_sha256="a" * 64,
        plan_sha256="b" * 64,
        literature_sha256="c" * 64,
        figure_strategy_sha256="d" * 64,
        generated_at="2026-09-07T00:00:00Z",
    ).model_dump(mode="json")
    row = {
        "run_id": "run_inner",
        "project_dir": str(wrapper),
        "scientific_configuration_sha256": digest,
        "research_input_state": "prepared",
    }
    manifest = {"research_input_state": "prepared"}
    record = SimpleNamespace(
        artifact_payloads={
            "source_run_manifest.json": manifest,
            "scientific_plan_review.json": review,
        }
    )
    seed = SimpleNamespace(
        schema_version="easyicu.web-review-recovery-seed/4",
        budget_mode="full_reviewed",
        prepared_package_binding={"sha256": "e" * 64},
        study=study,
        scientific_configuration_sha256=digest,
        pipeline_config={},
        pipeline_config_sha256="f" * 64,
    )
    config = SimpleNamespace(
        require_human_plan_review=True,
        workdir=run_dir.parent,
        bound_plan_revision_contract="Preserve age, sex, admission type and Charlson.",
        required_primary_cohort_selection_mode="all_input_rows",
        bound_analysis_inputs=None,
    )
    checkpoint = SimpleNamespace(
        run_id="run_inner",
        pipeline_config_sha256="f" * 64,
        approved_decisions=[],
        execution_start_receipt=None,
        run_input_capsule_sha256=hashlib.sha256(capsule.read_bytes()).hexdigest(),
    )
    calls = []
    monkeypatch.setattr(
        owner.agent_runs, "list_run_history", lambda **kw: {"runs": [row]}
    )
    monkeypatch.setattr(owner.agent_runs, "read_run_record", lambda path: record)
    monkeypatch.setattr(owner, "load_recovery_seed", lambda path: seed)
    monkeypatch.setattr(
        owner.PipelineConfig, "from_recovery_payload", lambda *a, **kw: config
    )
    monkeypatch.setattr(owner, "load_checkpoint", lambda path, **kw: checkpoint)
    monkeypatch.setattr(
        owner,
        "load_verified_run_input_capsule",
        lambda **kw: calls.append(("input", kw)),
    )
    monkeypatch.setattr(owner, "research_input_state", lambda path: "prepared")
    monkeypatch.setattr(
        owner.dataio,
        "validate_research_pipeline_source",
        lambda *a, **kw: calls.append(("package", kw)),
    )
    return SimpleNamespace(
        study=study,
        root=tmp_path,
        wrapper=wrapper,
        run_dir=run_dir,
        capsule=capsule,
        row=row,
        record=record,
        manifest=manifest,
        seed=seed,
        config=config,
        checkpoint=checkpoint,
        review=review,
        calls=calls,
    )


def load(source: SimpleNamespace):
    return owner.load_prepared_plan_revision(
        study=source.study, project_root=str(source.root), source_run_id="run_inner"
    )


@pytest.fixture
def failed_source(source):
    import pandas as pd
    from tests.research_agent.planning.scientific_review_fixtures import _traditional_table_one_step

    step = _traditional_table_one_step().model_dump(mode="json")
    names = [v["name"] for v in step["table_one_spec"]["variables"]]
    group = step["table_one_spec"]["group_by"]
    plan = {"steps": [step, {
        "step_id": "risk", "population_scope": "primary_model",
        "expected_outputs": ["table:risk_context"],
    }]}
    source.review.update(approval_allowed=True, findings=[], plan_sha256=owner.canonical_sha256(plan))
    source.record.artifact_payloads["agent_plan.json"] = plan
    source.row.update(run_status="blocked", gate_reason="research_agent_pipeline_failed_closed")
    source.checkpoint.state = "completed"
    source.checkpoint.approved_decisions = [{"decision": "approved"}]
    source.checkpoint.execution_start_receipt = {"existing": "receipt"}
    source.checkpoint.plan_handoff = {"plan": plan}
    source.capsule.write_text(json.dumps({
        "scientific_identity": {"target_outcome": "death"},
        "cohort_relative_path": "cohort.parquet",
    }))
    source.checkpoint.run_input_capsule_sha256 = hashlib.sha256(source.capsule.read_bytes()).hexdigest()
    pd.DataFrame({name: [] for name in [group, *names]}).to_parquet(source.run_dir / "cohort.parquet")
    source.status_path = source.run_dir / "run_status.json"
    source.status_path.write_text(json.dumps({"gates": {"failed_steps": [{"step_id": "baseline"}]}}))
    return source


def test_failed_execution_replan_preserves_source_inputs_and_full_requirements(failed_source):
    source = failed_source
    before = source.capsule.read_bytes()
    result = load(source)
    assert result.failed_execution_replan is True
    assert result.run_dir == source.run_dir
    assert result.budget_mode == "full_reviewed"
    assert result.baseline_requirements.source_plan_sha256 == source.review["plan_sha256"]
    assert result.population_requirements.populations[0].population_scope == "primary_model"
    assert json.loads(result.prior_plan_contract.split("- source_plan_json: ")[1]) == source.record.artifact_payloads["agent_plan.json"]
    assert source.capsule.read_bytes() == before
    assert source.checkpoint.approved_decisions == [{"decision": "approved"}]
    assert not hasattr(result, "approved_decisions")
    assert not hasattr(result, "runner_image")


@pytest.mark.parametrize("fixture", ["source", "failed_source"])
def test_a_repair_keeps_the_accepted_candidates_primary_inputs(request, fixture) -> None:
    """Dev9 M3: the first package-bound plan kept its ten reviewed features, but
    its automatic repair re-planned without them and the phenotyping family was
    refused ("needs selectable feature candidates") before any Provider call."""

    source = request.getfixturevalue(fixture)
    accepted = {
        "schema_version": "easyicu.accepted_analysis_inputs/1",
        "source_plan_sha256": "b" * 64,
        "source_step_ids": ["primary_cluster_solution"],
        "concepts": ["hr", "wbc", "crea"],
    }
    source.config.bound_analysis_inputs = accepted

    result = load(source)

    assert result.failed_execution_replan is (fixture == "failed_source")
    assert result.analysis_inputs.model_dump(mode="json") == accepted
    source.config.bound_analysis_inputs = None
    assert load(source).analysis_inputs is None


@pytest.mark.parametrize("mutation", [
    "running", "successful", "other_gate", "no_failure", "unapproved", "no_execution",
    "rejected", "checkpoint_running", "plan", "handoff", "missing_column", "capsule",
])
def test_failed_execution_replan_rejects_unbound_or_nonterminal_source(failed_source, mutation):
    import pandas as pd

    source = failed_source
    if mutation == "running":
        source.row["run_status"] = "running"
    elif mutation == "successful":
        source.row["run_status"] = "completed"
    elif mutation == "other_gate":
        source.row["gate_reason"] = "independent_signoff_required"
    elif mutation == "no_failure":
        source.status_path.write_text('{"gates":{"execution_complete":true}}')
    elif mutation == "unapproved":
        source.checkpoint.approved_decisions = []
    elif mutation == "no_execution":
        source.checkpoint.execution_start_receipt = None
    elif mutation == "rejected":
        source.checkpoint.approved_decisions = [{"decision": "rejected"}]
    elif mutation == "checkpoint_running":
        source.checkpoint.state = "executing"
    elif mutation == "plan":
        source.review["plan_sha256"] = "0" * 64
    elif mutation == "handoff":
        source.checkpoint.plan_handoff = {"plan": {"steps": []}}
    elif mutation == "missing_column":
        pd.DataFrame({"unrelated": []}).to_parquet(source.run_dir / "cohort.parquet")
    elif mutation == "capsule":
        source.capsule.write_text("{}")
    with pytest.raises(owner.ResearchPipelineRunError) as raised:
        load(source)
    assert raised.value.code == "prepared_plan_revision_source_invalid"


@pytest.mark.parametrize("nonconvergent", [False, True])
def test_prepared_repair_reuses_bound_scope_without_consuming_review(source, nonconvergent) -> None:
    if nonconvergent:
        source.manifest["plan_revision_nonconvergent"] = True
    original_manifest = dict(source.manifest)
    before = source.capsule.read_bytes()
    result = load(source)
    assert result.run_dir == source.run_dir
    assert result.budget_mode == "full_reviewed"
    assert result.required_primary_cohort_selection_mode == "all_input_rows"
    assert "Charlson" in result.prior_plan_contract
    assert source.calls == [
        (
            "input",
            {
                "run_dir": source.run_dir,
                "scientific_identity": {"target_outcome": "death"},
            },
        ),
        (
            "package",
            {
                "database": "miiv",
                "expected_binding": source.seed.prepared_package_binding,
            },
        ),
    ]
    assert source.checkpoint.approved_decisions == []
    assert source.checkpoint.execution_start_receipt is None
    assert source.capsule.read_bytes() == before
    assert source.manifest == original_manifest


@pytest.mark.parametrize("state", ["metadata_only", "unavailable", None])
def test_unprepared_history_cannot_promote_a_candidate(source, state) -> None:
    source.row["research_input_state"] = state
    source.manifest["research_input_state"] = state
    assert load(source) is None
    assert source.calls == []


@pytest.mark.parametrize(
    "mutation",
    [
        "study",
        "legacy_seed",
        "budget",
        "package",
        "seed_study",
        "seed_digest",
        "no_review",
        "workdir",
        "checkpoint_run",
        "checkpoint_config",
        "approval",
        "execution",
        "capsule",
        "receipt",
        "scope",
        "approvable",
        "owner",
    ],
)
def test_prepared_repair_rejects_drift_instead_of_falling_back(
    source, mutation
) -> None:
    if mutation == "study":
        source.row["scientific_configuration_sha256"] = "x" * 64
    elif mutation == "legacy_seed":
        source.seed.schema_version = "easyicu.web-review-recovery-seed/1"
    elif mutation == "budget":
        source.seed.budget_mode = "planner_canary"
    elif mutation == "package":
        source.seed.prepared_package_binding = None
    elif mutation == "seed_study":
        source.seed.study = {**source.study, "id": "another-study"}
    elif mutation == "seed_digest":
        source.seed.scientific_configuration_sha256 = "x" * 64
    elif mutation == "no_review":
        source.config.require_human_plan_review = False
    elif mutation == "workdir":
        source.config.workdir = source.root / "another-run"
    elif mutation == "checkpoint_run":
        source.checkpoint.run_id = "another-run"
    elif mutation == "checkpoint_config":
        source.checkpoint.pipeline_config_sha256 = "x" * 64
    elif mutation == "approval":
        source.checkpoint.approved_decisions = [{"decision": "approved"}]
    elif mutation == "execution":
        source.checkpoint.execution_start_receipt = {"run_id": "run_inner"}
    elif mutation == "capsule":
        source.capsule.write_text("{}")
    elif mutation == "receipt":
        source.manifest.clear()
    elif mutation == "scope":
        source.root = source.root / "another-project"
    elif mutation == "approvable":
        source.review["approval_allowed"] = True
    elif mutation == "owner":
        source.review["findings"][0]["remediation_route"] = "runtime_capability"
    with pytest.raises(owner.ResearchPipelineRunError) as raised:
        load(source)
    assert raised.value.code == "prepared_plan_revision_source_invalid"
    assert source.calls == []


@pytest.mark.parametrize("boundary", ["checkpoint", "input", "package", "physical"])
def test_owner_validation_failure_cannot_be_downgraded_to_metadata(
    source, monkeypatch, boundary
) -> None:
    def invalid(*a, **kw):
        raise RunInputIdentityError("private input path must not escape")

    if boundary == "checkpoint":
        monkeypatch.setattr(owner, "load_checkpoint", invalid)
    elif boundary == "input":
        monkeypatch.setattr(owner, "load_verified_run_input_capsule", invalid)
    elif boundary == "package":
        monkeypatch.setattr(owner.dataio, "validate_research_pipeline_source", invalid)
    else:
        monkeypatch.setattr(owner, "research_input_state", lambda path: "metadata_only")
    with pytest.raises(owner.ResearchPipelineRunError) as raised:
        load(source)
    assert "private input path" not in str(raised.value)


def test_symlinked_capsule_is_not_restoration_authority(source) -> None:
    real = source.run_dir / "capsule_real.json"
    source.capsule.rename(real)
    source.capsule.symlink_to(real)
    with pytest.raises(owner.ResearchPipelineRunError):
        load(source)


def test_no_source_never_inherits_a_neighbouring_prepared_run(source) -> None:
    assert (
        owner.load_prepared_plan_revision(
            study=source.study, project_root=str(source.root), source_run_id=""
        )
        is None
    )
    assert source.calls == []


def _revision_plan(*steps):
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    return AnalysisPlan(
        research_question="Relate the exposure to death in adult stays.",
        analysis_type="association_study",
        steps=[AnalysisStep(
            step_id="primary", planned_analysis_role="primary",
            intent="Estimate the requested association.", method="logistic_regression",
            inputs=["exposure", "death"], expected_outputs=["table:association"],
        ), *steps],
    )


def _revision_guard(plan, *findings, facts=None):
    review = PlanScientificReview(
        status="changes_required", approval_allowed=False, top_journal_candidate=False,
        score=50, dimension_scores={}, findings=list(findings), facts=facts or {},
        context_sha256="a" * 64, plan_sha256=owner.canonical_sha256(plan.model_dump(mode="json")),
        literature_sha256="c" * 64, figure_strategy_sha256="d" * 64,
        generated_at="2026-09-01T00:00:00Z",
    )
    return owner.PlanRevisionGuard.from_review(plan=plan.model_dump(mode="json"), review=review)


@pytest.mark.parametrize("method,role", [
    ("complete_case_sensitivity", "sensitivity"),
    ("restricted_cubic_spline_sensitivity", "sensitivity"),
    ("subgroup_analysis", "secondary"),
    ("ordinal_trend", "secondary"),
    ("e_value", "auxiliary"),
    ("multiple_imputation_sensitivity", "auxiliary"),
    ("visualization", "auxiliary"),
    ("extra_table", "auxiliary"),
])
def test_revision_rejects_unrequested_analysis_and_keeps_previous_plan(method, role):
    from easyicu.research_agent.schema import AnalysisStep
    from tests.research_agent.planning.scientific_review_fixtures import _context

    previous = _revision_plan()
    before = previous.model_dump(mode="json")
    guard = _revision_guard(previous)
    extra = AnalysisStep(
        step_id="new", planned_analysis_role=role, intent="Add an optional analysis.",
        method=method, inputs=["exposure", "death", "age"],
        expected_outputs=["table:extra"],
    )
    revised = previous.model_copy(update={"steps": [*previous.steps, extra]})
    with pytest.raises(owner.ResearchPipelineRunError) as raised:
        guard.validate_plan(plan=revised, context=_context())
    assert raised.value.code == "revision_added_unrequested_analysis"
    assert raised.value.details["module"] == method
    assert raised.value.details["role"] == role
    assert guard.previous_plan == before == previous.model_dump(mode="json")


def _revision_step(method="complete_case_sensitivity", role="sensitivity", **kwargs):
    from easyicu.research_agent.schema import AnalysisStep

    return AnalysisStep(**{
        "step_id": "extra", "planned_analysis_role": role, "intent": "Synthetic planned analysis.",
        "method": method, "inputs": ["exposure", "death", "age"],
        "expected_outputs": ["table:extra"], **kwargs,
    })


def _blocker(code, **kwargs):
    return PlanScientificFinding(**{
        "code": code, "severity": "blocker", "dimension": "statistical_design",
        "message": "A required module is missing.", "remediation": "Supply the required module.",
        "remediation_route": "agent_plan_revision", **kwargs,
    })


@pytest.mark.parametrize("change", ["remove", "rename", "reorder", "binding"])
def test_revision_allows_removal_and_binding_repair_without_using_step_ids(change):
    from tests.research_agent.planning.scientific_review_fixtures import _context, _plan

    typed = _plan().steps[0]
    producer = _revision_step(expected_outputs=["cohort:corrected"])
    previous = _revision_plan(producer).model_copy(update={"steps": [typed, producer]})
    steps = list(previous.steps)
    if change == "remove":
        steps.pop()
    elif change == "rename":
        steps = [step.model_copy(update={"step_id": f"renamed_{index}"}) for index, step in enumerate(steps)]
    elif change == "reorder":
        steps.reverse()
    else:
        steps[0] = typed.model_copy(update={"inputs": [*typed.inputs, "cohort:corrected"]})
    _revision_guard(previous).validate_plan(plan=previous.model_copy(update={"steps": steps}), context=_context())


@pytest.mark.parametrize("change", ["module", "role", "coordinates", "duplicate"])
def test_revision_same_ids_do_not_authorize_different_or_additional_science(change):
    from tests.research_agent.planning.scientific_review_fixtures import _context

    extra = _revision_step()
    previous = _revision_plan(extra)
    update = {
        "module": {"method": "subgroup_analysis"},
        "role": {"planned_analysis_role": "secondary"},
        "coordinates": {"inputs": ["exposure", "death", "other_source"]},
        "duplicate": {},
    }[change]
    steps = [previous.steps[0], extra.model_copy(update=update)]
    if change == "duplicate":
        steps.append(extra.model_copy(update={"step_id": "another_id"}))
    with pytest.raises(owner.ResearchPipelineRunError, match="unrequested"):
        _revision_guard(previous).validate_plan(plan=previous.model_copy(update={"steps": steps}), context=_context())


@pytest.mark.parametrize("authority", ["blocker", "question"])
def test_revision_allows_a_required_landmark(authority):
    from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec
    from tests.research_agent.planning.scientific_review_fixtures import _context

    previous = _revision_plan()
    context = _context()
    finding = _blocker("POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED")
    step = _revision_step("landmark_analysis")
    if authority == "question":
        context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(update={
            "sensitivity_specs": [PrespecifiedSensitivitySpec(
                spec_id="timing", axis="timing", strategy="landmark", landmark_hours=24,
            )],
        })})
        step = step.model_copy(update={"sensitivity_spec_ids": ["timing"]})
    guard = _revision_guard(previous, *([finding] if authority == "blocker" else []))
    guard.validate_plan(plan=_revision_plan(step), context=context)


@pytest.mark.parametrize("audit,missing", [("missingness_profile", "quality"), ("cohort_flow", "cohort")])
def test_revision_allows_only_the_audit_named_by_blocking_typed_remediation(audit, missing):
    from tests.research_agent.planning.scientific_review_fixtures import _context

    step = _revision_step("measurement_audit", "auxiliary", measurement_audit_spec={
        "products": [{"product_id": "extra", "audit": audit}],
    })
    guard = _revision_guard(_revision_plan(), _blocker("ARTICLE_CONTENT_ROLES_INCOMPLETE"),
                            facts={"content_roles": {"missing_roles": [missing]}})
    guard.validate_plan(plan=_revision_plan(step), context=_context())
    with pytest.raises(owner.ResearchPipelineRunError):
        guard.validate_plan(plan=_revision_plan(_revision_step("multiple_imputation_sensitivity")), context=_context())


@pytest.mark.parametrize("change", ["major", "other_owner", "authorization", "prose", "wrong_role"])
def test_revision_suggestions_and_unrelated_blockers_grant_no_additional_analysis(change):
    from tests.research_agent.planning.scientific_review_fixtures import _context

    kwargs = {
        "major": {"severity": "major"}, "other_owner": {"remediation_route": "runtime_capability"},
        "authorization": {"requires_user_authorization": True},
        "prose": {"code": "OTHER_BLOCKER", "remediation": "Add a landmark_analysis step."},
        "wrong_role": {},
    }[change]
    finding = _blocker("POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED", **kwargs) if change != "prose" else _blocker(**kwargs)
    step = _revision_step("landmark_analysis", "auxiliary" if change == "wrong_role" else "sensitivity")
    with pytest.raises(owner.ResearchPipelineRunError):
        _revision_guard(_revision_plan(), finding).validate_plan(plan=_revision_plan(step), context=_context())


def test_revision_allows_a_complete_case_refit_required_by_a_blocker():
    from easyicu.research_agent.planning.robustness_contract import RobustnessSpec
    from tests.research_agent.planning.scientific_review_fixtures import _context

    plan = _revision_plan(_revision_step()).model_copy(update={"robustness_specs": [RobustnessSpec(
        spec_id="cc", axis="missing", description="Required complete-case examination.",
        missing_override={"strategy": "complete_case", "variables": ["age"]},
    )]})
    guard = _revision_guard(_revision_plan(), _blocker("MISSING_CATEGORY_WITHOUT_COMPLETE_CASE_SENSITIVITY"))
    guard.validate_plan(plan=plan, context=_context())
    with pytest.raises(owner.ResearchPipelineRunError):
        _revision_guard(_revision_plan()).validate_plan(plan=plan, context=_context())
    with pytest.raises(owner.ResearchPipelineRunError):
        guard.validate_plan(plan=plan.model_copy(update={"steps": [plan.steps[0], _revision_step("multiple_imputation_sensitivity")]}), context=_context())


@pytest.mark.parametrize("stated", [True, False])
def test_revision_allows_a_subgroup_only_when_the_question_requires_it(stated):
    from tests.research_agent.planning.scientific_review_fixtures import _context

    context = _context()
    if stated:
        context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(update={
            "data_constraints": json.dumps({"stated_subgroups": [{
                "variable": "age", "concept": "age", "words": "within age groups", "source_status": "available",
            }]}),
        })})
    step = _revision_step("subgroup_analysis", "secondary", scientific_action_id="association.effect_modification")
    guard = _revision_guard(_revision_plan())
    if stated:
        guard.validate_plan(plan=_revision_plan(step), context=context)
    else:
        with pytest.raises(owner.ResearchPipelineRunError):
            guard.validate_plan(plan=_revision_plan(step), context=context)


def test_revision_rejects_growth_before_authority_or_any_plan_write(tmp_path):
    import inspect

    from easyicu.research_agent.orchestration.services import PipelineServices
    from easyicu.research_agent.pipeline import ResearchAgentPipeline
    from tests.research_agent.planning.scientific_review_fixtures import _context

    previous = _revision_plan()
    path = tmp_path / "analysis_plan.json"
    path.write_text(previous.model_dump_json())
    before = path.read_bytes()
    guard = _revision_guard(previous)
    pipeline = SimpleNamespace(
        _services=PipelineServices(plan_revision_guard=guard),
        _scientific_runtime_authorities=SimpleNamespace(current_case=None, trajectory=None),
    )
    method = ResearchAgentPipeline._validate_and_persist_plan
    kwargs = {name: None for name in inspect.signature(method).parameters if name != "self"}
    kwargs.update(generation=SimpleNamespace(plan=_revision_plan(_revision_step())), agent_context=_context())
    with pytest.raises(owner.ResearchPipelineRunError) as raised:
        method(pipeline, **kwargs)
    assert raised.value.code == "revision_added_unrequested_analysis"
    assert path.read_bytes() == before
    assert guard.previous_plan == previous.model_dump(mode="json")
    from easyicu.webserver import agent_pipeline_runs

    wrapper = tmp_path / "rejected"
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=wrapper, study={"id": "synthetic"}, provider={},
        exc=raised.value, code=raised.value.code, execution_retry_id=None,
    )
    assert json.loads((wrapper / "quality_gate.json").read_text())["gate"]["reason"] == "revision_added_unrequested_analysis"
    assert path.read_bytes() == before


def test_revision_guard_loads_only_the_reviewed_plan(source):
    previous = _revision_plan()
    source.record.artifact_payloads["agent_plan.json"] = previous.model_dump(mode="json")
    source.review["plan_sha256"] = owner.canonical_sha256(previous.model_dump(mode="json"))
    kwargs = dict(study=source.study, project_root=str(source.root), source_run_id="run_inner",
                  review=PlanScientificReview.model_validate(source.review))
    guard = owner.load_plan_revision_guard(**kwargs)
    assert guard.previous_plan == previous.model_dump(mode="json")
    assert "Repair or remove" in guard.prompt_contract()
    assert "source_plan_json: " + guard.source_plan_json in guard.prompt_contract()
    source.record.artifact_payloads["agent_plan.json"]["steps"].append(_revision_step().model_dump(mode="json"))
    with pytest.raises(owner.ResearchPipelineRunError) as raised:
        owner.load_plan_revision_guard(**kwargs)
    assert raised.value.code == "plan_revision_source_plan_mismatch"
    source.record.artifact_payloads.pop("agent_plan.json")
    with pytest.raises(owner.ResearchPipelineRunError) as raised:
        owner.load_plan_revision_guard(**kwargs)
    assert raised.value.code == "plan_revision_source_plan_missing"


@pytest.mark.parametrize("change", [None, "unknown_id", "wrong_method", "wrong_input", "host_suggestion", "wrong_target"])
def test_revision_sensitivity_permission_is_bound_to_the_question_and_exact_analysis(change):
    from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec
    from tests.research_agent.planning.scientific_review_fixtures import _context

    context = _context()
    spec = PrespecifiedSensitivitySpec(spec_id="required", axis="functional_form", strategy="restricted_cubic_spline",
                                     execution_variables=("age",))
    context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(update={"sensitivity_specs": [spec]})})
    step = _revision_step("restricted_cubic_spline_sensitivity", sensitivity_spec_ids=["required"],
                          functional_form_spec={"target_column": "age", "knot_quantiles": [0.1, 0.5, 0.9]})
    if change == "unknown_id":
        step = step.model_copy(update={"sensitivity_spec_ids": ["invented"]})
    elif change == "wrong_method":
        step = step.model_copy(update={"method": "multiple_imputation_sensitivity", "functional_form_spec": None})
    elif change == "wrong_input":
        step = step.model_copy(update={"inputs": ["exposure", "death"]})
    elif change == "wrong_target":
        step = step.model_copy(update={"functional_form_spec": step.functional_form_spec.model_copy(update={"target_column": "exposure"})})
    guard = _revision_guard(_revision_plan())
    if change == "host_suggestion":
        from dataclasses import replace
        guard = replace(guard, question_sensitivity_ids=frozenset())
    if change is None:
        guard.validate_plan(plan=_revision_plan(step), context=context)
    else:
        with pytest.raises(owner.ResearchPipelineRunError):
            guard.validate_plan(plan=_revision_plan(step), context=context)


@pytest.mark.parametrize("change", ["adjustment", "outcome", "repeated_outcome", "other_exposure", "sensitivity"])
def test_revision_model_permission_repairs_required_bindings_without_adding_optional_models(change):
    from tests.research_agent.planning.scientific_review_fixtures import _context, _plan

    primary = _plan().steps[0]
    previous = _revision_plan().model_copy(update={"steps": [primary]})
    requirement = primary.model_requirements[0]
    context = _context()
    if change == "adjustment":
        requirement = requirement.model_copy(update={"covariates": [], "model_terms": requirement.model_terms[:1]})
        previous = previous.model_copy(update={"steps": [primary.model_copy(update={"model_requirements": [requirement]})]})
        revised = _revision_plan().model_copy(update={"steps": [primary]})
    else:
        requirement = requirement.model_copy(update={
            "analysis_role": "sensitivity" if change == "sensitivity" else "secondary",
            "outcome": "another_outcome" if change in {"outcome", "other_exposure", "sensitivity"} else "death",
        })
        if change == "other_exposure":
            requirement = requirement.model_copy(update={"exposure_source": "other_exposure", "model_terms": [
                requirement.model_terms[0].model_copy(update={"name": "other_exposure"}), *requirement.model_terms[1:],
            ]})
        inputs = ["exposure", "death", "age", "another_outcome"]
        if change == "other_exposure":
            inputs.append("other_exposure")
        step = _revision_step(primary.method, requirement.analysis_role, inputs=inputs,
                              expected_outputs=primary.expected_outputs, model_requirements=[requirement])
        revised = previous.model_copy(update={"steps": [primary, step]})
        context = context.model_copy(update={"cohort": context.cohort.model_copy(update={"requested_outcome_columns": ["another_outcome"]})})
    guard = _revision_guard(previous)
    if change in {"adjustment", "outcome"}:
        guard.validate_plan(plan=revised, context=context)
    else:
        with pytest.raises(owner.ResearchPipelineRunError):
            guard.validate_plan(plan=revised, context=context)


def test_revision_preserves_ordered_coordinates():
    from tests.research_agent.planning.scientific_review_fixtures import _context, _plan

    primary = _plan().steps[0]
    previous = _revision_plan().model_copy(update={"steps": [primary]})
    # The coordinates of an optional step include the model's ordered levels;
    # changing them is new science even when its local ids stay unchanged.
    optional = primary.model_copy(update={"planned_analysis_role": "sensitivity"})
    previous = previous.model_copy(update={"steps": [optional]})
    requirement = optional.model_requirements[0]
    term = requirement.model_terms[0].model_copy(update={"levels": ["1", "0"]})
    changed = optional.model_copy(update={"model_requirements": [requirement.model_copy(update={"model_terms": [term, *requirement.model_terms[1:]]})]})
    with pytest.raises(owner.ResearchPipelineRunError):
        _revision_guard(previous).validate_plan(plan=previous.model_copy(update={"steps": [changed]}), context=_context())


def test_revision_refuses_an_optional_analysis_added_by_host_shaping(monkeypatch):
    import inspect

    from easyicu.research_agent import pipeline as module
    from easyicu.research_agent.orchestration.services import PipelineServices
    from tests.research_agent.planning.scientific_review_fixtures import _context

    previous = _revision_plan()
    calls = []
    class Runtime:
        current_case = trajectory = None

        def seal_for_plan(self, plan):
            calls.append("seal")
            return self

        def bind_plan(self, plan):
            calls.append("bind")
            return plan, []

    pipeline = SimpleNamespace(_services=PipelineServices(plan_revision_guard=_revision_guard(previous)),
                               _scientific_runtime_authorities=Runtime())
    monkeypatch.setattr(module, "_shape_fresh_plan", lambda **kw: _revision_plan(_revision_step()))
    method = module.ResearchAgentPipeline._validate_and_persist_plan
    kwargs = {name: None for name in inspect.signature(method).parameters if name != "self"}
    kwargs.update(generation=SimpleNamespace(
        plan=previous, reused_prior_plan=False, reused_plan_path=None, migrated_plan_path=None,
        plan_generation_mode="llm", used_mock_llm=False, planner_prompt_metrics=None,
    ), agent_context=_context(), findings=[])
    with pytest.raises(owner.ResearchPipelineRunError) as raised:
        method(pipeline, **kwargs)
    assert raised.value.code == "revision_added_unrequested_analysis"
    assert calls == ["seal", "seal", "bind"]


@pytest.mark.parametrize("runtime_kind", ["current_case", "trajectory"])
@pytest.mark.parametrize("extra_stage", [None, "draft", "shape", "runtime", "figure"])
def test_revision_compares_the_runtime_bound_plan_before_persisting(
    tmp_path, monkeypatch, runtime_kind, extra_stage,
):
    """A compiler-owned method change is not additional scientific scope."""
    import inspect

    from easyicu.research_agent import pipeline as module
    from easyicu.research_agent.orchestration.services import PipelineServices
    from tests.research_agent.planning.scientific_review_fixtures import _context

    previous = _revision_plan()
    before = previous.model_dump(mode="json")
    guard = _revision_guard(previous)
    draft = previous.model_copy(update={"steps": [previous.steps[0].model_copy(
        update={"method": "synthetic_planner_template"},
    )]})
    source_file = tmp_path / "analysis_plan.json"
    source_file.write_text(previous.model_dump_json())
    source_bytes = source_file.read_bytes()
    calls = []

    def add_optional(plan, stage):
        return plan.model_copy(update={"steps": [*plan.steps, _revision_step()]}) if extra_stage == stage else plan

    class Runtime:
        current_case = object() if runtime_kind == "current_case" else None
        trajectory = object() if runtime_kind == "trajectory" else None

        def seal_for_plan(self, plan):
            return self

        def bind_plan(self, plan):
            calls.append("runtime_binding")
            normalized = plan.model_copy(update={"steps": [
                plan.steps[0].model_copy(update={"method": previous.steps[0].method}),
                *plan.steps[1:],
            ]})
            return add_optional(normalized, "runtime"), []

    class LaterValidation(Exception):
        pass

    def next_validation(*args, **kwargs):
        raise LaterValidation

    def bind_figures(plan, findings):
        calls.append("figure_binding")
        return add_optional(plan, "figure")

    monkeypatch.setattr(module, "_shape_fresh_plan", lambda **kw: add_optional(kw["plan"], "shape"))
    monkeypatch.setattr(module._figure_plan, "apply_runtime_bound_figure_contracts", bind_figures)
    monkeypatch.setattr(module._final_plan, "endpoint_contract_findings", next_validation)
    pipeline = SimpleNamespace(
        _services=PipelineServices(plan_revision_guard=guard),
        _scientific_runtime_authorities=Runtime(),
    )
    method = module.ResearchAgentPipeline._validate_and_persist_plan
    kwargs = {name: None for name in inspect.signature(method).parameters if name != "self"}
    kwargs.update(generation=SimpleNamespace(
        plan=add_optional(draft, "draft"), reused_prior_plan=False, reused_plan_path=None,
        migrated_plan_path=None, plan_generation_mode="llm", used_mock_llm=False,
        planner_prompt_metrics=None,
    ), agent_context=_context(), findings=[])
    expected = owner.ResearchPipelineRunError if extra_stage else LaterValidation
    with pytest.raises(expected) as raised:
        method(pipeline, **kwargs)
    if extra_stage:
        assert raised.value.code == "revision_added_unrequested_analysis"
        assert raised.value.details["step_id"] == "extra"
    assert calls == ["runtime_binding", "figure_binding"]
    assert source_file.read_bytes() == source_bytes
    assert guard.previous_plan == before == previous.model_dump(mode="json")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["analysis_plan.json"]


@pytest.mark.parametrize("allowed", [True, False])
def test_revision_checks_coordinates_of_an_existing_replay_step(allowed):
    from easyicu.research_agent.planning.robustness_contract import RobustnessSpec
    from tests.research_agent.planning.scientific_review_fixtures import _context

    step = _revision_step("robustness_analysis", robustness_replay_spec={
        "products": [{"product_id": "extra", "output": "robustness_matrix"}],
    })
    spec = RobustnessSpec(spec_id="original", axis="missing", description="Synthetic sensitivity.",
                          missing_override={"strategy": "complete_case", "variables": ["age"]})
    previous = _revision_plan(step).model_copy(update={"robustness_specs": [spec]})
    renamed = RobustnessSpec(spec_id="renamed", axis="missing", description="Renamed sensitivity.", missing_override=spec.missing_override)
    specs = [renamed] if allowed else [spec, RobustnessSpec(
        spec_id="unrequested", axis="outcome", description="An unrequested alternate endpoint.", outcome_override={"column": "other"},
    )]
    revised = previous.model_copy(update={"robustness_specs": specs})
    if allowed:
        _revision_guard(previous).validate_plan(plan=revised, context=_context())
    else:
        with pytest.raises(owner.ResearchPipelineRunError):
            _revision_guard(previous).validate_plan(plan=revised, context=_context())


@pytest.mark.parametrize("change", ["not_named", "wrong_audit", "mixed_audits", "no_spec"])
def test_revision_an_audit_blocker_cannot_authorize_unrelated_reports(change):
    from tests.research_agent.planning.scientific_review_fixtures import _context

    audits = ["cohort_flow"] if change == "wrong_audit" else ["missingness_profile"]
    if change == "mixed_audits":
        audits.append("cohort_flow")
    products = [{"product_id": f"audit_{index}", "audit": audit} for index, audit in enumerate(audits)]
    step = _revision_step("measurement_audit", "auxiliary", measurement_audit_spec={"products": products},
                          expected_outputs=[f"table:{item['product_id']}" for item in products])
    if change == "no_spec":
        step = step.model_copy(update={"measurement_audit_spec": None})
    guard = _revision_guard(_revision_plan(), _blocker("ARTICLE_CONTENT_ROLES_INCOMPLETE"),
                            facts={"content_roles": {"missing_roles": [] if change == "not_named" else ["quality"]}})
    with pytest.raises(owner.ResearchPipelineRunError):
        guard.validate_plan(plan=_revision_plan(step), context=_context())


@pytest.mark.parametrize("change", ["no_spec", "wrong_axis", "imputation", "outcome_override", "no_variables", "wrong_role"])
def test_revision_a_complete_case_blocker_requires_that_exact_refit(change):
    from easyicu.research_agent.planning.robustness_contract import RobustnessSpec
    from tests.research_agent.planning.scientific_review_fixtures import _context

    spec = RobustnessSpec(spec_id="cc", axis="outcome" if change == "wrong_axis" else "missing", description="Synthetic examination.",
                          missing_override={"strategy": "multiple_imputation" if change == "imputation" else "complete_case",
                                            "variables": [] if change == "no_variables" else ["age"]},
                          outcome_override={"column": "other"} if change == "outcome_override" else None)
    plan = _revision_plan(_revision_step(role="auxiliary" if change == "wrong_role" else "sensitivity")).model_copy(
        update={"robustness_specs": [] if change == "no_spec" else [spec]},
    )
    guard = _revision_guard(_revision_plan(), _blocker("MISSING_CATEGORY_WITHOUT_COMPLETE_CASE_SENSITIVITY"))
    with pytest.raises(owner.ResearchPipelineRunError):
        guard.validate_plan(plan=plan, context=_context())


@pytest.mark.parametrize("change", ["absent", "invalid_record"])
def test_revision_guard_does_not_load_a_neighbouring_or_unverified_source(source, monkeypatch, change):
    if change == "absent":
        source.row["run_id"] = "different_run"
    else:
        monkeypatch.setattr(owner.agent_runs, "read_run_record", lambda path: owner.RunRecordReadError(ok=False, error="invalid"))
    with pytest.raises(owner.ResearchPipelineRunError) as raised:
        owner.load_plan_revision_guard(study=source.study, project_root=str(source.root), source_run_id="run_inner",
                                       review=PlanScientificReview.model_validate(source.review))
    assert raised.value.code == "plan_revision_source_plan_missing"


def test_revision_a_stated_subgroup_does_not_authorize_another_axis():
    from easyicu.research_agent.contracts.post_analysis import SubgroupAnalysisSpec
    from tests.research_agent.planning.scientific_review_fixtures import _context

    context = _context()
    context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(update={
        "data_constraints": json.dumps({"stated_subgroups": [{
            "variable": "age", "concept": "age", "words": "within age groups", "source_status": "available",
        }]}),
    })})
    step = _revision_step("subgroup_analysis", "secondary", scientific_action_id="association.effect_modification")
    plan = _revision_plan(step).model_copy(update={"subgroup_analysis_spec": SubgroupAnalysisSpec(
        primary_model_requirement_id="primary", predictor="exposure", outcome="death",
        subgroup_columns=["age", "another_axis"], multiplicity_family_id="subgroups",
    )})
    with pytest.raises(owner.ResearchPipelineRunError):
        _revision_guard(_revision_plan()).validate_plan(plan=plan, context=context)


def test_revision_repairs_an_untyped_primary_binding_to_the_required_outcome():
    from tests.research_agent.planning.scientific_review_fixtures import _context

    correct = _revision_plan()
    previous = correct.model_copy(update={"steps": [correct.steps[0].model_copy(update={"inputs": ["exposure", "wrong_outcome"]})]})
    _revision_guard(previous).validate_plan(plan=correct, context=_context())


def test_revision_cannot_pack_an_additional_model_into_an_existing_step():
    from tests.research_agent.planning.scientific_review_fixtures import _context, _plan

    primary = _plan().steps[0]
    extra = primary.model_requirements[0].model_copy(update={
        "requirement_id": "optional_complete_case", "analysis_set": "complete_case",
    })
    previous = _revision_plan().model_copy(update={"steps": [primary]})
    changed = primary.model_copy(update={"model_requirements": [*primary.model_requirements, extra]})
    with pytest.raises(owner.ResearchPipelineRunError):
        _revision_guard(previous).validate_plan(plan=previous.model_copy(update={"steps": [changed]}), context=_context())


def test_revision_an_unrelated_blocker_cannot_authorize_a_missingness_report():
    from tests.research_agent.planning.scientific_review_fixtures import _context

    step = _revision_step("measurement_audit", "auxiliary", measurement_audit_spec={
        "products": [{"product_id": "extra", "audit": "missingness_profile"}],
    })
    guard = _revision_guard(_revision_plan(), _blocker("UNRELATED_BLOCKER"),
                            facts={"content_roles": {"missing_roles": ["quality"]}})
    with pytest.raises(owner.ResearchPipelineRunError):
        guard.validate_plan(plan=_revision_plan(step), context=_context())


@pytest.mark.parametrize("change", [None, "input_mention", "not_required", "duplicate", "primary_role", "wrong_exposure"])
def test_revision_adds_only_a_missing_required_outcome_inside_the_existing_model_roster(change):
    from tests.research_agent.planning.scientific_review_fixtures import _context, _plan

    primary = _plan().steps[0]
    if change == "input_mention":
        primary = primary.model_copy(update={"inputs": [*primary.inputs, "another_outcome"]})
    previous = _revision_plan().model_copy(update={"steps": [primary]})
    requirement = primary.model_requirements[0].model_copy(update={
        "requirement_id": "required_endpoint", "analysis_role": "secondary", "outcome": "another_outcome",
    })
    if change == "primary_role":
        requirement = requirement.model_copy(update={"analysis_role": "primary"})
    elif change == "wrong_exposure":
        requirement = requirement.model_copy(update={"exposure_source": "other_exposure"})
    additions = [requirement]
    if change == "duplicate":
        additions.append(requirement.model_copy(update={"requirement_id": "unrequested_repeat"}))
    changed = primary.model_copy(update={"model_requirements": [*primary.model_requirements, *additions],
                                          "inputs": [*primary.inputs, "another_outcome"]})
    context = _context()
    if change != "not_required":
        context = context.model_copy(update={"cohort": context.cohort.model_copy(update={"requested_outcome_columns": ["another_outcome"]})})
    guard = _revision_guard(previous)
    if change in {None, "input_mention"}:
        guard.validate_plan(plan=previous.model_copy(update={"steps": [changed]}), context=context)
    else:
        with pytest.raises(owner.ResearchPipelineRunError):
            guard.validate_plan(plan=previous.model_copy(update={"steps": [changed]}), context=context)


def test_revision_fills_a_missing_typed_primary_model_binding():
    from tests.research_agent.planning.scientific_review_fixtures import _context, _plan

    primary = _plan().steps[0]
    correct = _revision_plan().model_copy(update={"steps": [primary]})
    previous = correct.model_copy(update={"steps": [primary.model_copy(update={"model_requirements": []})]})
    _revision_guard(previous).validate_plan(plan=correct, context=_context())


@pytest.mark.parametrize("primary_outcome", ["death", "wrong_outcome"])
def test_revision_question_outcomes_do_not_authorize_duplicate_secondary_models(primary_outcome):
    from tests.research_agent.planning.scientific_review_fixtures import _context, _plan

    primary = _plan().steps[0]
    context = _context()
    if primary_outcome == "death":
        extra_outcome = "another_outcome"
        context = context.model_copy(update={"cohort": context.cohort.model_copy(update={"requested_outcome_columns": [extra_outcome]})})
        covered = primary.model_requirements[0].model_copy(update={
            "requirement_id": "covered", "analysis_role": "secondary", "outcome": extra_outcome,
        })
        primary = primary.model_copy(update={"model_requirements": [*primary.model_requirements, covered]})
        previous = _revision_plan().model_copy(update={"steps": [primary]})
    else:
        extra_outcome = "death"
        wrong = primary.model_requirements[0].model_copy(update={"outcome": primary_outcome})
        previous = _revision_plan().model_copy(update={"steps": [primary.model_copy(update={"model_requirements": [wrong]})]})
    extra = primary.model_requirements[0].model_copy(update={
        "requirement_id": "optional_repeat", "analysis_role": "secondary", "outcome": extra_outcome,
    })
    step = primary.model_copy(update={"model_requirements": [*primary.model_requirements, extra]})
    if primary_outcome == "death":
        step = primary.model_copy(update={"step_id": "repeat", "planned_analysis_role": "secondary", "model_requirements": [extra]})
        revised = previous.model_copy(update={"steps": [primary, step]})
    else:
        revised = previous.model_copy(update={"steps": [step]})
    with pytest.raises(owner.ResearchPipelineRunError):
        _revision_guard(previous).validate_plan(plan=revised, context=context)


@pytest.mark.parametrize("change", [None, "wrong_group", "optional_table", "nonblocker"])
def test_revision_keeps_required_accepted_baseline_repairs_available(change):
    from tests.support.baseline_requirement_plans import _bound_context, _plan

    previous = _revision_plan(_plan("age").steps[0])
    step = _plan("age", "cci_value", group="death" if change == "wrong_group" else "exposure").steps[0]
    if change == "optional_table":
        step = _revision_step("extra_table", "auxiliary")
    finding = _blocker("ACCEPTED_BASELINE_CONTENT_MISSING", severity="major" if change == "nonblocker" else "blocker")
    guard = _revision_guard(previous, finding)
    revised = previous.model_copy(update={"steps": [previous.steps[0], step]})
    if change is None:
        guard.validate_plan(plan=revised, context=_bound_context("age", "charlson"))
    else:
        with pytest.raises(owner.ResearchPipelineRunError):
            guard.validate_plan(plan=revised, context=_bound_context("age", "charlson"))
