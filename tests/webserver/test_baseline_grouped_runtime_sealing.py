"""Synthetic grouped exposures preserve their signed adjustment and dependence."""

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.baseline_association_runtime import build_baseline_authority
from easyicu.research_agent.authority.baseline_association_runtime import BaselineAssociationAuthorityError
from easyicu.research_agent.orchestration.exposure_grouping_phase import planning_run_groupings
from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
from easyicu.research_agent.planning.dependence_authority import bind_context_dependence_authority, context_dependence_authority
from easyicu.research_agent.planning.family_spec import build_family_spec_request
from tests.support.baseline_association import baseline_context, baseline_plan_attempt, baseline_runtime, grouped_case
from tests.support.node import run_node


@pytest.mark.parametrize("source", ["age", "bmi", "assay"])
@pytest.mark.parametrize("covariates", [("sex",), ()])
def test_verified_bands_seal_and_execute_without_adjusting_for_their_source(tmp_path, source, covariates):
    from easyicu.research_agent.execution.runners.baseline_association_executor import run_baseline_step

    context, runtime, catalog = grouped_case(tmp_path, source, covariates=covariates)
    attempt, _ = baseline_plan_attempt(context)
    plan = attempt.output
    rebound = runtime.for_planning_context(context=context, run_dir=tmp_path, cohort_path=catalog)
    sealed = rebound.seal_for_plan(plan)
    bound, _ = sealed.bind_plan(plan)
    sealed.validate_plan(bound)
    authority = sealed.current_case
    assert authority.exposure_column == context.primary_exposure
    assert authority.required_adjustment_columns == covariates
    assert authority.exposure_reference_level == ("1" if source == "assay" else "2")
    assert authority.primary_contrast_level == ("2" if source == "assay" else "1")
    assert authority.protocol_content_sha256 == runtime.current_case.protocol_content_sha256
    assert authority.dependence == runtime.current_case.dependence
    assert authority.execution_contract_sha256 != runtime.current_case.execution_contract_sha256
    assert sealed.for_planning_context(context=context, run_dir=tmp_path, cohort_path=catalog) is sealed
    assert sealed.seal_for_plan(bound) is sealed
    assert source not in bound.steps[3].model_requirements[0].covariates
    rng = np.random.default_rng(27)
    levels = [1, 2] if source == "assay" else [1, 2, 3]
    frame = pd.DataFrame({context.primary_exposure: np.resize(levels, 300),
                          "sex": rng.integers(0, 2, 300), "death": rng.integers(0, 2, 300).astype(bool)})
    result = run_baseline_step(step_id="adjusted_association", plan=bound, frame=frame, authority=authority,
                               runtime_projection_sha256="a" * 64, output_dir=tmp_path / "primary")
    assert result["status"] == "ok"
    primary_table = pd.read_csv(tmp_path / "primary/adjusted_association_estimates.csv")
    risk = run_baseline_step(step_id="absolute_risk_context", plan=bound, frame=frame, authority=authority,
                            runtime_projection_sha256="a" * 64, output_dir=tmp_path / "risk", primary_table=primary_table)
    assert risk["status"] == "ok"


@pytest.mark.parametrize("change", ["other_primary", "other_source", "not_metadata", "sealed"])
def test_group_rebinding_cannot_change_unrelated_or_sealed_authority(tmp_path, change):
    context, runtime, catalog = grouped_case(tmp_path)
    if change == "other_primary":
        context = context.model_copy(update={"primary_exposure": "another_group"})
    elif change == "not_metadata":
        context.cohort.provenance.clear()
    else:
        fields = runtime.current_case.model_dump(mode="python", exclude={"execution_contract_sha256"})
        fields.update({"exposure_column": "other"} if change == "other_source" else {"model_requirement_sha256": "c" * 64})
        runtime = ScientificRuntimeAuthorities(trajectory=None, current_case=build_baseline_authority(**fields))
    assert runtime.for_planning_context(context=context, run_dir=tmp_path, cohort_path=catalog) is runtime


def test_group_rebinding_checks_the_record_against_the_empty_input(tmp_path):
    context, runtime, catalog = grouped_case(tmp_path)
    record = tmp_path / "exposure_groupings.json"
    record.write_text(record.read_text().replace('"value": 30', '"value": 31'))
    with pytest.raises(ValueError, match="does not declare the group columns"):
        runtime.for_planning_context(context=context, run_dir=tmp_path, cohort_path=catalog)


@pytest.mark.parametrize("conflict,field", [("window", "grouping.window"), ("source", "grouping.source_columns"), ("adjustment", "covariates")])
def test_group_binding_cannot_change_timing_or_absorb_an_adjustment(tmp_path, conflict, field):
    context, runtime, catalog = grouped_case(tmp_path)
    column = planning_run_groupings(tmp_path, cohort_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest()).columns[0]
    authority = runtime.current_case
    reason = "baseline_association_design_mismatch"
    if conflict == "window":
        column = replace(column, window=(0, 24))
    elif conflict == "source":
        column = replace(column, parameters={
            **column.parameters, "source_columns": {"value": "bmi", "max": "other"},
            "groups": [{**group, "rule": {**group["rule"], "summary": "max"}} if group["id"] == "g3" else group
                       for group in column.parameters["groups"]],
        })
    else:
        coordinates = authority.model_dump(mode="python", exclude={"execution_contract_sha256"})
        authority = build_baseline_authority(**{**coordinates, "required_adjustment_columns": (context.primary_exposure,)})
        reason = "association_adjustment_exposure_conflict"
    with pytest.raises(BaselineAssociationAuthorityError) as caught:
        authority.for_grouped_exposure(column)
    assert caught.value.reason_code == reason
    assert caught.value.mismatch_fields == (field,)


@pytest.mark.parametrize("requested", ["exposure", "source"])
@pytest.mark.parametrize("selection", ["exact", "planner_selectable"])
def test_a_requested_exposure_or_its_source_is_refused_not_dropped(tmp_path, requested, selection):
    context, _, _ = grouped_case(tmp_path)
    name = context.primary_exposure if requested == "exposure" else "bmi"
    context.user_preferences.covariate_selection = selection
    context.user_preferences.covariates = [name] if selection == "exact" else []
    context.user_preferences.data_constraints = json.dumps({"stated_adjustments": [
        {"variable": name, "concept": name, "words": "Adjust for the named variable", "source_status": "available"},
    ]})
    with pytest.raises(BaselineAssociationAuthorityError) as caught:
        build_family_spec_request(context, analysis_types=("association_study",),
                                  variable_roster=[v.name for v in context.variables], allowed_literature_citation_keys=())
    assert caught.value.reason_code == "association_adjustment_exposure_conflict"


def test_pipeline_binds_patient_dependence_before_the_first_roster_seal(tmp_path, monkeypatch):
    from easyicu.research_agent import pipeline as owner

    context, runtime, catalog = grouped_case(tmp_path, clustered=True)
    runtime = runtime.for_planning_context(context=context, run_dir=tmp_path, cohort_path=catalog)
    dependence = context_dependence_authority(context)
    assert dependence is not None
    attempt, _ = baseline_plan_attempt(context)
    plan = attempt.output
    class ReachedSeal(Exception):
        pass
    def seal(plan):
        assert plan.steps[3].model_requirements[0].dependence == dependence
        assert runtime.seal_for_plan(plan).current_case.dependence == dependence
        raise ReachedSeal
    monkeypatch.setattr(owner, "seal_plan_tier", lambda plan, *_args, **_kwargs: plan)
    pipeline = SimpleNamespace(_config=None, _services=SimpleNamespace(validate_plan_revision=lambda **_kwargs: None),
                               _scientific_runtime_authorities=SimpleNamespace(seal_for_plan=seal))
    generation = SimpleNamespace(plan=plan, reused_prior_plan=False, reused_plan_path=None, migrated_plan_path=None,
                                 plan_generation_mode="synthetic", used_mock_llm=True, planner_prompt_metrics=None)
    arguments = {name: None for name in inspect.signature(owner.ResearchAgentPipeline._validate_and_persist_plan).parameters if name != "self"}
    arguments.update(generation=generation, agent_context=context)
    with pytest.raises(ReachedSeal):
        owner.ResearchAgentPipeline._validate_and_persist_plan(pipeline, **arguments)


@pytest.mark.parametrize("violation,reason,words,expected_fields", [
    ("omitted_covariate", "baseline_association_roster_mismatch", "调整变量", ["covariates"]),
    ("changed_exposure", "baseline_association_design_mismatch", "暴露", ["exposure_source"]),
    ("changed_dependence", "baseline_association_design_mismatch", "患者聚类", ["dependence"]),
    ("requested_exposure", "association_adjustment_exposure_conflict", "暴露本身", []),
])
def test_roster_refusal_reaches_record_gate_and_chinese_stop_text(tmp_path, violation, reason, words, expected_fields):
    from easyicu.webserver import agent_pipeline_runs as web
    from easyicu.webserver.pi_copilot.projections import gate_detail_projection

    context = baseline_context("sex", covariates=("age",))
    attempt, _ = baseline_plan_attempt(context)
    runtime = baseline_runtime(context, attempt.output)
    draft = attempt.output.model_copy(deep=True)
    requirement = draft.steps[3].model_requirements[0]
    if violation == "omitted_covariate":
        draft.steps[3].model_requirements = [requirement.model_copy(update={
            "covariates": [], "covariate_rationales": {}, "covariate_temporal_roles": {},
            "model_terms": [term for term in requirement.model_terms if term.role == "exposure"],
        })]
    elif violation in {"changed_exposure", "changed_dependence"}:
        fields = runtime.current_case.model_dump(mode="python", exclude={"execution_contract_sha256"})
        from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
        change = {"exposure_column": "other"} if violation == "changed_exposure" else {
            "dependence": PlannedDependenceRequirement(group_source="stay_id", group_derivation="identity"),
        }
        runtime = ScientificRuntimeAuthorities(trajectory=None, current_case=build_baseline_authority(**{**fields, **change}))
    with pytest.raises(BaselineAssociationAuthorityError) as caught:
        if violation == "requested_exposure":
            context.user_preferences.covariates = ["sex"]
            build_family_spec_request(context, analysis_types=("association_study",),
                                      variable_roster=[v.name for v in context.variables], allowed_literature_citation_keys=())
        else:
            runtime.seal_for_plan(draft)
    assert caught.value.reason_code == reason
    wrapper = RuntimeError("private synthetic message")
    wrapper.__cause__ = caught.value
    code = web._pipeline_failure_code(wrapper)
    assert code == "research_pipeline_scientific_authority_refused"
    path = web._record_pipeline_failure(wrapper_dir=tmp_path, study={"id": "synthetic"}, provider={},
                                        exc=wrapper, code=code, execution_retry_id=None)
    record = json.loads((tmp_path / path).read_text())
    assert record["typed_failure"]["reason_code"] == reason
    assert record["typed_failure"].get("mismatch_fields", []) == expected_fields
    assert "BaselineAssociationAuthorityError" in record["exception_types"]
    assert "private synthetic message" not in json.dumps(record)
    gate = json.loads((tmp_path / "quality_gate.json").read_text())["gate"]
    assert gate_detail_projection(gate["detail"])["gate_detail_code"] == reason
    node = shutil.which("node")
    assert node
    script = r"""
global.window = {EU_HTML: {esc: String}, EasyICU: {guidedPi: {declare: (_name, owner) => {global.owner = owner;}}}};
require(process.argv[1]);
const errorText = global.owner.create({tr: (_en, zh) => zh, staticPreview: () => false});
process.stdout.write(errorText.runFailureText(process.argv[2], {code: process.argv[3]}));
"""
    result = run_node(node, script, str(Path(web.__file__).parent / "static/js/screens-guided-pi-error-text.js"), code, reason, check=True)
    assert words in result.stdout
    assert "分析尚未开始" in result.stdout
    assert "请" in result.stdout
    assert "有一个分析步骤在运行中失败了" not in result.stdout


def test_group_sources_are_not_offered_as_adjustment_candidates(tmp_path):
    from easyicu.research_agent.planning.adjustment_authority import adjusted_model_term_planning_authority

    context, _, _ = grouped_case(tmp_path)
    candidates = adjusted_model_term_planning_authority(context, [v.name for v in context.variables])
    assert "bmi" not in {row["name"] for row in candidates["eligible_covariates"]}
    assert {"name": "bmi", "reason": "exposure_group_source"} in candidates["excluded_covariates"]


def test_rebinding_keeps_the_launch_receipt_needed_to_render_the_host_executor(tmp_path):
    from easyicu.research_agent.pipeline import ResearchAgentPipeline
    from easyicu.research_agent.execution.runners.selection import select_standard_executor

    context, runtime, catalog = grouped_case(tmp_path)
    pipeline = SimpleNamespace(_scientific_runtime_authorities=runtime, _scientific_runtime_projection_sha256="b" * 64)
    ResearchAgentPipeline._plan_run_formed_groups_without_source_spline(
        pipeline, context=context, run_dir=tmp_path, cohort_path=catalog, run_id="synthetic",
        emit_progress=lambda *_args, **_kwargs: None,
    )
    assert pipeline._scientific_runtime_projection_sha256 == "b" * 64
    attempt, _ = baseline_plan_attempt(context)
    sealed = pipeline._scientific_runtime_authorities.seal_for_plan(attempt.output)
    bound, _ = sealed.bind_plan(attempt.output)
    selection = select_standard_executor(bound.steps[3], plan=bound, current_case_scientific_runtime_authority=sealed.current_case,
                                          scientific_runtime_projection_sha256=pipeline._scientific_runtime_projection_sha256)
    assert selection is not None
    assert "run_baseline_step" in selection.code


@pytest.mark.parametrize("source_present", [True, False])
@pytest.mark.parametrize("binding", ["source", "group"])
@pytest.mark.parametrize("middle_band", [False, True])
@pytest.mark.parametrize("source_hint", [None, "age"])
def test_accepted_grouped_plan_projects_from_the_source_only_universe(tmp_path, source_present, binding, middle_band, source_hint):
    from easyicu.webserver.scientific_runtime_projection import compile_web_scientific_runtime_projection

    context, _, catalog = grouped_case(tmp_path, source="age", clustered=True, middle_band=middle_band)
    groups = planning_run_groupings(tmp_path, cohort_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest())
    source = tmp_path / "source.parquet"
    frame = pd.DataFrame({"age": pd.Series(dtype="float64"), "sex": pd.Series(dtype="bool"),
                          "death": pd.Series(dtype="bool")})
    if not source_present:
        frame = frame.drop(columns="age")
    frame.to_parquet(source, index=False)
    projection = compile_web_scientific_runtime_projection(
        study={**json.loads(context.user_preferences.data_constraints), "covariate_selection": "exact"},
        sensitivity_specs=(), primary_exposure="age" if binding == "source" else context.primary_exposure, primary_exposure_source=source_hint,
        target_outcome="death", declared_covariates=("sex",), covariate_operationalizations={},
        target_is_event_status=True, universe_path=source, scientific_configuration_sha256="a" * 64,
        dependence=context_dependence_authority(context), exposure_groupings=groups,
    )
    if not source_present:
        assert projection is None
        return
    assert projection is not None
    runtime = ScientificRuntimeAuthorities.load(trajectory=None, current_case=projection.authority)
    authority = runtime.current_case
    assert authority.exposure_column == context.primary_exposure
    assert authority.exposure_encoding == "treatment_contrast"
    assert authority.exposure_levels == (("1", "2", "3") if middle_band else ("1", "2"))
    assert authority.exposure_reference_level == "2"
    assert authority.primary_contrast_level == "1"
    assert authority.dependence == context_dependence_authority(context)
    attempt, _ = baseline_plan_attempt(context)
    draft = bind_context_dependence_authority(plan=attempt.output, context=context)
    sealed = runtime.seal_for_plan(draft)
    bound, _ = sealed.bind_plan(draft)
    sealed.validate_plan(bound)
    assert bound.steps[3].model_requirements[0].covariates == ["sex"]


@pytest.mark.parametrize("field", [
    "primary_model", "method", "model_requirements", "outcome_type", "outcome", "exposure_levels",
    "exposure_reference_level", "primary_contrast_level", "exposure_encoding", "scientific_action_id",
    "inputs", "effect_modifiers", "reviewed_requirement",
])
def test_design_refusal_names_its_coordinate_without_weakening_the_check(tmp_path, field):
    context, runtime, catalog = grouped_case(tmp_path)
    authority = runtime.for_planning_context(context=context, run_dir=tmp_path, cohort_path=catalog).current_case
    plan = baseline_plan_attempt(context)[0].output
    primary = plan.steps[3]
    requirement = primary.model_requirements[0]
    if field == "primary_model":
        primary.planned_analysis_role = "secondary"
    elif field == "method":
        primary.method = "synthetic_wrong_method"
    elif field == "model_requirements":
        primary.model_requirements = []
    elif field == "outcome_type":
        primary.model_requirements = [requirement.model_copy(update={
            "outcome_type": "continuous", "method_family": "linear_regression", "estimator": "ols",
        })]
    elif field in {"outcome", "exposure_levels", "exposure_reference_level", "primary_contrast_level", "exposure_encoding"}:
        updates = {
            "outcome": {"outcome_column": "synthetic_other_outcome"},
            "exposure_levels": {"exposure_levels": ("3", "2", "1")},
            "exposure_reference_level": {"exposure_reference_level": "3"},
            "primary_contrast_level": {"primary_contrast_level": "3"},
            "exposure_encoding": {"exposure_encoding": "restricted_cubic_spline", "exposure_levels": (),
                                  "exposure_reference_level": None, "primary_contrast_level": None},
        }[field]
        authority = build_baseline_authority(**{
            **authority.model_dump(mode="python", exclude={"execution_contract_sha256"}), **updates,
        })
    elif field == "scientific_action_id":
        primary.scientific_action_id = "association.effect_modification"
    elif field == "inputs":
        primary.inputs = [name for name in primary.inputs if name != "sex"]
    else:
        authority = authority.seal_adjustment_roster(plan)
        plan = authority.bind_plan(plan)
        primary = plan.steps[3]
        requirement = primary.model_requirements[0]
        change = {"covariate_rationales": {"sex": "Changed reviewed rationale"}} if field == "reviewed_requirement" else {
            "effect_modifiers": [next(term for term in requirement.model_terms if term.role == "covariate")],
        }
        primary.model_requirements = [requirement.model_copy(update=change)]
    with pytest.raises(BaselineAssociationAuthorityError) as caught:
        authority.seal_adjustment_roster(plan)
    assert caught.value.mismatch_fields == ("model_requirements" if field == "reviewed_requirement" else field,)


def test_a_group_without_a_source_hint_cannot_adjust_for_its_source(tmp_path):
    from easyicu.webserver.scientific_runtime_projection import compile_web_scientific_runtime_projection, WebScientificRuntimeProjectionError

    context, _, catalog = grouped_case(tmp_path, source="age")
    groups = planning_run_groupings(tmp_path, cohort_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest())
    with pytest.raises(WebScientificRuntimeProjectionError) as caught:
        compile_web_scientific_runtime_projection(
            study={"analysis_design": {"analysis_family": "association_study"}, "covariate_selection": "exact"},
            sensitivity_specs=(), primary_exposure=context.primary_exposure, primary_exposure_source=None,
            target_outcome="death", declared_covariates=("age",), covariate_operationalizations={},
            target_is_event_status=True, universe_path=catalog, scientific_configuration_sha256="a" * 64,
            exposure_groupings=groups,
        )
    assert caught.value.code == "web_baseline_association_incomplete"
    assert caught.value.__cause__.reason_code == "association_adjustment_exposure_conflict"


def test_diagnostic_fields_are_bounded_and_do_not_copy_exception_values(tmp_path):
    from easyicu.webserver.agent_pipeline_runs import _record_pipeline_failure, _safe_pipeline_typed_failure

    private = "synthetic-private-value"
    error = BaselineAssociationAuthorityError(private, mismatch_fields=(private, "exposure_source", "exposure_source", "dependence"))
    wrapper = RuntimeError(private)
    wrapper.__cause__ = error
    typed = _safe_pipeline_typed_failure(wrapper)
    assert typed["mismatch_fields"] == ["exposure_source", "dependence"]
    path = _record_pipeline_failure(wrapper_dir=tmp_path, study={"id": "synthetic"}, provider={}, exc=wrapper,
                                    code="research_pipeline_scientific_authority_refused", execution_retry_id=None)
    receipt = json.loads((tmp_path / path).read_text())
    assert receipt["typed_failure"] == typed
    assert private not in json.dumps(receipt)
