"""Baseline compilation and bounded repair use the same patient dependence."""

import json

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.authority.baseline_association_runtime import (
    BaselineAssociationAuthorityError, build_baseline_authority,
)
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.intake.materialized_metadata import FIRST_ICU_STAY_RESTRICTION_SCHEMA
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificAuthorityRepairError, ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning.dependence_authority import (
    bind_context_dependence_authority, context_dependence_authority,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.structured_retry import (
    StructuredResponseFailure, call_llm_with_structured_retry,
)
from easyicu.research_agent.schema import AnalysisPlan, ConceptDescriptor
from tests.support.baseline_association import baseline_context, baseline_plan_attempt


def _case(database="eicu", population="all_stays", exposure="chronic_flag"):
    context = baseline_context(exposure, covariates=("age",))
    context.cohort.database = database
    context.cohort.provenance.update(evidence_stage="metadata_only_planning", analysis_unit="icu_stay")
    if database == "eicu":
        context.cohort.provenance["replacement_row_identity"] = {
            "output_identity_column": "stay_id", "mapping_file_sha256": "a" * 64,
            "patient_group_derivation": {"algorithm": "prefix_before_:s", "delimiter": ":s"},
        }
    else:
        context.cohort.id_columns.append("subject_id")
        context.variables.append(ConceptDescriptor(name="subject_id", role="id", dtype="int64"))
        context.cohort.provenance["patient_id_columns"] = ["subject_id"]
    design = {"analysis_family": "association_study", "analysis_unit": "icu_stay", "variance_estimator": "model_based"}
    if population == "all_stays":
        design.update(variance_estimator="cluster_robust", cluster_unit="patient")
    else:
        context.cohort.provenance["first_icu_stay_restriction"] = {
            "schema_version": FIRST_ICU_STAY_RESTRICTION_SCHEMA, "coordinate_sha256": "b" * 64,
        }
    context.user_preferences.data_constraints = json.dumps({"analysis_design": design})
    levels = ("0", "1") if exposure == "chronic_flag" else ("elective", "emergency", "urgent")
    authority = build_baseline_authority(
        protocol_content_sha256="c" * 64, exposure_column=exposure, outcome_column="death",
        exposure_encoding="treatment_contrast", exposure_levels=levels,
        exposure_reference_level=levels[0], primary_contrast_level=levels[-1],
        required_adjustment_columns=("age",), dependence=context_dependence_authority(context),
    )
    return context, ScientificRuntimeAuthorities(trajectory=None, current_case=authority)


def _draft(context):
    # Templates require one row identity. The ordinary compiler also handles
    # a source carrying a separate, explicitly declared patient identifier.
    template_context = context.model_copy(deep=True)
    template_context.cohort.id_columns = ["stay_id"]
    return baseline_plan_attempt(template_context)


@pytest.mark.parametrize("database", ["eicu", "miiv"])
@pytest.mark.parametrize("population", ["all_stays", "first_stay"])
@pytest.mark.parametrize("exposure", ["chronic_flag", "admission_type"])
def test_compiler_binds_dependence_before_authority_acceptance(database, population, exposure):
    context, runtime = _case(database, population, exposure)
    attempt, llm = _draft(context)
    agent = ProgressivePlannerAgent(llm, scientific_authorities=runtime)
    plan, receipt = agent._compile_and_accept(
        attempt.facts.skeleton, agent_context=context, article_context=context,
        allowed_literature_citation_keys=(), direct_comparator_literature_keys=(),
        allowed_know_how_decisions=None, enforce_article_contract=False,
    )
    primary = plan.steps[3]
    expected = context_dependence_authority(context)
    assert primary.model_requirements[0].dependence == expected
    assert (expected is None) == (population == "first_stay")
    if expected is not None:
        assert expected.group_source in primary.inputs
        assert expected.group_derivation == ("identity" if database == "miiv" else "prefix_before_delimiter")
    step_receipt = next(row for row in receipt.compiled_steps if row.step_id == primary.step_id)
    assert step_receipt.compiled_step_sha256 == canonical_sha256(primary.model_dump(mode="json"))
    assert receipt.analysis_plan_sha256 == canonical_sha256(plan.model_dump(mode="json"))
    sealed = runtime.seal_for_plan(plan)
    bound, _ = sealed.bind_plan(plan)
    sealed.validate_plan(bound)
    assert bind_context_dependence_authority(plan=plan, context=context) == plan
    assert len(llm.calls) == 1


@pytest.mark.parametrize("exposure", ["chronic_flag", "admission_type"])
def test_family_attempt_with_live_authority_needs_no_repair(exposure):
    context, runtime = _case(exposure=exposure)
    attempt, llm = baseline_plan_attempt(context, scientific_authorities=runtime)
    assert attempt.output.steps[3].model_requirements[0].dependence == runtime.current_case.dependence
    assert len(llm.calls) == 1


def _candidate(database="eicu", population="all_stays"):
    context, runtime = _case(database, population)
    attempt, _ = _draft(context)
    # Construct a valid candidate independently of the compiler regression.
    plan = bind_context_dependence_authority(plan=attempt.output, context=context)
    if runtime.current_case.dependence:
        primary = plan.steps[3]
        primary.inputs = list(dict.fromkeys([*primary.inputs, runtime.current_case.dependence.group_source]))
    runtime.check_planner_plan(plan, context)
    return context, runtime, plan


def _drift(plan, population):
    plan = plan.model_copy(deep=True)
    plan.steps[3].model_requirements[0].dependence = (
        None if population == "all_stays" else PlannedDependenceRequirement(group_source="stay_id", group_derivation="identity")
    )
    return plan


@pytest.mark.parametrize("database", ["eicu", "miiv"])
@pytest.mark.parametrize("population", ["all_stays", "first_stay"])
def test_dependence_only_drift_uses_existing_repair_budget(database, population):
    context, runtime, repaired = _candidate(database, population)
    original = _drift(repaired, population)
    before = original.model_dump_json()
    llm = ScriptedMockLLMClient([before, repaired.model_dump_json()])
    events = []
    result = call_llm_with_structured_retry(
        llm, [], parser=lambda raw: runtime.check_planner_plan(AnalysisPlan.model_validate_json(raw), context),
        retryable_parse_error=runtime.planner_retryable_error, max_retries=1, progress_callback=events.append,
    )
    assert result == repaired and original.model_dump_json() == before
    assert len(llm.calls) == 2
    assert [event.phase for event in events] == ["started", "rejected", "started", "accepted"]
    feedback = llm.calls[1][0][-1].content
    assert "dependence" in feedback and "adjusted_association" in feedback
    assert "cluster_robust" in feedback if population == "all_stays" else '"dependence": null' in feedback
    with pytest.raises(BaselineAssociationAuthorityError):
        runtime.seal_for_plan(original)  # Feedback never changes the final owner.


@pytest.mark.parametrize("conflict", ["context", "invalid_context", "missing_context", "sealed", "roster", "outcome", "steps"])
def test_dependence_feedback_cannot_relax_other_authority_coordinates(conflict):
    context, runtime, repaired = _candidate()
    if conflict == "sealed":
        runtime = runtime.seal_for_plan(repaired)
    candidate = _drift(repaired, "all_stays")
    if conflict == "context":
        context.user_preferences.data_constraints = "{}"
    elif conflict == "invalid_context":
        context.user_preferences.data_constraints = json.dumps({"analysis_design": {"variance_estimator": "invented"}})
    elif conflict == "missing_context":
        context = None
    elif conflict == "roster":
        candidate.steps[3].model_requirements[0].covariates = []
    elif conflict == "outcome":
        candidate.steps[3].model_requirements[0].outcome_type = "continuous"
    elif conflict == "steps":
        candidate.steps = candidate.steps[:-1]
    with pytest.raises(BaselineAssociationAuthorityError) as caught:
        runtime.check_planner_plan(candidate, context)
    assert not runtime.planner_retryable_error(caught.value)


def test_dependence_refusal_exhausts_the_budget_without_changing_candidate(tmp_path):
    context, runtime, repaired = _candidate()
    candidate = _drift(repaired, "all_stays")
    before = candidate.model_dump_json()
    llm = ScriptedMockLLMClient([before] * 4)
    with pytest.raises(StructuredResponseFailure) as caught:
        call_llm_with_structured_retry(
            llm, [], parser=lambda raw: runtime.check_planner_plan(AnalysisPlan.model_validate_json(raw), context),
            retryable_parse_error=runtime.planner_retryable_error, max_retries=2,
        )
    assert len(llm.calls) == 3
    assert len(caught.value.easyicu_structured_attempt_metadata) == 3
    assert isinstance(caught.value.__cause__, ScientificAuthorityRepairError)
    assert candidate.model_dump_json() == before
    from easyicu.webserver import agent_pipeline_runs

    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path, study={"id": "synthetic"}, provider={}, exc=caught.value,
        code=agent_pipeline_runs._pipeline_failure_code(caught.value), execution_retry_id=None,
    )
    receipt = json.loads((tmp_path / "diagnostics/research_pipeline_failure.json").read_text())
    assert len(receipt["structured_retry_history"]) == 3
    assert receipt["typed_failure"]["mismatch_fields"] == ["dependence"]
    assert not receipt["raw_model_output_recorded"]


def test_wrong_group_derivation_is_repairable_without_mutating_it():
    context, runtime, plan = _candidate()
    requirement = plan.steps[3].model_requirements[0]
    requirement.dependence = requirement.dependence.model_copy(update={"delimiter": ":other"})
    with pytest.raises(ScientificAuthorityRepairError) as caught:
        runtime.check_planner_plan(plan, context)
    support = caught.value.details["findings"][0]["supported_repair"]
    assert support["dependence"] == runtime.current_case.dependence.model_dump(mode="json")
    assert requirement.dependence.delimiter == ":other"
