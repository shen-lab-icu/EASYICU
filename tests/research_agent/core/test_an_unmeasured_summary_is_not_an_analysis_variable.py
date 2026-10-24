"""Window summaries retain denominators without authorizing empty analyses."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.cohort import materializer
from easyicu.research_agent.execution.model_matrix import primary_model_rows
from easyicu.research_agent.execution.runners.descriptive_association_executor import (
    run_descriptive_association,
)
from easyicu.research_agent.execution.runners.cohort_summary_executor import run_cohort_summary_from_env
from easyicu.research_agent.execution.runners.descriptive_distribution_executor import grouped_distribution_table
from easyicu.research_agent.execution.runners.deterministic_descriptive import run_absolute_risk_context
from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import _distribution_rows
from easyicu.research_agent.gates.data_answerability import analysis_answerability_findings
from easyicu.research_agent.methods.descriptive_inputs import RequestedConceptNoDataError
from easyicu.research_agent.methods.table_one import build_grouped_table_one
from easyicu.research_agent.orchestration.scientific_plan_review_gate import (
    ScientificPlanReviewArtifactError, persist_or_validate_scientific_plan_review,
)
from easyicu.research_agent.planning.baseline_requirements import (
    baseline_requirement_coverage, bind_baseline_requirements,
)
from easyicu.research_agent.planning.question_requirements import (
    JudgedRequirement, QuestionRequirement, question_requirement_coverage,
)
from easyicu.research_agent.planning.requirement_coverage import (
    stated_adjustment_coverage,
)
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, VariableRole
from easyicu.research_agent.contracts.exposure_outcome_distribution import ExposureOutcomeDistributionSpec


REASON = "requested_concept_no_data"
EXPOSURE = "fixture_signal_max"
OUTCOME = "fixture_response_max"
COVARIATE = "fixture_adjuster_max"


def _materialize(monkeypatch, tmp_path, *, empty=None, partial=False, partial_column=COVARIATE):
    ids = list(range(1, 9))
    sources = {"age": pd.DataFrame({"stay_id": ids, "age": range(40, 48)})}
    for concept in ("fixture_signal", "fixture_response", "fixture_adjuster"):
        times = [48.0] * 8 if f"{concept}_max" == empty else [1.0] * 8
        if partial and f"{concept}_max" == partial_column:
            times[-2:] = [48.0, 48.0]
        sources[concept] = pd.DataFrame({
            "stay_id": ids, "charttime": times,
            concept: [1.0, 3.0, 2.0, 4.0, 6.0, 5.0, 8.0, 7.0],
        })
    monkeypatch.setattr(materializer, "_resolve_source", lambda *_: ("converted", tmp_path))
    monkeypatch.setattr(
        materializer, "_load_concept",
        lambda _mode, _root, concept, *_args, **_kwargs: sources[concept].copy(),
    )
    cohort, _ = materializer.materialize_cohort(
        feature_concepts=list(sources)[1:], data_path=tmp_path,
        static_concepts=["age"], outcome_concepts=[],
    )
    # dc2's left join must still retain every stay and publish zero flags.
    assert len(cohort) == 8
    if empty is not None:
        assert cohort[empty].isna().all()
        assert cohort[empty.removesuffix("_max") + "_measured"].eq(0).all()
    return cohort


def _context(frame):
    return build_research_context(
        research_question="Describe the synthetic response by the signal.",
        cohort=frame, cohort_name="synthetic", database="synthetic",
        target_outcome=OUTCOME, primary_exposure=EXPOSURE,
        user_preferences={"covariates": [COVARIATE], "covariate_selection": "exact"},
    )


def _plan():
    return AnalysisPlan(
        research_question="Describe the synthetic response by the signal.",
        analysis_type="descriptive_study",
        steps=[AnalysisStep(
            step_id="association", planned_analysis_role="primary",
            intent="Describe the declared pair.", method="descriptive_association",
            inputs=["artifact:analysis_cohort", EXPOSURE, OUTCOME],
            expected_outputs=["statistic:association"],
        )],
    )


@pytest.mark.parametrize("empty", [OUTCOME, COVARIATE, EXPOSURE])
def test_unmeasured_requested_summary_blocks_before_any_provider(
    ra, monkeypatch, tmp_path, empty,
):
    cohort = _materialize(monkeypatch, tmp_path, empty=empty)
    llm = ScriptedMockLLMClient([AssertionError("no-data must stop before the provider")])
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "run", llm=llm, enable_literature=False,
    )
    result = pipeline.run(
        question="Describe the synthetic response by the signal.",
        cohort=cohort, cohort_name="synthetic", database="eicu_demo",
        target_outcome=OUTCOME, primary_exposure=EXPOSURE,
        user_preferences={"covariates": [COVARIATE], "covariate_selection": "exact"},
    )
    assert result.plan_path == ""
    assert llm.calls == []
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["notes"] == "aborted: data_answerability_failed"
    finding = next(item for item in manifest["findings"] if (item["detail"] or {}).get("kind") == REASON)
    assert finding["severity"] == "error"
    assert empty in finding["detail"]["columns"]
    assert finding["message"] == (
        f"The data contain no recorded values for {empty}, which the question asks to summarize."
    )


@pytest.mark.parametrize("empty", [OUTCOME, COVARIATE, EXPOSURE])
def test_review_does_not_approve_an_owner_of_an_unmeasured_summary(
    monkeypatch, tmp_path, empty,
):
    context = _context(_materialize(monkeypatch, tmp_path, empty=empty))
    review = build_plan_scientific_review(context=context, plan=_plan())
    finding = next(item for item in review.findings if item.code == "REQUESTED_CONCEPT_NO_DATA")
    assert finding.severity == "blocker"
    assert finding.message == (
        f"The data contain no recorded values for {empty}, which the question asks to summarize."
    )
    assert finding.remediation == (
        "Choose a variable or data source that records these measurements, then plan again."
    )
    assert finding.remediation_route == "runtime_capability"
    assert not review.approval_allowed


def test_an_approval_from_before_the_no_data_gate_cannot_authorize_resume(monkeypatch, tmp_path):
    context = _context(_materialize(monkeypatch, tmp_path, empty=OUTCOME))
    current = build_plan_scientific_review(context=context, plan=_plan())
    archived = current.model_copy(update={
        "schema_version": "easyicu.plan_scientific_review/17", "approval_allowed": True,
        "status": "ready_for_approval", "findings": [],
    })
    run_dir = tmp_path / "resume"
    evidence = EvidenceStore(run_dir)
    persist_or_validate_scientific_plan_review(
        run_dir=run_dir, evidence=evidence, current_review=archived,
    )
    with pytest.raises(ScientificPlanReviewArtifactError) as caught:
        persist_or_validate_scientific_plan_review(
            run_dir=run_dir, evidence=evidence, current_review=current, reuse_existing_review=True,
        )
    assert caught.value.code == "scientific_plan_review_binding_drift"


def test_stated_adjustment_coverage_cannot_call_an_empty_value_covered(monkeypatch, tmp_path):
    context = _context(_materialize(monkeypatch, tmp_path, empty=COVARIATE))
    prefs = context.user_preferences.model_copy(update={"data_constraints": json.dumps({
        "stated_adjustments": [{
            "variable": COVARIATE, "concept": "fixture_adjuster",
            "words": "adjust for the synthetic adjuster", "source_status": "available",
        }],
    })})
    context = context.model_copy(update={"user_preferences": prefs})
    record, = stated_adjustment_coverage(context, _plan()).records
    assert record.status == "unsupported"
    assert record.reason_code == REASON
    assert record.owner_step_ids == ()


def _terms():
    return [
        {"name": name, "role": role, "coding": "continuous", "transform": "identity"}
        for name, role in [(EXPOSURE, "exposure"), (COVARIATE, "covariate")]
    ]


@pytest.mark.parametrize("empty", [OUTCOME, COVARIATE, EXPOSURE])
def test_model_rows_stop_with_a_typed_reason_before_complete_case_selection(
    monkeypatch, tmp_path, empty,
):
    cohort = _materialize(monkeypatch, tmp_path, empty=empty)
    with pytest.raises(ValueError) as caught:
        primary_model_rows(cohort, terms=_terms(), exposure=EXPOSURE, outcome=OUTCOME)
    assert caught.value.reason_code == REASON
    assert empty in caught.value.columns
    assert str(caught.value) == (
        f"{REASON}: the extracted data have no recorded values for {empty}"
    )


def test_no_data_messages_name_every_affected_variable(monkeypatch, tmp_path):
    cohort = _materialize(monkeypatch, tmp_path, empty=COVARIATE)
    cohort[OUTCOME] = float("nan")
    cohort[OUTCOME.removesuffix("_max") + "_measured"] = 0
    context = _context(cohort)
    columns = f"{COVARIATE}, {OUTCOME}"
    message = f"The data contain no recorded values for {columns}, which the question asks to summarize."
    finding = next(
        item for item in analysis_answerability_findings(context)
        if item.detail.get("kind") == REASON
    )
    assert finding.message == message
    review = build_plan_scientific_review(context=context, plan=_plan())
    assert next(item for item in review.findings if item.code == "REQUESTED_CONCEPT_NO_DATA").message == message
    with pytest.raises(RequestedConceptNoDataError) as caught:
        build_grouped_table_one(cohort, {**_table_spec(), "group_by": OUTCOME})
    assert str(caught.value) == f"{REASON}: the extracted data have no recorded values for {columns}"


def _table_spec():
    return {
        "group_by": "arm", "group_levels": [0, 1],
        "variables": [{
            "name": COVARIATE, "variable_kind": "continuous",
            "summary": "median_iqr", "test": "mann_whitney_or_kruskal",
        }],
    }


def test_table_one_cannot_publish_an_empty_requested_covariate(monkeypatch, tmp_path):
    cohort = _materialize(monkeypatch, tmp_path, empty=COVARIATE)
    cohort["arm"] = [0, 0, 0, 0, 1, 1, 1, 1]
    with pytest.raises(ValueError) as caught:
        build_grouped_table_one(cohort, _table_spec())
    assert caught.value.reason_code == REASON


@pytest.mark.parametrize("empty", [EXPOSURE, OUTCOME])
def test_descriptive_association_names_the_unobserved_requested_variable(
    monkeypatch, tmp_path, empty,
):
    cohort = _materialize(monkeypatch, tmp_path, empty=empty)
    with pytest.raises(ValueError) as caught:
        run_descriptive_association(
            frame=cohort, predictor=EXPOSURE, outcome=OUTCOME,
            statistic_product="association", typed_cohort_input="artifact:analysis_cohort",
            source_cohort=tmp_path / "cohort.parquet", out_dir=tmp_path / "out",
        )
    assert caught.value.reason_code == REASON
    assert not (tmp_path / "out" / "association.json").exists()


@pytest.mark.parametrize("column", [EXPOSURE, OUTCOME, COVARIATE])
def test_partial_measurement_keeps_existing_denominators_and_execution(monkeypatch, tmp_path, column):
    cohort = _materialize(monkeypatch, tmp_path, partial=True, partial_column=column)
    assert cohort[column.removesuffix("_max") + "_measured"].tolist() == [1] * 6 + [0] * 2
    context = _context(cohort)
    assert analysis_answerability_findings(context) == []
    review = build_plan_scientific_review(context=context, plan=_plan())
    assert not any(item.code == "REQUESTED_CONCEPT_NO_DATA" for item in review.findings)
    compiled = primary_model_rows(cohort, terms=_terms(), exposure=EXPOSURE, outcome=OUTCOME)
    assert len(compiled.design) == 6
    cohort["arm"] = [0, 0, 0, 0, 1, 1, 1, 1]
    table = build_grouped_table_one(cohort, _table_spec())
    overall = table.loc[table["group"] == "Overall"].iloc[0]
    assert overall["denominator_n"] == 8
    assert overall["missing_n"] == (2 if column == COVARIATE else 0)
    result = run_descriptive_association(
        frame=cohort, predictor=EXPOSURE, outcome=OUTCOME,
        statistic_product="association", typed_cohort_input="artifact:analysis_cohort",
        source_cohort=tmp_path / "cohort.parquet", out_dir=tmp_path / "out",
    )
    assert result["status"] == "ok"


@pytest.mark.parametrize("requested,alternative,blocked", [
    ("fixture_adjuster", False, True),
    ("fixture_adjuster", True, False),
    (COVARIATE, True, True),
])
def test_concept_coverage_reads_values_and_respects_the_exact_requested_coordinate(
    monkeypatch, tmp_path, requested, alternative, blocked,
):
    cohort = _materialize(monkeypatch, tmp_path, empty=COVARIATE)
    if alternative:
        cohort["fixture_adjuster_alternative"] = [2., 3., None, None, None, None, None, None]
    context = _context(cohort)
    variables = []
    for variable in context.variables:
        if variable.name.startswith("fixture_adjuster_"):
            role = VariableRole.META if variable.name.endswith(("_measured", "_n")) else variable.role
            variable = variable.model_copy(update={"source_concept": "fixture_adjuster", "role": role})
        variables.append(variable)
    context = context.model_copy(update={
        "variables": variables,
        "user_preferences": context.user_preferences.model_copy(update={"covariates": [requested]}),
    })
    findings = analysis_answerability_findings(context)
    assert any(item.detail.get("kind") == REASON for item in findings) is blocked


def test_an_unrequested_empty_summary_is_retained_without_blocking(monkeypatch, tmp_path):
    cohort = _materialize(monkeypatch, tmp_path, empty=COVARIATE)
    context = _context(cohort).model_copy(update={"user_preferences": None})
    assert analysis_answerability_findings(context) == []
    review = build_plan_scientific_review(context=context, plan=_plan())
    assert not any(item.code == "REQUESTED_CONCEPT_NO_DATA" for item in review.findings)
    assert cohort[COVARIATE].isna().all() and len(cohort) == 8


def test_a_plan_chosen_table_one_variable_is_checked_even_without_a_context_request(
    monkeypatch, tmp_path,
):
    cohort = _materialize(monkeypatch, tmp_path, empty=COVARIATE)
    context = _context(cohort).model_copy(update={"user_preferences": None})
    step = AnalysisStep(
        step_id="baseline", planned_analysis_role="auxiliary", method="table_one",
        intent="Describe the requested adjuster.", inputs=["arm", COVARIATE],
        expected_outputs=["table:table_one"], table_one_spec=_table_spec(),
    )
    plan = _plan().model_copy(update={"steps": [step]})
    review = build_plan_scientific_review(context=context, plan=plan)
    assert any(item.code == "REQUESTED_CONCEPT_NO_DATA" for item in review.findings)


@pytest.mark.parametrize("column", [OUTCOME, COVARIATE])
def test_question_coverage_does_not_turn_an_owner_into_data(monkeypatch, tmp_path, column):
    context = _context(_materialize(monkeypatch, tmp_path, empty=column))
    requirement = QuestionRequirement(
        id="r1", kind="subgroup", quote="describe the synthetic subgroup",
        concepts=[column], coverage="plan",
    )
    coverage = question_requirement_coverage(
        [JudgedRequirement(
            requirement=requirement, disposition="covered", reason_code="typed_plan_owner_present",
            owner_step_ids=("association",),
        )], context=context, plan=_plan(),
    )
    row, = coverage.records
    assert not coverage.complete
    assert row.status == "unsupported" and row.reason_code == REASON
    assert row.owner_step_ids == ()


def test_accepted_baseline_coverage_does_not_offer_empty_value_columns(monkeypatch, tmp_path):
    context = _context(_materialize(monkeypatch, tmp_path, empty=COVARIATE))
    context = bind_baseline_requirements(context, {
        "schema_version": "easyicu.accepted_baseline_requirements/1",
        "source_plan_sha256": "a" * 64,
        "tables": [{"source_step_id": "baseline", "group_by": {"name": EXPOSURE},
                    "variables": [{"name": COVARIATE}]}],
    })
    step = AnalysisStep(
        step_id="baseline", intent="Describe the adjuster.", method="table_one",
        inputs=[EXPOSURE, COVARIATE], expected_outputs=["table:table_one"],
        table_one_spec={**_table_spec(), "group_by": EXPOSURE},
    )
    coverage = baseline_requirement_coverage(context, _plan().model_copy(update={"steps": [step]}))
    assert coverage["status"] == "incomplete"
    assert coverage["tables"][0]["variables"][0]["available_columns"] == []


@pytest.mark.parametrize("owner", ["cohort_summary", "grouped_distribution", "binary_distribution", "absolute_risk"])
@pytest.mark.parametrize("empty", [EXPOSURE, OUTCOME])
def test_descriptive_owners_cannot_publish_an_all_unmeasured_variable(
    monkeypatch, tmp_path, owner, empty,
):
    cohort = pd.DataFrame({
        EXPOSURE: [0., 0., 0., 1., 1., 1.], OUTCOME: [0., 1., 0., 1., 0., 1.],
        "fixture_signal_measured": [1] * 6, "fixture_response_measured": [1] * 6,
    })
    cohort[empty] = float("nan")
    cohort[empty.removesuffix("_max") + "_measured"] = 0
    context = {"target_outcome": OUTCOME, "primary_exposure": EXPOSURE, "variables": [
        {"name": EXPOSURE, "role": "other", "observed_domain": {"levels": [0, 1]}},
        {"name": OUTCOME, "role": "outcome", "unit": "unitless"},
    ]}
    (tmp_path / "research_context.json").write_text(json.dumps(context))
    out_dir = tmp_path / "out"
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("EASYICU_STEP_ID", "summary")
    monkeypatch.setenv("COHORT_PARQUET", str(tmp_path / "cohort.parquet"))
    (tmp_path / "analysis_plan.json").write_text(json.dumps({"steps": [{
        "step_id": "summary", "inputs": [EXPOSURE, OUTCOME],
        "expected_outputs": ["table:exposure_outcome_summary"],
    }]}))
    with pytest.raises(RequestedConceptNoDataError) as caught:
        if owner == "cohort_summary":
            run_cohort_summary_from_env(
                declared_columns=[EXPOSURE, OUTCOME], typed_cohort_input="artifact:analysis_cohort",
                frame=cohort, cohort_path=tmp_path / "cohort.parquet",
            )
        elif owner == "grouped_distribution":
            grouped_distribution_table(
                frame=cohort, grouping_variable=EXPOSURE, value_variable=OUTCOME, run_dir=tmp_path,
            )
        elif owner == "binary_distribution":
            _distribution_rows(cohort, spec=ExposureOutcomeDistributionSpec(
                exposure=EXPOSURE, exposure_levels=[0, 1], outcome=OUTCOME,
                outcome_levels=[0, 1], outcome_positive_value=1,
                level_match_policy="exact_typed", denominator_policy="all_declared_rows",
                missing_outcome_policy="structural_absence_is_non_event", confidence_level=0.95,
            ))
        else:
            run_absolute_risk_context(population_frame=cohort)
    assert caught.value.reason_code == REASON
    assert caught.value.columns == (empty,)
    assert not list(out_dir.glob("*.csv"))
