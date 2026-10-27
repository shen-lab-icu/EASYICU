"""A binary association reports a risk ratio only when its plan declares one.

The odds ratio stays every binary model's default.  When the question asks for
a risk ratio (relative risk), the plan declares ``effect_measure`` and the
compiler binds the modified-Poisson estimator; nothing switches it from the
data.  Where an owner reports odds ratios only (the signed landmark spline
suite, the ordered logistic trend), a requested risk ratio is a typed,
host-verified gap or a stable refusal, never an odds ratio under the
risk-ratio name.  Synthetic contexts and small synthetic cohorts only.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.family_spec_planner import (
    family_spec_response_shape,
    family_spec_structured_output_request,
    family_spec_user_prompt,
)
from easyicu.research_agent.audits.validators import PrimaryModelContractValidator
from easyicu.research_agent.authority.scientific_claims import derive_scientific_claim_drafts
from easyicu.research_agent.contracts.association_execution import primary_binary_effect_measure
from easyicu.research_agent.contracts.model_tokens import (
    ASSOCIATION_LOGIT_ESTIMATOR,
    ASSOCIATION_OLS_ESTIMATOR,
    ASSOCIATION_RR_ESTIMATOR,
)
from easyicu.research_agent.execution.runners.baseline_association_executor import run_baseline_step
from easyicu.research_agent.planning.family_spec import (
    LANDMARK_CATEGORICAL_FAMILY_ID,
    LANDMARK_SPLINE_FAMILY_ID,
    FamilySpecError,
    validate_family_plan_spec,
)
from easyicu.research_agent.planning.family_spec.contract import (
    family_effect_measures,
    spec_from_mapping,
)
from easyicu.research_agent.planning.progressive_compiler import _compile_adjusted_association
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressiveSkeletonStep,
)
from easyicu.research_agent.planning.question_requirements import (
    QuestionRequirement,
    judge_question_requirements,
)
from easyicu.research_agent.providers.structured_diagnostics import safe_validation_issues
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    EndpointSpec,
    ResearchContext,
    VariableRole,
)
from tests.research_agent.planning.family_spec_fixtures import (
    PLANNER_ROSTER,
    _context as _landmark_context,
    _request as _landmark_request,
    _run as _landmark_run,
    _spec_payload as _landmark_payload,
)
from tests.support.baseline_association import baseline_context, baseline_plan_attempt, baseline_runtime

RR_WORDS = "reported as a risk ratio"
GAP = {
    "requirement": "estimand_unsupported", "concept": None, "element": "analysis",
    "detail": "The signed spline model of this design reports odds ratios only.",
}


def _rr_requirement(**updates):
    return {
        "id": "r1", "kind": "estimand", "quote": RR_WORDS, "concepts": [], "coverage": "plan",
        "gap": None, "note": None, "effect_measure": "risk_ratio", **updates,
    }


# ---------------------------------------------------------------------------
# The compiler: one declared coordinate selects the estimator.
# ---------------------------------------------------------------------------


NEUTRAL = "Assess an ordered stage against mortality and length of stay."


def _ordinal_context(question: str = NEUTRAL) -> ResearchContext:
    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(
            cohort_name="synthetic", database="synthetic", n_stays=0, id_columns=["stay_id"],
            outcome_columns=["los_days", "death"],
        ),
        variables=[
            ConceptDescriptor(name="stage", role=VariableRole.ORDINAL_SCORE, dtype="float64",
                              is_ordinal=True, ordinal_levels=[0, 1, 2, 3]),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="float64"),
            ConceptDescriptor(name="los_days", role=VariableRole.OUTCOME, dtype="float64"),
        ],
        primary_exposure="stage", target_outcome="death",
        endpoint=EndpointSpec(name="death", kind="binary", absence_semantics="no_absent_rows", levels=[0, 1]),
    )


def _primary(**updates) -> ProgressiveSkeletonStep:
    return ProgressiveSkeletonStep.model_validate({
        "step_id": "primary_adjusted_model", "planned_analysis_role": "primary",
        "module_id": "adjusted_association", "objective": "Estimate the adjusted stage and mortality association.",
        "raw_inputs": ["stage", "death"], "primary_exposure": "stage", "outcome": "death",
        "outcome_type": "binary",
        "model_terms": [{"name": "stage", "role": "exposure", "coding": "categorical", "reference_level_index": 0}],
        "event_level_index": 1, "primary_contrast_level_index": 3, "confidence_level": 0.95,
        **updates,
    })


def _compiled(step, question: str = NEUTRAL):
    context = _ordinal_context(question)
    (requirement,) = _compile_adjusted_association(
        context=context, variables={item.name: item for item in context.variables}, step=step, step_index=0,
    )
    return requirement


def test_the_compiler_binds_the_declared_measure_and_keeps_the_odds_ratio_default():
    default = _primary()
    # Absent, the field adds nothing to the step's bytes or digest.
    assert "effect_measure" not in default.model_dump(mode="json")
    assert _compiled(default).method_family == ASSOCIATION_LOGIT_ESTIMATOR
    assert _compiled(_primary(effect_measure="odds_ratio")).method_family == ASSOCIATION_LOGIT_ESTIMATOR
    asked = NEUTRAL.rstrip(".") + f", {RR_WORDS}."
    assert _compiled(_primary(effect_measure="risk_ratio"), asked).method_family == ASSOCIATION_RR_ESTIMATOR


def test_a_risk_ratio_needs_a_binary_outcome():
    continuous = {"outcome": "los_days", "outcome_type": "continuous", "raw_inputs": ["stage", "los_days"],
                  "event_level_index": None}
    assert _compiled(_primary(**continuous)).method_family == ASSOCIATION_OLS_ESTIMATOR
    with pytest.raises(ProgressivePlanCompileError) as refused:
        _compiled(_primary(**continuous, effect_measure="risk_ratio"))
    assert refused.value.reason_code == "progressive_effect_measure_requires_binary_outcome"


def test_an_unknown_measure_is_refused_and_named_in_the_diagnostics():
    with pytest.raises(ValidationError) as refused:
        _primary(effect_measure="hazard_ratio")
    assert ["effect_measure"] in [issue["location"] for issue in safe_validation_issues(refused.value)]


# ---------------------------------------------------------------------------
# The baseline family: declared, compiled, fitted and reported as a risk ratio.
# ---------------------------------------------------------------------------


def _frame(exposure: str) -> pd.DataFrame:
    rng = np.random.default_rng(20261026)
    n = 600
    frame = pd.DataFrame({
        "stay_id": np.arange(n), "age": rng.uniform(20, 85, n), "sex": rng.integers(0, 2, n),
        "chronic_flag": rng.integers(0, 2, n),
        "admission_type": rng.choice(["elective", "emergency", "urgent"], n),
    })
    exposed = frame["sex"] if exposure == "sex" else (frame["age"] - 20) / 65
    frame["death"] = rng.binomial(1, 0.15 + 0.2 * exposed + 0.05 * frame["chronic_flag"]).astype(bool)
    return frame


def _rr_context(exposure, covariates):
    context = baseline_context(exposure, covariates=covariates)
    return context.model_copy(update={
        "research_question": context.research_question.rstrip(".") + f", {RR_WORDS}.",
    })


def _executed(context, plan, tmp_path):
    runtime = baseline_runtime(context, plan)
    bound, _ = runtime.bind_plan(plan)
    sealed = runtime.seal_for_plan(plan).current_case
    frame = _frame(context.primary_exposure)
    summary = run_baseline_step(
        step_id="adjusted_association", plan=bound, authority=sealed, runtime_projection_sha256="f" * 64,
        frame=frame, output_dir=tmp_path / "primary",
    )
    estimates = pd.read_csv(tmp_path / "primary" / "adjusted_association_estimates.csv")
    risk = run_baseline_step(
        step_id="absolute_risk_context", plan=bound, authority=sealed, runtime_projection_sha256="f" * 64,
        frame=frame, output_dir=tmp_path / "risk", primary_table=estimates,
    )
    return bound, frame, summary, estimates, risk


def test_the_odds_ratio_stays_the_default(tmp_path):
    context = baseline_context("sex", covariates=("chronic_flag",))
    default, _ = baseline_plan_attempt(context)
    explicit, _ = baseline_plan_attempt(context, spec_fields={"effect_measure": None})
    # Declaring nothing and declaring null are the same plan, step for step.
    assert [step.model_dump(mode="json") for step in default.output.steps] == [
        step.model_dump(mode="json") for step in explicit.output.steps
    ]
    plan = default.output
    assert plan.steps[3].model_requirements[0].method_family == ASSOCIATION_LOGIT_ESTIMATOR
    assert plan.design_selection.candidates[0].primary_method.startswith("Logistic model")
    _, _, summary, estimates, _ = _executed(context, plan, tmp_path)
    assert (summary["effect_scale"], summary["estimator_kind"]) == ("odds_ratio", "logistic")
    assert summary["primary_or"] == summary["primary_estimate"] and "primary_rr" not in summary
    assert set(estimates["effect_scale"]) == {"odds_ratio"}


@pytest.mark.parametrize("exposure,covariates", [
    ("sex", ("chronic_flag",)),
    ("admission_type", ("age", "chronic_flag")),
    ("age", ("sex",)),
])
def test_a_declared_risk_ratio_is_fitted_and_reported_as_a_risk_ratio(exposure, covariates, tmp_path):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    context = _rr_context(exposure, covariates)
    attempt, _ = baseline_plan_attempt(context, spec_fields={
        "effect_measure": "risk_ratio", "question_requirements": [_rr_requirement()],
    })
    plan = attempt.output
    primary = plan.steps[3]
    assert primary.model_requirements[0].method_family == ASSOCIATION_RR_ESTIMATOR
    assert primary_binary_effect_measure(primary) == "risk_ratio"
    design = plan.design_selection.candidates[0]
    assert design.primary_method.startswith("Modified Poisson model (log link, robust standard errors)")
    assert not any("logistic" in line.casefold() for line in design.reviewable_plan)

    bound, frame, summary, estimates, risk = _executed(context, plan, tmp_path)
    assert (summary["effect_scale"], summary["estimator_kind"]) == ("risk_ratio", "glm_poisson")
    issues, _ = PrimaryModelContractValidator._planned_model_requirement_issues(
        step=bound.steps[3], contracts=summary["model_contracts"],
    )
    assert issues == []
    assert summary["primary_rr"] == summary["primary_estimate"] and "primary_or" not in summary
    assert summary["model_contracts"][0]["method_family"] == ASSOCIATION_RR_ESTIMATOR
    assert set(estimates["effect_scale"]) == {"risk_ratio"}
    assert set(estimates["estimator_kind"]) == {"glm_poisson"}
    assert risk["status"] == "ok"
    [claim] = derive_scientific_claim_drafts(summary)
    assert claim.estimand == "adjusted risk ratio"
    if exposure == "sex":
        # The same modified-Poisson model, fitted independently.
        independent = smf.glm(
            "y ~ sex + chronic_flag", data=frame.assign(y=frame["death"].astype(int)),
            family=sm.families.Poisson(),
        ).fit(cov_type="HC0")
        headline = estimates.loc[estimates["is_primary_contrast"]].iloc[0]
        assert headline["estimate"] == pytest.approx(math.exp(independent.params["sex"]), rel=1e-6)
        low, high = np.exp(independent.conf_int().loc["sex"])
        assert (headline["ci_low"], headline["ci_high"]) == (pytest.approx(low, rel=1e-6), pytest.approx(high, rel=1e-6))
    if exposure == "age":
        # Standardized risks under the log link stay risks, and their contrast adds up.
        rows = pd.read_csv(tmp_path / "risk" / "absolute_risk_context.csv")
        assert rows["absolute_risk"].between(0, 1).all() and rows["reference_risk"].between(0, 1).all()
        assert (rows["absolute_risk"] - rows["reference_risk"] - rows["risk_difference"]).abs().max() < 1e-10

    judged = judge_question_requirements(
        [QuestionRequirement.model_validate(_rr_requirement())], plan=bound, family_template=True,
        relatives=lambda concept: frozenset({concept}), check_gap=lambda gap: ("unverifiable", ""),
    )
    assert (judged[0].disposition, judged[0].host_judged) == ("covered", True)
    assert judged[0].owner_step_ids == ("adjusted_association",)


def test_a_question_asking_for_a_risk_ratio_is_not_answered_by_an_odds_ratio():
    # An odds-ratio plan (its question names no measure), judged on a stated
    # risk-ratio requirement.
    plan = baseline_plan_attempt(baseline_context("sex", covariates=("chronic_flag",)))[0].output
    requirement = QuestionRequirement.model_validate(_rr_requirement())

    def judged(item, *, family_template=True):
        return judge_question_requirements(
            [item], plan=plan, family_template=family_template,
            relatives=lambda concept: frozenset({concept}),
            check_gap=lambda gap: ("unverifiable", "an estimand gap needs review"),
        )[0]

    refused = judged(requirement)
    assert (refused.disposition, refused.host_judged) == ("not_covered", True)
    assert refused.row()["effect_measure"] == "risk_ratio"
    gap = QuestionRequirement.model_validate({**_rr_requirement(), "coverage": "capability_gap", "gap": GAP})
    verified = judged(gap)
    assert (verified.disposition, verified.verification, verified.host_judged) == ("capability_gap", "verified", True)
    assert verified.reason_code == "question_effect_measure_unavailable_in_family_template"
    assert "risk ratio" in verified.gap.detail
    # The outline route has no family to verify against: the gap stays a claim.
    assert judged(gap, family_template=False).verification == "unverifiable"
    # A requirement that names no measure keeps every recorded row byte.
    plain = QuestionRequirement.model_validate({**_rr_requirement(), "effect_measure": None})
    assert "effect_measure" not in judged(plain).row()
    with pytest.raises(ValidationError, match="only an estimand"):
        QuestionRequirement.model_validate({**_rr_requirement(), "kind": "analysis"})


# ---------------------------------------------------------------------------
# The spec contract: a measure only where the family reports it.
# ---------------------------------------------------------------------------


def _baseline_request():
    from easyicu.research_agent.planning.family_spec import build_family_spec_request

    context = _rr_context("sex", ("chronic_flag",))
    return build_family_spec_request(
        context, analysis_types=("association_study",), variable_roster=[v.name for v in context.variables],
        allowed_literature_citation_keys=(), required_primary_cohort_selection_mode="all_input_rows",
    )


def _baseline_spec(request, **fields):
    return spec_from_mapping({
        "schema_version": "easyicu.family_plan_spec/1", "family_id": request.family_id,
        "request_sha256": request.request_sha256, "adjustment_set": [],
        "reader_display_labels": [{"key": key, "value": "Clinical " + key.replace("_", " ")}
                                  for key in request.required_reader_label_keys],
        "comparator_applications": [], "roster_decision_note": "Keep the study's exact stated adjustment roster.",
        **fields,
    })


def test_the_spec_contract_holds_the_measure_to_the_question():
    request = _baseline_request()
    assert family_effect_measures(request) == ("odds_ratio", "risk_ratio")
    validate_family_plan_spec(_baseline_spec(request, effect_measure="risk_ratio",
                                             question_requirements=[_rr_requirement()]), request)
    for fields, code in (
        # Claiming the question's risk ratio while the plan reports an odds ratio.
        ({"question_requirements": [_rr_requirement()]}, "family_spec_effect_measure_mismatch"),
        # A gap for a measure this family reports is not a gap.
        ({"question_requirements": [_rr_requirement(coverage="capability_gap", gap=GAP)]},
         "family_spec_effect_measure_gap_refuted"),
    ):
        with pytest.raises(FamilySpecError) as refused:
            validate_family_plan_spec(_baseline_spec(request, **fields), request)
        assert refused.value.reason_code == code


def test_the_measure_is_offered_where_the_family_reports_it():
    request = _baseline_request()
    schema = json.loads(family_spec_structured_output_request(request).schema_json)
    assert schema["properties"]["effect_measure"]["anyOf"][0]["enum"] == ["odds_ratio", "risk_ratio"]
    assert "effect_measure" in schema["required"]
    items = schema["properties"]["question_requirements"]["items"]
    assert "effect_measure" in items["required"]
    shape = family_spec_response_shape(request)
    assert '"effect_measure": null (odds ratio) or one of ["odds_ratio", "risk_ratio"]' in shape
    prompt = family_spec_user_prompt(request, variable_descriptions={})
    assert "set effect_measure to risk_ratio (modified Poisson, robust SE) only when" in prompt


def _spline_context() -> ResearchContext:
    base = _landmark_context(exact=False)
    lactate = ConceptDescriptor(
        name="peak_lactate", description="lactate", role=VariableRole.LAB, dtype="float32", unit="mmol/L",
        source_concept="lact", analysis_window="icu_admission[0,24]h",
    )
    keep = {"stay_id", "patient_stay_id", "age", "sex", "death_time_hours", "followup_time_hours",
            "first_stay_flag", "death"}
    constraints = json.loads(base.user_preferences.data_constraints)
    constraints["stated_adjustments"] = [
        item for item in constraints["stated_adjustments"] if item["concept"] in {"age", "sex"}
    ]
    preferences = base.user_preferences.model_copy(update={
        "sensitivity_specs": [spec for spec in base.user_preferences.sensitivity_specs
                              if spec.axis != "exposure_definition"],
        "data_constraints": json.dumps(constraints),
    })
    return base.model_copy(update={
        "research_question": (
            "Among adult ICU stays, is the peak lactate in the first 24 h associated with in-hospital "
            f"death after a 24 h landmark, {RR_WORDS}?"
        ),
        "primary_exposure": "peak_lactate",
        "variables": [item for item in base.variables if item.name in keep] + [lactate],
        "user_preferences": preferences,
        "cohort": base.cohort.model_copy(update={"outcome_columns": ["death"], "requested_outcome_columns": ["death"]}),
    })


def _spline_spec(request, **fields):
    return spec_from_mapping({
        "schema_version": "easyicu.family_plan_spec/1", "family_id": request.family_id,
        "request_sha256": request.request_sha256, "adjustment_set": PLANNER_ROSTER[:2],
        "reader_display_labels": [{"key": key, "value": "Clinical " + key.replace("_", " ")}
                                  for key in request.required_reader_label_keys],
        "comparator_applications": [
            {"citation_key": key, "application": f"Compare population, exposure and estimand with {key}."}
            for key in request.direct_comparator_literature_keys
        ],
        "roster_decision_note": "Planner-selected baseline covariates within the host timing authority.",
        **fields,
    })


def test_a_spline_family_reports_odds_ratios_so_a_requested_risk_ratio_is_a_typed_gap():
    request = _landmark_request(_spline_context())
    assert request.family_id == LANDMARK_SPLINE_FAMILY_ID
    assert family_effect_measures(request) == ("odds_ratio",)
    for fields, code in (
        ({"effect_measure": "risk_ratio"}, "family_spec_effect_measure_unsupported"),
        ({"question_requirements": [_rr_requirement()]}, "family_spec_effect_measure_mismatch"),
        # The question asks for a risk ratio: the spec states the gap, not silence.
        ({}, "family_spec_effect_measure_gap_required"),
    ):
        with pytest.raises(FamilySpecError) as refused:
            validate_family_plan_spec(_spline_spec(request, **fields), request)
        assert refused.value.reason_code == code
    validate_family_plan_spec(
        _spline_spec(request, question_requirements=[_rr_requirement(coverage="capability_gap", gap=GAP)]),
        request,
    )
    # The spec is not offered a measure it cannot report; the requirement can name the one asked for.
    schema = json.loads(family_spec_structured_output_request(request).schema_json)
    assert "effect_measure" not in schema["properties"]
    items = schema["properties"]["question_requirements"]["items"]["properties"]
    assert items["effect_measure"]["anyOf"][0]["enum"] == ["odds_ratio", "risk_ratio"]
    assert "reports odds ratios only" in family_spec_user_prompt(request, variable_descriptions={})
    # The signed spline primary is read as an odds-ratio primary.
    signed = SimpleNamespace(scientific_capability="association_landmark_spline_v1", model_requirements=[])
    assert primary_binary_effect_measure(signed) == "odds_ratio"


# ---------------------------------------------------------------------------
# The categorical landmark family: a risk ratio unless its logistic trend is drafted.
# ---------------------------------------------------------------------------


def _categorical_without_trend() -> ResearchContext:
    base = _landmark_context()
    return base.model_copy(update={
        "research_question": base.research_question.rstrip("?") + f", {RR_WORDS}?",
        "cohort": base.cohort.model_copy(update={"outcome_columns": ["death"], "requested_outcome_columns": ["death"]}),
        "user_preferences": base.user_preferences.model_copy(update={
            "must_have_outputs": "Generate figures for the primary result.",
        }),
    })


def test_the_categorical_landmark_trend_keeps_the_odds_ratio():
    request = _landmark_request(_landmark_context())
    assert request.family_id == LANDMARK_CATEGORICAL_FAMILY_ID
    assert family_effect_measures(request) == ("odds_ratio",)
    with pytest.raises(FamilySpecError) as refused:
        validate_family_plan_spec(spec_from_mapping(_landmark_payload(request, effect_measure="risk_ratio")), request)
    assert refused.value.reason_code == "family_spec_effect_measure_unsupported"


def test_a_categorical_landmark_plan_compiles_a_declared_risk_ratio():
    context = _categorical_without_trend()
    request = _landmark_request(context)
    assert request.family_id == LANDMARK_CATEGORICAL_FAMILY_ID
    assert family_effect_measures(request) == ("odds_ratio", "risk_ratio")
    payload = _landmark_payload(request, effect_measure="risk_ratio", question_requirements=[_rr_requirement()])
    _, result = _landmark_run(context, [json.dumps(payload)])
    plan = result.output
    primary = next(step for step in plan.steps if step.planned_analysis_role == "primary")
    assert primary.model_requirements[0].method_family == ASSOCIATION_RR_ESTIMATOR
    assert "ordinal_trend" not in {step.step_id for step in plan.steps}
    design = plan.design_selection.candidates[0]
    assert "modified Poisson model (log link, robust standard errors)" in design.primary_method
    assert any(line.startswith("Modified Poisson model") for line in design.reviewable_plan)
