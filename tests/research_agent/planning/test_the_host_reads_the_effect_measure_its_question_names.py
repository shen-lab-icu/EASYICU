"""The host reads the effect measure a question names; the Planner neither drops nor invents one.

Whether a binary primary reports a risk ratio is decided by the question's own
words, read deterministically (``question_effect_measures``), never by the
Planner alone.  A question that asks for a risk ratio is not answered by an
odds ratio, and a risk ratio the question does not ask for is not fitted: the
outline compiler and the family spec contract refuse both within the Planner's
repair budget, and the question-requirements owner stops approval of any plan
that still reports the wrong measure, on the compiled plan and on the plan
offered for review.  Synthetic contexts only.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.contracts.model_tokens import (
    ASSOCIATION_LOGIT_ESTIMATOR,
    ASSOCIATION_RR_ESTIMATOR,
)
from easyicu.research_agent.orchestration.progressive_planning import question_requirement_outcome
from easyicu.research_agent.planning.family_spec import (
    FamilySpecError,
    build_family_spec_request,
    validate_family_plan_spec,
)
from easyicu.research_agent.planning.family_spec.contract import spec_from_mapping
from easyicu.research_agent.planning.progressive_compiler import _compile_adjusted_association
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressiveSkeletonStep,
)
from easyicu.research_agent.planning.question_requirements import (
    QUESTION_REQUIREMENTS_FILENAME,
    QuestionRequirement,
    judge_recorded_requirements,
    question_effect_measures,
    question_requirements_on_plan_under_review,
)
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    EndpointSpec,
    ResearchContext,
    VariableRole,
)
from tests.support.baseline_association import baseline_context, baseline_plan_attempt

EICU_RR = (
    "在 eICU 中，入 ICU 第一天最高乳酸超过 2 mmol/L 的患者，院内死亡风险是否更高？"
    "调整年龄、性别和疾病严重程度后，用风险比（RR）表示。"
)
RR, OR = "risk_ratio", "odds_ratio"


# ---------------------------------------------------------------------------
# The reader.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question,measures", [
    (EICU_RR, (RR,)),
    ("最高乳酸与院内死亡是否相关？调整年龄和性别后还成立吗？用风险比（RR）表示。", (RR,)),
    ("计算校正后的相对危险度", (RR,)),
    ("以优势比表示", (OR,)),
    ("报告OR值及95%置信区间", (OR,)),
    ("What is the relative risk of death?", (RR,)),
    ("Express the association as an odds ratio.", (OR,)),
    ("Report the adjusted OR (95% CI).", (OR,)),
    ("aOR for in-hospital death", (OR,)),
    # Both named: the question asks for both, and a primary reports one.
    ("OR 和 RR 都要", (OR, RR)),
    ("Report the odds ratio or RR.", (OR, RR)),
    # Negated or replaced: only the measure asked for counts.
    ("用 RR 而不是 OR 表示", (RR,)),
    ("用RR代替OR", (RR,)),
    ("不要用比值比，用风险比", (RR,)),
    ("Report the risk ratio rather than the odds ratio.", (RR,)),
    ("Do not report an odds ratio; use a risk ratio.", (RR,)),
    ("不要报告风险比", ()),
    # Not a measure: a conjunction, an operating room, a respiratory rate,
    # renal replacement therapy, "risk than", "relatively", a hazard ratio.
    ("Is lactate associated with death or survival?", ()),
    ("Is sepsis OR shock associated with death?", ()),
    ("Patients admitted from the OR after surgery: is lactate associated with death?", ()),
    ("Is a respiratory rate (RR) above 22 associated with death?", ()),
    ("RR ≥ 22 次/分的患者死亡率是否更高？", ()),
    ("Does RRT use predict death?", ()),
    ("高乳酸组的死亡风险比低乳酸组更高吗？", ()),
    ("患者的死亡风险比对照组高吗？", ()),
    ("死亡风险比较高的患者", ()),
    ("相对风险较高的人群", ()),
    ("用风险比（HR）表示生存差异", ()),
    ("Is peak lactate associated with in-hospital death?", ()),
])
def test_the_reader_reads_the_measure_the_question_names(question, measures):
    assert tuple(item.measure for item in question_effect_measures(question)) == measures


def test_the_reading_quotes_the_question():
    (reading,) = question_effect_measures(EICU_RR)
    assert reading.quote == "风险比" and reading.quote in EICU_RR
    assert question_effect_measures("计算相对危险度")[0].quote == "相对危险度"


# ---------------------------------------------------------------------------
# The outline route: the compiler refuses within the Planner's repair budget.
# ---------------------------------------------------------------------------

NEUTRAL = "Assess an ordered stage against in-hospital death."


def _compiled(question: str, **updates):
    context = ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_stays=0,
                                id_columns=["stay_id"], outcome_columns=["death"]),
        variables=[
            ConceptDescriptor(name="stage", role=VariableRole.ORDINAL_SCORE, dtype="float64",
                              is_ordinal=True, ordinal_levels=[0, 1, 2, 3]),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="float64"),
        ],
        primary_exposure="stage", target_outcome="death",
        endpoint=EndpointSpec(name="death", kind="binary", absence_semantics="no_absent_rows", levels=[0, 1]),
    )
    step = ProgressiveSkeletonStep.model_validate({
        "step_id": "primary_adjusted_model", "planned_analysis_role": "primary",
        "module_id": "adjusted_association", "objective": "Estimate the adjusted stage and death association.",
        "raw_inputs": ["stage", "death"], "primary_exposure": "stage", "outcome": "death",
        "outcome_type": "binary",
        "model_terms": [{"name": "stage", "role": "exposure", "coding": "categorical", "reference_level_index": 0}],
        "event_level_index": 1, "primary_contrast_level_index": 3, "confidence_level": 0.95,
        **updates,
    })
    (requirement,) = _compile_adjusted_association(
        context=context, variables={item.name: item for item in context.variables}, step=step, step_index=0,
    )
    return requirement.method_family


@pytest.mark.parametrize("question,declared,expected", [
    (NEUTRAL, None, ASSOCIATION_LOGIT_ESTIMATOR),
    (NEUTRAL + " Report the odds ratio.", None, ASSOCIATION_LOGIT_ESTIMATOR),
    (NEUTRAL + " Report the odds ratio.", OR, ASSOCIATION_LOGIT_ESTIMATOR),
    (NEUTRAL + " Report the risk ratio.", RR, ASSOCIATION_RR_ESTIMATOR),
    (EICU_RR, RR, ASSOCIATION_RR_ESTIMATOR),
    # Both named: either compiles; approval then stops on the other.
    ("OR 和 RR 都要", None, ASSOCIATION_LOGIT_ESTIMATOR),
    # (a) the question asks for a risk ratio, the step leaves it undeclared.
    (EICU_RR, None, "progressive_effect_measure_not_declared"),
    (EICU_RR, OR, "progressive_effect_measure_not_declared"),
    # (b) the step declares a risk ratio the question does not ask for.
    (NEUTRAL, RR, "progressive_effect_measure_unrequested"),
    (NEUTRAL + " Report the odds ratio.", RR, "progressive_effect_measure_unrequested"),
])
def test_the_compiler_holds_the_step_to_the_question(question, declared, expected):
    updates = {"effect_measure": declared} if declared else {}
    if expected.startswith("progressive_"):
        with pytest.raises(ProgressivePlanCompileError) as refused:
            _compiled(question, **updates)
        assert refused.value.reason_code == expected
        assert refused.value.path == "effect_measure"
    else:
        assert _compiled(question, **updates) == expected


# ---------------------------------------------------------------------------
# The family route: the spec contract refuses, and the Planner repairs.
# ---------------------------------------------------------------------------


def _requirement(quote: str, **updates) -> dict:
    return {"id": "r1", "kind": "estimand", "quote": quote, "concepts": [], "coverage": "plan",
            "gap": None, "note": None, "effect_measure": RR, **updates}


def _context(question: str | None = None) -> ResearchContext:
    context = baseline_context("sex", covariates=("chronic_flag",))
    return context if question is None else context.model_copy(update={"research_question": question})


def _spec(context, **fields):
    request = build_family_spec_request(
        context, analysis_types=("association_study",), variable_roster=[v.name for v in context.variables],
        allowed_literature_citation_keys=(), required_primary_cohort_selection_mode="all_input_rows",
    )
    return spec_from_mapping({
        "schema_version": "easyicu.family_plan_spec/1", "family_id": request.family_id,
        "request_sha256": request.request_sha256, "adjustment_set": [],
        "reader_display_labels": [{"key": key, "value": "Clinical " + key.replace("_", " ")}
                                  for key in request.required_reader_label_keys],
        "comparator_applications": [], "roster_decision_note": "Keep the study's exact stated adjustment roster.",
        **fields,
    }), request


RR_QUESTION = "Association of sex with death, adjusted for chronic_flag, reported as a risk ratio."
OR_QUESTION = "Association of sex with death, adjusted for chronic_flag, reported as an odds ratio."


@pytest.mark.parametrize("question,fields,code", [
    (None, {}, None),
    (OR_QUESTION, {}, None),
    (RR_QUESTION, {"effect_measure": RR, "question_requirements": [_requirement("reported as a risk ratio")]}, None),
    (RR_QUESTION, {}, "family_spec_effect_measure_not_declared"),
    (None, {"effect_measure": RR}, "family_spec_effect_measure_unrequested"),
    (OR_QUESTION, {"effect_measure": RR}, "family_spec_effect_measure_unrequested"),
])
def test_the_spec_contract_holds_the_spec_to_the_question(question, fields, code):
    spec, request = _spec(_context(question), **fields)
    if code is None:
        validate_family_plan_spec(spec, request)
        return
    with pytest.raises(FamilySpecError) as refused:
        validate_family_plan_spec(spec, request)
    assert refused.value.reason_code == code


def test_the_family_planner_repairs_a_dropped_risk_ratio_within_its_budget():
    context = _context(RR_QUESTION)
    repaired = {"effect_measure": RR, "question_requirements": [_requirement("reported as a risk ratio")]}
    result, llm = baseline_plan_attempt(context, retry_spec_fields=[repaired])
    assert len(llm.calls) == 2
    assert "family_spec_effect_measure_not_declared" in llm.calls[1][0][-1].content
    # The host told the Planner what it read, before the first answer.
    assert "The question names: risk_ratio ('risk ratio')" in llm.calls[0][0][-1].content
    assert result.output.steps[3].model_requirements[0].method_family == ASSOCIATION_RR_ESTIMATOR


# ---------------------------------------------------------------------------
# The question-requirements owner: approval stops on the wrong measure.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plans():
    odds = baseline_plan_attempt(_context())[0].output
    risk = baseline_plan_attempt(_context(RR_QUESTION), spec_fields={
        "effect_measure": RR, "question_requirements": [_requirement("reported as a risk ratio")],
    })[0].output
    return {OR: odds, RR: risk}


def _outcome(plan, question, tmp_path, requirements=()):
    return question_requirement_outcome(
        context=_context(question), plan=plan, run_dir=tmp_path,
        facts=SimpleNamespace(family_template=True, question_requirements=tuple(requirements)),
    )


def _stops(findings):
    return [finding for finding in findings if finding.detail.get("approval_allowed") is False]


def test_a_question_asking_for_a_risk_ratio_stops_an_odds_ratio_plan(plans, tmp_path):
    # (a): the Planner named no measure and fitted the odds-ratio default.
    findings = _outcome(plans[OR], EICU_RR, tmp_path)
    (stop,) = _stops(findings)
    assert stop.detail["reason"] == "question_requirement_not_covered"
    (row,) = stop.detail["requirements"]
    assert (row["reason_code"], row["effect_measure"], row["quote"]) == (
        "question_effect_measure_not_reported", RR, "风险比",
    )
    assert row["read_from_question"] is True and row["verified_by_host"] is True
    assert "asks for a risk ratio ('风险比'), but no primary step reports one" in stop.message
    record = json.loads((tmp_path / QUESTION_REQUIREMENTS_FILENAME).read_text())
    assert record["question_effect_measures"] == [{"measure": RR, "quote": "风险比"}]
    assert not record["coverage"]["complete"]
    # The plan offered for review is judged again from the record alone.
    (again,) = _stops(question_requirements_on_plan_under_review([], plan=plans[OR], run_dir=tmp_path))
    assert again.detail["requirements"][0]["reason_code"] == "question_effect_measure_not_reported"
    # The same record passes the risk-ratio plan.
    assert not _stops(judge_recorded_requirements(record, plan=plans[RR]).findings())


def test_a_risk_ratio_the_question_does_not_ask_for_stops_the_plan(plans, tmp_path):
    # (b): the Planner switched the estimator on its own initiative.
    (stop,) = _stops(_outcome(plans[RR], None, tmp_path))
    assert stop.detail["reason"] == "question_requirement_not_covered"
    assert stop.detail["reason_code"] == "question_effect_measure_unrequested"
    (row,) = stop.detail["requirements"]
    assert (row["reason_code"], row["quote"], row["read_from_question"]) == (
        "question_effect_measure_unrequested", "", False,
    )
    assert "reports a risk ratio, which the question does not ask for" in stop.message
    record = json.loads((tmp_path / QUESTION_REQUIREMENTS_FILENAME).read_text())
    assert record["question_effect_measures"] == []
    assert _stops(judge_recorded_requirements(record, plan=plans[RR]).findings())
    assert not _stops(judge_recorded_requirements(record, plan=plans[OR]).findings())


@pytest.mark.parametrize("plan,question", [(RR, RR_QUESTION), (RR, EICU_RR), (OR, OR_QUESTION), (OR, None)])
def test_a_plan_reporting_the_measure_its_question_names_is_not_stopped(plans, plan, question, tmp_path):
    requirements = (
        [QuestionRequirement.model_validate(_requirement("reported as a risk ratio"))]
        if question == RR_QUESTION else []
    )
    findings = _outcome(plans[plan], question, tmp_path, requirements)
    assert not _stops(findings)
    if question is None:
        # Nothing named and nothing to judge: no record, as before.
        assert findings == [] and not (tmp_path / QUESTION_REQUIREMENTS_FILENAME).exists()
    elif question != RR_QUESTION:
        record = json.loads((tmp_path / QUESTION_REQUIREMENTS_FILENAME).read_text())
        (row,) = record["judged"]
        assert (row["disposition"], row["owner_step_ids"]) == ("covered", ["adjusted_association"])


def test_a_question_naming_both_measures_cannot_be_answered_by_one_primary(plans, tmp_path):
    (stop,) = _stops(_outcome(plans[RR], "OR 和 RR 都要", tmp_path))
    (row,) = stop.detail["requirements"]
    assert (row["effect_measure"], row["reason_code"]) == (OR, "question_effect_measure_not_reported")
    assert "a primary model reports one measure, so ask for one of the two" in stop.message


def test_a_planner_requirement_for_the_measure_is_judged_once(plans, tmp_path):
    # The Planner's own gap for the measure carries the stop; the host does not add a second.
    gap = QuestionRequirement.model_validate(_requirement("风险比", coverage="capability_gap", gap={
        "requirement": "estimand_unsupported", "concept": None, "element": "analysis",
        "detail": "This family's signed primary reports odds ratios only.",
    }))
    (stop,) = _stops(_outcome(plans[OR], EICU_RR, tmp_path, [gap]))
    assert stop.detail["reason"] == "question_requirement_capability_gap"
    assert [row["id"] for row in stop.detail["requirements"]] == ["r1"]


def test_a_record_written_before_the_reading_judges_no_measure(plans, tmp_path):
    _outcome(plans[OR], EICU_RR, tmp_path)
    record = json.loads((tmp_path / QUESTION_REQUIREMENTS_FILENAME).read_text())
    record.pop("question_effect_measures")
    record["judged"] = [row for row in record["judged"] if row["id"] != "measure"]
    assert not _stops(judge_recorded_requirements(record, plan=plans[RR]).findings())


# Unlabelled 风险比 is also how a Cox hazard ratio is written in Chinese
# (「调整后的 Cox 风险比」).  The host holds only a binary primary to it, so a
# time-to-event plan answers the same sentence with its hazard ratio.
LOOSE = "肾脏替代治疗与 90 天死亡时间有什么关系？用 Cox 回归，调整年龄和性别，结果用风险比表示。"


def test_an_unlabelled_risk_ratio_does_not_stop_a_time_to_event_plan(tmp_path):
    from tests.support.survival_proposal import (
        AGE, SEX, proposed_survival_plan, survival_context, survival_request, survival_spec,
    )

    assert [item.measure for item in question_effect_measures(LOOSE)] == [RR]
    context = survival_context(research_question=LOOSE)
    plan, _llm = proposed_survival_plan(context, survival_spec(survival_request(context), [AGE, SEX]))
    findings = question_requirement_outcome(
        context=context, plan=plan, run_dir=tmp_path,
        facts=SimpleNamespace(family_template=True, question_requirements=()),
    )
    assert not _stops(findings)
    record = json.loads((tmp_path / QUESTION_REQUIREMENTS_FILENAME).read_text())
    assert record["question_effect_measures"] == [{"measure": RR, "quote": "风险比"}]
    assert not any(row.get("reason_code", "").startswith("question_effect_measure")
                   for row in record.get("measures", []))


def test_the_same_unlabelled_risk_ratio_holds_a_binary_primary_to_it(plans, tmp_path):
    (stop,) = _stops(_outcome(plans[OR], LOOSE, tmp_path))
    (row,) = stop.detail["requirements"]
    assert (row["reason_code"], row["effect_measure"]) == ("question_effect_measure_not_reported", RR)
    (tmp_path / "rr").mkdir()
    assert not _stops(_outcome(plans[RR], LOOSE, tmp_path / "rr"))
