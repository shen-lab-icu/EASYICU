"""Bilingual P1 shadow typing through the live legacy judgment boundary."""

from __future__ import annotations

import json
import logging
from collections import Counter
from types import SimpleNamespace

import pytest

from easyicu.research_agent.planning.question_requirements import (
    QuestionRequirement,
    judge_question_requirements,
    question_requirement_host_facts,
)
from easyicu.research_agent.planning.requirement_typing.shadow import FLAG, LOGGER
from easyicu.research_agent.planning.requirement_typing.contract import CohortPredicate, Method
from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec
from easyicu.research_agent.schema import UserPreferences

from .test_typed_question_requirement_coverage import _context, _model_plan
from .test_typed_question_requirement_regressions import (
    CLUSTER, KDIGO_48, MAX_POTASSIUM, REPORT_STAYS, STUDY_POPULATION, VASOPRESSOR16,
    bind, interval, key,
)

BILINGUAL_CASES = (
    ("time_zero", "Clarify vasopressor initiation time zero at 16 hours", "明确升压药开始的时间零点为16小时"),
    ("window", "Use peak potassium in the first six hours after ICU entry", "入ICU后前六小时取最高血钾"),
    ("grading", "Specify KDIGO 48-hour creatinine staging", "明确肌酐的KDIGO 48小时分级"),
    ("unit", "Specify albumin units in mg/24 hours", "明确白蛋白单位为mg/24小时"),
    ("method", "External validation", "外部验证"),
    ("cohort_predicate", "Include stays with potassium at least 5.5 in 24 hours", "纳入24小时内血钾至少为5.5的住院"),
    ("repeat_stay_reporting", "Specify handling of repeated ICU stays", "说明重复入住ICU的处理"),
    ("repeat_stay_policy", "Specify that repeated ICU stays are retained with clustering by patient", "说明保留重复入住ICU并按患者聚类"),
    ("delivery", "Deliver the result tables", "交付结果表"),
)


def _legacy(facet: str, quote: str, proposals=()):
    context = _context().model_copy(update={"research_question": quote})
    plan = _model_plan().model_copy(update={"research_question": quote})
    kwargs = {"kind": "analysis", "concepts": []}
    if facet == "time_zero":
        # Deliberately misbound legacy proposal. The old engine can treat the
        # dialysis landmark as evidence for the vasopressor clause.
        kwargs["concepts"] = ["rrt"]
        context = context.model_copy(update={
            "primary_exposure": "rrt",
            "user_preferences": UserPreferences(sensitivity_specs=[
                PrespecifiedSensitivitySpec(spec_id="dialysis", axis="timing", strategy="landmark", landmark_hours=16),
            ]),
        })
    elif facet == "method":
        kwargs["clauses"] = [quote]
        plan.steps[0].intent = quote
    elif facet == "delivery":
        kwargs.update({"kind": "delivery", "deliverable_kinds": ["result_tables"]})
    requirement = QuestionRequirement(id="r1", quote=quote, coverage="plan", **kwargs)
    host_facts = question_requirement_host_facts(context=context, plan=plan, requirements=[requirement])
    return lambda: judge_question_requirements(
        [requirement], plan=plan, family_template=True,
        relatives=lambda concept: frozenset({concept}),
        check_gap=lambda gap: pytest.fail("fixture declared no gap"),
        host_facts=host_facts, question=quote, shadow_context=context,
        typing_proposals=proposals,
    )


def test_live_v1_credits_dialysis_landmark_to_vasopressor_clause(monkeypatch) -> None:
    monkeypatch.delenv(FLAG, raising=False)
    [entry] = _legacy("time_zero", BILINGUAL_CASES[0][1])()
    assert entry.requirement.concepts == ["rrt"]
    assert entry.disposition == "covered"
    assert entry.reason_code == "question_methods_clauses_covered"
    [clause] = entry.clauses
    assert clause["evidence"]["source"] == "host_default"
    assert clause["evidence"]["source_ref"] == (
        "research_context.json.user_preferences.sensitivity_specs[dialysis]"
    )
    assert clause["evidence"]["statement"] == {"landmark_hours": 16, "strategy": "landmark"}


_SCRIPTED_KEYS = {
    "time_zero": VASOPRESSOR16,
    "window": MAX_POTASSIUM,
    "grading": KDIGO_48,
    "method": key(Method(action_id="prediction.external_validation"), STUDY_POPULATION),
    "cohort_predicate": key(CohortPredicate(
        operation="max", operator=">=", value=5.5, polarity="include", missing_policy="exclude",
    ), bind("k", "population"), time=interval(0, 24)),
    "repeat_stay_reporting": REPORT_STAYS,
    "repeat_stay_policy": CLUSTER,
}


def proposal(part_id, quote, asked, *, facet=None, alternatives=(), start=0):
    return {
        "part_id": part_id, "source_span": {"start": start, "end": start + len(quote)},
        "facet_hint": facet or (asked.facet.kind if asked else None),
        "key": asked.model_dump(mode="json") if asked else None,
        "alternatives": [choice.model_dump(mode="json") for choice in alternatives],
    }


def _shadow_event(caplog):
    [event] = [entry.requirement_coverage_shadow for entry in caplog.records
               if hasattr(entry, "requirement_coverage_shadow")
               and entry.requirement_coverage_shadow["boundary"] == "question_requirements"]
    return event


@pytest.mark.parametrize(
    ("facet", "language", "quote"),
    [pytest.param(facet, language, quote, id=f"{facet}-{language}")
     for facet, en, zh in BILINGUAL_CASES for language, quote in (("en", en), ("zh", zh))],
)
def test_bilingual_shadow_parts(facet, language, quote, monkeypatch, caplog) -> None:
    judge = _legacy(facet, quote)
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    monkeypatch.delenv(FLAG, raising=False)
    prior = judge()
    assert not caplog.records

    monkeypatch.setenv(FLAG, "shadow")
    shadowed = judge()
    assert shadowed == prior
    event = _shadow_event(caplog)
    [row] = event["parts"]
    assert row["source_span"] is not None
    assert row["v2"] == ("covered" if facet == "delivery" else "unknown")
    assert (row["result"]["part"]["key"] is not None) == (facet == "delivery")
    assert quote not in str(event)


def test_same_18_parts_with_scripted_intake(monkeypatch, caplog) -> None:
    monkeypatch.setenv(FLAG, "shadow")
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    counts = {facet: Counter() for facet, _, _ in BILINGUAL_CASES}
    differences = Counter()
    unknown_by_language = Counter()
    for facet, english, chinese in BILINGUAL_CASES:
        for language, quote in (("en", english), ("zh", chinese)):
            caplog.clear()
            proposals = [] if facet == "delivery" else [proposal("r1", quote, _SCRIPTED_KEYS.get(facet), facet=facet)]
            _legacy(facet, quote, proposals)()
            [row] = _shadow_event(caplog)["parts"]
            counts[facet][row["v2"]] += 1
            if row["v2"] == "unknown":
                unknown_by_language[language] += 1
            bucket = None
            if row["v1"] == "covered" and row["v2"] != "covered":
                assert facet in {"method", "time_zero"}
                bucket = "prose_proven" if row["v1_reason"] == "plan_declared" else "true_false_pass"
                differences[bucket] += 1
            print("RC1C_ROW", json.dumps({
                "dataset": 18, "id": facet, "language": language, "facet": facet,
                "v1": row["v1"], "v2": row["v2"], "reason": row["v1_reason"], "bucket": bucket,
            }))
    assert sum(group["unknown"] for group in counts.values()) <= 4
    assert unknown_by_language == {"en": 2, "zh": 2}
    assert counts["delivery"] == {"covered": 2}
    assert counts["time_zero"] == {"unknown": 2}
    assert counts["unit"] == {"unknown": 2}
    assert differences == {"true_false_pass": 1, "prose_proven": 2}


def test_legacy_plan_without_a_typed_contract_reports_unknown(monkeypatch, caplog) -> None:
    monkeypatch.setenv(FLAG, "shadow")
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    requirement = QuestionRequirement(
        id="r1", kind="delivery", quote="Deliver result tables",
        deliverable_kinds=["result_tables"], coverage="plan",
    )
    judge_question_requirements(
        [requirement], plan=SimpleNamespace(steps=[]), family_template=True,
        relatives=lambda concept: frozenset({concept}),
        check_gap=lambda gap: pytest.fail("fixture declared no gap"),
        question=requirement.quote,
    )
    [event] = [entry.requirement_coverage_shadow for entry in caplog.records
               if hasattr(entry, "requirement_coverage_shadow")]
    [row] = event["parts"]
    assert event["authority"] is None
    assert row["v2"] == "unknown"
    assert row["v2_reason"] == "plan_contract_unavailable"


@pytest.mark.parametrize("quote", ["Report the odds ratio", "Report the odds ratio for magnesium and kdigo_aki"])
def test_legacy_secondary_effect_measure_never_inherits_primary_pair(quote, monkeypatch, caplog) -> None:
    primary = "Report the odds ratio for potassium and mortality"
    question = f"{primary}. {quote}."
    plan = _model_plan().model_copy(update={"research_question": question})
    items = [
        QuestionRequirement(id="r1", kind="estimand", quote=primary, concepts=["k", "death"],
                            effect_measure="odds_ratio", coverage="plan"),
        QuestionRequirement(id="r2", kind="estimand", quote=quote, concepts=["mg", "kdigo_aki"],
                            effect_measure="odds_ratio", coverage="plan"),
    ]
    if quote == "Report the odds ratio":
        question = question.replace(f"{primary}. ", "Primary analysis. ")
        items = items[1:]
    monkeypatch.setenv(FLAG, "shadow")
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    judge_question_requirements(
        items, plan=plan, family_template=True, relatives=lambda concept: frozenset({concept}),
        check_gap=lambda gap: pytest.fail("fixture declared no gap"), question=question, shadow_context=_context(),
    )
    row = _shadow_event(caplog)["parts"][-1]
    assert row["source_span"] is not None
    assert row["v2"] == "unknown"
    assert row["result"]["part"]["key"] is None


def test_legacy_effect_measures_on_delivery_and_hazard_ratio_are_invalid() -> None:
    from pydantic import ValidationError

    for kind, measure in (("delivery", "odds_ratio"), ("estimand", "hazard_ratio")):
        with pytest.raises(ValidationError, match="effect_measure|effect measure"):
            QuestionRequirement(id="r1", quote="Report the effect", kind=kind, coverage="plan", effect_measure=measure)


@pytest.mark.parametrize(("parent", "clauses", "starts"), [
    ("external validation, calibration", ("calibration", "external validation"), (21, 0)),
    ("calibration of predictions", ("calibration of predictions", "calibration"), (0, 0)),
    ("magnesium in mg/dL, phosphate in mg/dL, calibration", ("calibration", "in mg/dL", "in mg/dL"), (40, 10, 30)),
    ("calibration of risks, calibration", ("calibration", "calibration of risks"), (22, 0)),
    ("include figures and tables", ("tables", "tables"), (20, None)),
])
def test_reordered_or_overlapping_clauses_keep_their_source_spans(parent, clauses, starts, monkeypatch, caplog) -> None:
    from easyicu.research_agent.planning.question_requirements import JudgedRequirement
    from easyicu.research_agent.planning.requirement_typing.shadow import observe_requirements

    entry = JudgedRequirement(
        requirement=QuestionRequirement(id="r1", kind="analysis", quote=parent, coverage="plan"),
        disposition="covered", reason_code="plan_declared",
        clauses=tuple({"quote": clause, "disposition": "covered"} for clause in clauses),
    )
    monkeypatch.setenv(FLAG, "shadow")
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    observe_requirements([entry], plan=_model_plan(), question=f"Methods. {parent}.", context=_context())
    rows = _shadow_event(caplog)["parts"]
    assert [row["source_span"] for row in rows] == [
        {"start": 9 + start, "end": 9 + start + len(clause)} if start is not None else None
        for clause, start in zip(clauses, starts, strict=True)
    ]
    assert all(row["v2_reason"] == "source_span_unavailable" for row in rows if row["source_span"] is None)
