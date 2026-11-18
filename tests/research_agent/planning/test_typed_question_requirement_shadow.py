"""Bilingual P1 shadow typing through the live legacy judgment boundary."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from easyicu.research_agent.planning.question_requirements import (
    QuestionRequirement,
    judge_question_requirements,
    question_requirement_host_facts,
)
from easyicu.research_agent.planning.requirement_typing.shadow import FLAG, LOGGER
from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec
from easyicu.research_agent.schema import UserPreferences

from .test_typed_question_requirement_coverage import _context, _model_plan

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


def _legacy(facet: str, quote: str):
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
    )


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
    [event] = [entry.requirement_coverage_shadow for entry in caplog.records
               if hasattr(entry, "requirement_coverage_shadow")]
    [row] = event["parts"]
    assert row["source_span"] is not None
    assert row["v2"] == ("covered" if facet == "delivery" else "unknown")
    assert (row["result"]["part"]["key"] is not None) == (facet == "delivery")
    assert quote not in str(event)


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
