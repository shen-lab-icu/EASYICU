"""Bilingual P1 shadow typing through the live legacy judgment boundary."""

from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import pytest

from easyicu.research_agent.orchestration.progressive_planning import question_requirement_outcome
from easyicu.research_agent.planning.question_requirements import (
    QUESTION_REQUIREMENTS_FILENAME,
    QuestionRequirement,
    judge_question_requirements,
    question_requirement_host_facts,
)
from easyicu.research_agent.planning.requirement_typing import shadow
from easyicu.research_agent.planning.requirement_typing.contract import CohortPredicate, Method
from easyicu.research_agent.planning.requirement_typing.shadow import FLAG, observe_requirements
from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec
from easyicu.research_agent.schema import UserPreferences

from tests.support.typed_question_requirements import (
    CLUSTER, KDIGO_48, MAX_POTASSIUM, REPORT_STAYS, STUDY_POPULATION, VASOPRESSOR16,
    _context, _model_plan, _shadow_event, bind, interval, key, proposal,
)
from tests.support.typed_question_requirements import shadow_events as shadow_events

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


def _legacy_inputs(facet: str, quote: str):
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
    return context, plan, requirement


def _judge_legacy(context, plan, requirement):
    return judge_question_requirements(
        [requirement], plan=plan, family_template=True,
        relatives=lambda concept: frozenset({concept}),
        check_gap=lambda gap: pytest.fail("fixture declared no gap"),
        host_facts=question_requirement_host_facts(context=context, plan=plan, requirements=[requirement]),
        question=context.research_question,
    )


def test_live_v1_credits_dialysis_landmark_to_vasopressor_clause() -> None:
    [entry] = _judge_legacy(*_legacy_inputs("time_zero", BILINGUAL_CASES[0][1]))
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


@pytest.mark.parametrize(
    ("facet", "language", "quote"),
    [
        pytest.param(facet, language, quote, id=f"{facet}-{language}")
        for facet, en, zh in (
            *BILINGUAL_CASES,
            ("untyped", "Include a discussion of assay drift", "请讨论化验结果随时间漂移的问题"),
        )
        for language, quote in (("en", en), ("zh", zh))
    ],
)
def test_bilingual_shadow_parts(facet, language, quote, monkeypatch, shadow_events, tmp_path) -> None:
    context, plan, requirement = _legacy_inputs(facet, quote)
    facts = SimpleNamespace(
        family_template=True, question_requirements=[requirement], question_requirement_typings=[],
    )

    def run(name):
        run_dir = tmp_path / name
        run_dir.mkdir()
        metrics = {}
        findings = question_requirement_outcome(
            context=context, plan=plan, facts=facts, run_dir=run_dir, prompt_metrics=metrics,
        )
        return (findings, (run_dir / QUESTION_REQUIREMENTS_FILENAME).read_bytes()), metrics

    monkeypatch.delenv(FLAG, raising=False)
    baseline, off_metrics = run("off")
    assert shadow.SHADOW_METRICS_FIELD not in off_metrics
    assert not shadow_events

    monkeypatch.setenv(FLAG, "shadow")
    shadowed, metrics = run("shadow")
    assert shadowed == baseline
    [row] = _shadow_event(shadow_events)["parts"]
    covered = facet == "delivery"
    assert row["source_span"] is not None
    assert row["v2"] == ("covered" if covered else "unknown")
    assert (row["result"]["part"]["key"] is not None) == covered

    summary = metrics[shadow.SHADOW_METRICS_FIELD]
    assert summary["parts_total"] == 1
    assert summary["by_language"] == {language: {"covered": int(covered), "not_covered": 0, "unknown": int(not covered)}}
    assert quote not in json.dumps(summary, ensure_ascii=False)


def test_same_18_parts_with_scripted_intake(shadow_events) -> None:
    counts = {facet: Counter() for facet, _, _ in BILINGUAL_CASES}
    differences = Counter()
    unknown_by_language = Counter()
    for facet, english, chinese in BILINGUAL_CASES:
        for language, quote in (("en", english), ("zh", chinese)):
            shadow_events.clear()
            proposals = [] if facet == "delivery" else [proposal("r1", quote, _SCRIPTED_KEYS.get(facet), facet=facet)]
            context, plan, requirement = _legacy_inputs(facet, quote)
            summary = observe_requirements(
                _judge_legacy(context, plan, requirement), plan=plan, question=quote,
                context=context, proposals=proposals,
            )
            [row] = _shadow_event(shadow_events)["parts"]
            counts[facet][row["v2"]] += 1
            if row["v2"] == "unknown":
                unknown_by_language[language] += 1
            disagreements = summary["disagreements"]
            differences.update(disagreements)
            bucket = next((name for name, count in disagreements.items() if count), None)
            print("RC1C_ROW", json.dumps({
                "dataset": 18, "id": facet, "language": language, "facet": facet,
                "v1": row["v1"], "v2": row["v2"], "reason": row["v1_reason"], "bucket": bucket,
            }))
    assert sum(group["unknown"] for group in counts.values()) <= 4
    assert unknown_by_language == {"en": 2, "zh": 2}
    assert counts["delivery"] == {"covered": 2}
    assert counts["time_zero"] == {"unknown": 2}
    assert counts["unit"] == {"unknown": 2}
    assert differences == Counter(prose_proven=2, v2_under_typing=1)


def test_legacy_plan_without_a_typed_contract_reports_unknown(monkeypatch, shadow_events) -> None:
    monkeypatch.setenv(FLAG, "shadow")
    requirement = QuestionRequirement(
        id="r1", kind="delivery", quote="Deliver result tables",
        deliverable_kinds=["result_tables"], coverage="plan",
    )
    plan = SimpleNamespace(steps=[])
    judged = judge_question_requirements(
        [requirement], plan=plan, family_template=True,
        relatives=lambda concept: frozenset({concept}),
        check_gap=lambda gap: pytest.fail("fixture declared no gap"),
        question=requirement.quote,
    )
    observe_requirements(judged, plan=plan, question=requirement.quote, context=None, proposals=())
    [row] = _shadow_event(shadow_events)["parts"]
    assert row["v2"] == "unknown"
    assert row["v2_reason"] == "plan_contract_unavailable"


@pytest.mark.parametrize("quote", ["Report the odds ratio", "Report the odds ratio for magnesium and kdigo_aki"])
def test_legacy_secondary_effect_measure_never_inherits_primary_pair(quote, monkeypatch, shadow_events) -> None:
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
    judged = judge_question_requirements(
        items, plan=plan, family_template=True, relatives=lambda concept: frozenset({concept}),
        check_gap=lambda gap: pytest.fail("fixture declared no gap"), question=question,
    )
    observe_requirements(judged, plan=plan, question=question, context=_context(), proposals=())
    row = _shadow_event(shadow_events)["parts"][-1]
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
    (
        "magnesium in mg/dL, phosphate in mg/dL",
        ("in mg/dL", "magnesium in mg/dL", "IN MG/DL"),
        (10, 0, 30),
    ),
    (
        "血镁以 mg/dL 表示，血磷以 ｍｇ/ｄＬ 表示",
        ("以 mg/dL", "血镁以 mg/dL", "以 ｍｇ/ｄＬ"),
        (2, 0, 15),
    ),
    (
        "adjust mortality with age; adjust readmission with age",
        ("with age", "mortality with age", "with age"),
        (17, 7, 46),
    ),
    ("include figures and tables", ("tables", "tables"), (20, None)),
])
def test_reordered_or_overlapping_clauses_keep_their_source_spans(parent, clauses, starts, monkeypatch, shadow_events) -> None:
    from easyicu.research_agent.planning.question_requirements import JudgedRequirement

    entry = JudgedRequirement(
        requirement=QuestionRequirement(id="r1", kind="analysis", quote=parent, coverage="plan"),
        disposition="covered", reason_code="plan_declared",
        clauses=tuple({"quote": clause, "disposition": "covered"} for clause in clauses),
    )
    monkeypatch.setenv(FLAG, "shadow")
    observe_requirements([entry], plan=_model_plan(), question=f"Methods. {parent}.", context=_context(), proposals=())
    rows = _shadow_event(shadow_events)["parts"]
    assert [row["source_span"] for row in rows] == [
        {"start": 9 + start, "end": 9 + start + len(clause)} if start is not None else None
        for clause, start in zip(clauses, starts, strict=True)
    ]
    assert all(row["v2_reason"] == "source_span_unavailable" for row in rows if row["source_span"] is None)
