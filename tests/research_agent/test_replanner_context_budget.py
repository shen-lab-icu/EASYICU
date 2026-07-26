"""Replanner prompt context-budget guards.

The replanner prompt embeds every completed step's record (including its full
``step_summary.json``) plus the probe summary. Neither is byte-capped at the
source, so a step that dumps a wide matrix/table into its summary would inflate
the prompt without bound, multiplied by up to ``max_total_steps``. These tests
pin the projection that keeps the prompt bounded while leaving the on-disk /
validator records untouched.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.core import (
    PlannerPromptBudgetError,
    ReplannerAgent,
    _REPLANNER_STEP_SUMMARY_CHAR_BUDGET,
    _REPLANNER_TOTAL_RECORDS_CHAR_BUDGET,
    _clip_json,
    _slim_completed_records_for_prompt,
    _slim_record_for_replanner,
)
from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient
from easyicu.research_agent.schema import (
    AnalysisPlan,
    CohortDescriptor,
    ResearchContext,
)


def _pathological_records(n: int = 12) -> list[dict]:
    return [
        {
            "step_id": f"{i:02d}_step",
            "intent": "fit model",
            "status": "ok",
            # input the replanner already has via CURRENT PLAN -> must be dropped
            "analysis_request": "X" * 5000,
            "step_summary": {"matrix": [{"k": j, "v": j * 1.234} for j in range(400)]},
            "usage_findings": [
                {
                    "validator": "concept_usage",
                    "severity": "warning",
                    "message": "m" * 1000,
                    "extra": "Z" * 2000,
                }
                for _ in range(30)
            ],
        }
        for i in range(n)
    ]


def test_slim_drops_inputs_and_compacts_findings():
    rec = _pathological_records(1)[0]
    slim = _slim_record_for_replanner(rec)

    assert "analysis_request" not in slim  # input, not needed by replanner
    assert slim["step_id"] == "00_step"
    assert slim["status"] == "ok"
    # findings list compacted to top-k with only validator/severity/message
    assert len(slim["usage_findings"]) <= 8
    for f in slim["usage_findings"]:
        assert set(f).issubset({"validator", "severity", "message"})
        assert len(f["message"]) <= 240


def test_oversized_step_summary_is_clipped_with_marker():
    rec = _pathological_records(1)[0]
    slim = _slim_record_for_replanner(rec)
    summary = slim["step_summary"]
    # the 400-row matrix exceeds the per-step budget -> clipped to a string marker
    assert isinstance(summary, str)
    assert "truncated" in summary
    assert len(summary) <= _REPLANNER_STEP_SUMMARY_CHAR_BUDGET + 80


def test_small_step_summary_is_preserved_verbatim():
    rec = {"step_id": "00", "intent": "x", "status": "ok", "step_summary": {"auroc": 0.81}}
    slim = _slim_record_for_replanner(rec)
    assert slim["step_summary"] == {"auroc": 0.81}


def test_total_blob_stays_within_global_budget():
    recs = _pathological_records(12)
    slim = _slim_completed_records_for_prompt(recs)
    blob = json.dumps(slim, ensure_ascii=False, default=str)
    assert len(blob) <= _REPLANNER_TOTAL_RECORDS_CHAR_BUDGET
    # oldest records collapse first; the newest step is never collapsed
    assert "collapsed" not in slim[-1]


def test_under_budget_records_pass_through_unchanged_order():
    recs = [
        {"step_id": "00", "intent": "a", "status": "ok", "step_summary": {"n": 1}},
        {"step_id": "01", "intent": "b", "status": "ok", "step_summary": {"n": 2}},
    ]
    slim = _slim_completed_records_for_prompt(recs)
    assert [r["step_id"] for r in slim] == ["00", "01"]
    assert all("collapsed" not in r for r in slim)


def test_clip_json_marks_truncation_and_respects_budget():
    out = _clip_json({"a": "q" * 100}, char_budget=20)
    assert out.endswith("budget]")
    assert out.startswith("{")


def test_oversized_replanner_directive_fails_before_provider_call():
    context = ResearchContext(
        research_question="Describe the ICU cohort.",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="miiv",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
    )
    plan = AnalysisPlan(research_question=context.research_question, steps=[])
    llm = PatternScriptedMockLLMClient([])

    with pytest.raises(PlannerPromptBudgetError, match="Replanner prompt"):
        ReplannerAgent(llm).run(
            context=context,
            current_plan=plan,
            directive="x" * 90_000,
        )

    assert llm.calls == []
