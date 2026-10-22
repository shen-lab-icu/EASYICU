"""The plan a researcher reviews is asked for in the language of their question.

The outline's six ``reviewable_plan`` items are what the researcher reads
before approving.  The Planner was told to write them "in the research
question's language" and, on the hosted pilot, wrote them in English for a
question asked in Chinese.  The outline prompt now names the language for a
Chinese question; an English question adds nothing.  Synthetic context only.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.agents.progressive_planner import (
    ProgressivePlannerAgent,
    select_progressive_variables,
)
from easyicu.research_agent.planning.outline_action_menu import outline_action_catalog
from easyicu.research_agent.planning.outline_design_selection import (
    reviewable_plan_language_rule,
)
from tests.research_agent.planning.progressive_planner_fixtures import _context


def _anchor_block(question: str) -> str:
    context = _context().model_copy(update={"research_question": question})
    _ids, rows = outline_action_catalog(("association_study",))
    prompt = ProgressivePlannerAgent._user_prompt(
        context,
        analysis_types=("association_study",),
        variables=select_progressive_variables(context),
        action_rows=rows,
    )
    return prompt.split("Research question and sealed study anchors:\n", 1)[1].split("\n\n", 1)[0]


def test_a_chinese_question_asks_for_a_chinese_recommendation() -> None:
    rule = reviewable_plan_language_rule("比较两组患者的院内死亡率")

    assert "Simplified Chinese" in rule and "reviewable_plan" in rule
    assert rule in _anchor_block("比较两组患者的院内死亡率")


@pytest.mark.parametrize("question", ["Compare in-hospital death between two groups.", ""])
def test_an_english_question_adds_no_rule(question: str) -> None:
    assert reviewable_plan_language_rule(question) == ""
    assert "Simplified Chinese" not in _anchor_block(question or "Compare outcomes.")
