"""Planner prompt compaction must not alter the illustrated scientific shape."""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent import schema
from easyicu.research_agent.agents import planner


def _example_slice(prompt: str) -> tuple[str, str, str]:
    marker = prompt.index("OUTPUT FORMAT — VERY IMPORTANT:\n")
    start = prompt.index("{\n", marker)
    end = prompt.index("\n}\n\n", start) + len("\n}\n\n")
    return prompt[:start], prompt[start:end], prompt[end:]


@pytest.fixture
def minimal_research_context() -> schema.ResearchContext:
    return schema.ResearchContext(
        research_question="Does lactate predict mortality?",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="synthetic", n_patients=100, n_stays=100,
        ),
        variables=[
            schema.ConceptDescriptor(name="lactate", dtype="float64"),
            schema.ConceptDescriptor(name="death", dtype="int64"),
        ],
        target_outcome="death", primary_exposure="lactate",
    )


def test_planner_compacts_only_its_static_json_example(
    minimal_research_context, monkeypatch,
) -> None:
    context = minimal_research_context.model_copy(update={
        "research_question": "First  ICU  stay?\nPreserve   all spaces.",
    })
    compact = planner._build_planner_user_prompt(context)
    with monkeypatch.context() as patch:
        patch.setattr(planner._payload, "compact_static_planner_example", lambda text: text)
        original = planner._build_planner_user_prompt(context)

    old_prefix, old_example, old_suffix = _example_slice(original)
    new_prefix, new_example, new_suffix = _example_slice(compact)
    assert (new_prefix, new_suffix) == (old_prefix, old_suffix)
    assert json.loads(old_example) == json.loads(new_example)
    assert len(new_example.encode("utf-8")) < len(old_example.encode("utf-8"))


def test_strict_schema_prompt_and_user_text_are_not_compacted(
    minimal_research_context, monkeypatch,
) -> None:
    context = minimal_research_context.model_copy(update={
        "research_question": "ICU    research\n  unchanged indentation",
    })
    compact = planner._build_planner_user_prompt(context, strict_transport_schema=True)
    with monkeypatch.context() as patch:
        patch.setattr(planner._payload, "compact_static_planner_example", lambda text: text)
        original = planner._build_planner_user_prompt(context, strict_transport_schema=True)
    assert compact == original
