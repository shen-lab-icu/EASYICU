"""Planner capability guidance stays data-driven instead of inlined."""

from __future__ import annotations

import inspect

from easyicu.research_agent.agents import planner


def test_base_planner_prompt_delegates_guides_to_data_modules() -> None:
    source = inspect.getsource(planner._base_planner_user_prompt)

    # The capability/method guides are owned by data-driven modules, not by
    # the prompt builder itself.
    assert "_payload." in source
    assert "_scientific_actions.planner_scientific_action_guide" in source
    assert "planner_descriptive_method_guidance" in source
    assert "planner_descriptive_robustness_guidance" in source


def test_capability_guide_modules_own_the_heavy_prompt_text() -> None:
    from easyicu.research_agent.agents import plan_payload as payload
    from easyicu.research_agent.planning import (
        scientific_action_catalog as actions,
    )

    payload_source = inspect.getsource(payload)
    action_source = inspect.getsource(actions)

    assert "def planner_scientific_action_guide" in action_source
    assert "def planner_descriptive_method_guidance" in payload_source
    assert "def planner_descriptive_robustness_guidance" in payload_source
