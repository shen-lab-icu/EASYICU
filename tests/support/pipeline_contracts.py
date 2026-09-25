"""Switches shared by scripted end-to-end pipeline tests.

Test modules may not import one another (``tests/governance/test_test_organization.py``);
these helpers moved here from ``test_pipeline`` when a second module needed them.
"""

from __future__ import annotations

import pytest


def disable_article_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep focused lifecycle fixtures independent of manuscript completeness."""

    import easyicu.research_agent.agents.core as agent_core
    import easyicu.research_agent.pipeline as pipeline_module
    import easyicu.research_agent.planning.final_plan_shape as final_plan_module
    from easyicu.research_agent.agents.core import PlannerAgent

    original_run = PlannerAgent.run

    def run_without_article_contract(self, context, **kwargs):
        kwargs["enforce_article_contract"] = False
        return original_run(self, context, **kwargs)

    monkeypatch.setattr(PlannerAgent, "run", run_without_article_contract)
    monkeypatch.setattr(
        agent_core,
        "_validate_required_primary_result",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        final_plan_module,
        "_enforce_advanced_plan_contract",
        lambda *, plan, context, **_kwargs: (plan, []),
    )
    monkeypatch.setattr(
        final_plan_module,
        "_ensure_publication_figure_step_in_plan",
        lambda *, plan, context, force: (plan, []),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_ensure_audit_panel_step_in_plan",
        lambda *, plan, context, **_kwargs: (plan, []),
    )


def stable_plan_rules(plan: str):
    """Return initial and probe-replan routes for a fixed focused test plan."""

    return [
        ("Produce an ICU-AWARE RESEARCH PLAN as JSON", [plan] * 8),
        ("REVISE THE ICU-AWARE RESEARCH PLAN", [plan] * 8),
    ]
