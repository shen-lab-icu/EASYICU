from types import SimpleNamespace
import json

import pytest

from easyicu.research_agent.planning import figure_strategy as owner
from easyicu.research_agent.reporting.completion import publication_authorized


@pytest.mark.parametrize("broken", [None, "missing_role", "unknown_chart"])
def test_visual_variety_is_advice_while_scientific_coverage_still_gates(
    monkeypatch, tmp_path, broken
):
    roles = ["primary_estimand", "descriptive_result", "robustness"]
    strategy = owner.ArticleFigureStrategy(
        analysis_family="association",
        archetype="quantitative_grid",
        hero_role=roles[0],
        minimum_distinct_chart_types=3,
        role_strategies=[
            owner.FigureRoleStrategy(
                role=role,
                rationale="Required scientific information",
                acceptable_chart_types=["forest", "bar"],
            )
            for role in roles
        ],
    )
    panels = [
        {
            "panel_id": str(index),
            "role": role,
            "chart_type": "forest" if index == 0 else "bar",
            "_figure_id": "primary",
            "_primary_publication_contract": True,
        }
        for index, role in enumerate(roles)
    ]
    if broken == "missing_role":
        panels.pop()
    elif broken == "unknown_chart":
        panels[0]["chart_type"] = "unspecified"
    monkeypatch.setattr(
        owner, "build_article_figure_strategy", lambda *args, **kwargs: strategy
    )
    directory = tmp_path / "publication_figures"
    directory.mkdir()
    (directory / "primary.figure_contract.json").write_text(
        json.dumps({"figure_id": "primary", "panels": panels}), encoding="utf-8",
    )
    status = owner.summarize_article_figure_strategy_coverage(
        context=SimpleNamespace(), run_dir=tmp_path
    )
    assert len(status["article_figure_strategy_design_advice"]) == 2
    assert status["article_figure_strategy_complete"] is (broken is None)
    assert publication_authorized(
        manuscript_ready=True,
        publication_figure_bundle_ready=True,
        publication_provenance_ready=True,
        display_suite_complete=True,
        article_contract_complete=True,
        article_figure_strategy_complete=status["article_figure_strategy_complete"],
    ) is (broken is None)


def test_actual_planner_transports_share_research_display_requirements():
    from easyicu.research_agent.contracts.research_display import RESEARCH_DISPLAY_GUIDE
    from easyicu.research_agent.agents import planner, progressive_planner
    # Protect the real transport inputs, not an unused strategy renderer.
    assert RESEARCH_DISPLAY_GUIDE in planner._PRINCIPLES_GUIDE
    assert RESEARCH_DISPLAY_GUIDE in progressive_planner._GUIDE
