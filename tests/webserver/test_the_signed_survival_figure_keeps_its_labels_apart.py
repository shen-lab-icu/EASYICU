"""The signed survival figure keeps its labels apart and its charts in its roles.

A real run's publication figure failed visual QA: the "Number at risk" heading
sat on the first column's time label, and the proportional-hazards panel gave
six model terms (a categorical covariate adds one per level) the fixed height
of four.  The one-row estimate panel also ran a long exposure label across
the Kaplan-Meier panel, which the text-collision audit cannot see.  And the
article figure strategy rejected the interval-specific hazard-ratio forest the
suite draws when the PH assumption is rejected.  Synthetic study and seeded
synthetic rows only (renal replacement therapy and 90-day mortality).
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.figures import publication
from easyicu.research_agent.gates.visual_qa import audit_svg_text_layout
from easyicu.research_agent.planning.figure_strategy import (
    build_article_figure_strategy,
    figure_panel_covers_role,
)
from tests.support.survival_sealed import (
    SURVIVAL_SOURCES,
    render_signed_survival_figure,
    run_signed_suite,
    sealed_survival,
    synthetic_survival_rows,
)


def _render(tmp_path, *, extra_terms: int = 0):
    """Run the suite, give its PH table ``extra_terms`` more model terms, and render."""

    _context, authority = sealed_survival(tmp_path)
    out = tmp_path / "suite"
    run_signed_suite(authority, synthetic_survival_rows(), out)
    if extra_terms:
        ph = pd.read_csv(out / SURVIVAL_SOURCES["ph_product"])
        # One row per level of a categorical covariate, as the suite writes it.
        level = ph.loc[ph["covariate"].astype(str) != "global"].iloc[[0]]
        ph = pd.concat(
            [ph, *(level.assign(covariate=f"admission_type_{index}") for index in range(extra_terms))],
            ignore_index=True,
        )
        ph.to_csv(out / SURVIVAL_SOURCES["ph_product"], index=False)
    render_signed_survival_figure(authority, out, tmp_path / "figure")
    return authority, tmp_path / "figure" / "landmark_survival_suite.svg"


@pytest.mark.parametrize("extra_terms", [0, 3, 12, 20])
def test_no_two_figure_labels_overlap_however_many_model_terms(tmp_path, extra_terms):
    _authority, svg = _render(tmp_path, extra_terms=extra_terms)

    findings = audit_svg_text_layout(svg)
    assert [finding.detail for finding in findings if finding.severity == "error"] == []


def test_the_one_row_estimate_label_stays_beside_its_own_panel(tmp_path, monkeypatch):
    drawn = []
    save = publication.save_publication_figure

    def keep(fig, *args, **kwargs):
        drawn.append(fig)
        return save(fig, *args, **kwargs)

    monkeypatch.setattr(publication, "save_publication_figure", keep)
    authority, _svg = _render(tmp_path)

    (fig,) = drawn
    fig.canvas.draw()
    by_title = {axes.get_title(loc="left"): axes for axes in fig.axes}
    km = by_title["Unadjusted landmark Kaplan-Meier survival"]
    estimate = next(
        axes for title, axes in by_title.items()
        if title in {"Adjusted Cox association", "PH-free survival contrast"}
    )
    (label,) = estimate.get_yticklabels()
    # Only line breaks are added; the label still names the exposed group.
    assert label.get_text().replace("\n", " ") == authority.exposed_group_label
    assert label.get_window_extent().x0 >= km.get_window_extent().x1


def test_every_chart_the_signed_suite_may_draw_meets_its_article_role(tmp_path):
    context, authority = sealed_survival(tmp_path)
    roles = {
        role.role: role
        for role in build_article_figure_strategy(context, analysis_family="time_to_event").role_strategies
    }

    drawn = {
        (panel.article_role, chart)
        for panel in authority.figure_panel_templates()
        for chart in (panel.chart_type, *panel.policy_alternative_chart_types)
        if panel.article_role in roles
    }
    assert ("survival_effect", "time_varying_hazard_ratio_forest") in drawn
    for role, chart in sorted(drawn):
        panel = SimpleNamespace(article_role=role, chart_type=chart)
        assert figure_panel_covers_role(panel, roles[role]), (role, chart)

    # The alias names one geometry; it does not open the role to any chart.
    assert not figure_panel_covers_role(
        SimpleNamespace(article_role="survival_effect", chart_type="profile_heatmap"),
        roles["survival_effect"],
    )
