"""The signed survival figure draws what its legend states.

The risk-set legend starts at the source records, but its panel drew only the
last four stages of the suite's risk-set flow, so the first exclusion was not
shown.  The proportional-hazards panel drew every model term against one line
at the prespecified alpha.  The signed rule judges the exposure term at that
alpha and the other terms only through the Bonferroni-adjusted global test, so
a covariate bar past the line read as a rejection the rule did not make.  The
risk-set panel now draws every stage; the PH panel sets the two judged tests
apart, and its legend states the rule.  A PH table without either test draws
no panel.

Synthetic study and seeded synthetic rows only (renal replacement therapy and
90-day mortality).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from matplotlib.colors import to_hex

from easyicu.research_agent.figures import publication
from easyicu.research_agent.figures.display_labels import display_label
from easyicu.research_agent.figures.publication import PALETTE_CLINICAL, FigureContract
from tests.support.survival_sealed import (
    SURVIVAL_SOURCES,
    render_signed_survival_figure,
    run_signed_suite,
    sealed_survival,
    synthetic_survival_rows,
)


def _render(tmp_path, monkeypatch, *, edit_ph=None):
    """Run the suite, optionally edit its PH table, render, and keep the figure."""

    drawn = []
    save = publication.save_publication_figure

    def keep(fig, *args, **kwargs):
        drawn.append(fig)
        return save(fig, *args, **kwargs)

    monkeypatch.setattr(publication, "save_publication_figure", keep)
    _context, authority = sealed_survival(tmp_path)
    suite = tmp_path / "suite"
    run_signed_suite(authority, synthetic_survival_rows(), suite)
    if edit_ph is not None:
        path = suite / SURVIVAL_SOURCES["ph_product"]
        edit_ph(pd.read_csv(path), authority).to_csv(path, index=False)
    render_signed_survival_figure(authority, suite, tmp_path / "figure")
    (fig,) = drawn
    contract = FigureContract.model_validate_json(
        (tmp_path / "figure" / "landmark_survival_suite.figure_contract.json").read_text(encoding="utf-8")
    )
    axes = {ax.get_title(loc="left"): ax for ax in fig.axes}
    return authority, suite, axes, contract.reader_caption


def _more_terms(ph, _authority):
    """Three more model terms, as a categorical covariate's levels add them."""

    level = ph.loc[ph["covariate"].astype(str) != "global"].iloc[[0]]
    return pd.concat(
        [ph, *(level.assign(covariate=f"admission_type_{index}") for index in range(3))],
        ignore_index=True,
    )


def test_the_risk_set_panel_draws_every_stage_from_the_source_records(tmp_path, monkeypatch):
    _authority, suite, axes, legend = _render(tmp_path, monkeypatch)

    flow = pd.read_csv(suite / SURVIVAL_SOURCES["risk_set_product"])
    panel = axes["Risk-set accounting"]

    labels = [label.get_text() for label in panel.get_yticklabels()]
    assert labels[0] == "Source records" and labels[-1] == "Landmark analysis population"
    assert [round(bar.get_width()) for bar in panel.patches] == flow["count"].tolist()
    assert "(c) Risk-set accounting from the source records" in legend


def test_the_ph_panel_sets_apart_the_tests_its_rule_judges(tmp_path, monkeypatch):
    authority, suite, axes, legend = _render(tmp_path, monkeypatch, edit_ph=_more_terms)

    ph = pd.read_csv(suite / SURVIVAL_SOURCES["ph_product"])
    panel = axes["Proportional-hazards diagnostics"]
    labels = [label.get_text() for label in panel.get_yticklabels()]
    colors = {
        labels[round(bar.get_y() + bar.get_height() / 2)]: to_hex(bar.get_facecolor())
        for bar in panel.patches
    }

    judged = {"Global", display_label(authority.derived_exposure_column)}
    assert len(colors) == len(ph)
    assert {label for label, color in colors.items() if color == PALETTE_CLINICAL["orange"].lower()} == judged
    assert {colors[label] for label in set(colors) - judged} == {PALETTE_CLINICAL["orange_soft"].lower()}
    (line,) = panel.lines
    assert line.get_xdata()[0] == pytest.approx(-math.log10(authority.proportional_hazards_alpha))
    assert "the exposure term and the global test were judged" in legend
    assert "enter the decision only through the global test" in legend


@pytest.mark.parametrize("term", ["global", "exposure"])
def test_a_ph_table_without_a_judged_test_draws_no_panel(tmp_path, monkeypatch, term):
    def drop(ph, authority):
        name = "global" if term == "global" else authority.derived_exposure_column
        return ph.loc[ph["covariate"].astype(str) != name]

    with pytest.raises(ValueError, match="lack the global or exposure test"):
        _render(tmp_path, monkeypatch, edit_ph=drop)
