"""The signed survival figure carries the legend its manuscript projects.

A real run's manuscript failed evidence_complete with "A reader figure has no
source-bound explanatory legend": the suite's figure contract, which the
publication figure copies, had no reader caption, so no signed survival figure
could enter a manuscript.  The contract now gives each panel the suite draws a
legend entry, the estimate entry following the PH decision as drawn, and states
no value of its own.  Synthetic study and seeded synthetic rows only (renal
replacement therapy and 90-day mortality); the second set has crossing hazards,
so the PH test rejects.
"""

from __future__ import annotations

import re

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.figures.publication import FigureContract
from easyicu.research_agent.reporting.manuscript_figures import build_manuscript_figures
from tests.support.survival_sealed import (
    render_signed_survival_figure,
    run_signed_suite,
    sealed_survival,
    synthetic_crossing_hazard_rows,
    synthetic_survival_rows,
)

STEP = "02_authority_compiled_survival_figure"


def _registered(run_dir, figure_dir) -> list:
    """The figure step's export and contract, registered as the host runner does."""

    store = EvidenceStore(run_dir)
    for kind, name in (
        ("figure", "landmark_survival_suite.pdf"),
        ("log", "landmark_survival_suite.figure_contract.json"),
    ):
        store.register_file(
            kind=kind, description=name, source_path=figure_dir / name,
            produced_by_step=STEP, producer="runner", generation_mode="deterministic_standard",
        )
    return store.records()


@pytest.mark.parametrize(
    ("rows", "estimate"),
    [
        (synthetic_survival_rows, "The adjusted Cox hazard ratio"),
        (synthetic_crossing_hazard_rows, "Adjusted interval-specific hazard ratios"),
    ],
)
def test_the_manuscript_projects_a_legend_for_every_drawn_panel(tmp_path, rows, estimate):
    _context, authority = sealed_survival(tmp_path)
    run_dir = tmp_path / "run"
    suite = run_dir / "steps" / "primary_survival_suite" / "outputs"
    figure_dir = run_dir / "steps" / STEP / "outputs"
    run_signed_suite(authority, rows(), suite)
    render_signed_survival_figure(authority, suite, figure_dir)

    contract = FigureContract.model_validate_json(
        (figure_dir / "landmark_survival_suite.figure_contract.json").read_text(encoding="utf-8")
    )
    legend = contract.reader_caption
    assert legend is not None
    # One entry per drawn panel, in order; the estimate entry is the panel drawn.
    assert re.findall(r"\(([a-z])\) ", legend) == [panel.panel_id for panel in contract.panels]
    assert f"(b) {estimate}" in legend
    # The legend describes; every value stays in the bound tables.
    assert re.search(r"\d", legend) is None

    projected = build_manuscript_figures(
        evidence_records=_registered(run_dir, figure_dir), run_dir=run_dir,
    )
    assert projected.findings == ()
    (figure,) = projected.figures
    assert figure.caption == legend
