"""Anchor-integrity gate between the REAL prompt pack and the guide scoper.

``research_context/prompt_scope._guide_segments`` locates 24 sections of the
coder guide by exact prose substrings and deliberately fails closed at
runtime when an anchor is missing or reordered. That is the right runtime
behavior, but a prompt-pack wording edit should fail HERE, in tests, not in
the middle of a paid run. This suite validates the anchors against the exact
guide production uses (``agents.core._CODER_GUIDE``), never a fixture copy.

Longer term the pack should carry explicit section markers instead of prose
anchors; that migration is deferred until no run is in flight, because the
pack bytes participate in provider receipts and resume identity.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.agents.core import _CODER_GUIDE
from easyicu.research_agent.research_context.prompt_scope import (
    _guide_segments,
    coder_guide_for_step,
    coder_rewrite_guide_for_step,
)
from easyicu.research_agent.schema import AnalysisStep


def test_every_anchor_resolves_in_the_production_guide() -> None:
    segments = _guide_segments(_CODER_GUIDE)
    assert "core" in segments
    assert len(segments) == 25  # core + 24 anchored sections
    assert all(segments[name] for name in segments), "empty guide segment"


def test_anchors_are_unique_in_the_production_guide() -> None:
    """A duplicated anchor would silently split at the wrong occurrence."""

    anchored = _guide_segments(_CODER_GUIDE)
    for name, body in anchored.items():
        if name == "core":
            continue
        first_line = body.splitlines()[0]
        assert (
            _CODER_GUIDE.count(first_line) == 1
        ), f"anchor line for section {name!r} appears more than once"


@pytest.mark.parametrize(
    ("method", "outputs", "expected_fragment"),
    [
        (
            "descriptive_summary",
            ["table:cohort_summary"],
            "TABLE-ONE / DESCRIPTIVE SUMMARIES:",
        ),
        (
            "cox_proportional_hazards",
            ["table:hazard_ratio"],
            "STATISTICS APIs:",
        ),
        (
            "publication_figure",
            ["figure:overview"],
            "- For rendering-only figure steps,",
        ),
    ],
)
def test_scoped_guide_selects_the_matching_section(
    method: str, outputs: list, expected_fragment: str
) -> None:
    step = AnalysisStep(
        step_id="01_probe",
        planned_analysis_role="auxiliary",
        intent="anchor integrity probe",
        inputs=["stay_id"],
        expected_outputs=outputs,
        method=method,
        icu_rule_refs=[],
    )
    scoped = coder_guide_for_step(_CODER_GUIDE, step)
    assert scoped
    assert expected_fragment in scoped
    rewrite = coder_rewrite_guide_for_step(_CODER_GUIDE, step)
    assert rewrite


def test_broken_anchor_fails_closed_with_the_section_name() -> None:
    mutated = _CODER_GUIDE.replace("STATISTICS APIs:", "STATISTIC APIs:", 1)
    with pytest.raises(ValueError, match="statistics"):
        _guide_segments(mutated)
