"""Regression: a concept-SET idea is executable only if it resolves to enough
DISTINCT concepts that are actually available.

Before 2026-06-22 the default floor was 1, so a "set" whose many named terms all
collapsed to a single concept (e.g. seven fluid terms -> fluid_balance), or that
resolved to a single thin concept, was wrongly flagged executable as a
multi-variable analysis. The floor is now 2 for multi-variable families, with an
explicit floor of 1 for genuinely single-variable threshold/policy families.
"""

from __future__ import annotations

from easyicu.research_agent.concept_catalog import load_concept_catalog
from easyicu.research_agent.idea_mining import (
    _build_concept_lookup,
    _map_concept_set_candidate,
)
from easyicu.research_agent.idea_mining_schema import LiteratureIdeaCandidate

# A small AVAILABLE set (as if from an export catalog). fluid_balance present
# once; bun_creatinine_ratio present; crea + map present.
_AVAILABLE = ["crea", "map", "fluid_balance", "bun_creatinine_ratio", "sofa2"]


def _lookup():
    # Build the lookup the way the mine does: available concepts from the export,
    # synonym aliases from the (restricted) dictionary catalog.
    cat = load_concept_catalog(restrict_to=_AVAILABLE)
    return _build_concept_lookup(_AVAILABLE, concept_aliases=cat.concept_aliases)


def _candidate(family, concepts):
    return LiteratureIdeaCandidate(
        literature_idea_id="x",
        source_snapshot_id="s",
        citation_key="k",
        analysis_family=family,
        analysis_concepts=concepts,
        population="ICU adults",
        exposure_or_predictor="",
        outcome="",
        source_quote="q",
        source_adapter_level="metadata_only",
        rationale="r",
    )


def _map(family, concepts):
    ec = _map_concept_set_candidate(_candidate(family, concepts), lookup=_lookup())
    return ec.resolved_analysis_concepts, not bool(ec.non_executable_reasons)


def test_multivariable_set_collapsing_to_one_concept_is_not_executable():
    # Five fluid terms that all map to the single fluid_balance concept must NOT
    # count as an executable multi-variable descriptive analysis.
    resolved, executable = _map(
        "descriptive_epidemiology",
        ["fluid resuscitation", "fluid overload", "fluid removal", "fluid type",
         "fluid status"],
    )
    assert resolved == ["fluid_balance"]
    assert executable is False


def test_multivariable_set_with_two_distinct_concepts_is_executable():
    resolved, executable = _map(
        "descriptive_epidemiology", ["creatinine", "mean arterial pressure"]
    )
    assert set(resolved) == {"crea", "map"}
    assert executable is True


def test_single_variable_policy_family_with_one_concept_stays_executable():
    # A single-variable threshold/policy idea legitimately operates on one
    # concept (e.g. a urea-to-creatinine ratio cutoff sensitivity).
    resolved, executable = _map(
        "score_policy_sensitivity",
        ["urea-to-creatinine ratio cutoff threshold"],
    )
    assert resolved == ["bun_creatinine_ratio"]
    assert executable is True


def test_set_resolving_to_zero_available_concepts_is_not_executable():
    resolved, executable = _map(
        "score_policy_sensitivity", ["a concept that does not exist anywhere"]
    )
    assert resolved == []
    assert executable is False
