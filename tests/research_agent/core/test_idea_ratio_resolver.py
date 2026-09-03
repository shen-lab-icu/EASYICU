"""Regression: a two-component clinical ratio must not mis-map to an unrelated
unified ratio concept (G-5).

Root cause: in the inexact resolution fallback the generic word "ratio" scored
as a clinical signal token, so ``pafi`` (whose alias is "P/F ratio") tied the
real components of "urea-to-creatinine ratio" (urea->bun, creatinine->crea) and
won by insertion order. The discovery dry run then reported
``resolved_predictor: pafi`` for a BUN/creatinine-ratio topic.

Fix: "ratio" is desensitised in the resolution fallback only (exact-phrase
lookups for genuine unified ratio concepts are untouched), and an X-to-Y ratio
that has no exact unified concept but resolves to >=2 distinct component
concepts is flagged ``requires_derived_feature`` naming those components.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.concept_catalog import load_concept_catalog
from easyicu.research_agent.discovery.idea_mining import (
    _build_concept_lookup,
    _feature_derivation_status,
    _resolve_concept,
    _token_specificity,
)


@pytest.fixture(scope="module")
def lookup():
    cat = load_concept_catalog()
    return _build_concept_lookup(
        list(cat.available_concepts), concept_aliases=cat.concept_aliases
    )


@pytest.mark.parametrize(
    "term",
    ["urea-to-creatinine ratio", "urea/creatinine ratio", "BUN/creatinine ratio"],
)
def test_urea_creatinine_ratio_resolves_to_unified_concept_not_pafi(lookup, term):
    # bun_creatinine_ratio was added as a first-class derived concept
    # (2026-06-22) with explicit "urea-to-creatinine ratio"/"UCR" synonyms, so a
    # urea/creatinine-ratio phrase now resolves to THAT unified concept via the
    # exact variant table — exactly like "P/F ratio" -> pafi — never to the
    # unrelated pafi, and no longer needs on-the-fly derivation from bun+crea.
    resolved = _resolve_concept(term, lookup)
    assert resolved == "bun_creatinine_ratio"
    assert resolved != "pafi"
    status, _, _ = _feature_derivation_status(
        term, resolved_key=resolved, lookup=lookup
    )
    assert status == "raw_concept_available"


@pytest.mark.parametrize(
    "term",
    [
        "urea-to-creatinine ratio cutoff threshold",
        "admission urea-to-creatinine ratio",
        "ICU admission urea-to-creatinine ratio measurement timing",
    ],
)
def test_noisy_concept_set_phrase_recovers_embedded_unified_concept(lookup, term):
    # A concept-SET / threshold idea often arrives as a noisy phrase that embeds
    # a real first-class concept ("urea-to-creatinine ratio cutoff threshold").
    # The embedded multi-word alias must win over decomposing the ratio into
    # bun+crea, and must never fall through to a spurious leading token
    # ("admission" -> adm). Regression for the 2026-06-22 embedded-alias fix.
    resolved = _resolve_concept(term, lookup)
    assert resolved == "bun_creatinine_ratio"
    assert resolved not in {"adm", "bun", "crea", "pafi"}


def test_generic_shared_token_is_down_weighted(lookup):
    # The general cure for the bug class: a token shared across many concepts
    # ("ratio" -> pafi, safi, nlr...) must score strictly lower than a more
    # specific token ("urea"). This is what stops a generic word from tying a
    # real clinical signal, without enumerating stop-words. NB: since
    # bun_creatinine_ratio was added, "urea" names two concepts (bun +
    # bun_creatinine_ratio) so its specificity is 0.5 rather than 1.0 — but the
    # generic "ratio" is shared far more widely and still ranks strictly below.
    spec = _token_specificity(lookup)
    assert spec["ratio"] < spec["urea"]


@pytest.mark.parametrize(
    "term,expected",
    [
        ("P/F ratio", "pafi"),
        ("PaO2/FiO2 ratio", "pafi"),
        ("S/F ratio", "safi"),
        ("RDW", "rdw"),
    ],
)
def test_unified_ratio_concepts_still_resolve_as_raw(lookup, term, expected):
    # Genuine single-concept ratios resolve via the exact variant table and must
    # stay raw_concept_available (no spurious derived-feature flag).
    resolved = _resolve_concept(term, lookup)
    assert resolved == expected
    status, _, _ = _feature_derivation_status(
        term, resolved_key=resolved, lookup=lookup
    )
    assert status == "raw_concept_available"


@pytest.mark.parametrize(
    "term,expected",
    [("creatinine", "crea"), ("urea", "bun"), ("peep", "peep"), ("lactate", "lact")],
)
def test_plain_concepts_unaffected(lookup, term, expected):
    assert _resolve_concept(term, lookup) == expected
    status, _, _ = _feature_derivation_status(
        term, resolved_key=expected, lookup=lookup
    )
    assert status == "raw_concept_available"


@pytest.mark.parametrize(
    "term,expected",
    [
        ("serum creatinine", "crea"),
        ("plasma creatinine", "crea"),
        ("serum lactate", "lact"),
    ],
)
def test_specimen_qualified_lab_phrases_resolve_to_lab_concepts(
    lookup, term, expected
):
    assert _resolve_concept(term, lookup) == expected
