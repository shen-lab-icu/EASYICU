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
from easyicu.research_agent.idea_mining import (
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
def test_urea_creatinine_ratio_is_derived_not_pafi(lookup, term):
    # Must anchor to a REAL component (bun/crea), never the unrelated P/F-ratio.
    resolved = _resolve_concept(term, lookup)
    assert resolved in {"bun", "crea"}
    status, requirements, _ = _feature_derivation_status(
        term, resolved_key=resolved, lookup=lookup
    )
    assert status == "requires_derived_feature"
    # The requirement must name BOTH component concepts (bun + crea).
    joined = " ".join(requirements).lower()
    assert "bun" in joined and "crea" in joined


def test_generic_shared_token_is_down_weighted(lookup):
    # The general cure for the bug class: a token shared across many concepts
    # ("ratio" -> pafi, safi, nlr...) must score strictly lower than a token that
    # names a single concept ("urea"). This is what stops a generic word from
    # tying a real clinical signal, without enumerating stop-words.
    spec = _token_specificity(lookup)
    assert spec["ratio"] < spec["urea"]
    assert spec["urea"] == 1.0  # urea names exactly one concept (bun)


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
