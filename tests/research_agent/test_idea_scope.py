"""Tests for the idea-mining literature scope → PubMed query builder."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from easyicu.research_agent.idea_scope import (
    JOURNAL_PRESETS,
    LiteratureScopeSpec,
    build_pubmed_query_from_scope,
    resolve_journals,
    resolve_year_range,
)


def test_empty_scope_is_rejected_as_too_broad() -> None:
    with pytest.raises(ValidationError, match="too broad"):
        LiteratureScopeSpec()


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown journal_preset"):
        LiteratureScopeSpec(journal_preset="not_a_real_preset")


def test_journal_preset_and_explicit_merge_and_dedup() -> None:
    scope = LiteratureScopeSpec(
        journals=["Intensive Care Med", "Chest"],
        journal_preset="critical_care_top3",
    )
    journals = resolve_journals(scope)
    # explicit first, then preset extras, case-insensitive dedup of ICM.
    assert journals[0] == "Intensive Care Med"
    assert "Chest" in journals
    assert journals.count("Intensive Care Med") == 1
    assert "Lancet Respir Med" in journals  # from preset


def test_last_n_years_is_deterministic_with_reference_year() -> None:
    scope = LiteratureScopeSpec(journal_preset="critical_care_top3", last_n_years=2)
    assert resolve_year_range(scope, reference_year=2026) == (2025, 2026)


def test_explicit_year_range_used_as_is() -> None:
    scope = LiteratureScopeSpec(topic_terms=["sepsis"], start_year=2020, end_year=2024)
    assert resolve_year_range(scope) == (2020, 2024)


def test_start_after_end_rejected() -> None:
    with pytest.raises(ValidationError, match="start_year cannot be after end_year"):
        LiteratureScopeSpec(topic_terms=["sepsis"], start_year=2025, end_year=2020)


def test_relative_and_explicit_window_conflict_rejected() -> None:
    with pytest.raises(ValidationError, match="not both"):
        LiteratureScopeSpec(topic_terms=["x"], last_n_years=2, start_year=2024)


def test_build_query_top3_reviews_editorials_last_2_years() -> None:
    # The user's concrete example: top-3 IF critical-care journals, last 2y,
    # editorials/reviews.
    scope = LiteratureScopeSpec(
        journal_preset="critical_care_top3",
        pub_types=["review", "editorial"],
        last_n_years=2,
    )
    query = build_pubmed_query_from_scope(scope, reference_year=2026)
    assert '"Intensive Care Med"[Journal]' in query
    assert '"Lancet Respir Med"[Journal]' in query
    assert "review[pt]" in query and "editorial[pt]" in query
    assert "2025:2026[dp]" in query
    assert " AND " in query


def test_broad_specialty_preset_includes_critical_care_journals() -> None:
    scope = LiteratureScopeSpec(
        journal_preset="critical_care_specialty_broad",
        pub_types=["review", "editorial", "letter"],
        last_n_years=3,
    )
    journals = resolve_journals(scope)
    assert "J Crit Care" in journals
    assert "Shock" in journals
    assert "Crit Care Explor" in journals
    assert "Ann Intensive Care" in journals
    assert "Crit Care Med" in journals
    assert "Resuscitation" in journals

    query = build_pubmed_query_from_scope(scope, reference_year=2026)
    assert '"J Crit Care"[Journal]' in query
    assert '"Crit Care Explor"[Journal]' in query
    assert '"Ann Intensive Care"[Journal]' in query
    assert '"Crit Care Med"[Journal]' in query
    assert "letter[pt]" in query
    assert "2024:2026[dp]" in query
    assert "sepsis" not in query.lower()


def test_wide_specialty_preset_adds_subspecialty_critical_care_journals() -> None:
    scope = LiteratureScopeSpec(
        journal_preset="critical_care_specialty_wide",
        pub_types=["review", "editorial", "letter"],
        last_n_years=3,
    )
    journals = resolve_journals(scope)
    assert "Ann Intensive Care" in journals
    assert "Crit Care Med" in journals
    assert "Curr Opin Crit Care" in journals
    assert "Crit Care Clin" in journals
    assert "Neurocrit Care" in journals
    assert "Pediatr Crit Care Med" in journals
    assert "J Trauma Acute Care Surg" in journals
    assert "Acute Crit Care" in journals

    query = build_pubmed_query_from_scope(scope, reference_year=2026)
    assert '"Curr Opin Crit Care"[Journal]' in query
    assert '"Neurocrit Care"[Journal]' in query
    assert '"Pediatr Crit Care Med"[Journal]' in query
    assert "letter[pt]" in query
    assert "2024:2026[dp]" in query
    assert "sepsis" not in query.lower()


def test_topic_terms_are_quoted_when_multiword_and_ored() -> None:
    scope = LiteratureScopeSpec(topic_terms=["sepsis", "septic shock"])
    query = build_pubmed_query_from_scope(scope)
    assert '(sepsis OR "septic shock")' in query


def test_preset_only_query_carries_no_clinical_terms() -> None:
    # G1 sanity: with no user topic, the query is pure publication metadata.
    scope = LiteratureScopeSpec(
        journal_preset="critical_care_top3",
        pub_types=["review"],
        start_year=2024,
        end_year=2026,
    )
    query = build_pubmed_query_from_scope(scope)
    expected = (
        '("Lancet Respir Med"[Journal] OR "Intensive Care Med"[Journal] '
        'OR "Crit Care"[Journal]) AND (review[pt]) AND 2024:2026[dp]'
    )
    assert query == expected


def test_unknown_pub_type_passes_through_as_pt_tag() -> None:
    scope = LiteratureScopeSpec(topic_terms=["x"], pub_types=["letter"])
    query = build_pubmed_query_from_scope(scope)
    assert "letter[pt]" in query


def test_presets_are_nonempty_lists() -> None:
    assert JOURNAL_PRESETS
    for name, journals in JOURNAL_PRESETS.items():
        assert journals, f"preset {name} is empty"
