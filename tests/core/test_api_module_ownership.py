"""Regression tests for the domain-owned public data API."""

from __future__ import annotations

import easyicu.api as api


def test_public_data_entry_points_resolve_to_domain_owners() -> None:
    assert api.load_concepts.__module__ == "easyicu.api.concepts"
    assert api.load_demographics.__module__ == "easyicu.api.convenience"
    assert api.list_available_concepts.__module__ == "easyicu.api.special_concepts"
    assert api.align_to_icu_admission.__module__ == "easyicu.api.compat"
    assert api.extract_database.__module__ == "easyicu.api.extraction"


def test_public_api_facade_stays_thin() -> None:
    source = api.__file__
    assert source is not None
    with open(source, encoding="utf-8") as handle:
        assert sum(1 for _ in handle) < 600
