"""Focused contracts for the concept-dictionary bounds loader."""

from __future__ import annotations

import pytest

from easyicu.concept import loader


def test_packaged_concept_dictionary_supplies_age_bounds() -> None:
    loader._load_concept_dict_cached.cache_clear()

    assert loader._get_concept_bounds("age", "min") == 0.0
    assert loader._get_concept_bounds("age", "max") == 100.0


def test_unknown_concept_or_missing_bound_remains_optional() -> None:
    assert loader._get_concept_bounds("not_a_real_concept", "min") is None
    assert loader._get_concept_bounds("alp", "max") is None


@pytest.mark.parametrize("invalid", ["not-a-number", float("inf")])
def test_malformed_bound_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    invalid: object,
) -> None:
    monkeypatch.setattr(
        loader,
        "_load_concept_dict_cached",
        lambda: {"age": {"min": invalid}},
    )

    with pytest.raises(ValueError, match="concept 'age'.*'min' bound"):
        loader._get_concept_bounds("age", "min")


def test_dictionary_load_error_is_not_silently_downgraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load() -> dict[str, object]:
        raise OSError("fixture resource unavailable")

    monkeypatch.setattr(loader, "_load_concept_dict_cached", fail_load)

    with pytest.raises(OSError, match="resource unavailable"):
        loader._get_concept_bounds("age", "min")
