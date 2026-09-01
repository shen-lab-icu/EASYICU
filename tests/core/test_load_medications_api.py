"""Structural / behavioral tests for the expanded ``load_medications`` API.

Validates without touching the real data pipeline that:
  - The legacy ricu 14-concept list is still the default when ``include_new=False``.
  - ``include_new=True`` (default) delivers the full expanded catalog.
  - ``groups=`` selector filters correctly (single str and list).
  - An unknown group raises ValueError with a helpful message.
  - Every concept named in the groups actually exists in concept-dict.json.

These tests monkey-patch the inner ``load_concepts`` call to avoid hitting any
database, so they run in the same environment as the rest of the dict-level
tests in this suite.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from easyicu import api  # noqa: E402


DICT = REPO / "src" / "easyicu" / "data" / "concept-dict.json"


def test_medication_errors_are_part_of_the_public_api():
    import easyicu

    assert easyicu.MedicationLoadError is api.MedicationLoadError
    assert easyicu.MedicationMergeError is api.MedicationMergeError


# ── helpers ──
class _CaptureCalls:
    """Monkey-patch target for ``api.load_concepts``. Records the concepts
    that would have been loaded and returns a tiny valid DataFrame."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, concepts, **kwargs):
        if isinstance(concepts, str):
            concepts = [concepts]
        self.calls.append(list(concepts))
        # Return a dataframe with the concept name column so the merge path
        # inside load_medications keeps working.
        col = concepts[0]
        return pd.DataFrame({"stay_id": [1], "charttime": [0.0], col: [True]})


@pytest.fixture
def capture(monkeypatch):
    cap = _CaptureCalls()
    monkeypatch.setattr(api, "load_concepts", cap)
    # Also stub out _validate_concepts so we can test purely against our config
    monkeypatch.setattr(api, "_validate_concepts", lambda cs, verbose=False: cs)
    return cap


@pytest.fixture(scope="module")
def cdict():
    with DICT.open() as f:
        return json.load(f)


# ── legacy behavior ──
def test_legacy_default_concept_list_unchanged(capture):
    api.load_medications(include_new=False, verbose=False)
    loaded = [c for call in capture.calls for c in call]
    expected = [
        "abx", "adh_rate", "cort", "dex", "dobu_dur", "dobu_rate", "dobu60",
        "epi_dur", "epi_rate", "ins", "norepi_dur", "norepi_equiv",
        "norepi_rate", "vaso_ind",
    ]
    assert sorted(loaded) == sorted(expected), (
        f"Legacy mode diverged from ricu baseline: {sorted(loaded)}"
    )


def test_legacy_mode_preserves_exact_count(capture):
    api.load_medications(include_new=False)
    loaded = [c for call in capture.calls for c in call]
    assert len(loaded) == 14, f"Legacy ricu baseline must stay at 14 concepts, got {len(loaded)}"


# ── expanded catalog ──
def test_full_catalog_includes_new_meds(capture):
    api.load_medications()  # default include_new=True
    loaded = set(c for call in capture.calls for c in call)
    # New additions must all be included
    for must in [
        "furosemide", "propofol", "propofol_rate", "midazolam", "midazolam_rate",
        "fentanyl", "fentanyl_rate", "heparin", "amiodarone",
        "lorazepam", "ketamine", "vecuronium", "cisatracurium", "nitroglycerin",
        "pantoprazole", "vancomycin", "meropenem", "calcium_iv",
        "potassium_iv", "magnesium_iv", "albumin_iv", "packed_rbc",
        "bicarbonate", "dextrose50", "ffp", "platelets",
    ]:
        assert must in loaded, f"{must} missing from expanded catalog"


def test_full_catalog_still_has_legacy(capture):
    api.load_medications()
    loaded = set(c for call in capture.calls for c in call)
    for must in ["abx", "norepi_rate", "vaso_ind"]:
        assert must in loaded, f"Legacy {must} disappeared from expanded catalog"


def test_no_duplicate_concepts_in_full_catalog(capture):
    api.load_medications()
    loaded = [c for call in capture.calls for c in call]
    assert len(loaded) == len(set(loaded)), (
        f"Duplicate concepts loaded: {[c for c in loaded if loaded.count(c) > 1]}"
    )


# ── group filtering ──
def test_groups_sedation_only(capture):
    api.load_medications(groups="sedation")
    loaded = [c for call in capture.calls for c in call]
    expected = {
        "propofol", "propofol_rate", "midazolam", "midazolam_rate",
        "dexmedetomidine", "lorazepam", "ketamine",
    }
    assert set(loaded) == expected


def test_groups_multiple_as_list(capture):
    api.load_medications(groups=["sedation", "analgesia"])
    loaded = set(c for call in capture.calls for c in call)
    # Sedation
    assert "propofol" in loaded
    assert "midazolam" in loaded
    # Analgesia
    assert "fentanyl" in loaded
    assert "morphine" in loaded
    # Must not include unrelated groups
    assert "abx" not in loaded
    assert "norepi_rate" not in loaded


def test_groups_neuromuscular_blockers(capture):
    api.load_medications(groups="neuromuscular")
    loaded = set(c for call in capture.calls for c in call)
    assert loaded == {"rocuronium", "vecuronium", "cisatracurium"}


def test_unknown_group_raises(capture):
    with pytest.raises(ValueError, match="Unknown medication group"):
        api.load_medications(groups="unicorn")


def test_groups_overrides_include_new(capture):
    """When ``groups`` is specified, ``include_new=False`` is irrelevant."""
    api.load_medications(groups="sedation", include_new=False)
    loaded = set(c for call in capture.calls for c in call)
    assert "propofol" in loaded  # sedation must still load
    assert "abx" not in loaded   # legacy list ignored


def test_load_failure_is_not_silently_returned_as_complete(monkeypatch):
    monkeypatch.setattr(
        api, "_validate_concepts", lambda concepts, verbose=False: concepts
    )

    def load_one(concepts, **kwargs):
        concept = concepts[0]
        if concept == "vancomycin":
            raise OSError("synthetic read failure")
        return pd.DataFrame(
            {"stay_id": [1], "charttime": [0.0], concept: [True]}
        )

    monkeypatch.setattr(api, "load_concepts", load_one)

    with pytest.raises(api.MedicationLoadError) as caught:
        api.load_medications(groups="antibiotics")

    assert caught.value.report["loaded"] == ["abx", "meropenem"]
    assert caught.value.report["failed"] == {
        "vancomycin": {"reason": "load_error", "error_type": "OSError"}
    }
    assert "synthetic read failure" not in str(caught.value)


def test_explicit_partial_result_has_warning_and_structured_report(monkeypatch):
    monkeypatch.setattr(
        api, "_validate_concepts", lambda concepts, verbose=False: concepts
    )

    def load_one(concepts, **kwargs):
        concept = concepts[0]
        if concept == "vancomycin":
            return pd.DataFrame()
        return pd.DataFrame(
            {"stay_id": [1], "charttime": [0.0], concept: [True]}
        )

    monkeypatch.setattr(api, "load_concepts", load_one)

    with pytest.warns(RuntimeWarning, match="partial result"):
        result = api.load_medications(groups="antibiotics", allow_partial=True)

    report = result.attrs["easyicu_medication_load_report"]
    assert report["loaded"] == ["abx", "meropenem"]
    assert report["failed"] == {"vancomycin": {"reason": "empty_result"}}
    assert {"abx", "meropenem"}.issubset(result.columns)


def test_medication_merge_rejects_many_to_many_row_multiplication(monkeypatch):
    monkeypatch.setattr(
        api, "_validate_concepts", lambda concepts, verbose=False: concepts
    )

    def load_one(concepts, **kwargs):
        concept = concepts[0]
        rows = 2 if concept == "abx" else 1
        return pd.DataFrame(
            {
                "stay_id": [1] * rows,
                "charttime": [0.0] * rows,
                concept: [True] * rows,
            }
        )

    monkeypatch.setattr(api, "load_concepts", load_one)

    with pytest.raises(api.MedicationMergeError, match="could multiply rows"):
        api.load_medications(groups="antibiotics")


# ── catalog ↔ concept-dict consistency ──
@pytest.mark.parametrize("group_name", [
    "vasopressors", "sedation", "analgesia", "neuromuscular",
    "antibiotics", "cardiac", "diuretics", "anticoagulation",
    "endocrine", "vasodilators", "gi_prophylaxis", "electrolytes",
    "colloids_blood", "other",
])
def test_every_group_concept_exists_in_dict(cdict, group_name, capture):
    """Regression guard: every concept named in a group must be in the dict."""
    api.load_medications(groups=group_name)
    loaded = [c for call in capture.calls for c in call]
    for concept in loaded:
        assert concept in cdict, (
            f"Group '{group_name}' references '{concept}' which is not in "
            f"concept-dict.json — orphan reference"
        )
