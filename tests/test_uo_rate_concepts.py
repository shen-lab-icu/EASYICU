"""Structural regression tests for uo_6h / uo_12h / uo_24h rolling-window
urine-output rate concepts.

These are rec_cncpt entries that depend on ``urine`` and ``weight`` and are
computed by the existing Python callbacks in ``easyicu.callbacks`` (``uo_6h``,
``uo_12h``, ``uo_24h``). Prior to this patch the callbacks existed and were
dispatched via ``CONCEPT_CALLBACKS``, but there was no first-class JSON
concept entry — ``rrt_criteria`` referenced them via ``concepts:`` alone.
Adding the JSON entries lets users load them directly.

End-to-end validation (MIIV, 200 stays, 2026-05-13):
  - Median ~1.37 mL/kg/h (clinical normal 0.5-2)
  - 99 % of rows within [0, 10] mL/kg/h
  - 188/200 stays have data, matching ICU near-universal urine documentation
"""
from __future__ import annotations

import json
import pathlib

import pytest

DICT = pathlib.Path(__file__).resolve().parents[1] / "src" / "easyicu" / "data" / "concept-dict.json"


UO_CONCEPTS = [
    ("uo_6h",  6),
    ("uo_12h", 12),
    ("uo_24h", 24),
]


@pytest.fixture(scope="module")
def cdict():
    with DICT.open() as f:
        return json.load(f)


@pytest.mark.parametrize("name,_w", UO_CONCEPTS, ids=[c[0] for c in UO_CONCEPTS])
def test_is_rec_cncpt(cdict, name, _w):
    assert name in cdict
    entry = cdict[name]
    assert entry["class_name"] == "rec_cncpt"
    assert entry["category"] == "renal"
    assert entry["unit"] == "mL/kg/h"


@pytest.mark.parametrize("name,_w", UO_CONCEPTS, ids=[c[0] for c in UO_CONCEPTS])
def test_depends_on_urine_and_weight(cdict, name, _w):
    entry = cdict[name]
    assert set(entry["depends_on"]) == {"urine", "weight"}
    assert set(entry["concepts"]) == {"urine", "weight"}


@pytest.mark.parametrize("name,window", UO_CONCEPTS, ids=[c[0] for c in UO_CONCEPTS])
def test_callback_name_matches_window(cdict, name, window):
    """Callback name must exactly match the registered dispatch key in
    ``CONCEPT_CALLBACKS``."""
    assert cdict[name]["callback"] == f"uo_{window}h"


@pytest.mark.parametrize("name,_w", UO_CONCEPTS, ids=[c[0] for c in UO_CONCEPTS])
def test_callback_is_dispatchable(name, _w):
    """The declared callback name must resolve in the registry."""
    from easyicu.concept_callbacks import CALLBACK_REGISTRY  # type: ignore
    assert name in CALLBACK_REGISTRY, (
        f"{name} callback not registered in CALLBACK_REGISTRY dispatch"
    )


@pytest.mark.parametrize("name,_w", UO_CONCEPTS, ids=[c[0] for c in UO_CONCEPTS])
def test_registered_in_webapp_catalog(name, _w):
    from easyicu.concept_catalog import (
        CONCEPT_DB_COVERAGE,
        CONCEPT_DICTIONARY,
    )
    assert name in CONCEPT_DICTIONARY
    en, zh, unit = CONCEPT_DICTIONARY[name]
    assert "Urine" in en
    assert "尿量" in zh
    assert unit == "mL/kg/h"
    assert CONCEPT_DB_COVERAGE[name] == 6


def test_rrt_criteria_still_references_all_three(cdict):
    """Regression guard: the rec_cncpt that motivated their existence must
    keep the dependency declaration."""
    rrt = cdict["rrt_criteria"]
    for n in ["uo_6h", "uo_12h", "uo_24h"]:
        assert n in rrt["concepts"], f"rrt_criteria lost {n} dependency"
