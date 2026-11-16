"""Contract and synthetic extraction for the opt-in cardiac assay markers."""

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.api.extraction import EXTRACT_MODULE_ORDER, EXTRACT_MODULES
from easyicu.concept import _apply_callback
from easyicu.concept.callback_apply import _drop_declared_row_exclusions
from easyicu.concept.catalog import CARDIAC_MARKERS_METADATA, HIDDEN_DICTIONARY_CONCEPTS
from easyicu.resources import load_dictionary

DICTIONARY = json.loads(
    (Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json").read_text()
)


@pytest.mark.parametrize(
    "name,unit,sources",
    [
        ("ckmb_activity", "U/L", {"aumc": [("numericitems", [6824])],
                                   "sic": [("laboratory", [612])]}),
        ("tnt_hs", "ng/mL", {"hirid": [("observations", [24000806])],
                              "sic": [("laboratory", [481])]}),
    ],
)
def test_dictionary_contract(name, unit, sources):
    definition = DICTIONARY[name]
    assert definition.get("class_name", "num_cncpt") == "num_cncpt"
    assert definition["unit"] == unit and definition["min"] == 0
    assert "max" not in definition
    assert {db: [(s["table"], s["ids"]) for s in rows]
            for db, rows in definition["sources"].items()} == sources
    assert name in EXTRACT_MODULES["cardiac_markers"]
    assert name in HIDDEN_DICTIONARY_CONCEPTS and name in CARDIAC_MARKERS_METADATA
    assert "cardiac_markers" not in EXTRACT_MODULE_ORDER
    assert all(name not in EXTRACT_MODULES[module] for module in EXTRACT_MODULE_ORDER)


def test_activity_unit_gate_and_conversion():
    source = load_dictionary()["ckmb_activity"].sources["aumc"][0]
    assert source.unit_var == "unit"
    assert source.params["require_regex"] == {"unit": r"(?i)^\s*IE/l\s*$"}
    frame = pd.DataFrame({"ckmb_activity": [0., 20., 30.], "unit": ["IE/l", "IE/l", "U/L"]})
    filtered = _drop_declared_row_exclusions(frame, source, "ckmb_activity")
    result = _apply_callback(filtered, source, "ckmb_activity", unit_column="unit")
    assert result.ckmb_activity.tolist() == [0., 20.]
    assert result.unit.tolist() == ["U/L", "U/L"]


@pytest.mark.parametrize("database", ["hirid", "sic"])
def test_high_sensitivity_conversion(database):
    source = load_dictionary()["tnt_hs"].sources[database][0]
    frame = pd.DataFrame({"tnt_hs": [0., 1000., 2500.]})
    result = _apply_callback(frame, source, "tnt_hs")
    assert result.tnt_hs.tolist() == [0., 1., 2.5]


def test_sic_activity_conversion():
    source = load_dictionary()["ckmb_activity"].sources["sic"][0]
    result = _apply_callback(pd.DataFrame({"ckmb_activity": [0., 23.]}), source, "ckmb_activity")
    assert result.ckmb_activity.tolist() == [0., 23.]


@pytest.mark.parametrize("database,name", [
    ("aumc", "ckmb_activity"), ("sic", "ckmb_activity"),
    ("hirid", "tnt_hs"), ("sic", "tnt_hs"),
])
def test_synthetic_source_extracts(tmp_path, database, name):
    from easyicu import load_concepts
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / database
    generate_source(root, database, stays=12)
    try:
        result = load_concepts([name], database=database, data_path=root, merge=False,
                               verbose=False, concept_workers=1, parallel_workers=1)
        frame = result if isinstance(result, pd.DataFrame) else result[name]
        assert not frame.empty and frame[name].notna().any()
    finally:
        clear_global_loader()
