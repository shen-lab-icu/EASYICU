"""Absolute blood counts and IL-6 remain separate opt-in measurements."""

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.api.extraction import EXTRACT_MODULE_ORDER, EXTRACT_MODULES
from easyicu.concept import _apply_callback
from easyicu.concept.callback_apply import _drop_declared_row_exclusions
from easyicu.concept.catalog import HIDDEN_DICTIONARY_CONCEPTS, INFLAM_DETAIL_METADATA
from easyicu.resources import load_dictionary

DICTIONARY = json.loads(
    (Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json").read_text()
)


@pytest.mark.parametrize("name,unit,maximum,sources", [
    ("il6", "pg/mL", None, {"sic": [("laboratory", [569])]}),
    ("lymph_abs", "10^9/L", 1000, {
        "aumc": [("numericitems", [14258]), ("numericitems", [6780])],
        "miiv": [("chartevents", [229358]), ("labevents", [51133]), ("labevents", [52769])],
        "sic": [("laboratory", [302]), ("laboratory", [303])],
    }),
    ("neut_abs", "10^9/L", 1000, {
        "aumc": [("numericitems", [14254])],
        "miiv": [("chartevents", [229355]), ("chartevents", [229357]),
                 ("labevents", [52075])],
    }),
])
def test_dictionary_contract(name, unit, maximum, sources):
    definition = DICTIONARY[name]
    assert definition.get("class_name", "num_cncpt") == "num_cncpt"
    assert definition["unit"] == unit and definition["min"] == 0
    assert definition.get("max") == maximum
    assert {db: [(s["table"], s["ids"]) for s in rows]
            for db, rows in definition["sources"].items()} == sources
    assert name in EXTRACT_MODULES["inflam_detail"]
    assert name in HIDDEN_DICTIONARY_CONCEPTS and name in INFLAM_DETAIL_METADATA
    assert "inflam_detail" not in EXTRACT_MODULE_ORDER
    assert all(name not in EXTRACT_MODULES[module] for module in EXTRACT_MODULE_ORDER)


@pytest.mark.parametrize("name,database", [
    ("lymph_abs", "aumc"), ("lymph_abs", "miiv"), ("lymph_abs", "sic"),
    ("neut_abs", "aumc"), ("neut_abs", "miiv"),
])
def test_each_conversion_and_unit_gate(name, database):
    definition = DICTIONARY[name]
    for entry, source in zip(definition["sources"][database], load_dictionary()[name].sources[database]):
        expected = 0.001 if entry["ids"] == [52769] else 1.
        column = entry.get("unit_var")
        if column:
            valid = "10^9/L" if database == "aumc" else ("#/uL" if expected == 0.001 else "K/uL")
            frame = pd.DataFrame({name: [0., 1000., 1000.], column: [valid, valid, "wrong"]})
            frame = _drop_declared_row_exclusions(frame, source, name)
            assert len(frame) == 2
        else:
            frame = pd.DataFrame({name: [0., 1000.]})
        result = _apply_callback(frame, source, name, unit_column=column)
        assert result[name].tolist() == [0., 1000. * expected]


@pytest.mark.parametrize("database,name", [
    ("aumc", "lymph_abs"), ("aumc", "neut_abs"),
    ("sic", "il6"), ("sic", "lymph_abs"),
])
def test_generated_synthetic_source(tmp_path, database, name):
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


@pytest.mark.parametrize("name,table,item,unit,value,expected", [
    ("lymph_abs", "chartevents", 229358, "K/uL", 2., 2.),
    ("lymph_abs", "labevents", 51133, "K/uL", 2., 2.),
    ("lymph_abs", "labevents", 52769, "#/uL", 2000., 2.),
    ("neut_abs", "chartevents", 229355, "K/uL", 4., 4.),
    ("neut_abs", "chartevents", 229357, "K/uL", 4., 4.),
    ("neut_abs", "labevents", 52075, "K/uL", 4., 4.),
])
def test_miiv_synthetic_source(tmp_path, name, table, item, unit, value, expected):
    from tests.support.mimic_synthetic_source import mimic_synthetic_load, mimic_synthetic_source

    now = pd.Timestamp("2150-01-01 00:15:00")
    frame = pd.DataFrame({"subject_id": [11, 11], "hadm_id": [21, 21],
                          "itemid": [item, item], "charttime": [now, now + pd.Timedelta(hours=1)],
                          "storetime": [now, now + pd.Timedelta(hours=1)],
                          "valuenum": [value, value], "value": [str(value), str(value)],
                          "valueuom": [unit, "wrong"]})
    if table == "chartevents":
        frame["stay_id"] = 1
    other = frame.copy()
    other["itemid"] = -1
    other["stay_id"] = 1
    tables = {"chartevents": other, "labevents": other, table: frame}
    source = mimic_synthetic_source(tmp_path, "miiv", **tables)
    actual = mimic_synthetic_load(source, [name])[name].data[name].dropna()
    assert actual.tolist() == [expected]
