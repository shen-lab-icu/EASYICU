"""S732 urine concentrations: exact source extensions and conversion boundaries."""

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.concept import _apply_callback
from easyicu.concept.callback_apply import _drop_declared_row_exclusions
from easyicu.resources import load_dictionary

DICTIONARY = json.loads((Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json").read_text())
EXPECTED_SOURCES = {'urine_crea': {'aumc': [{'ids': [7109],
                          'table': 'numericitems',
                          'sub_var': 'itemid',
                          'unit_var': 'unit',
                          'require_regex': {'unit': '(?i)^\\s*mmol/l\\s*$'},
                          'callback': "convert_unit(binary_op(`*`, 11.312217194570135), 'mg/dL')"}],
                'hirid': [{'ids': [24000572],
                           'table': 'observations',
                           'sub_var': 'variableid',
                           'class_name': 'hrd_itm',
                           'callback': "convert_unit(binary_op(`*`, 0.011312217194570135), 'mg/dL')"}]},
 'urine_na': {'aumc': [{'ids': [10270],
                        'table': 'numericitems',
                        'sub_var': 'itemid',
                        'unit_var': 'unit',
                        'require_regex': {'unit': '(?i)^\\s*mmol/l\\s*$'}}],
              'hirid': [{'ids': [20003200],
                         'table': 'observations',
                         'sub_var': 'variableid',
                         'class_name': 'hrd_itm'}],
              'sic': [{'ids': [470], 'table': 'laboratory', 'sub_var': 'LaboratoryID'}]},
 'urine_osm': {'aumc': [{'ids': [11921],
                         'table': 'numericitems',
                         'sub_var': 'itemid',
                         'unit_var': 'unit',
                         'require_regex': {'unit': '(?i)^\\s*mosmol/kg\\s*$'},
                         'callback': "convert_unit(binary_op(`*`, 1), 'mOsm/kg')"},
                        {'ids': [11922],
                         'table': 'numericitems',
                         'sub_var': 'itemid',
                         'unit_var': 'unit',
                         'require_regex': {'unit': '(?i)^\\s*mosmol/kg\\s*$'},
                         'callback': "convert_unit(binary_op(`*`, 1), 'mOsm/kg')"}],
               'eicu': [{'ids': ['urinary osmolality'],
                         'table': 'lab',
                         'sub_var': 'labname',
                         'unit_var': 'labmeasurenameinterface',
                         'require_regex': {'labmeasurenameinterface': '(?i)^\\s*mOsm/kg\\s*$'}}],
               'sic': [{'ids': [475], 'table': 'laboratory', 'sub_var': 'LaboratoryID'}]},
 'urine_urea': {'aumc': [{'ids': [10280],
                          'table': 'numericitems',
                          'sub_var': 'itemid',
                          'unit_var': 'unit',
                          'require_regex': {'unit': '(?i)^\\s*mmol/l\\s*$'},
                          'callback': "convert_unit(binary_op(`*`, 2.80134), 'mg/dL')"},
                         {'ids': [10281],
                          'table': 'numericitems',
                          'sub_var': 'itemid',
                          'unit_var': 'unit',
                          'require_regex': {'unit': '(?i)^\\s*mmol/l\\s*$'},
                          'callback': "convert_unit(binary_op(`*`, 2.80134), 'mg/dL')"},
                         {'ids': [7846],
                          'table': 'numericitems',
                          'sub_var': 'itemid',
                          'unit_var': 'unit',
                          'require_regex': {'unit': '(?i)^\\s*mmol/l\\s*$'},
                          'callback': "convert_unit(binary_op(`*`, 2.80134), 'mg/dL')"}],
                'hirid': [{'ids': [24000754],
                           'table': 'observations',
                           'sub_var': 'variableid',
                           'class_name': 'hrd_itm',
                           'callback': "convert_unit(binary_op(`*`, 2.80134), 'mg/dL')"}],
                'sic': [{'ids': [356],
                         'table': 'laboratory',
                         'sub_var': 'LaboratoryID',
                         'callback': "convert_unit(binary_op(`*`, 0.466459), 'mg/dL')"},
                        {'ids': [396],
                         'table': 'laboratory',
                         'sub_var': 'LaboratoryID',
                         'callback': "convert_unit(binary_op(`*`, 0.1), 'mg/dL')"}]}}


@pytest.mark.parametrize("name", list(EXPECTED_SOURCES))
def test_dictionary_contract(name):
    definition = DICTIONARY[name]
    expected = {"urine_na": ("mmol/L", 400), "urine_crea": ("mg/dL", 1000),
                "urine_osm": ("mOsm/kg", 2000), "urine_urea": ("mg/dL", 5000)}
    unit, maximum = expected[name]
    assert definition.get("class_name", "num_cncpt") == "num_cncpt"
    assert unit in ([definition["unit"]] if isinstance(definition["unit"], str) else definition["unit"])
    assert definition["min"] == 0 and definition["max"] == maximum
    for database, sources in EXPECTED_SOURCES[name].items():
        assert all(entry in definition["sources"][database] for entry in sources)
    from easyicu.api.extraction import EXTRACT_MODULES
    assert name in EXTRACT_MODULES["renal_urine_chem"]


@pytest.mark.parametrize("name,database,item,unit,factor", [
    ("urine_crea", "aumc", 7109, "mmol/l", 11.312217194570135),
    ("urine_crea", "hirid", 24000572, None, 0.011312217194570135),
    ("urine_osm", "aumc", 11921, "mOsmol/kg", 1.),
    ("urine_osm", "aumc", 11922, "mOsmol/kg", 1.),
    ("urine_urea", "aumc", 10280, "mmol/l", 2.80134),
    ("urine_urea", "aumc", 10281, "mmol/l", 2.80134),
    ("urine_urea", "aumc", 7846, "mmol/l", 2.80134),
    ("urine_urea", "hirid", 24000754, None, 2.80134),
    ("urine_urea", "sic", 356, None, 0.466459),
    ("urine_urea", "sic", 396, None, 0.1),
])
def test_conversion_and_wrong_unit(name, database, item, unit, factor):
    source = next(s for s in load_dictionary()[name].sources[database] if s.ids == [item])
    column = source.unit_var
    frame = pd.DataFrame({name: [0., 10., 10.]})
    if column:
        frame[column] = [unit, unit, "wrong"]
        frame = _drop_declared_row_exclusions(frame, source, name)
        assert len(frame) == 2
    else:
        frame = frame.iloc[:2]
    actual = _apply_callback(frame, source, name, unit_column=column)
    assert actual[name].tolist() == pytest.approx([0., 10 * factor])


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    from easyicu.synthetic_sources import generate_source
    from easyicu.api.concepts import clear_global_loader

    root = tmp_path_factory.mktemp("urine-batch1")
    for database in ("aumc", "hirid", "sic"):
        generate_source(root / database, database, stays=12)
    yield root
    clear_global_loader()


@pytest.mark.parametrize("name,database", [(name, db) for name, dbs in EXPECTED_SOURCES.items()
                                          for db in dbs if db in {"aumc", "hirid", "sic"}])
def test_synthetic_extraction(sources, name, database):
    from easyicu import load_concepts

    actual = load_concepts([name], database=database, data_path=sources / database, merge=False,
                           verbose=False, concept_workers=1, parallel_workers=1)
    frame = actual if isinstance(actual, pd.DataFrame) else actual[name]
    assert not frame.empty and frame[name].notna().any()


def test_eicu_osmolality_unit_gate_in_loader(tmp_path):
    from easyicu.concept import ConceptResolver
    from tests.support.urine_chemistry_source import urine_chemistry_source

    source, _ = urine_chemistry_source(tmp_path, "eicu", bucketed=False)
    lab = pd.read_parquet(tmp_path / "lab.parquet").iloc[:2].copy()
    lab["labname"] = "urinary osmolality"
    lab["labresult"] = [600., 700.]
    lab["labmeasurenameinterface"] = ["mOsm/kg", "mOsm/L"]
    lab.to_parquet(tmp_path / "lab.parquet", index=False)
    result = ConceptResolver(load_dictionary()).load_concepts(
        ["urine_osm"], source, merge=False, r_compatible=False, concept_workers=1, verbose=False
    )["urine_osm"].data
    assert result.urine_osm.dropna().tolist() == [600.]
