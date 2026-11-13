"""Venous gases require unanimous, same-subject specimen evidence."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.api.extraction import EXTRACT_MODULES, EXTRACT_MODULE_ORDER
from easyicu.concept import _apply_callback
from easyicu.concept.catalog import HIDDEN_DICTIONARY_CONCEPTS, GAS_SPECIMEN_METADATA
from easyicu.resources import load_dictionary

DICTIONARY = json.loads((Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json").read_text())
EXPECTED = {'be_venous': {'unit': 'mEq/L',
               'min': -25,
               'max': 25,
               'description': 'Venous arterial base excess',
               'category': 'blood gas',
               'sources': {'miiv': [{'ids': [50802],
                                     'table': 'labevents',
                                     'sub_var': 'itemid',
                                     'unit_var': 'valueuom',
                                     'require_regex': {'valueuom': '(?i)^\\s*mEq/L\\s*$'},
                                     'callback': 'batch1_miiv_gas_site',
                                     'extra_vars': ['specimen_id', 'subject_id'],
                                     'specimen_token': 'VEN.'}]}},
 'lact_venous': {'unit': 'mmol/L',
                 'min': 0,
                 'max': 50,
                 'description': 'Venous lactate',
                 'category': 'cardiovascular',
                 'sources': {'hirid': [{'ids': [24000485],
                                        'table': 'observations',
                                        'sub_var': 'variableid',
                                        'class_name': 'hrd_itm'},
                                       {'ids': [24000732],
                                        'table': 'observations',
                                        'sub_var': 'variableid',
                                        'class_name': 'hrd_itm'}],
                             'miiv': [{'ids': [50813],
                                       'table': 'labevents',
                                       'sub_var': 'itemid',
                                       'unit_var': 'valueuom',
                                       'require_regex': {'valueuom': '(?i)^\\s*mmol/L\\s*$'},
                                       'callback': 'batch1_miiv_gas_site',
                                       'extra_vars': ['specimen_id', 'subject_id'],
                                       'specimen_token': 'VEN.'}]}},
 'pco2_mixed_venous': {'unit': 'mmHg',
                       'min': 10,
                       'max': 150,
                       'description': 'Mixed-venous CO2 partial pressure',
                       'category': 'blood gas',
                       'sources': {'miiv': [{'ids': [50818],
                                             'table': 'labevents',
                                             'sub_var': 'itemid',
                                             'unit_var': 'valueuom',
                                             'require_regex': {'valueuom': '(?i)^\\s*mm\\ Hg\\s*$'},
                                             'callback': 'batch1_miiv_gas_site',
                                             'extra_vars': ['specimen_id', 'subject_id'],
                                             'specimen_token': 'MIX.'}]}},
 'pco2_venous': {'unit': 'mmHg',
                 'min': 10,
                 'max': 150,
                 'description': 'Venous CO2 partial pressure',
                 'category': 'blood gas',
                 'sources': {'miiv': [{'ids': [50818],
                                       'table': 'labevents',
                                       'sub_var': 'itemid',
                                       'unit_var': 'valueuom',
                                       'require_regex': {'valueuom': '(?i)^\\s*mm\\ Hg\\s*$'},
                                       'callback': 'batch1_miiv_gas_site',
                                       'extra_vars': ['specimen_id', 'subject_id'],
                                       'specimen_token': 'VEN.'}]}},
 'ph_mixed_venous': {'min': 6.8,
                     'max': 8,
                     'description': 'Mixed-venous pH of blood',
                     'category': 'blood gas',
                     'sources': {'miiv': [{'ids': [50820],
                                           'table': 'labevents',
                                           'sub_var': 'itemid',
                                           'unit_var': 'valueuom',
                                           'require_regex': {'valueuom': '(?i)^\\s*units\\s*$'},
                                           'callback': 'batch1_miiv_gas_site',
                                           'extra_vars': ['specimen_id', 'subject_id'],
                                           'specimen_token': 'MIX.'}]}},
 'ph_venous': {'min': 6.8,
               'max': 8,
               'description': 'Venous pH of blood',
               'category': 'blood gas',
               'sources': {'miiv': [{'ids': [50820],
                                     'table': 'labevents',
                                     'sub_var': 'itemid',
                                     'unit_var': 'valueuom',
                                     'require_regex': {'valueuom': '(?i)^\\s*units\\s*$'},
                                     'callback': 'batch1_miiv_gas_site',
                                     'extra_vars': ['specimen_id', 'subject_id'],
                                     'specimen_token': 'VEN.'}],
                           'sic': [{'ids': [697], 'table': 'laboratory', 'sub_var': 'LaboratoryID'}]}},
 'po2_mixed_venous': {'unit': 'mmHg',
                      'min': 20,
                      'max': 600,
                      'description': 'Mixed-venous O2 partial pressure',
                      'category': 'blood gas',
                      'sources': {'miiv': [{'ids': [50821],
                                            'table': 'labevents',
                                            'sub_var': 'itemid',
                                            'unit_var': 'valueuom',
                                            'require_regex': {'valueuom': '(?i)^\\s*mm\\ Hg\\s*$'},
                                            'callback': 'batch1_miiv_gas_site',
                                            'extra_vars': ['specimen_id', 'subject_id'],
                                            'specimen_token': 'MIX.'}]}},
 'po2_venous': {'unit': 'mmHg',
                'min': 20,
                'max': 600,
                'description': 'Venous O2 partial pressure',
                'category': 'blood gas',
                'sources': {'miiv': [{'ids': [50821],
                                      'table': 'labevents',
                                      'sub_var': 'itemid',
                                      'unit_var': 'valueuom',
                                      'require_regex': {'valueuom': '(?i)^\\s*mm\\ Hg\\s*$'},
                                      'callback': 'batch1_miiv_gas_site',
                                      'extra_vars': ['specimen_id', 'subject_id'],
                                      'specimen_token': 'VEN.'}],
                            'sic': [{'ids': [694], 'table': 'laboratory', 'sub_var': 'LaboratoryID'}]}}}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_dictionary_contract(name):
    assert DICTIONARY[name] == EXPECTED[name]
    assert name in EXTRACT_MODULES["gas_specimen"]
    assert name in HIDDEN_DICTIONARY_CONCEPTS and name in GAS_SPECIMEN_METADATA
    assert "gas_specimen" not in EXTRACT_MODULE_ORDER
    assert all(name not in EXTRACT_MODULES[module] for module in EXTRACT_MODULE_ORDER)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_miiv_specimen_selection_unit_and_bounds(tmp_path, name):
    from tests.support.mimic_synthetic_source import mimic_synthetic_load, mimic_synthetic_source

    entry = EXPECTED[name]["sources"]["miiv"][0]
    item, site = entry["ids"][0], entry["specimen_token"]
    unit = {"be_venous": "mEq/L", "lact_venous": "mmol/L",
            "ph_venous": "units", "ph_mixed_venous": "units"}.get(name, "mm Hg")
    value = 7.4 if name.startswith("ph_") else (2. if name == "lact_venous" else 40.)
    if name == "be_venous":
        value = -5.
    origin = pd.Timestamp("2150-01-01")
    times = origin + pd.to_timedelta([15, 75, 135, 195], unit="m")
    analytes = pd.DataFrame({"subject_id": [11] * 4, "hadm_id": [21] * 4, "itemid": [item] * 4,
        "specimen_id": [101, 102, 103, 104], "charttime": times, "storetime": times,
        "valuenum": [value] * 4, "value": [str(value)] * 4,
        "valueuom": [unit, unit, unit, "wrong"]})
    labels = pd.DataFrame({"subject_id": [11] * 4, "hadm_id": [21] * 4, "itemid": [52033] * 4,
        "specimen_id": [101, 102, 103, 104], "charttime": times, "storetime": times,
        "valuenum": [None] * 4, "value": [site, "ART.", "CENTRAL VENOUS.", site],
        "valueuom": [None] * 4})
    source = mimic_synthetic_source(tmp_path, "miiv", labevents=pd.concat([analytes, labels]))
    result = mimic_synthetic_load(source, [name])[name].data[name].dropna()
    assert result.tolist() == pytest.approx([value])


@pytest.mark.parametrize("token", ["VEN.", "MIX."])
def test_exact_specimen_unanimity_and_subject_isolation(token):
    name = "ph_venous" if token == "VEN." else "ph_mixed_venous"
    source = load_dictionary()[name].sources["miiv"][0]
    specimen = pd.DataFrame({"subject_id": [1, 1, 1, 1, 1, 2, 1, 1],
        "specimen_id": [1, 2, 2, 3, 4, 1, 5, 5], "itemid": [52033] * 8,
        "value": ["  " + token.lower() + "  ", token, "ART.", "", "CENTRAL VENOUS.",
                  "ART.", token, ""]})
    frame = pd.DataFrame({"subject_id": [1] * 7, "specimen_id": [1, 2, 3, 4, 5, 6, None],
                          "charttime": pd.Timestamp("2150-01-01"), name: [7.4] * 7, "valueuom": ["units"] * 7})
    ds = SimpleNamespace(load_table=lambda *a, **kw: specimen)
    result = _apply_callback(frame, source, name, data_source=ds)
    assert result.specimen_id.tolist() == [1., 5.]
    batched = pd.concat([_apply_callback(frame.iloc[:3], source, name, data_source=ds),
                         _apply_callback(frame.iloc[3:], source, name, data_source=ds)])
    pd.testing.assert_frame_equal(result, batched)


@pytest.mark.parametrize("name,database", [(n, db) for n, d in EXPECTED.items()
                                          for db in d["sources"] if db in {"hirid", "sic"}])
def test_other_synthetic_source_extraction(tmp_path, name, database):
    from easyicu import load_concepts
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / database
    generate_source(root, database, stays=12)
    try:
        actual = load_concepts([name], database=database, data_path=root, merge=False,
                               verbose=False, concept_workers=1, parallel_workers=1)
        frame = actual if isinstance(actual, pd.DataFrame) else actual[name]
        assert not frame.empty and frame[name].notna().any()
    finally:
        clear_global_loader()


def test_venous_base_excess_retains_both_signs():
    source = load_dictionary()["be_venous"].sources["miiv"][0]
    frame = pd.DataFrame({"subject_id": [1, 1], "specimen_id": [1, 2],
        "charttime": pd.Timestamp("2150-01-01"), "be_venous": [-10., 10.],
        "valueuom": ["mEq/L", "mEq/L"]})
    specimen = pd.DataFrame({"subject_id": [1, 1], "specimen_id": [1, 2],
                              "itemid": [52033, 52033], "value": ["VEN.", "VEN."]})
    data_source = SimpleNamespace(load_table=lambda *a, **kw: specimen)
    result = _apply_callback(frame, source, "be_venous", data_source=data_source)
    assert result.be_venous.tolist() == [-10., 10.]
