"""Opt-in haemodynamics and ECMO measurements retain distinct units."""

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.api.extraction import EXTRACT_MODULES, EXTRACT_MODULE_ORDER
from easyicu.concept import _apply_callback
from easyicu.concept.callback_apply import _drop_declared_row_exclusions
from easyicu.concept.catalog import HIDDEN_DICTIONARY_CONCEPTS, HAEMO_ADVANCED_METADATA
from easyicu.resources import load_dictionary

DICTIONARY = json.loads((Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json").read_text())
EXPECTED = {'ci': {'unit': 'L/min/m2',
        'min': 0,
        'max': 20,
        'description': 'Cardiac index',
        'category': 'cardiovascular',
        'sources': {'aumc': [{'ids': [13150],
                              'table': 'numericitems',
                              'sub_var': 'itemid',
                              'unit_var': 'unit',
                              'require_regex': {'unit': '(?i)^\\s*l/min/m2\\s*$'}}],
                    'miiv': [{'ids': [228177],
                              'table': 'chartevents',
                              'sub_var': 'itemid',
                              'unit_var': 'valueuom',
                              'require_regex': {'valueuom': '(?i)^\\s*L/min/m2\\s*$'}},
                             {'ids': [228368],
                              'table': 'chartevents',
                              'sub_var': 'itemid',
                              'unit_var': 'valueuom',
                              'require_regex': {'valueuom': '(?i)^\\s*L/min/m2\\s*$'}}]}},
 'ecmo_blood_flow': {'unit': 'L/min',
                     'min': 0,
                     'max': 20,
                     'description': 'ECMO blood flow',
                     'category': 'ventilator',
                     'sources': {'aumc': [{'ids': [20648],
                                           'table': 'numericitems',
                                           'sub_var': 'itemid',
                                           'unit_var': 'unit',
                                           'require_regex': {'unit': '(?i)^\\s*l/min\\s*$'}}],
                                 'miiv': [{'ids': [229270],
                                           'table': 'chartevents',
                                           'sub_var': 'itemid',
                                           'unit_var': 'valueuom',
                                           'require_regex': {'valueuom': '(?i)^\\s*L/min\\s*$'}}]}},
 'ecmo_fio2': {'unit': '%',
               'min': 0,
               'max': 100,
               'description': 'ECMO oxygenator oxygen concentration',
               'category': 'ventilator',
               'sources': {'aumc': [{'ids': [20656],
                                     'table': 'numericitems',
                                     'sub_var': 'itemid',
                                     'unit_var': 'unit',
                                     'require_regex': {'unit': '(?i)^\\s*Geen\\s*$'},
                                     'callback': "convert_unit(binary_op(`*`, 1), '%')"}],
                           'miiv': [{'ids': [229280],
                                     'table': 'chartevents',
                                     'sub_var': 'itemid',
                                     'unit_var': 'valueuom',
                                     'require_regex': {'valueuom': '(?i)^\\s*%\\s*$'}}],
                           'sic': [{'ids': [2025], 'table': 'data_float_h', 'sub_var': 'DataID'}]}},
 'ecmo_pump_speed': {'unit': 'rpm',
                     'min': 0,
                     'max': 15000,
                     'description': 'ECMO pump speed',
                     'category': 'ventilator',
                     'sources': {'aumc': [{'ids': [20649],
                                           'table': 'numericitems',
                                           'sub_var': 'itemid',
                                           'unit_var': 'unit',
                                           'require_regex': {'unit': '(?i)^\\s*/min\\s*$'},
                                           'callback': "convert_unit(binary_op(`*`, 1), 'rpm')"}],
                                 'miiv': [{'ids': [229277],
                                           'table': 'chartevents',
                                           'sub_var': 'itemid',
                                           'unit_var': 'valueuom',
                                           'require_regex': {'valueuom': '(?i)^\\s*RPM\\s*$'}}],
                                 'sic': [{'ids': [2023],
                                          'table': 'data_float_h',
                                          'sub_var': 'DataID',
                                          'callback': "convert_unit(binary_op(`*`, 1), 'rpm')"}]}},
 'ecmo_sweep_flow': {'unit': 'L/min',
                     'min': 0,
                     'max': 30,
                     'description': 'ECMO oxygenator sweep gas flow',
                     'category': 'ventilator',
                     'sources': {'aumc': [{'ids': [20657],
                                           'table': 'numericitems',
                                           'sub_var': 'itemid',
                                           'unit_var': 'unit',
                                           'require_regex': {'unit': '(?i)^\\s*l/min\\s*$'}}],
                                 'miiv': [{'ids': [228192],
                                           'table': 'chartevents',
                                           'sub_var': 'itemid',
                                           'unit_var': 'valueuom',
                                           'require_regex': {'valueuom': '(?i)^\\s*L/min\\s*$'}}]}},
 'evlwi': {'unit': 'mL/kg',
           'min': 0,
           'max': 100,
           'description': 'Extravascular lung water index',
           'category': 'cardiovascular',
           'sources': {'aumc': [{'ids': [13153],
                                 'table': 'numericitems',
                                 'sub_var': 'itemid',
                                 'unit_var': 'unit',
                                 'require_regex': {'unit': '(?i)^\\s*ml/kg\\s*$'}}],
                       'miiv': [{'ids': [228179],
                                 'table': 'chartevents',
                                 'sub_var': 'itemid',
                                 'unit_var': 'valueuom',
                                 'require_regex': {'valueuom': '(?i)^\\s*ml/kg\\s*$'}}]}},
 'gedvi': {'unit': 'mL/m2',
           'min': 0,
           'max': 3000,
           'description': 'Global end-diastolic volume index',
           'category': 'cardiovascular',
           'sources': {'aumc': [{'ids': [13152],
                                 'table': 'numericitems',
                                 'sub_var': 'itemid',
                                 'unit_var': 'unit',
                                 'require_regex': {'unit': '(?i)^\\s*ml/m2\\s*$'}}],
                       'miiv': [{'ids': [228180],
                                 'table': 'chartevents',
                                 'sub_var': 'itemid',
                                 'unit_var': 'valueuom',
                                 'require_regex': {'valueuom': '(?i)^\\s*mL/m2\\s*$'}}]}},
 'gef': {'unit': '%',
         'min': 0,
         'max': 100,
         'description': 'Global ejection fraction',
         'category': 'cardiovascular',
         'sources': {'aumc': [{'ids': [14064],
                               'table': 'numericitems',
                               'sub_var': 'itemid',
                               'unit_var': 'unit',
                               'require_regex': {'unit': '(?i)^\\s*Geen\\s*$'},
                               'callback': "convert_unit(binary_op(`*`, 1), '%')"}]}},
 'ppv': {'unit': '%',
         'min': 0,
         'max': 100,
         'description': 'Pulse pressure variation',
         'category': 'cardiovascular',
         'sources': {'aumc': [{'ids': [13155],
                               'table': 'numericitems',
                               'sub_var': 'itemid',
                               'unit_var': 'unit',
                               'require_regex': {'unit': '(?i)^\\s*Geen\\s*$'},
                               'callback': "convert_unit(binary_op(`*`, 1), '%')"}]}},
 'svr': {'unit': 'dyn.s/cm5',
         'min': 0,
         'max': 10000,
         'description': 'Systemic vascular resistance, unindexed',
         'category': 'cardiovascular',
         'sources': {'aumc': [{'ids': [14074],
                               'table': 'numericitems',
                               'sub_var': 'itemid',
                               'unit_var': 'unit',
                               'require_regex': {'unit': '(?i)^\\s*dynes\\.sec\\.cm\\-5\\s*$'},
                               'callback': "convert_unit(binary_op(`*`, 1), 'dyn.s/cm5')"}]}},
 'svv': {'unit': '%',
         'min': 0,
         'max': 100,
         'description': 'Stroke volume variation',
         'category': 'cardiovascular',
         'sources': {'aumc': [{'ids': [13154],
                               'table': 'numericitems',
                               'sub_var': 'itemid',
                               'unit_var': 'unit',
                               'require_regex': {'unit': '(?i)^\\s*Geen\\s*$'},
                               'callback': "convert_unit(binary_op(`*`, 1), '%')"}],
                     'miiv': [{'ids': [228184],
                               'table': 'chartevents',
                               'sub_var': 'itemid',
                               'unit_var': 'valueuom',
                               'require_regex': {'valueuom': '(?i)^\\s*%\\s*$'}}]}}}

UNITS = {
    "ci": "L/min/m2", "ecmo_blood_flow": "L/min", "ecmo_fio2": "%",
    "ecmo_pump_speed": "RPM", "ecmo_sweep_flow": "L/min", "evlwi": "mL/kg",
    "gedvi": "mL/m2", "gef": "Geen", "ppv": "Geen",
    "svr": "dynes.sec.cm-5", "svv": "%",
}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_dictionary_contract(name):
    assert DICTIONARY[name] == EXPECTED[name]
    assert name in EXTRACT_MODULES["haemo_advanced"]
    assert name in HIDDEN_DICTIONARY_CONCEPTS and name in HAEMO_ADVANCED_METADATA
    assert "haemo_advanced" not in EXTRACT_MODULE_ORDER
    assert all(name not in EXTRACT_MODULES[module] for module in EXTRACT_MODULE_ORDER)


@pytest.mark.parametrize("name,database", [(name, db) for name, d in EXPECTED.items()
                                          for db in d["sources"]])
def test_unit_gates_and_conversions(name, database):
    for entry, source in zip(EXPECTED[name]["sources"][database], load_dictionary()[name].sources[database]):
        column = entry.get("unit_var")
        unit = UNITS[name]
        if database == "aumc" and name in {"ecmo_fio2", "svv"}:
            unit = "Geen"
        if database == "aumc" and name == "ecmo_pump_speed":
            unit = "/min"
        frame = pd.DataFrame({name: [0., 10., 10.]})
        if column:
            frame[column] = [unit, unit, "wrong"]
            frame = _drop_declared_row_exclusions(frame, source, name)
            assert len(frame) == 2
        else:
            frame = frame.iloc[:2]
        actual = _apply_callback(frame, source, name, unit_column=column)
        assert actual[name].tolist() == [0., 10.]


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    from easyicu.synthetic_sources import generate_source
    from easyicu.api.concepts import clear_global_loader

    root = tmp_path_factory.mktemp("haemo-batch1")
    for database in ("aumc", "sic"):
        generate_source(root / database, database, stays=12)
    yield root
    clear_global_loader()


@pytest.mark.parametrize("name,database", [(name, db) for name, d in EXPECTED.items()
                                          for db in d["sources"] if db in {"aumc", "sic"}])
def test_synthetic_extraction(sources, name, database):
    from easyicu import load_concepts

    actual = load_concepts([name], database=database, data_path=sources / database, merge=False,
                           verbose=False, concept_workers=1, parallel_workers=1)
    frame = actual if isinstance(actual, pd.DataFrame) else actual[name]
    assert not frame.empty and frame[name].notna().any()


@pytest.mark.parametrize("name,entry", [(name, entry) for name, d in EXPECTED.items()
                                       for entry in d["sources"].get("miiv", [])])
def test_miiv_source_extraction_and_wrong_unit(tmp_path, name, entry):
    from tests.support.mimic_synthetic_source import mimic_synthetic_load, mimic_synthetic_source

    now = pd.Timestamp("2150-01-01 00:15:00")
    frame = pd.DataFrame({"subject_id": [11, 11], "hadm_id": [21, 21], "stay_id": [1, 1],
                          "itemid": entry["ids"] * 2,
                          "charttime": [now, now + pd.Timedelta(hours=1)],
                          "storetime": [now, now + pd.Timedelta(hours=1)],
                          "valuenum": [10., 10.], "value": ["10", "10"],
                          "valueuom": [UNITS[name], "wrong"]})
    source = mimic_synthetic_source(tmp_path, "miiv", chartevents=frame)
    result = mimic_synthetic_load(source, [name])[name].data[name].dropna()
    assert result.tolist() == [10.]
