"""Explicit opt-in ventilation measurements keep their source unit gates."""

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.api.extraction import EXTRACT_MODULES, EXTRACT_MODULE_ORDER
from easyicu.concept import _apply_callback
from easyicu.concept.callback_apply import _drop_declared_row_exclusions
from easyicu.concept.catalog import HIDDEN_DICTIONARY_CONCEPTS, VENT_MECHANICS_METADATA
from easyicu.resources import load_dictionary

DICTIONARY = json.loads((Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json").read_text())
EXPECTED = {'co2_tidal': {'unit': 'mL',
               'min': 0,
               'max': 200,
               'description': 'Exhaled CO2 volume per breath',
               'category': 'ventilator',
               'sources': {'aumc': [{'ids': [12270],
                                     'table': 'numericitems',
                                     'sub_var': 'itemid',
                                     'unit_var': 'unit',
                                     'require_regex': {'unit': '(?i)^\\s*ml\\s*$'}},
                                    {'ids': [12352],
                                     'table': 'numericitems',
                                     'sub_var': 'itemid',
                                     'unit_var': 'unit',
                                     'require_regex': {'unit': '(?i)^\\s*ml\\s*$'}}]}},
 'insp_time': {'unit': 's',
               'min': 0,
               'max': 20,
               'description': 'Charted inspiratory time',
               'category': 'ventilator',
               'sources': {'miiv': [{'ids': [224738],
                                     'table': 'chartevents',
                                     'sub_var': 'itemid',
                                     'unit_var': 'valueuom',
                                     'require_regex': {'valueuom': '(?i)^\\s*sec\\s*$'},
                                     'callback': "convert_unit(binary_op(`*`, 1), 's')"}]}},
 'minute_vol_spont': {'unit': 'L/min',
                      'min': 0,
                      'max': 100,
                      'description': 'Spontaneous minute ventilation',
                      'category': 'ventilator',
                      'sources': {'aumc': [{'ids': [12578],
                                            'table': 'numericitems',
                                            'sub_var': 'itemid',
                                            'unit_var': 'unit',
                                            'require_regex': {'unit': '(?i)^\\s*l/min\\s*$'}},
                                           {'ids': [8875],
                                            'table': 'numericitems',
                                            'sub_var': 'itemid',
                                            'unit_var': 'unit',
                                            'require_regex': {'unit': '(?i)^\\s*l/min\\s*$'}},
                                           {'ids': [9663],
                                            'table': 'numericitems',
                                            'sub_var': 'itemid',
                                            'unit_var': 'unit',
                                            'require_regex': {'unit': '(?i)^\\s*l/min\\s*$'}}]}},
 'pc_above_peep_set': {'unit': 'cmH2O',
                       'min': 0,
                       'max': 100,
                       'description': 'Set inspiratory pressure above PEEP',
                       'category': 'ventilator',
                       'sources': {'aumc': [{'ids': [12285],
                                             'table': 'numericitems',
                                             'sub_var': 'itemid',
                                             'unit_var': 'unit',
                                             'require_regex': {'unit': '(?i)^\\s*cmH2O\\s*$'}},
                                            {'ids': [12335],
                                             'table': 'numericitems',
                                             'sub_var': 'itemid',
                                             'unit_var': 'unit',
                                             'require_regex': {'unit': '(?i)^\\s*cmH2O\\s*$'}}],
                                   'sic': [{'ids': [3040],
                                            'table': 'data_float_h',
                                            'sub_var': 'DataID',
                                            'callback': 'convert_unit(binary_op(`*`, '
                                                        "1.0197162129779282), 'cmH2O')"}]}},
 'peep_intrinsic': {'unit': 'cmH2O',
                    'min': 0,
                    'max': 60,
                    'description': 'Measured intrinsic PEEP',
                    'category': 'ventilator',
                    'sources': {'aumc': [{'ids': [8882],
                                          'table': 'numericitems',
                                          'sub_var': 'itemid',
                                          'unit_var': 'unit',
                                          'require_regex': {'unit': '(?i)^\\s*mbar\\s*$'},
                                          'callback': 'convert_unit(binary_op(`*`, 1.0197162129779282), '
                                                      "'cmH2O')"},
                                         {'ids': [9661],
                                          'table': 'numericitems',
                                          'sub_var': 'itemid',
                                          'unit_var': 'unit',
                                          'require_regex': {'unit': '(?i)^\\s*mbar\\s*$'},
                                          'callback': 'convert_unit(binary_op(`*`, 1.0197162129779282), '
                                                      "'cmH2O')"}]}},
 'prone': {'unit': 'boolean',
           'description': 'Recorded prone positioning evidence',
           'category': 'ventilator',
           'class_name': 'lgl_cncpt',
           'aggregate': 'max',
           'sources': {'aumc': [{'ids': [7927],
                                 'table': 'processitems',
                                 'sub_var': 'itemid',
                                 'val_var': 'item',
                                 'index_var': 'start',
                                 'dur_var': 'stop',
                                 'callback': 'transform_fun(set_val(TRUE))'}],
                       'eicu': [{'ids': ['pulmonary|ventilation and oxygenation|prone position'],
                                 'table': 'treatment',
                                 'sub_var': 'treatmentstring',
                                 'val_var': 'treatmentstring',
                                 'index_var': 'treatmentoffset',
                                 'callback': 'transform_fun(set_val(TRUE))'}]}},
 'resp_spont': {'unit': 'breaths/min',
                'min': 0,
                'max': 100,
                'description': 'Spontaneous respiratory rate',
                'category': 'ventilator',
                'sources': {'aumc': [{'ids': [12577],
                                      'table': 'numericitems',
                                      'sub_var': 'itemid',
                                      'unit_var': 'unit',
                                      'require_regex': {'unit': '(?i)^\\s*/min\\s*$'},
                                      'callback': "convert_unit(binary_op(`*`, 1), 'breaths/min')"}],
                            'eicu': [{'ids': ['Spontaneous Rate'],
                                      'table': 'lab',
                                      'sub_var': 'labname',
                                      'unit_var': 'labmeasurenameinterface',
                                      'require_regex': {'labmeasurenameinterface': '(?i)^\\s*Breaths\\s*$'},
                                      'callback': "convert_unit(binary_op(`*`, 1), 'breaths/min')"}],
                            'miiv': [{'ids': [224422],
                                      'table': 'chartevents',
                                      'sub_var': 'itemid',
                                      'unit_var': 'valueuom',
                                      'require_regex': {'valueuom': '(?i)^\\s*bpm\\s*$'},
                                      'callback': "convert_unit(binary_op(`*`, 1), 'breaths/min')"},
                                     {'ids': [224689],
                                      'table': 'chartevents',
                                      'sub_var': 'itemid',
                                      'unit_var': 'valueuom',
                                      'require_regex': {'valueuom': '(?i)^\\s*insp/min\\s*$'},
                                      'callback': "convert_unit(binary_op(`*`, 1), 'breaths/min')"}],
                            'sic': [{'ids': [2280],
                                     'table': 'data_float_h',
                                     'sub_var': 'DataID',
                                     'callback': "convert_unit(binary_op(`*`, 1), 'breaths/min')"}]}},
 'tidal_vol_spont': {'unit': 'mL',
                     'min': 0,
                     'max': 3000,
                     'description': 'Spontaneous tidal volume',
                     'category': 'ventilator',
                     'sources': {'eicu': [{'ids': ['Vt Spontaneous (mL)'],
                                           'table': 'respiratorycharting',
                                           'sub_var': 'respchartvaluelabel',
                                           'val_var': 'respchartvalue'}],
                                 'miiv': [{'ids': [224421],
                                           'table': 'chartevents',
                                           'sub_var': 'itemid',
                                           'unit_var': 'valueuom',
                                           'require_regex': {'valueuom': '(?i)^\\s*mL\\s*$'}},
                                          {'ids': [224686],
                                           'table': 'chartevents',
                                           'sub_var': 'itemid',
                                           'unit_var': 'valueuom',
                                           'require_regex': {'valueuom': '(?i)^\\s*mL\\s*$'}}]}},
 'transpulm_pres_exp': {'unit': 'cmH2O',
                        'min': -100,
                        'max': 100,
                        'description': 'Expiratory-hold transpulmonary pressure',
                        'category': 'ventilator',
                        'sources': {'miiv': [{'ids': [224746],
                                              'table': 'chartevents',
                                              'sub_var': 'itemid',
                                              'unit_var': 'valueuom',
                                              'require_regex': {'valueuom': '(?i)^\\s*cmH2O\\s*$'}}]}},
 'transpulm_pres_insp': {'unit': 'cmH2O',
                         'min': -100,
                         'max': 100,
                         'description': 'Inspiratory-hold transpulmonary pressure',
                         'category': 'ventilator',
                         'sources': {'miiv': [{'ids': [224747],
                                               'table': 'chartevents',
                                               'sub_var': 'itemid',
                                               'unit_var': 'valueuom',
                                               'require_regex': {'valueuom': '(?i)^\\s*cmH2O\\s*$'}}]}},
 'vco2': {'unit': 'mL/min',
          'min': 0,
          'max': 3000,
          'description': 'Exhaled CO2 production per minute',
          'category': 'ventilator',
          'sources': {'aumc': [{'ids': [12350],
                                'table': 'numericitems',
                                'sub_var': 'itemid',
                                'unit_var': 'unit',
                                'require_regex': {'unit': '(?i)^\\s*ml/min\\s*$'}},
                               {'ids': [12432],
                                'table': 'numericitems',
                                'sub_var': 'itemid',
                                'unit_var': 'unit',
                                'require_regex': {'unit': '(?i)^\\s*ml/min\\s*$'}},
                               {'ids': [8080],
                                'table': 'numericitems',
                                'sub_var': 'itemid',
                                'unit_var': 'unit',
                                'require_regex': {'unit': '(?i)^\\s*ml/min\\s*$'}}]}}}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_dictionary_contract(name):
    assert DICTIONARY[name] == EXPECTED[name]
    assert name in EXTRACT_MODULES["vent_mechanics"]
    assert name in HIDDEN_DICTIONARY_CONCEPTS and name in VENT_MECHANICS_METADATA
    assert "vent_mechanics" not in EXTRACT_MODULE_ORDER
    assert all(name not in EXTRACT_MODULES[module] for module in EXTRACT_MODULE_ORDER)


@pytest.mark.parametrize("name,database,item,unit,factor", [
    ("insp_time", "miiv", 224738, "sec", 1.),
    ("pc_above_peep_set", "sic", 3040, None, 1.0197162129779282),
    ("peep_intrinsic", "aumc", 8882, "mbar", 1.0197162129779282),
    ("peep_intrinsic", "aumc", 9661, "mbar", 1.0197162129779282),
    ("resp_spont", "aumc", 12577, "/min", 1.),
    ("resp_spont", "eicu", "Spontaneous Rate", "Breaths", 1.),
    ("resp_spont", "miiv", 224422, "bpm", 1.),
    ("resp_spont", "miiv", 224689, "insp/min", 1.),
    ("resp_spont", "sic", 2280, None, 1.),
])
def test_unit_conversion_and_wrong_units(name, database, item, unit, factor):
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
    assert actual[name].tolist() == pytest.approx([0., 10. * factor])


@pytest.mark.parametrize("database", ["aumc", "eicu"])
def test_prone_callback_marks_observed_events(database):
    source = load_dictionary()["prone"].sources[database][0]
    actual = _apply_callback(pd.DataFrame({"prone": ["recorded", "recorded"]}), source, "prone")
    assert actual.prone.tolist() == [True, True]


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    from easyicu.synthetic_sources import generate_source
    from easyicu.api.concepts import clear_global_loader

    root = tmp_path_factory.mktemp("vent-batch1")
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

    units = {"insp_time": "sec", "resp_spont": {224422: "bpm", 224689: "insp/min"},
             "tidal_vol_spont": "mL", "transpulm_pres_exp": "cmH2O",
             "transpulm_pres_insp": "cmH2O"}
    item = entry["ids"][0]
    unit = units[name][item] if isinstance(units[name], dict) else units[name]
    now = pd.Timestamp("2150-01-01 00:15:00")
    frame = pd.DataFrame({"subject_id": [11, 11], "hadm_id": [21, 21], "stay_id": [1, 1],
                          "itemid": [item, item],
                          "charttime": [now, now + pd.Timedelta(hours=1)],
                          "storetime": [now, now + pd.Timedelta(hours=1)],
                          "valuenum": [10., 10.], "value": ["10", "10"],
                          "valueuom": [unit, "wrong"]})
    source = mimic_synthetic_source(tmp_path, "miiv", chartevents=frame)
    result = mimic_synthetic_load(source, [name])[name].data[name].dropna()
    assert result.tolist() == [10.]


@pytest.mark.parametrize("name", ["prone", "resp_spont", "tidal_vol_spont"])
def test_eicu_synthetic_extraction(tmp_path, name):
    from tests.support.eicu_source import _eicu_source, _hourly

    tables = {
        "treatment": pd.DataFrame({"treatmentoffset": [15], "treatmentstring":
            ["pulmonary|ventilation and oxygenation|prone position"]}),
        "lab": pd.DataFrame({"labresultoffset": [15], "labname": ["Spontaneous Rate"],
            "labresult": [10.], "labmeasurenameinterface": ["Breaths"]}),
        "respiratorycharting": pd.DataFrame({"respchartoffset": [15],
            "respchartvaluelabel": ["Vt Spontaneous (mL)"], "respchartvalue": ["500"]}),
    }
    source = _eicu_source(tmp_path, "eicu", **tables)
    assert list(_hourly(source, name).values()) == [{"prone": True, "resp_spont": 10.,
                                                    "tidal_vol_spont": 500.}[name]]
