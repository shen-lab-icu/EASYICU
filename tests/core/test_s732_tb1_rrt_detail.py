"""Separate charted and prescribed RRT measurements and coded modality."""

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.api.extraction import EXTRACT_MODULES, EXTRACT_MODULE_ORDER
from easyicu.concept import _apply_callback
from easyicu.concept.callback_apply import _drop_declared_row_exclusions
from easyicu.concept.catalog import HIDDEN_DICTIONARY_CONCEPTS, RRT_DETAIL_METADATA
from easyicu.resources import load_dictionary

DICTIONARY = json.loads((Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json").read_text())
EXPECTED = {'crrt_blood_flow': {'unit': 'mL/min',
                     'min': 0,
                     'max': 1000,
                     'description': 'Charted CRRT blood flow',
                     'category': 'renal',
                     'sources': {'aumc': [{'ids': [10736],
                                           'table': 'numericitems',
                                           'sub_var': 'itemid',
                                           'unit_var': 'unit',
                                           'require_regex': {'unit': '(?i)^\\s*ml/min\\s*$'}},
                                          {'ids': [12460],
                                           'table': 'numericitems',
                                           'sub_var': 'itemid',
                                           'unit_var': 'unit',
                                           'require_regex': {'unit': '(?i)^\\s*ml/min\\s*$'}}],
                                 'miiv': [{'ids': [224144],
                                           'table': 'chartevents',
                                           'sub_var': 'itemid',
                                           'unit_var': 'valueuom',
                                           'require_regex': {'valueuom': '(?i)^\\s*ml/min\\s*$'}}],
                                 'sic': [{'ids': [723], 'table': 'data_float_h', 'sub_var': 'DataID'}]}},
 'crrt_blood_flow_set': {'unit': 'mL/min',
                         'min': 0,
                         'max': 1000,
                         'description': 'Prescribed CRRT blood flow',
                         'category': 'renal',
                         'sources': {'aumc': [{'ids': [12444],
                                               'table': 'numericitems',
                                               'sub_var': 'itemid',
                                               'unit_var': 'unit',
                                               'require_regex': {'unit': '(?i)^\\s*ml/min\\s*$'}},
                                              {'ids': [14850],
                                               'table': 'numericitems',
                                               'sub_var': 'itemid',
                                               'unit_var': 'unit',
                                               'require_regex': {'unit': '(?i)^\\s*ml/min\\s*$'}},
                                              {'ids': [20538],
                                               'table': 'numericitems',
                                               'sub_var': 'itemid',
                                               'unit_var': 'unit',
                                               'require_regex': {'unit': '(?i)^\\s*ml/min\\s*$'}}]}},
 'crrt_citrate_flow': {'unit': 'mL/h',
                       'min': 0,
                       'max': 3000,
                       'description': 'Charted CRRT citrate solution flow',
                       'category': 'renal',
                       'sources': {'miiv': [{'ids': [228004],
                                             'table': 'chartevents',
                                             'sub_var': 'itemid',
                                             'unit_var': 'valueuom',
                                             'require_regex': {'valueuom': '(?i)^\\s*ml/hr\\s*$'},
                                             'callback': "convert_unit(binary_op(`*`, 1), 'mL/h')"}]}},
 'crrt_dialysate_flow': {'unit': 'mL/h',
                         'min': 0,
                         'max': 15000,
                         'description': 'Charted CRRT dialysate flow',
                         'category': 'renal',
                         'sources': {'miiv': [{'ids': [224154],
                                               'table': 'chartevents',
                                               'sub_var': 'itemid',
                                               'unit_var': 'valueuom',
                                               'require_regex': {'valueuom': '(?i)^\\s*ml/hr\\s*$'},
                                               'callback': "convert_unit(binary_op(`*`, 1), 'mL/h')"}],
                                     'sic': [{'ids': [730],
                                              'table': 'data_float_h',
                                              'sub_var': 'DataID'}]}},
 'crrt_replacement_flow': {'unit': 'mL/h',
                           'min': 0,
                           'max': 15000,
                           'description': 'Charted total CRRT replacement flow',
                           'category': 'renal',
                           'sources': {'aumc': [{'ids': [10735],
                                                 'table': 'numericitems',
                                                 'sub_var': 'itemid',
                                                 'unit_var': 'unit',
                                                 'require_regex': {'unit': '(?i)^\\s*ml/uur\\s*$'},
                                                 'callback': "convert_unit(binary_op(`*`, 1), 'mL/h')"},
                                                {'ids': [12461],
                                                 'table': 'numericitems',
                                                 'sub_var': 'itemid',
                                                 'unit_var': 'unit',
                                                 'require_regex': {'unit': '(?i)^\\s*ml/uur\\s*$'},
                                                 'callback': "convert_unit(binary_op(`*`, 1), 'mL/h')"}],
                                       'miiv': [{'ids': [224153],
                                                 'table': 'chartevents',
                                                 'sub_var': 'itemid',
                                                 'unit_var': 'valueuom',
                                                 'require_regex': {'valueuom': '(?i)^\\s*ml/hr\\s*$'},
                                                 'callback': 'convert_unit(binary_op(`*`, 1), '
                                                             "'mL/h')"}]}},
 'crrt_replacement_flow_set': {'unit': 'mL/h',
                               'min': 0,
                               'max': 15000,
                               'description': 'Prescribed total CRRT replacement flow',
                               'category': 'renal',
                               'sources': {'aumc': [{'ids': [12452],
                                                     'table': 'numericitems',
                                                     'sub_var': 'itemid',
                                                     'unit_var': 'unit',
                                                     'require_regex': {'unit': '(?i)^\\s*ml/uur\\s*$'},
                                                     'callback': 'convert_unit(binary_op(`*`, 1), '
                                                                 "'mL/h')"}]}},
 'crrt_replacement_post_flow': {'unit': 'mL/h',
                                'min': 0,
                                'max': 15000,
                                'description': 'Charted postfilter CRRT replacement flow',
                                'category': 'renal',
                                'sources': {'miiv': [{'ids': [228006],
                                                      'table': 'chartevents',
                                                      'sub_var': 'itemid',
                                                      'unit_var': 'valueuom',
                                                      'require_regex': {'valueuom': '(?i)^\\s*ml/hr\\s*$'},
                                                      'callback': 'convert_unit(binary_op(`*`, 1), '
                                                                  "'mL/h')"}],
                                            'sic': [{'ids': [732],
                                                     'table': 'data_float_h',
                                                     'sub_var': 'DataID'}]}},
 'crrt_replacement_post_flow_set': {'unit': 'mL/h',
                                    'min': 0,
                                    'max': 15000,
                                    'description': 'Prescribed postfilter CRRT replacement flow',
                                    'category': 'renal',
                                    'sources': {'aumc': [{'ids': [20716],
                                                          'table': 'numericitems',
                                                          'sub_var': 'itemid',
                                                          'unit_var': 'unit',
                                                          'require_regex': {'unit': '(?i)^\\s*ml/uur\\s*$'},
                                                          'callback': 'convert_unit(binary_op(`*`, 1), '
                                                                      "'mL/h')"}]}},
 'crrt_replacement_pre_flow': {'unit': 'mL/h',
                               'min': 0,
                               'max': 15000,
                               'description': 'Charted prefilter CRRT replacement flow',
                               'category': 'renal',
                               'sources': {'miiv': [{'ids': [228005],
                                                     'table': 'chartevents',
                                                     'sub_var': 'itemid',
                                                     'unit_var': 'valueuom',
                                                     'require_regex': {'valueuom': '(?i)^\\s*ml/hr\\s*$'},
                                                     'callback': 'convert_unit(binary_op(`*`, 1), '
                                                                 "'mL/h')"}],
                                           'sic': [{'ids': [731],
                                                    'table': 'data_float_h',
                                                    'sub_var': 'DataID'}]}},
 'crrt_replacement_pre_flow_set': {'unit': 'mL/h',
                                   'min': 0,
                                   'max': 15000,
                                   'description': 'Prescribed prefilter CRRT replacement flow',
                                   'category': 'renal',
                                   'sources': {'aumc': [{'ids': [20710],
                                                         'table': 'numericitems',
                                                         'sub_var': 'itemid',
                                                         'unit_var': 'unit',
                                                         'require_regex': {'unit': '(?i)^\\s*ml/uur\\s*$'},
                                                         'callback': 'convert_unit(binary_op(`*`, 1), '
                                                                     "'mL/h')"}]}}}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_dictionary_contract(name):
    assert DICTIONARY[name] == EXPECTED[name]
    assert name in EXTRACT_MODULES["rrt_detail"]
    assert name in HIDDEN_DICTIONARY_CONCEPTS and name in RRT_DETAIL_METADATA
    assert "rrt_detail" not in EXTRACT_MODULE_ORDER
    assert all(name not in EXTRACT_MODULES[module] for module in EXTRACT_MODULE_ORDER)


@pytest.mark.parametrize("name,database,unit", [
    ("crrt_citrate_flow", "miiv", "ml/hr"),
    ("crrt_dialysate_flow", "miiv", "ml/hr"),
    ("crrt_replacement_flow", "miiv", "ml/hr"),
    ("crrt_replacement_flow", "aumc", "ml/uur"),
    ("crrt_replacement_flow_set", "aumc", "ml/uur"),
    ("crrt_replacement_post_flow", "miiv", "ml/hr"),
    ("crrt_replacement_post_flow_set", "aumc", "ml/uur"),
    ("crrt_replacement_pre_flow", "miiv", "ml/hr"),
    ("crrt_replacement_pre_flow_set", "aumc", "ml/uur"),
])
def test_all_unit_conversions_and_gates(name, database, unit):
    for source in load_dictionary()[name].sources[database]:
        column = source.unit_var
        frame = pd.DataFrame({name: [0., 100., 100.], column: [unit, unit, "wrong"]})
        filtered = _drop_declared_row_exclusions(frame, source, name)
        assert len(filtered) == 2
        converted = _apply_callback(filtered, source, name, unit_column=column)
        assert converted[name].tolist() == [0., 100.]
        assert converted[column].tolist() == ["mL/h", "mL/h"]


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    from easyicu.synthetic_sources import generate_source
    from easyicu.api.concepts import clear_global_loader

    root = tmp_path_factory.mktemp("rrt-batch1")
    for database in ("aumc", "sic"):
        generate_source(root / database, database, stays=12)
    yield root
    clear_global_loader()


@pytest.mark.parametrize("name,database", [(name, db) for name, definition in EXPECTED.items()
                                          for db in definition["sources"] if db in {"aumc", "sic"}])
def test_synthetic_extraction(sources, name, database):
    from easyicu import load_concepts

    actual = load_concepts([name], database=database, data_path=sources / database, merge=False,
                           verbose=False, concept_workers=1, parallel_workers=1)
    frame = actual if isinstance(actual, pd.DataFrame) else actual[name]
    assert not frame.empty and frame[name].notna().any()


@pytest.mark.parametrize("name", [name for name, d in EXPECTED.items() if "miiv" in d["sources"]])
def test_miiv_synthetic_source_extraction(tmp_path, name):
    from tests.support.mimic_synthetic_source import mimic_synthetic_load, mimic_synthetic_source

    entry = EXPECTED[name]["sources"]["miiv"][0]
    categorical = name == "rrt_modality"
    value = "CVVH" if categorical else "100"
    unit = "ml/min" if name == "crrt_blood_flow" else "ml/hr"
    now = pd.Timestamp("2150-01-01 00:15:00")
    frame = pd.DataFrame({"subject_id": [11], "hadm_id": [21], "stay_id": [1],
                          "itemid": entry["ids"], "charttime": [now], "storetime": [now],
                          "valuenum": [None if categorical else 100.], "value": [value],
                          "valueuom": [unit]})
    if categorical:
        other = frame.copy()
        other["value"] = "unmapped"
        other["charttime"] += pd.Timedelta(hours=1)
        frame = pd.concat([frame, other], ignore_index=True)
    source = mimic_synthetic_source(tmp_path, "miiv", chartevents=frame)
    result = mimic_synthetic_load(source, [name])[name].data[name].dropna()
    assert result.tolist() == (["cvvh"] if categorical else [100.])


def test_flow_and_setting_sources_remain_distinct():
    for stem in ("crrt_blood_flow", "crrt_replacement_flow",
                 "crrt_replacement_post_flow", "crrt_replacement_pre_flow"):
        actual, prescribed = EXPECTED[stem]["sources"], EXPECTED[stem + "_set"]["sources"]
        for db in set(actual) & set(prescribed):
            assert not {str(s["ids"]) for s in actual[db]} & {str(s["ids"]) for s in prescribed[db]}
