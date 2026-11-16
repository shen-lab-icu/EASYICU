"""Opt-in inotropic and vasoactive inputs retain their own molecules and units."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.api.extraction import EXTRACT_MODULES, EXTRACT_MODULE_ORDER
from easyicu.concept import _apply_callback
from easyicu.concept.catalog import HIDDEN_DICTIONARY_CONCEPTS, VIS_INPUT_METADATA
from easyicu.resources import load_dictionary

DICTIONARY = json.loads((Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json").read_text())
EXPECTED = {'angiotensin_ii_rate': {'unit': 'ng/kg/min',
                         'min': 0,
                         'max': 1000,
                         'description': 'Continuous intravenous angiotensin II infusion rate',
                         'category': 'cardiovascular',
                         'sources': {'miiv': [{'ids': [229709],
                                               'table': 'inputevents',
                                               'sub_var': 'itemid',
                                               'stop_var': 'endtime',
                                               'callback': 'miiv_angiotensin_rate',
                                               'extra_vars': ['ordercategorydescription',
                                                              'rateuom',
                                                              'statusdescription'],
                                               'status_var': 'statusdescription',
                                               'excluded_statuses': ['Rewritten',
                                                                     'Cancelled',
                                                                     'Canceled',
                                                                     'Bolus'],
                                               'require_regex': {'rateuom': '(?i)^\\s*mcg/kg/min\\s*$',
                                                                 'ordercategorydescription': '^Continuous '
                                                                                             'Med$'}},
                                              {'ids': [229764],
                                               'table': 'inputevents',
                                               'sub_var': 'itemid',
                                               'stop_var': 'endtime',
                                               'callback': 'miiv_angiotensin_rate',
                                               'extra_vars': ['ordercategorydescription',
                                                              'rateuom',
                                                              'statusdescription'],
                                               'status_var': 'statusdescription',
                                               'excluded_statuses': ['Rewritten',
                                                                     'Cancelled',
                                                                     'Canceled',
                                                                     'Bolus'],
                                               'require_regex': {'rateuom': '(?i)^\\s*ng/kg/min\\s*$',
                                                                 'ordercategorydescription': '^Continuous '
                                                                                             'Med$'}}]}},
 'levosimendan': {'unit': 'boolean',
                  'description': 'Levosimendan administration exposure',
                  'category': 'cardiovascular',
                  'class_name': 'lgl_cncpt',
                  'aggregate': 'max',
                  'sources': {'hirid': [{'ids': [1000606],
                                         'table': 'pharma',
                                         'sub_var': 'pharmaid',
                                         'callback': 'transform_fun(comp_na(`>`, 0.000001))',
                                         'extra_vars': ['route'],
                                         'require_regex': {'route': '(?i)^\\s*(?:[ci]v(?:-(?:inf|inj))?|i\\.?v\\.?|intravenous)\\s*$'}}],
                              'sic': [{'ids': [1548],
                                       'table': 'medication',
                                       'sub_var': 'DrugID',
                                       'val_var': 'Amount',
                                       'callback': 'sicdb_exposure'}]}},
 'terlipressin': {'unit': 'boolean',
                  'description': 'Terlipressin administration exposure',
                  'category': 'cardiovascular',
                  'class_name': 'lgl_cncpt',
                  'aggregate': 'max',
                  'sources': {'aumc': [{'ids': [12467],
                                        'table': 'drugitems',
                                        'sub_var': 'itemid',
                                        'val_var': 'dose',
                                        'callback': 'aumc_positive_dose',
                                        'index_var': 'start',
                                        'dur_var': 'stop',
                                        'extra_vars': ['ordercategory'],
                                        'exclude_regex': {'ordercategory': '(?i)^\\s*niet iv'}}],
                              'hirid': [{'ids': [1000793],
                                         'table': 'pharma',
                                         'sub_var': 'pharmaid',
                                         'callback': 'transform_fun(comp_na(`>`, 0.000001))',
                                         'extra_vars': ['route'],
                                         'require_regex': {'route': '(?i)^\\s*(?:[ci]v(?:-(?:inf|inj))?|i\\.?v\\.?|intravenous)\\s*$'}}],
                              'sic': [{'ids': [1483],
                                       'table': 'medication',
                                       'sub_var': 'DrugID',
                                       'val_var': 'Amount',
                                       'callback': 'sicdb_exposure'}]}}}


EXPECTED["angiotensin_ii_rate"]["_hourly_rule"] = "time_weighted_mean_active_rate"


@pytest.mark.parametrize("name", list(EXPECTED))
def test_dictionary_contract(name):
    assert DICTIONARY[name] == EXPECTED[name]
    assert name in EXTRACT_MODULES["vis_inputs"]
    assert name in HIDDEN_DICTIONARY_CONCEPTS and name in VIS_INPUT_METADATA
    assert "vis_inputs" not in EXTRACT_MODULE_ORDER
    assert all(name not in EXTRACT_MODULES[module] for module in EXTRACT_MODULE_ORDER)


@pytest.mark.parametrize("item,unit,rate", [(229709, "mcg/kg/min", .01), (229764, "ng/kg/min", 10.)])
def test_angiotensin_units_exact_stop_and_rejections(item, unit, rate):
    source = next(s for s in load_dictionary()["angiotensin_ii_rate"].sources["miiv"] if s.ids == [item])
    origin = pd.Timestamp("2150-01-01 00:15")
    starts = [origin + pd.Timedelta(minutes=30)] * 6
    ends = [origin + pd.Timedelta(hours=2)] * 5 + [starts[-1]]
    frame = pd.DataFrame({"stay_id": [1] * 6, "starttime": starts, "endtime": ends,
        "angiotensin_ii_rate": [rate, rate, rate, rate, 0., rate],
        "rateuom": [unit, "wrong", unit, unit, unit, unit],
        "ordercategorydescription": ["Continuous Med", "Continuous Med", "Bolus",
                                     "Continuous Med", "Continuous Med", "Continuous Med"],
        "statusdescription": ["FinishedRunning", "FinishedRunning", "FinishedRunning", "Bolus",
                              "FinishedRunning", "FinishedRunning"]})
    stays = pd.DataFrame({"stay_id": [1], "intime": [origin]})
    data_source = SimpleNamespace(load_table=lambda *a, **kw: stays)
    out = _apply_callback(frame, source, "angiotensin_ii_rate", data_source=data_source)
    assert out.angiotensin_ii_rate.tolist() == pytest.approx([10., 10.])
    assert out.starttime.tolist() == [origin, origin + pd.Timedelta(hours=1)]
    assert out.rateuom.tolist() == ["ng/kg/min", "ng/kg/min"]


def test_angiotensin_units_must_match_each_specific_item():
    source = load_dictionary()["angiotensin_ii_rate"].sources["miiv"][0]
    f = pd.DataFrame({"angiotensin_ii_rate": [10.], "rateuom": ["ng/kg/min"],
                      "ordercategorydescription": ["Continuous Med"], "statusdescription": ["Running"]})
    assert _apply_callback(f, source, "angiotensin_ii_rate").empty


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    from easyicu.synthetic_sources import generate_source
    from easyicu.api.concepts import clear_global_loader

    root = tmp_path_factory.mktemp("vis-batch1")
    for database in ("aumc", "hirid", "sic"):
        generate_source(root / database, database, stays=12)
    yield root
    clear_global_loader()


@pytest.mark.parametrize("name,database", [(name, db) for name, d in EXPECTED.items()
    for db in d["sources"] if db in {"aumc", "hirid", "sic"}])
def test_synthetic_extraction(sources, name, database):
    from easyicu import load_concepts

    actual = load_concepts([name], database=database, data_path=sources / database, merge=False,
                           verbose=False, concept_workers=1, parallel_workers=1)
    frame = actual if isinstance(actual, pd.DataFrame) else actual[name]
    assert not frame.empty and frame[name].notna().any()


@pytest.mark.parametrize("name,item,unit,rate", [("angiotensin_ii_rate", 229709, "mcg/kg/min", .01),
                                                ("angiotensin_ii_rate", 229764, "ng/kg/min", 10.)])
def test_miiv_source_extraction(tmp_path, name, item, unit, rate):
    from tests.support.mimic_synthetic_source import mimic_synthetic_load, mimic_synthetic_source

    origin = pd.Timestamp("2150-01-01")
    frame = pd.DataFrame({"subject_id": [11], "hadm_id": [21], "stay_id": [1],
        "itemid": [item], "orderid": [1], "linkorderid": [1],
        "starttime": [origin + pd.Timedelta(minutes=30)], "endtime": [origin + pd.Timedelta(hours=2)],
        "rate": [rate], "rateuom": [unit], "amount": [1.], "amountuom": ["mg"],
        "storetime": [origin + pd.Timedelta(minutes=30)], "ordercategorydescription": ["Continuous Med"],
        "statusdescription": ["FinishedRunning"]})
    source = mimic_synthetic_source(tmp_path, "miiv", inputevents=frame)
    values = mimic_synthetic_load(source, [name])[name].data[name].dropna()
    assert values.tolist() == pytest.approx([10., 10.])


@pytest.mark.parametrize("name", ["levosimendan", "terlipressin"])
def test_hirid_route_and_positive_threshold(name):
    source = load_dictionary()[name].sources["hirid"][0]
    frame = pd.DataFrame({name: [0., 0.000001, 0.000002, 1.], "route": ["i.v.", "i.v.", "i.v.", "po"]})
    assert _apply_callback(frame, source, name)[name].tolist() == [False, False, True]


@pytest.mark.parametrize("name", ["levosimendan", "terlipressin"])
def test_sic_positive_amount_and_missing(name):
    source = load_dictionary()[name].sources["sic"][0]
    frame = pd.DataFrame({name: [0., 1., -1., None]})
    values = _apply_callback(frame, source, name)[name]
    assert values.iloc[:2].tolist() == [False, True]
    assert values.iloc[2:].isna().all()


def test_terlipressin_positive_iv_dose_only():
    source = load_dictionary()["terlipressin"].sources["aumc"][0]
    frame = pd.DataFrame({"terlipressin": [0., 1., 1.],
                          "ordercategory": ["Infuus", "Infuus", "Niet IV"]})
    assert _apply_callback(frame, source, "terlipressin").terlipressin.tolist() == [True]


def test_angiotensin_batch_invariance():
    source = load_dictionary()["angiotensin_ii_rate"].sources["miiv"][0]
    origins = pd.to_datetime(["2150-01-01 00:15", "2150-01-01 00:45"])
    starts = origins + pd.Timedelta(minutes=30)
    frame = pd.DataFrame({"stay_id": [1, 2], "starttime": starts,
        "endtime": origins + pd.Timedelta(hours=2), "angiotensin_ii_rate": [.01, .01],
        "rateuom": ["mcg/kg/min"] * 2, "ordercategorydescription": ["Continuous Med"] * 2,
        "statusdescription": ["FinishedRunning"] * 2})
    stays = pd.DataFrame({"stay_id": [1, 2], "intime": origins})
    data_source = SimpleNamespace(load_table=lambda *a, **kw: stays)
    whole = _apply_callback(frame, source, "angiotensin_ii_rate", data_source=data_source)
    chunks = pd.concat([_apply_callback(frame.iloc[[i]], source, "angiotensin_ii_rate",
                                        data_source=data_source) for i in range(len(frame))])
    pd.testing.assert_frame_equal(whole.reset_index(drop=True), chunks.reset_index(drop=True))
