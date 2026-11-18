"""Reviewed nephrotoxin exposure sources and co-exposure dependencies."""

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.api.extraction import EXTRACT_MODULES, EXTRACT_MODULE_ORDER
from easyicu.concept import _apply_callback
from easyicu.concept.catalog import HIDDEN_DICTIONARY_CONCEPTS, DIURETIC_NEPHROTOXIN_METADATA
from easyicu.resources import load_dictionary

DICTIONARY = json.loads((Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json").read_text())
EXPECTED = {'furosemide_active': {'class_name': 'lgl_cncpt',
                       'description': 'furosemide given in the hour: an administration at that hour or '
                                      'a running infusion covering it (any route)',
                       'category': 'medications',
                       'aggregate': 'max',
                       'sources': {'miiv': [{'ids': [221794, 228340],
                                             'table': 'inputevents',
                                             'sub_var': 'itemid',
                                             'callback': 'mimic_positive_dose',
                                             'val_var': 'rate',
                                             'extra_vars': ['amount', 'statusdescription'],
                                             'status_var': 'statusdescription',
                                             'excluded_statuses': ['Rewritten'],
                                             'index_var': 'starttime',
                                             'dur_var': 'endtime'}],
                                   'aumc': [{'ids': [7244],
                                             'table': 'drugitems',
                                             'sub_var': 'itemid',
                                             'callback': 'aumc_positive_dose',
                                             'val_var': 'dose',
                                             'index_var': 'start',
                                             'dur_var': 'stop'}],
                                   'hirid': [{'ids': [4, 482, 1000232, 1000747],
                                              'table': 'pharma',
                                              'sub_var': 'pharmaid',
                                              'callback': 'transform_fun(comp_na(`>`, 0.000001))'}],
                                   'sic': [{'ids': [1417, 1654],
                                            'table': 'medication',
                                            'sub_var': 'DrugID',
                                            'val_var': 'Amount',
                                            'callback': 'sicdb_active_exposure',
                                            'extra_vars': ['OffsetDrugEnd']}],
                                   'eicu': [{'table': 'infusiondrug',
                                             'sub_var': 'drugname',
                                             'val_var': 'drugrate',
                                             'callback': 'eicu_positive_infusion',
                                             'extra_vars': ['drugrate', 'drugname'],
                                             'ids': ['Furosemide',
                                                     'Furosemide ()',
                                                     'Furosemide (Lasix) MAX 500 mg Sodium Chloride '
                                                     '0.9% 100 ml (mg/hr)',
                                                     'Furosemide (Lasix) STD 250 mg Sodium Chloride '
                                                     '0.9% 250 ml (mg/hr)',
                                                     'Furosemide (mg/hr)',
                                                     'Furosemide (ml/hr)',
                                                     'LASIX',
                                                     'LASIX (ml/hr)',
                                                     'Lasix (mg/hr)']}]}},
 'torasemide': {'class_name': 'lgl_cncpt',
                'description': 'torasemide (torsemide) administration, any route',
                'category': 'medications',
                'aggregate': 'max',
                'sources': {'hirid': [{'ids': [1000520, 1000521, 1000522, 1000986],
                                       'table': 'pharma',
                                       'sub_var': 'pharmaid',
                                       'callback': 'transform_fun(comp_na(`>`, 0.000001))'}],
                            'sic': [{'ids': [1450],
                                     'table': 'medication',
                                     'sub_var': 'DrugID',
                                     'callback': 'transform_fun(comp_na(`>`, 0))'}]}},
 'piperacillin_tazobactam': {'class_name': 'lgl_cncpt',
                             'description': 'piperacillin-tazobactam intravenous administration',
                             'category': 'medications',
                             'aggregate': 'max',
                             'sources': {'miiv': [{'ids': [225893],
                                                   'table': 'inputevents',
                                                   'sub_var': 'itemid',
                                                   'callback': 'mimic_positive_dose',
                                                   'val_var': 'rate',
                                                   'extra_vars': ['amount', 'statusdescription'],
                                                   'status_var': 'statusdescription',
                                                   'excluded_statuses': ['Rewritten']}],
                                         'hirid': [{'ids': [405, 1000837],
                                                    'table': 'pharma',
                                                    'sub_var': 'pharmaid',
                                                    'callback': 'transform_fun(comp_na(`>`, 0.000001))',
                                                    'extra_vars': ['route'],
                                                    'require_regex': {'route': '(?i)^\\s*(?:[ci]v(?:-(?:inf|inj))?|i\\.?v\\.?|intravenous)\\s*$'}}],
                                         'sic': [{'ids': [1459],
                                                  'table': 'medication',
                                                  'sub_var': 'DrugID',
                                                  'callback': 'sicdb_exposure',
                                                  'val_var': 'Amount'}]}},
 'vancomycin_piperacillin_tazobactam': {'class_name': 'rec_cncpt',
                                        'description': 'vancomycin and piperacillin-tazobactam '
                                                       'co-exposure: TRUE at an administration hour of '
                                                       'either agent when the other agent was given in '
                                                       'the same or the preceding 23 hours',
                                        'category': 'medications',
                                        'aggregate': 'max',
                                        'unit': 'boolean',
                                        'concepts': ['vancomycin_iv', 'piperacillin_tazobactam'],
                                        'callback': 'vpt_coexposure',
                                        'target': 'ts_tbl',
                                        'sources': {}},
 'nsaid': {'class_name': 'lgl_cncpt',
           'description': 'systemic non-steroidal anti-inflammatory drug administration (non-aspirin '
                          'COX inhibitors)',
           'category': 'medications',
           'aggregate': 'max',
           'sources': {'miiv': [{'ids': [229072],
                                 'table': 'inputevents',
                                 'sub_var': 'itemid',
                                 'callback': 'mimic_positive_dose',
                                 'val_var': 'rate',
                                 'extra_vars': ['amount', 'statusdescription'],
                                 'status_var': 'statusdescription',
                                 'excluded_statuses': ['Rewritten']}],
                       'aumc': [{'ids': [7407, 9063, 9064, 9081, 9103, 19162],
                                 'table': 'drugitems',
                                 'sub_var': 'itemid',
                                 'callback': 'aumc_positive_dose',
                                 'val_var': 'dose'}],
                       'hirid': [{'ids': [1000401, 1000809, 1000859, 1000883, 1000884, 1000885],
                                  'table': 'pharma',
                                  'sub_var': 'pharmaid',
                                  'callback': 'transform_fun(comp_na(`>`, 0.000001))'}],
                       'sic': [{'ids': [1575, 1732, 1733, 1753, 1804, 1865, 1971, 1972],
                                'table': 'medication',
                                'sub_var': 'DrugID',
                                'callback': 'transform_fun(comp_na(`>`, 0))'}]}},
 'vancomycin_iv': {'class_name': 'lgl_cncpt',
                   'description': 'vancomycin systemic (intravenous) administration; an input of the '
                                  'vancomycin and piperacillin-tazobactam co-exposure',
                   'category': 'medications',
                   'aggregate': 'max',
                   'sources': {'miiv': [{'ids': 225798,
                                         'table': 'inputevents',
                                         'sub_var': 'itemid',
                                         'callback': 'mimic_positive_dose',
                                         'val_var': 'rate',
                                         'extra_vars': ['amount', 'statusdescription'],
                                         'status_var': 'statusdescription',
                                         'excluded_statuses': ['Rewritten']}],
                               'hirid': [{'ids': [189],
                                          'table': 'pharma',
                                          'sub_var': 'pharmaid',
                                          'callback': 'transform_fun(comp_na(`>`, 0.000001))',
                                          'extra_vars': ['route'],
                                          'require_regex': {'route': '(?i)^\\s*(?:[ci]v(?:-(?:inf|inj))?|i\\.?v\\.?|intravenous)\\s*$'}}],
                               'sic': [{'ids': [1461],
                                        'table': 'medication',
                                        'sub_var': 'DrugID',
                                        'callback': 'sicdb_exposure',
                                        'val_var': 'Amount'}]}}}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_dictionary_contract(name):
    assert DICTIONARY[name] == EXPECTED[name]
    assert name in HIDDEN_DICTIONARY_CONCEPTS and name in DIURETIC_NEPHROTOXIN_METADATA
    assert (name in EXTRACT_MODULES["diuretic_nephrotoxin"]) == (name != "vancomycin_iv")
    assert "diuretic_nephrotoxin" not in EXTRACT_MODULE_ORDER
    assert all(name not in EXTRACT_MODULES[module] for module in EXTRACT_MODULE_ORDER)


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    from easyicu.synthetic_sources import generate_source
    from easyicu.api.concepts import clear_global_loader

    root = tmp_path_factory.mktemp("nephro-batch1")
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


@pytest.mark.parametrize("database", ["hirid", "sic"])
def test_coexposure_dependency_extraction(sources, database):
    from easyicu import load_concepts

    name = "vancomycin_piperacillin_tazobactam"
    actual = load_concepts([name], database=database, data_path=sources / database, merge=False,
                           verbose=False, concept_workers=1, parallel_workers=1)
    frame = actual if isinstance(actual, pd.DataFrame) else actual[name]
    assert not frame.empty and frame[name].eq(True).any()


@pytest.mark.parametrize("name", [n for n, d in EXPECTED.items() if "miiv" in d["sources"]])
def test_miiv_positive_administration_extraction(tmp_path, name):
    from tests.support.mimic_synthetic_source import mimic_synthetic_load, mimic_synthetic_source

    ids = EXPECTED[name]["sources"]["miiv"][0]["ids"]
    item = ids[0] if isinstance(ids, list) else ids
    start = pd.Timestamp("2150-01-01 00:15:00")
    frame = pd.DataFrame({"subject_id": [11, 11], "hadm_id": [21, 21], "stay_id": [1, 1],
        "itemid": [item, item], "orderid": [1, 2], "linkorderid": [1, 2],
        "starttime": [start, start + pd.Timedelta(hours=2)],
        "endtime": [start + pd.Timedelta(hours=1), start + pd.Timedelta(hours=3)],
        "amount": [2., 0.], "amountuom": ["mg", "mg"], "rate": [0., 0.],
        "rateuom": ["mg/hour", "mg/hour"], "statusdescription": ["FinishedRunning", "FinishedRunning"]})
    source = mimic_synthetic_source(tmp_path, "miiv", inputevents=frame)
    values = mimic_synthetic_load(source, [name])[name].data[name].dropna()
    assert not values.empty and values.eq(True).all()


def test_eicu_furosemide_positive_infusion_extraction(tmp_path):
    from tests.support.eicu_source import _eicu_source, _hourly

    frame = pd.DataFrame({"infusionoffset": [15, 90], "drugname": ["Furosemide (mg/hr)"] * 2,
                          "drugrate": ["1", "0"]})
    source = _eicu_source(tmp_path, "eicu", infusiondrug=frame)
    assert _hourly(source, "furosemide_active") == {0.: True, 1.: False}


@pytest.mark.parametrize("name", ["furosemide_active", "nsaid"])
def test_aumc_administered_amount_must_be_positive(name):
    source = load_dictionary()[name].sources["aumc"][0]
    frame = pd.DataFrame({name: [0., -1., None, 2.]})
    assert _apply_callback(frame, source, name)[name].tolist() == [True]


@pytest.mark.parametrize("name", [n for n, d in EXPECTED.items() if "hirid" in d["sources"]])
def test_hirid_positive_threshold_and_route(name):
    source = load_dictionary()[name].sources["hirid"][0]
    frame = pd.DataFrame({name: [0., 0.000001, 0.000002], "route": ["i.v."] * 3})
    result = _apply_callback(frame, source, name)[name]
    assert result.tolist() == [False, False, True]


@pytest.mark.parametrize("name", ["vancomycin_iv", "piperacillin_tazobactam"])
def test_hirid_oral_routes_are_excluded(name):
    source = load_dictionary()[name].sources["hirid"][0]
    frame = pd.DataFrame({name: [1., 1.], "route": ["i.v.", "po"]})
    assert _apply_callback(frame, source, name)[name].tolist() == [True]


@pytest.mark.parametrize("name", ["vancomycin_iv", "piperacillin_tazobactam"])
def test_sic_exposure_does_not_mark_missing_amount_positive(name):
    source = load_dictionary()[name].sources["sic"][0]
    frame = pd.DataFrame({name: [0., 1., -1., None]})
    values = _apply_callback(frame, source, name)[name]
    assert values.iloc[:2].tolist() == [False, True]
    assert values.iloc[2:].isna().all()
