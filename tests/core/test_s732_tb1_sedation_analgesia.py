"""Ported S732 exposure definitions with the reviewed source selections."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.api.extraction import EXTRACT_MODULES, EXTRACT_MODULE_ORDER
from easyicu.concept import _apply_callback
from easyicu.concept.catalog import HIDDEN_DICTIONARY_CONCEPTS, SEDATION_ANALGESIA_METADATA
from easyicu.resources import load_dictionary

DICTIONARY = json.loads((Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json").read_text())
EXPECTED = {'remifentanil': {'class_name': 'lgl_cncpt',
                  'description': 'remifentanil given in the hour (administration or running infusion)',
                  'category': 'medications',
                  'aggregate': 'max',
                  'sources': {'aumc': [{'ids': [12402],
                                        'table': 'drugitems',
                                        'sub_var': 'itemid',
                                        'index_var': 'start',
                                        'dur_var': 'stop',
                                        'callback': 'aumc_positive_dose',
                                        'val_var': 'dose'}],
                              'hirid': [{'ids': [1000705],
                                         'table': 'pharma',
                                         'sub_var': 'pharmaid',
                                         'callback': 'transform_fun(comp_na(`>`, 0.000001))'}],
                              'sic': [{'ids': [1546],
                                       'table': 'medication',
                                       'sub_var': 'DrugID',
                                       'val_var': 'Amount',
                                       'callback': 'sicdb_active_exposure',
                                       'extra_vars': ['OffsetDrugEnd']}]}},
 'sufentanil': {'class_name': 'lgl_cncpt',
                'description': 'sufentanil (intravenous) given in the hour',
                'category': 'medications',
                'aggregate': 'max',
                'sources': {'aumc': [{'ids': [7014],
                                      'table': 'drugitems',
                                      'sub_var': 'itemid',
                                      'index_var': 'start',
                                      'dur_var': 'stop',
                                      'callback': 'aumc_positive_dose',
                                      'val_var': 'dose',
                                      'extra_vars': ['ordercategory'],
                                      'exclude_regex': {'ordercategory': '(?i)niet '
                                                                         'iv|epidura|intratheca'}}],
                            'hirid': [{'ids': [1000771],
                                       'table': 'pharma',
                                       'sub_var': 'pharmaid',
                                       'callback': 'transform_fun(comp_na(`>`, 0.000001))',
                                       'extra_vars': ['route'],
                                       'require_regex': {'route': '(?i)^\\s*(?:[ci]v(?:-(?:inf|inj))?|i\\.?v\\.?|intravenous)\\s*$'}}],
                            'sic': [{'ids': [1547],
                                     'table': 'medication',
                                     'sub_var': 'DrugID',
                                     'val_var': 'Amount',
                                     'callback': 'sicdb_active_exposure',
                                     'extra_vars': ['OffsetDrugEnd']}]}},
 'piritramide': {'class_name': 'lgl_cncpt',
                 'description': 'piritramide given in the hour',
                 'category': 'medications',
                 'aggregate': 'max',
                 'sources': {'aumc': [{'ids': [6910],
                                       'table': 'drugitems',
                                       'sub_var': 'itemid',
                                       'index_var': 'start',
                                       'dur_var': 'stop',
                                       'callback': 'aumc_positive_dose',
                                       'val_var': 'dose'}],
                             'sic': [{'ids': [1521],
                                      'table': 'medication',
                                      'sub_var': 'DrugID',
                                      'val_var': 'Amount',
                                      'callback': 'sicdb_active_exposure',
                                      'extra_vars': ['OffsetDrugEnd']}]}},
 'esketamine': {'class_name': 'lgl_cncpt',
                'description': 'esketamine (S-ketamine) given in the hour',
                'category': 'medications',
                'aggregate': 'max',
                'sources': {'aumc': [{'ids': [9018],
                                      'table': 'drugitems',
                                      'sub_var': 'itemid',
                                      'index_var': 'start',
                                      'dur_var': 'stop',
                                      'callback': 'aumc_positive_dose',
                                      'val_var': 'dose'}],
                            'sic': [{'ids': [1578],
                                     'table': 'medication',
                                     'sub_var': 'DrugID',
                                     'val_var': 'Amount',
                                     'callback': 'sicdb_active_exposure',
                                     'extra_vars': ['OffsetDrugEnd']}]}},
 'hydromorphone': {'class_name': 'lgl_cncpt',
                   'description': 'hydromorphone given in the hour, any systemic route',
                   'category': 'medications',
                   'aggregate': 'max',
                   'sources': {'miiv': [{'ids': [221833],
                                         'table': 'inputevents',
                                         'sub_var': 'itemid',
                                         'index_var': 'starttime',
                                         'dur_var': 'endtime',
                                         'callback': 'mimic_positive_dose',
                                         'val_var': 'rate',
                                         'extra_vars': ['amount', 'statusdescription'],
                                         'status_var': 'statusdescription',
                                         'excluded_statuses': ['Rewritten']}],
                               'sic': [{'ids': [1652, 1671],
                                        'table': 'medication',
                                        'sub_var': 'DrugID',
                                        'val_var': 'Amount',
                                        'callback': 'sicdb_active_exposure',
                                        'extra_vars': ['OffsetDrugEnd']}]}},
 'methadone': {'class_name': 'lgl_cncpt',
               'description': 'methadone given in the hour, any systemic route',
               'category': 'medications',
               'aggregate': 'max',
               'sources': {'miiv': [{'ids': [227520],
                                     'table': 'inputevents',
                                     'sub_var': 'itemid',
                                     'index_var': 'starttime',
                                     'dur_var': 'endtime',
                                     'callback': 'mimic_positive_dose',
                                     'val_var': 'rate',
                                     'extra_vars': ['amount', 'statusdescription'],
                                     'status_var': 'statusdescription',
                                     'excluded_statuses': ['Rewritten']}],
                           'aumc': [{'ids': [9090, 12750],
                                     'table': 'drugitems',
                                     'sub_var': 'itemid',
                                     'index_var': 'start',
                                     'dur_var': 'stop',
                                     'callback': 'aumc_positive_dose',
                                     'val_var': 'dose'}],
                           'hirid': [{'ids': [215, 1000427],
                                      'table': 'pharma',
                                      'sub_var': 'pharmaid',
                                      'callback': 'transform_fun(comp_na(`>`, 0.000001))'}],
                           'sic': [{'ids': [1855],
                                    'table': 'medication',
                                    'sub_var': 'DrugID',
                                    'val_var': 'Amount',
                                    'callback': 'sicdb_active_exposure',
                                    'extra_vars': ['OffsetDrugEnd']}]}},
 'oxycodone': {'class_name': 'lgl_cncpt',
               'description': 'oxycodone given in the hour (enteral; immediate or prolonged release, '
                              'alone or combined)',
               'category': 'medications',
               'aggregate': 'max',
               'sources': {'aumc': [{'ids': [13883, 19221],
                                     'table': 'drugitems',
                                     'sub_var': 'itemid',
                                     'index_var': 'start',
                                     'dur_var': 'stop',
                                     'callback': 'aumc_positive_dose',
                                     'val_var': 'dose'}],
                           'hirid': [{'ids': [1000935, 1000936, 1000937, 1000989, 1000990],
                                      'table': 'pharma',
                                      'sub_var': 'pharmaid',
                                      'callback': 'transform_fun(comp_na(`>`, 0.000001))'}],
                           'sic': [{'ids': [1639, 1640],
                                    'table': 'medication',
                                    'sub_var': 'DrugID',
                                    'val_var': 'Amount',
                                    'callback': 'sicdb_active_exposure',
                                    'extra_vars': ['OffsetDrugEnd']}]}},
 'tramadol': {'class_name': 'lgl_cncpt',
              'description': 'tramadol given in the hour, any systemic route',
              'category': 'medications',
              'aggregate': 'max',
              'sources': {'aumc': [{'ids': [7052],
                                    'table': 'drugitems',
                                    'sub_var': 'itemid',
                                    'index_var': 'start',
                                    'dur_var': 'stop',
                                    'callback': 'aumc_positive_dose',
                                    'val_var': 'dose'}],
                          'hirid': [{'ids': [1000523, 1000984, 1000985],
                                     'table': 'pharma',
                                     'sub_var': 'pharmaid',
                                     'callback': 'transform_fun(comp_na(`>`, 0.000001))'}],
                          'sic': [{'ids': [1634],
                                   'table': 'medication',
                                   'sub_var': 'DrugID',
                                   'val_var': 'Amount',
                                   'callback': 'sicdb_active_exposure',
                                   'extra_vars': ['OffsetDrugEnd']}]}},
 'bps': {'unit': 'score',
         'min': 3,
         'max': 12,
         'description': 'Behavioral Pain Scale total; invalid 0–2 remain missing',
         'category': 'neurological',
         'sources': {'aumc': [{'ids': [19609],
                               'table': 'numericitems',
                               'sub_var': 'itemid',
                               'unit_var': 'unit',
                               'require_regex': {'unit': '(?i)^\\s*Geen\\s*$'},
                               'callback': "convert_unit(binary_op(`*`, 1), 'score')"}]}}}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_dictionary_contract(name):
    assert DICTIONARY[name] == EXPECTED[name]
    assert name in EXTRACT_MODULES["sedation_analgesia"]
    assert name in HIDDEN_DICTIONARY_CONCEPTS and name in SEDATION_ANALGESIA_METADATA
    assert "sedation_analgesia" not in EXTRACT_MODULE_ORDER
    assert all(name not in EXTRACT_MODULES[module] for module in EXTRACT_MODULE_ORDER)


@pytest.mark.parametrize("name", [n for n, d in EXPECTED.items() if d["sources"].get("aumc", [{}])[0].get("callback") == "aumc_positive_dose"])
def test_aumc_positive_administered_dose(name):
    source = load_dictionary()[name].sources["aumc"][0]
    frame = pd.DataFrame({name: [0., -1., None, 2.], "ordercategory": ["Infuus"] * 4})
    actual = _apply_callback(frame, source, name)
    assert actual[name].tolist() == [True]


def test_sic_active_exposure_interval_and_point():
    from easyicu.concept import ConceptResolver

    cases = pd.DataFrame({"CaseID": [1], "ICUOffset": [7200.], "TimeOfStay": [100000.]})
    ds = SimpleNamespace(config=SimpleNamespace(name="sic"), load_table=lambda *a, **kw: cases)
    source = load_dictionary()["remifentanil"].sources["sic"][0]
    frame = pd.DataFrame({"CaseID": [1, 1, 1], "Offset": [9000., 26100., 30000.],
                          "OffsetDrugEnd": [18000., 26100., 33600.],
                          "remifentanil": [1., 1., 0.]})
    actual = _apply_callback(frame, source, "remifentanil", data_source=ds)
    aligned = ConceptResolver.__new__(ConceptResolver)._align_time_to_admission(actual, ds, ["CaseID"], "Offset")
    assert sorted(aligned.Offset.astype(int).unique()) == [0, 1, 2, 5]
    assert aligned.remifentanil.eq(True).all()


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    from easyicu.synthetic_sources import generate_source
    from easyicu.api.concepts import clear_global_loader

    root = tmp_path_factory.mktemp("sedation-batch1")
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


@pytest.mark.parametrize("name", [n for n, d in EXPECTED.items() if "miiv" in d["sources"]])
def test_miiv_positive_administration_extraction(tmp_path, name):
    from tests.support.mimic_synthetic_source import mimic_synthetic_load, mimic_synthetic_source

    item = EXPECTED[name]["sources"]["miiv"][0]["ids"][0]
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


def test_sufentanil_neuraxial_route_is_excluded():
    source = load_dictionary()["sufentanil"].sources["aumc"][0]
    frame = pd.DataFrame({"sufentanil": [1., 1.], "ordercategory": ["Infuus", "epiduraal"]})
    assert _apply_callback(frame, source, "sufentanil").sufentanil.tolist() == [True]


@pytest.mark.parametrize("name", [n for n, d in EXPECTED.items() if "hirid" in d["sources"]])
def test_hirid_positive_threshold(name):
    source = load_dictionary()[name].sources["hirid"][0]
    frame = pd.DataFrame({name: [0., 0.000001, 0.000002], "route": ["i.v."] * 3})
    result = _apply_callback(frame, source, name)[name]
    assert result.iloc[:3].tolist() == [False, False, True]


def test_bps_unit_gate_and_boundaries():
    from easyicu.concept.callback_apply import _drop_declared_row_exclusions

    source = load_dictionary()["bps"].sources["aumc"][0]
    frame = pd.DataFrame({"bps": [3., 12., 4.], "unit": ["Geen", "Geen", "wrong"]})
    filtered = _drop_declared_row_exclusions(frame, source, "bps")
    result = _apply_callback(filtered, source, "bps", unit_column="unit")
    assert result.bps.tolist() == [3., 12.]
    assert result.unit.tolist() == ["score", "score"]


def test_bps_loader_excludes_invalid_scores_and_units(tmp_path):
    from easyicu.concept import ConceptResolver
    from easyicu.config import load_src_cfg
    from easyicu.datasource import ICUDataSource

    pd.DataFrame({"admissionid": [1], "patientid": [1], "admittedat": [0],
                  "dischargedat": [24 * 3600000]}).to_parquet(tmp_path / "admissions.parquet", index=False)
    frame = pd.DataFrame({"admissionid": [1] * 5, "itemid": [19609] * 5,
        "measuredat": [300000 + i * 3600000 for i in range(5)],
        "registeredat": [300000 + i * 3600000 for i in range(5)],
        "value": [2., 3., 12., 13., 4.], "unit": ["Geen", "Geen", "Geen", "Geen", "wrong"]})
    (tmp_path / "numericitems").mkdir()
    frame.to_parquet(tmp_path / "numericitems" / "part.parquet", index=False)
    source = ICUDataSource(load_src_cfg("aumc"), base_path=tmp_path, enable_cache=False)
    result = ConceptResolver(load_dictionary()).load_concepts(
        ["bps"], source, merge=False, r_compatible=False, interval=pd.Timedelta(hours=1),
        concept_workers=1, verbose=False)["bps"].data.bps.dropna()
    assert result.tolist() == [3., 12.]
