"""Reviewed S732 source extensions retain existing concept metadata."""

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.api.extraction import EXTRACT_MODULES
from easyicu.concept import _apply_callback
from easyicu.concept.callback_apply import _drop_declared_row_exclusions
from easyicu.resources import load_dictionary

DICTIONARY = json.loads((Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json").read_text())
EXPECTED = {'alb': {'aumc': [{'ids': [9975],
                   'table': 'numericitems',
                   'sub_var': 'itemid',
                   'unit_var': 'unit',
                   'require_regex': {'unit': '(?i)^\\s*g/l\\s*$'},
                   'callback': "convert_unit(binary_op(`*`, 0.1), 'g/dL')"}]},
 'bnp': {'miiv': [{'ids': [227446],
                   'table': 'chartevents',
                   'sub_var': 'itemid',
                   'unit_var': 'valueuom',
                   'require_regex': {'valueuom': '(?i)^\\s*pg/mL\\s*$'}}]},
 'pct': {'aumc': [{'ids': [15565],
                   'table': 'numericitems',
                   'sub_var': 'itemid',
                   'unit_var': 'unit',
                   'require_regex': {'unit': '(?i)^\\s*µg/l\\s*$'},
                   'callback': "convert_unit(binary_op(`*`, 1), 'ng/mL')"}]}}
BASELINE_METADATA = {'alb': {'unit': 'g/dL',
         'min': 0,
         'max': 6,
         'description': 'albumin',
         'omopid': 4017497,
         'category': 'chemistry'},
 'bnp': {'unit': ['pg/mL', 'ng/L'],
         'min': 0,
         'max': 50000,
         'description': 'B-type natriuretic peptide (BNP); distinct from NT-proBNP',
         'category': 'chemistry',
         '_comment': '2026-07-04 eICU-only: MIMIC/AUMC/SIC carry NT-proBNP (ntprobnp) not BNP; HiRID '
                     'none.'},
 'pct': {'unit': ['ng/mL', 'µg/L', 'ug/L'],
         'min': 0,
         'max': 1000,
         'description': 'procalcitonin (blood)',
         'category': 'chemistry'}}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_dictionary_contract(name):
    definition = DICTIONARY[name]
    assert {k: v for k, v in definition.items() if k != "sources"} == BASELINE_METADATA[name]
    for database, sources in EXPECTED[name].items():
        assert all(source in definition["sources"][database] for source in sources)
    assert name in EXTRACT_MODULES["chemistry"]


@pytest.mark.parametrize("name,database,entry", [(name, database, entry)
    for name, dbs in EXPECTED.items() for database, sources in dbs.items() for entry in sources])
def test_unit_gates_and_conversions(name, database, entry):
    item = entry["ids"]
    source = next(s for s in load_dictionary()[name].sources[database] if s.ids == item)
    valid = {"alb": "g/l", "bnp": "pg/mL", "co": "l/min", "dbp": "mmHg",
             "inr_pt": "sec", "map": "mmHg", "pct": "µg/l", "sbp": "mmHg"}[name]
    column = source.unit_var
    frame = pd.DataFrame({name: [0., 10., 10.]})
    if column:
        frame[column] = [valid, valid, "wrong"]
        frame = _drop_declared_row_exclusions(frame, source, name)
        assert len(frame) == 2
    else:
        frame = frame.iloc[:2]
    actual = _apply_callback(frame, source, name, unit_column=column)
    factor = .1 if name == "alb" else 1.
    assert actual[name].tolist() == pytest.approx([0., 10. * factor])


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    from easyicu.synthetic_sources import generate_source
    from easyicu.api.concepts import clear_global_loader

    root = tmp_path_factory.mktemp("existing-batch1")
    for database in sorted({db for dbs in EXPECTED.values() for db in dbs if db in {"aumc", "sic"}}):
        generate_source(root / database, database, stays=12)
    yield root
    clear_global_loader()


@pytest.mark.parametrize("name,database", [(name, db) for name, dbs in EXPECTED.items()
                                          for db in dbs if db in {"aumc", "sic"}])
def test_synthetic_extraction(sources, name, database):
    from easyicu import load_concepts

    actual = load_concepts([name], database=database, data_path=sources / database, merge=False,
                           verbose=False, concept_workers=1, parallel_workers=1)
    frame = actual if isinstance(actual, pd.DataFrame) else actual[name]
    assert not frame.empty and frame[name].notna().any()


def test_miiv_new_chart_source_unit_gate(tmp_path):
    from tests.support.mimic_synthetic_source import mimic_synthetic_load, mimic_synthetic_source

    now = pd.Timestamp("2150-01-01 00:15:00")
    frame = pd.DataFrame({"subject_id": [11, 11], "hadm_id": [21, 21], "stay_id": [1, 1],
                          "itemid": [227446, 227446],
                          "charttime": [now, now + pd.Timedelta(hours=1)],
                          "storetime": [now, now + pd.Timedelta(hours=1)],
                          "valuenum": [100., 100.], "value": ["100", "100"],
                          "valueuom": ["pg/mL", "wrong"]})
    source = mimic_synthetic_source(tmp_path, "miiv", chartevents=frame)
    result = mimic_synthetic_load(source, ["bnp"])["bnp"].data.bnp.dropna()
    assert result.tolist() == [100.]
