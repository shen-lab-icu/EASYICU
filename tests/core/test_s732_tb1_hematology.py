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
EXPECTED = {'inr_pt': {'sic': [{'ids': [3128], 'table': 'laboratory', 'sub_var': 'LaboratoryID'}],
            'aumc': [{'ids': [6789],
                      'table': 'numericitems',
                      'sub_var': 'itemid',
                      'unit_var': 'unit',
                      'require_regex': {'unit': '(?i)^\\s*sec\\s*$'},
                      'callback': "convert_unit(binary_op(`*`, 1), 'ratio', '(?i)^\\s*sec\\s*$')"}]}}
BASELINE_METADATA = {'inr_pt': {'class_name': 'num_cncpt',
            'description': 'prothrombin time/international normalized ratio',
            'omopid': 4306239,
            'category': 'hematology'}}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_dictionary_contract(name):
    definition = DICTIONARY[name]
    assert {k: v for k, v in definition.items() if k != "sources"} == BASELINE_METADATA[name]
    for database, sources in EXPECTED[name].items():
        assert all(source in definition["sources"][database] for source in sources)
    assert name in EXTRACT_MODULES["hematology"]


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
