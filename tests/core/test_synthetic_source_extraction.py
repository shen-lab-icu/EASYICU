"""Every mapped concept travels through the ordinary public extraction API."""

import json
from pathlib import Path

import pandas as pd
import pytest

DICTIONARY = json.loads((Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json").read_text())
CASES = [(db, name) for db in ("hirid", "aumc", "sic")
         for name, definition in DICTIONARY.items() if definition.get("sources", {}).get(db)]


@pytest.fixture(scope="module")
def synthetic_roots(tmp_path_factory):
    from easyicu.synthetic_sources import generate_source
    from easyicu.api.concepts import clear_global_loader

    root = tmp_path_factory.mktemp("synthetic-extraction")
    for db in ("hirid", "aumc", "sic"):
        generate_source(root / db, db, stays=12)
    yield root
    clear_global_loader()


@pytest.mark.parametrize("database,name", CASES)
def test_every_mapped_concept_extracts(synthetic_roots, database, name):
    from easyicu import load_concepts

    result = load_concepts([name], database=database, data_path=synthetic_roots / database,
                           merge=False, verbose=False, concept_workers=1, parallel_workers=1)
    frame = result if isinstance(result, pd.DataFrame) else result[name]
    identifier = {"hirid": "patientid", "aumc": "admissionid", "sic": "CaseID"}[database]
    assert identifier in frame
    assert name in frame
    assert not frame.empty, f"{database}.{name}: mapped source produced no records"
    assert frame[name].notna().any(), f"{database}.{name}: all values missing"
    assert pd.api.types.is_integer_dtype(frame[identifier])
    assert frame[identifier].between(1, 12).all()
    kind = DICTIONARY[name].get("class_name", "num_cncpt")
    if kind == "lgl_cncpt":
        assert set(frame[name].dropna().unique()) <= {True, False}
    elif kind in ("fct_cncpt", "chr_cncpt"):
        assert all(isinstance(value, str) for value in frame[name].dropna())
    else:
        assert pd.api.types.is_numeric_dtype(frame[name]), frame.dtypes
    for clock in ("charttime", "datetime", "start", "measuredat", "Offset"):
        if clock in frame:
            assert pd.api.types.is_numeric_dtype(frame[clock]), frame.dtypes


@pytest.mark.parametrize("database", ["hirid", "aumc", "sic"])
def test_core_units_and_plausibility_use_real_extraction(synthetic_roots, database):
    from easyicu import load_concepts

    names = ["map", "lact", "hgb", "crea", "k", "urine", "age", "sex", "weight", "death"]
    result = load_concepts(names, database=database, data_path=synthetic_roots / database,
                           merge=False, verbose=False, concept_workers=1, parallel_workers=1)
    result = {name: table.data if hasattr(table, "data") else table for name, table in result.items()}
    for name, low, high in [("map", 30, 110), ("lact", .5, 8), ("hgb", 8, 16),
                            ("crea", .4, 5), ("k", 3, 6), ("urine", 10, 100)]:
        values = result[name][name].dropna()
        assert values.between(low, high).all(), (database, name, values.describe())
    assert set(result["sex"].sex.dropna()) == {"Female", "Male"}
