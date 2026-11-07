"""Non-numeric source values are counted on every conversion path, never dropped silently (B-IMPLICIT-GATE).

A declared numeric concept (here ``temp``: numeric bounds) whose source holds
text keeps the text out of its values (NaN, as ricu's ``num_cncpt``) and
counts each such value per concept and source, in the bounds owner's records
(``non_numeric_values``). The same synthetic input gives the same count on the
raw path, the single-, multi- and wide-table SQL paths and the module export.
A concept with no class and no bounds is numeric only by default: any such
value refuses publication with ``ConceptValueKindUndeclared`` and leaves the
export unpublished. Synthetic rows only.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.aggregation_route import force_raw_aggregation, record_duckdb_routes
from easyicu.concept.value_kind import ConceptValueKindUndeclared
from easyicu.io.data_quality import SOURCE_BOUNDS_WARN_MIN_STAYS, source_value_coercion_warnings
from easyicu.source_bounds import collect_bounds_drops, non_numeric_by_concept
from tests.support.urine_raw_source import _load, _source

pytestmark = pytest.mark.filterwarnings("ignore")

_STAYS = SOURCE_BOUNDS_WARN_MIN_STAYS + 2
_TEXT = ["see note", "<35", "n/a"]  # one per stay, cycling
_NUMBERS = ["36.6", " 37.1 ", "38"]


def _cells(stay: int) -> list:
    return [*_NUMBERS, _TEXT[stay % len(_TEXT)]]


def _sic(root: Path):
    ids = np.arange(1, _STAYS + 1)
    rows = []
    for stay in ids:
        cells = _cells(int(stay))
        hours = np.arange(len(cells), dtype=float) + 1.25
        rows.append(pd.DataFrame({"CaseID": stay, "DataID": 709, "Offset": hours * 3600, "Val": cells}))
        rows.append(pd.DataFrame({"CaseID": stay, "DataID": 710, "Offset": hours * 3600, "Val": "96"}))
    return _source(
        root, "sic",
        cases=pd.DataFrame({
            "CaseID": ids, "PatientID": ids, "ICUOffset": 0.0,
            "TimeOfStay": 200.0 * 3600, "WeightOnAdmission": 70000.0,
        }),
        data_float_h=pd.concat(rows, ignore_index=True),
    )


def _read(source, names, *, raw=False):
    from contextlib import nullcontext

    with (force_raw_aggregation() if raw else nullcontext()), record_duckdb_routes() as routes, \
            collect_bounds_drops() as drops:
        out = _load(source, names, patient_ids={"CaseID": list(range(1, _STAYS + 1))})
    return out, drops, dict(routes)


def _non_numeric(drops, concept="temp"):
    return non_numeric_by_concept(drops).get(concept, 0)


def _published_values(out, name="temp"):
    data = out[name].data
    return sorted(pd.to_numeric(data[name], errors="coerce").dropna().round(6).tolist())


def test_one_input_gives_one_count_on_the_raw_and_sql_paths(tmp_path):
    single, single_drops, single_routes = _read(_sic(tmp_path / "s"), ["temp"])
    multi, multi_drops, multi_routes = _read(_sic(tmp_path / "m"), ["temp", "spo2"])
    raw, raw_drops, raw_routes = _read(_sic(tmp_path / "r"), ["temp"], raw=True)
    assert single_routes == {"bucketed_single": 1}
    assert multi_routes == {"long_table_batch": 1}
    assert raw_routes == {}
    # One text value per stay, on every path.
    assert _non_numeric(single_drops) == _non_numeric(multi_drops) == _non_numeric(raw_drops) == _STAYS
    for drops in (single_drops, multi_drops, raw_drops):
        [counts] = drops["temp"].values()
        assert counts["stays_with_non_numeric"] == _STAYS
        assert counts["stays_with_values"] == _STAYS
    # The text is NaN (no value), the numbers are kept: the same values everywhere.
    assert _published_values(single) == _published_values(multi) == _published_values(raw)
    assert len(_published_values(single)) > 0
    # spo2 (all numbers) counts nothing.
    assert _non_numeric(multi_drops, "spo2") == 0


def test_the_community_wide_path_counts_the_same(tmp_path):
    from easyicu.api.concepts import load_concepts

    pd.DataFrame({
        "subject_id": range(1, _STAYS + 1), "hadm_id": range(1, _STAYS + 1),
        "stay_id": range(1, _STAYS + 1), "INP_NO": range(1, _STAYS + 1),
        "intime": 1.0, "outtime": 40.0,
    }).to_parquet(tmp_path / "stays.parquet", index=False)
    (tmp_path / "dtnursingchart").mkdir()
    rows = [
        pd.DataFrame({
            "INP_NO": stay, "ChartTime": np.arange(4, dtype=float) + 2.25,
            "heart_rate": "88", "temperature": _cells(stay),
        })
        for stay in range(1, _STAYS + 1)
    ]
    pd.concat(rows, ignore_index=True).to_parquet(
        tmp_path / "dtnursingchart" / "part-00000.parquet", index=False
    )
    (tmp_path / "community_preparation_manifest.json").write_text(
        json.dumps({"database": "zigong"}), encoding="utf-8"
    )
    with record_duckdb_routes() as routes, collect_bounds_drops() as drops:
        load_concepts(
            ["hr", "temp"], patient_ids={"INP_NO": list(range(1, _STAYS + 1))},
            database="zigong", data_path=tmp_path, interval="1h",
            parallel_workers=1, concept_workers=1,
        )
    assert "wide_table_batch" in dict(routes)
    assert _non_numeric(drops) == _STAYS
    assert _non_numeric(drops, "hr") == 0


def test_the_module_export_counts_the_same(tmp_path):
    from easyicu.api.extraction import _normalise_counted

    frame = pd.DataFrame({
        "stay_id": [stay for stay in range(1, _STAYS + 1) for _ in range(4)],
        "charttime": [float(h) for _ in range(_STAYS) for h in range(4)],
        "temp": pd.Series([c for stay in range(1, _STAYS + 1) for c in _cells(stay)], dtype=object),
    })
    out, counts = _normalise_counted(frame, ["temp"], reorder=False)
    assert counts["temp"]["export"]["non_numeric_values"] == _STAYS
    assert out["temp"].isna().sum() == _STAYS


def test_a_declared_numeric_concept_warns_and_publishes(tmp_path):
    _out, drops, _ = _read(_sic(tmp_path / "s"), ["temp"])
    rows = source_value_coercion_warnings({"vitals": drops})
    assert [(r["concept"], r["reason"], r["non_numeric_values"]) for r in rows] == [
        ("temp", "declared_numeric_values_not_numeric", _STAYS)
    ]
    from easyicu.api.extraction import _refuse_classless_non_numeric

    _refuse_classless_non_numeric({"vitals": drops}, database="sic")  # temp has bounds: no refusal


@pytest.mark.parametrize("readers", [2, 5])
def test_a_source_read_by_several_modules_is_counted_once(tmp_path, readers):
    from easyicu.api.extraction import _refuse_classless_non_numeric

    _out, drops, _ = _read(_sic(tmp_path / "s"), ["temp"])
    one = {"vitals": drops}
    many = {f"module_{index}": drops for index in range(readers)}
    assert source_value_coercion_warnings(many) == [
        {**row, "module": "module_0"} for row in source_value_coercion_warnings(one)
    ]
    [row] = source_value_coercion_warnings(many)
    assert (row["non_numeric_values"], row["stays_with_non_numeric"]) == (_STAYS, _STAYS)
    assert non_numeric_by_concept(*many.values()) == non_numeric_by_concept(drops) == {"temp": _STAYS}
    _refuse_classless_non_numeric(many, database="sic")  # temp is declared: no refusal


def _classless(monkeypatch, name):
    from easyicu.api import extraction
    from easyicu.resources import load_dictionary as real

    def patched(*args, **kwargs):
        dictionary = real(*args, **kwargs)
        definition = dataclasses.replace(dictionary[name], class_name=None, minimum=None, maximum=None)
        dictionary._concepts[name] = definition
        return dictionary

    monkeypatch.setattr(extraction, "load_dictionary", patched)


def test_a_classless_concept_refuses_publication(tmp_path, monkeypatch):
    from easyicu.api.extraction import _refuse_classless_non_numeric

    _out, drops, _ = _read(_sic(tmp_path / "s"), ["temp"])
    _classless(monkeypatch, "temp")
    # Read by three modules, the refusal still names the unique count.
    with pytest.raises(ConceptValueKindUndeclared, match=rf"'temp'.*: {_STAYS} source value.*declare class"):
        _refuse_classless_non_numeric({"a": drops, "b": drops, "c": drops}, database="sic")


def test_a_refused_run_leaves_no_published_export(tmp_path, monkeypatch):
    from easyicu.api import extraction
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source
    from tests.support.inprocess_extraction import inprocess_mp_context

    root = tmp_path / "sic"
    generate_source(root, "sic", stays=6)
    _classless(monkeypatch, "temp")
    # The synthetic installation is numeric; the count stands in for text.
    monkeypatch.setattr(
        "easyicu.source_bounds.non_numeric_by_concept", lambda *parts: {"temp": 3}
    )
    monkeypatch.setattr(extraction, "_get_extraction_mp_context", inprocess_mp_context)
    out = tmp_path / "export"
    try:
        with pytest.raises(ConceptValueKindUndeclared, match="'temp'.*3 source value"):
            extraction.extract_database(
                "sic", data_path=str(root), output_dir=str(out), modules=["outcome", "vitals"],
                native_export_v2=True, verbose=False,
            )
    finally:
        clear_global_loader()
    assert not (out / "_manifest.json").exists()
