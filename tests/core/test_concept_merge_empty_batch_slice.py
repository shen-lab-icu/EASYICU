"""A concept with no rows in a patient batch merges like the unbatched load.

2026-11-06 r13mergeidx: under batch 50 on a MIMIC-III CareVue sample,
renal_urine_chem stopped with "All concepts must share identical index column
to merge".  A urine concept with no rows in one batch came back from the raw
source path without a declared index (native datetime ``charttime`` plus
``valueuom``), while its populated peers carried the aligned relative-hour
axis.  The merge owner now gives such an empty slice the batch's index name
and dtype; it never skips the table, and index or identifier mismatches
between populated tables (or a declared foreign index on an empty one) still
raise.
"""

import pandas as pd
import pytest

from easyicu.concept import ConceptResolver
from easyicu.concept.merge_time_index import RELATIVE_HOURS
from easyicu.table import ICUTable

HOUR = pd.Timedelta(hours=1)
NAMES = ("urine_na", "urine_crea", "urine_osm", "urine_urea")


def _aligned(name, rows):
    frame = pd.DataFrame(rows, columns=["icustay_id", "charttime", name])
    frame = frame.astype({"icustay_id": "int64", "charttime": "float64", name: "float64"})
    return ICUTable(frame, id_columns=["icustay_id"], index_column="charttime", value_column=name,
                    time_basis=RELATIVE_HOURS, interval=HOUR)


def _raw_empty(name, ids=("icustay_id",)):
    """The shape the CareVue batch returned: no index, raw datetime time, unit column."""
    frame = pd.DataFrame({
        **{col: pd.Series(dtype="int64") for col in ids},
        "charttime": pd.Series(dtype="datetime64[ns]"),
        name: pd.Series(dtype="float64"),
        "valueuom": pd.Series(dtype="object"),
    })
    return ICUTable(frame, id_columns=list(ids), index_column=None, value_column=name)


ROWS = {
    "urine_na": [(1, 0.0, 40.0), (2, 3.0, 55.0), (3, 1.0, 20.0)],
    "urine_crea": [(1, 0.0, 80.0), (2, 4.0, 90.0), (3, 1.0, 70.0)],
    "urine_osm": [(1, 1.0, 300.0), (2, 3.0, 410.0), (3, 2.0, 350.0)],
    "urine_urea": [(3, 2.0, 600.0)],  # only the second batch has urea rows
}


def _tables(stays, order):
    out = {}
    for name in order:
        rows = [r for r in ROWS[name] if r[0] in stays]
        out[name] = _aligned(name, rows) if rows else _raw_empty(name)
    return out


def _sorted(frame):
    return frame.sort_values(["icustay_id", "charttime"]).reset_index(drop=True)


@pytest.mark.parametrize("order", [NAMES, tuple(reversed(NAMES))], ids=["empty_last", "empty_first"])
def test_an_empty_batch_slice_merges_like_the_unbatched_load(order):
    resolver = ConceptResolver({})
    unbatched = _sorted(resolver._merge_tables(_tables({1, 2, 3}, order)))
    first = resolver._merge_tables(_tables({1, 2}, order))
    second = resolver._merge_tables(_tables({3}, order))
    assert first["charttime"].dtype == "float64"
    assert "valueuom" not in first.columns
    batched = _sorted(pd.concat([first, second], ignore_index=True))
    pd.testing.assert_frame_equal(batched[unbatched.columns], unbatched)
    assert list(first.columns) == list(unbatched.columns)


def test_the_carevue_b50_shape_reproduces_and_now_merges():
    tables = {name: _aligned(name, [r for r in ROWS[name] if r[0] == 1]) for name in NAMES[:3]}
    tables["urine_urea"] = _raw_empty("urine_urea")
    merged = ConceptResolver({})._merge_tables(tables)
    assert list(merged.columns) == ["icustay_id", "charttime", *NAMES]
    assert merged["urine_urea"].isna().all() and len(merged) == 2


def test_a_populated_table_without_the_batch_index_still_raises():
    tables = {name: _aligned(name, ROWS[name][:1]) for name in NAMES[:3]}
    tables["urine_urea"] = ICUTable(pd.DataFrame({"icustay_id": [1], "urine_urea": [5.0]}),
                                    id_columns=["icustay_id"], index_column=None, value_column="urine_urea")
    with pytest.raises(ValueError, match="identical index column"):
        ConceptResolver({})._merge_tables(tables)


def test_an_empty_table_with_a_declared_foreign_index_still_raises():
    tables = {name: _aligned(name, ROWS[name][:1]) for name in NAMES[:3]}
    frame = pd.DataFrame({"icustay_id": pd.Series(dtype="int64"), "starttime": pd.Series(dtype="float64"),
                          "urine_urea": pd.Series(dtype="float64")})
    tables["urine_urea"] = ICUTable(frame, id_columns=["icustay_id"], index_column="starttime",
                                    value_column="urine_urea")
    with pytest.raises(ValueError):
        ConceptResolver({})._merge_tables(tables)


def test_an_empty_slice_with_foreign_identifiers_still_raises():
    tables = {name: _aligned(name, ROWS[name][:1]) for name in NAMES[:3]}
    tables["urine_urea"] = _raw_empty("urine_urea", ids=("subject_id",))
    with pytest.raises(ValueError, match="identical identifier columns"):
        ConceptResolver({})._merge_tables(tables)


def test_the_fill_scope_counts_only_filled_empty_slices():
    from easyicu.concept.merge_time_index import merge_fill_scope

    resolver = ConceptResolver({})
    with merge_fill_scope() as normal:
        resolver._merge_tables(_tables({1, 2, 3}, NAMES))
    with merge_fill_scope() as repro:
        resolver._merge_tables(_tables({1, 2}, NAMES))
    assert normal == [] and repro == ["urine_urea"]


@pytest.mark.parametrize("drop_urea", [False, True], ids=["normal", "urea_empty_everywhere"])
def test_the_published_manifest_carries_the_fill_receipt(tmp_path, monkeypatch, drop_urea):
    import json

    from easyicu.api import extraction as api
    from easyicu.api.concepts import clear_global_loader
    from tests.support.urine_chemistry_source import urine_module_source

    monkeypatch.setenv("EASYICU_NATIVE_V2_REAL_ROOT", "")
    source = tmp_path / "source"
    urine_module_source(source, "mimic")
    if drop_urea:
        labs = pd.read_parquet(source / "labevents.parquet")
        labs[labs.itemid.ne(51104)].to_parquet(source / "labevents.parquet", index=False)
    out = tmp_path / "out"
    try:
        result = api.extract_database(
            "mimic", data_path=source, output_dir=out, modules=["outcome", "renal_urine_chem"],
            native_export_v2=True, resource_budget_mb=2048, verbose=False,
        )
        assert not {n: i["errors"] for n, i in result["modules"].items() if i.get("errors")}
        manifest = json.loads((out / "_manifest.json").read_text())
        entry = next(row for row in manifest["files"] if row["module"] == "renal_urine_chem")
        filled = entry["merge_empty_index_filled"]
        if drop_urea:
            assert [row["concept"] for row in filled] == ["urine_urea"]
            frame = pd.read_parquet(out / "renal_urine_chem.parquet")
            assert frame["charttime"].dtype == "float64" and frame["urine_urea"].isna().all()
        else:
            assert filled == []
    finally:
        clear_global_loader()
