"""Urine chemistry publication must preserve the established renal row set."""

import json

import pandas as pd
import pytest

from easyicu.api import extraction as api
from easyicu.api.concepts import clear_global_loader
from tests.support.urine_chemistry_source import urine_module_source


CONCEPTS = ["urine_na", "urine_crea", "urine_osm", "urine_urea",
            "sic_precondition_renal_dysfunction"]
MODULE = "renal_urine_chem"


@pytest.mark.parametrize("database", ["miiv", "mimic", "eicu", "aumc", "hirid", "sic"])
def test_native_urine_module_preserves_existing_module_bytes(tmp_path, monkeypatch, database):
    monkeypatch.setenv("EASYICU_NATIVE_V2_REAL_ROOT", "")
    source = tmp_path / "source"
    urine_module_source(source, database)
    existing = ["renal", "chemistry", "outcome"]
    # Reconstruct the pre-UC1 publication roster. This also detects mistakenly
    # retaining any of the new concepts in renal, including null-only columns.
    legacy_renal = [name for name in api.EXTRACT_MODULES["renal"] if name not in CONCEPTS]
    before, after = tmp_path / "before", tmp_path / "after"
    try:
        with monkeypatch.context() as baseline:
            baseline.setitem(api.EXTRACT_MODULES, "renal", legacy_renal)
            old = api.extract_database(
                database, data_path=source, output_dir=before, modules=existing,
                native_export_v2=True, resource_budget_mb=2048, verbose=False,
            )
        new = api.extract_database(
            database, data_path=source, output_dir=after, modules=[*existing, MODULE],
            native_export_v2=True, resource_budget_mb=2048, verbose=False,
        )
        for result in (old, new):
            assert not {name: info["errors"] for name, info in result["modules"].items()
                        if info.get("errors")}
        for module in existing:
            assert (before / f"{module}.parquet").read_bytes() == (after / f"{module}.parquet").read_bytes(), module
        manifest = json.loads((after / "_manifest.json").read_text())
        entry = next(row for row in manifest["files"] if row["module"] == MODULE)
        assert entry["physical_concept_ids"] == CONCEPTS
        frame = pd.read_parquet(after / f"{MODULE}.parquet")
        assert list(frame) == ["stay_id", "charttime", *CONCEPTS]
        assert frame[CONCEPTS].notna().any(axis=1).all()
        if database in {"aumc", "hirid"}:
            assert frame.empty
        elif database == "sic":
            assert frame.set_index("stay_id")[CONCEPTS[-1]].to_dict() == {1: True, 2: False}
        else:
            assert frame.charttime.tolist() == [0., 1., 2.]
            assert frame.urine_na.tolist() == [40., 0., 400.]
            assert frame.urine_crea.tolist() == [120., 0., 1000.]
            if database == "eicu":
                assert frame[["urine_osm", "urine_urea"]].isna().all().all()
    finally:
        clear_global_loader()


def test_urine_module_has_one_owner_and_uses_the_lab_group():
    assert api.EXTRACT_MODULES[MODULE] == CONCEPTS
    for name in CONCEPTS:
        assert [module for module, names in api.EXTRACT_MODULES.items() if name in names] == [MODULE]
    assert MODULE in api.EXTRACT_MODULE_ORDER
    grouped = api._group_modules_for_extraction(["chemistry", MODULE], [], True)
    assert len(grouped) == 1
    assert grouped[0]["modules"] == ["chemistry", MODULE]


@pytest.mark.parametrize("database,mapped", [
    ("miiv", CONCEPTS[:4]), ("eicu", CONCEPTS[:2]), ("sic", CONCEPTS[-1:]),
])
def test_sparse_module_keeps_zero_and_false_but_omits_value_less_keys(database, mapped):
    def load(**kwargs):
        assert kwargs["concepts"] == mapped
        assert kwargs["r_compatible"] is False
        # The last key has only missing measurements. A real zero / No
        # observation must survive the same rule that removes this key.
        values = pd.Series([True, False, pd.NA], dtype="boolean") if database == "sic" else [1., 0., None]
        return pd.DataFrame({"stay_id": [1, 2, 3], **{name: values for name in mapped}})

    frame = api._load_module_concepts(
        load, module_name=MODULE,
        load_kwargs={"concepts": CONCEPTS, "database": database, "merge": True},
    )
    assert frame.stay_id.tolist() == [1, 2]
    assert frame[mapped].iloc[1].eq(0).all()


@pytest.mark.parametrize("database,mapped,id_col,repeated", [
    # MIMIC-III: each zero-row concept frame keeps its own time column.
    ("mimic", CONCEPTS[:4], "icustay_id", "charttime"),
    # eICU: each keeps its lab source's unit/sub-item column.
    ("eicu", CONCEPTS[:2], "patientunitstayid", "labmeasurenameinterface"),
])
def test_zero_row_module_publishes_one_typed_axis_and_records_unmapped_concepts(
    database, mapped, id_col, repeated,
):
    def load(**kwargs):
        assert kwargs["concepts"] == mapped
        columns = [id_col, "charttime"]
        for name in mapped:
            columns += [name, repeated] if repeated != "charttime" else [name, "charttime"]
        frame = pd.DataFrame([], columns=pd.Index(columns))
        return frame.astype({id_col: "int64"})

    sink = {}
    frame = api._load_module_concepts(
        load, module_name=MODULE,
        load_kwargs={"concepts": CONCEPTS, "database": database, "merge": True,
                     "availability_sink": sink},
    )
    assert frame.empty
    assert list(frame) == [id_col, "charttime", *mapped]
    assert frame["charttime"].dtype == "float64"
    # Concepts with no source here are recorded exactly as the resolver
    # records an unmapped pair, so the export publishes their placeholder.
    unmapped = [name for name in CONCEPTS if name not in mapped]
    assert sorted(sink) == sorted(unmapped)
    assert {record.reason for record in sink.values()} == {"unmapped"}
    assert {record.status for record in sink.values()} == {"blocked"}


def test_module_with_rows_keeps_its_columns_and_records_unmapped_concepts():
    mapped = CONCEPTS[:2]

    def load(**kwargs):
        return pd.DataFrame({"patientunitstayid": [1], "charttime": [1.0],
                             "urine_na": [40.0], "urine_crea": [120.0]})

    sink = {}
    frame = api._load_module_concepts(
        load, module_name=MODULE,
        load_kwargs={"concepts": CONCEPTS, "database": "eicu", "merge": True,
                     "availability_sink": sink},
    )
    assert list(frame) == ["patientunitstayid", "charttime", *mapped]
    assert frame["urine_na"].tolist() == [40.0]
    assert sorted(sink) == sorted(set(CONCEPTS) - set(mapped))
