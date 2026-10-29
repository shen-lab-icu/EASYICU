"""Invented converted sources exercise the native module/publication boundary."""

import json

import pandas as pd
import pyarrow.parquet as pq
import pytest

import easyicu
from easyicu.api import extraction as api
from easyicu.api.concepts import clear_global_loader
from easyicu import synthetic_source_tables
from easyicu.concept.callbacks import _apply_locf_24h
from easyicu.synthetic_sources import generate_source


@pytest.fixture
def aumc_source(tmp_path, monkeypatch):
    raw_values = synthetic_source_tables.raw_values

    def ordinal_values(name, *args):
        values = raw_values(name, *args)
        # The general numeric generator adds noise to this ordinal channel.
        # Keep this fixture valid for the strict SOFA-2 pre-sedation GCS input.
        return values.round() if name == "sedated_gcs" else values

    monkeypatch.setattr(synthetic_source_tables, "raw_values", ordinal_values)
    source = tmp_path / "source"
    generate_source(source, "aumc", stays=3)
    yield source
    clear_global_loader()


@pytest.mark.parametrize("streamed", [False, True])
def test_sofa2_module_merges_source_named_cns_receipts(aumc_source, tmp_path, streamed):
    output = tmp_path / "export"
    output.mkdir()
    api._run_module_extraction(
        module_name="sofa2_score",
        concepts=api.EXTRACT_MODULES["sofa2_score"],
        database="aumc",
        data_path=str(aumc_source),
        patient_ids_filter={"admissionid": [1, 2, 3]},
        batch_size=3,
        output_dir=str(output),
        use_sofa2=True,
        stream_output_batches=streamed,
    )
    module = json.loads((output / "_manifest.json").read_text())
    assert module["errors"] == []
    frame = pd.read_parquet(output / "sofa2_score.parquet")
    assert not frame.empty
    assert frame["sofa2"].notna().any()
    assert "charttime" in frame
    assert "measuredat" not in frame
    receipts = [name for name in api.EXTRACT_MODULES["sofa2_score"] if "ascertainment" in name or "sensitivity" in name]
    assert frame[receipts].notna().any().all()
    (output / "_manifest.json").rename(output / "sofa2_score.manifest.json")

    api._publish_native_export_v2(
        database="aumc",
        data_path=str(aumc_source),
        output_dir=str(output),
        modules=["sofa2_score"],
        max_patients=None,
        result={"modules": {"sofa2_score": module}},
    )
    published = pd.read_parquet(output / "sofa2_score.parquet")
    assert published.columns[:2].tolist() == ["stay_id", "charttime"]
    assert published["charttime"].notna().all()
    assert set(published["charttime"]) >= {0.0, 6.0, 12.0, 24.0, 48.0}
    assert not set(published.columns) & {"admissionid", "measuredat", "start"}
    schema = pq.read_schema(output / "sofa2_score.parquet")
    assert str(schema.field("charttime").type) == "double"


def test_all_nineteen_modules_publish_from_synthetic_aumc(aumc_source, tmp_path):
    output = tmp_path / "full-export"
    result = api.extract_database(
        database="aumc", data_path=aumc_source, output_dir=output,
        patient_ids={"admissionid": [1, 2, 3]}, batch_size=3,
        native_export_v2=True, resource_budget_mb=2048, verbose=False,
    )
    assert len(result["modules"]) == 19
    assert all(not module["errors"] for module in result["modules"].values())
    manifest = json.loads((output / "_manifest.json").read_text())
    assert manifest["schema_version"] == "easyicu_native_export_v2"
    contract = manifest["canonical_physical_schema"]
    assert contract["identity_column"] == "stay_id"
    assert contract["time_column"] == "charttime"
    assert contract["time_origin"] == "icu_admission"
    assert contract["time_unit"] == "h"
    assert {entry["module"] for entry in manifest["files"]} == set(api.EXTRACT_MODULES)
    for entry in manifest["files"]:
        frame = pd.read_parquet(output / entry["file"])
        assert frame.columns[0] == "stay_id"
        if entry["module"] != "demographics":
            assert frame.columns[1] == "charttime"
        assert not set(frame.columns) & {"admissionid", "measuredat", "start"}
    score = pd.read_parquet(output / "sofa2_score.parquet")
    assert score["sofa2"].notna().any()
    assert score["charttime"].notna().all()


def test_raw_unaggregated_concepts_do_not_claim_a_resampled_grid(aumc_source):
    tables = easyicu.load_concepts(
        ["hr"], database="aumc", data_path=aumc_source,
        merge=False, aggregate=False, r_compatible=False,
        interval="1h", verbose=False, concept_workers=1, parallel_workers=1,
    )
    assert tables["hr"].time_basis is None
    assert tables["hr"].interval is None


@pytest.mark.filterwarnings("error:Setting an item of incompatible dtype:FutureWarning")
@pytest.mark.parametrize("dtype", ["bool", "int64"])
def test_locf_keeps_observed_values_in_their_original_dtype(dtype):
    # NEWS supplies a non-nullable boolean supp_o2 channel. Even an empty
    # expiry mask must not assign a float NaN into that boolean column.
    frame = pd.DataFrame({
        "stay_id": [1, 1], "charttime": [0.0, 49.0],
        "observed": pd.Series([True, False], dtype=dtype),
    })
    result = _apply_locf_24h(frame, ["stay_id"], "charttime", ["observed"])
    pd.testing.assert_frame_equal(result, frame)


@pytest.mark.parametrize("dtype,first,last", [
    ("boolean", True, False), ("Int64", 1, 2), ("float64", 1.0, 2.0),
    ("string", "A", "V"),
])
def test_locf_masks_expired_values_without_extending_the_original_window(dtype, first, last):
    frame = pd.DataFrame({
        "stay_id": [1] * 6, "charttime": [0.0, 24.0, 25.0, 26.0, 50.0, 51.0],
        "observed": pd.Series([first, None, None, last, None, None], dtype=dtype),
    })
    result = _apply_locf_24h(frame, ["stay_id"], "charttime", ["observed"])
    expected = pd.Series([first, first, None, last, last, None], dtype=dtype, name="observed")
    pd.testing.assert_series_equal(result["observed"], expected)
