"""Every Parquet reader outside load_table consumes the validated file set.

Final review B-STORAGE-INVENTORY: each reader below once listed its own
files (``dir/*.parquet``, ``dir/**``, a bucket glob), so a missing chunk or a
file no conversion manifest owns went unnoticed.  Each now reads
``resolve_table_files`` for the location it chose and refuses such a scan;
CSV reads are admitted by ``csv_table_scan`` and recorded.  All rows are
invented.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.datasource import ICUDataSource
from easyicu.resources import load_data_sources
from easyicu.storage_layout import StorageIntegrityError, scan_fallbacks
from tests.support.mimic_chartevents import chart

T = pd.Timestamp("2150-01-01 08:00")


def _chart(rows):
    return chart(rows, T)


def _source(database, root):
    return ICUDataSource(config=load_data_sources().get(database), base_path=root)


def _manifest(root, tables):
    entries = [
        {"table": table, "row_count": rows,
         "outputs": [{"relative_path": path} for path in paths]}
        for table, (paths, rows) in tables.items()
    ]
    (root / "conversion_manifest.json").write_text(json.dumps({"tables": entries}))


def _chunked(root, table, frames, *, extra=None):
    (root / table).mkdir()
    paths = []
    for index, frame in enumerate(frames, start=1):
        frame.to_parquet(root / table / f"{index}.parquet")
        paths.append(f"{table}/{index}.parquet")
    tables = {table: (paths, sum(len(f) for f in frames))}
    tables.update(extra or {})
    _manifest(root, tables)


def _stays(ids):
    return pd.DataFrame({
        "row_id": ids, "subject_id": [i * 10 for i in ids], "hadm_id": [i * 100 for i in ids],
        "icustay_id": ids, "intime": [T] * len(ids), "outtime": [T + pd.Timedelta(days=2)] * len(ids),
    })


def _missing_chunk_stays(root):
    _chunked(root, "icustays", [_stays([1, 2]), _stays([3])])
    (root / "icustays" / "2.parquet").unlink()


def _raised_at(caught, text):
    return any(text in str(entry.statement) for entry in caught.traceback)


# --- concept loader sites ---------------------------------------------------


def _resolver(name, **source):
    from easyicu.concept import ConceptResolver
    from easyicu.concept.schema import ConceptDefinition, ConceptDictionary, ConceptSource

    return ConceptResolver(ConceptDictionary({name: ConceptDefinition(
        name=name, sources={"mimic": [ConceptSource(table="chartevents", **source)]},
    )}))


def _mimic_chart(root):
    pd.DataFrame(_stays([1000, 2000]).assign(los=2.0)).to_parquet(root / "icustays.parquet")
    _chunked(
        root, "chartevents",
        [_chart([(1, 10, 100, 1000, 80.0)]), _chart([(2, 20, 200, 2000, 70.0)])],
        extra={"icustays": (["icustays.parquet"], 2)},
    )


def test_the_regex_prematch_refuses_a_missing_chunk(tmp_path):
    _mimic_chart(tmp_path)
    (tmp_path / "chartevents" / "2.parquet").unlink()
    resolver = _resolver("uom_probe", value_var="valuenum", sub_var="valueuom", regex="kg")
    with pytest.raises(StorageIntegrityError) as caught:
        resolver.load_concepts(["uom_probe"], _source("mimic", tmp_path), merge=False,
                               verbose=False, concept_workers=1)
    # The complete shared decision can reject before regex prematching.
    assert caught.value.reason == "table_scan_manifest_mismatch"


def test_the_single_column_fast_path_refuses_a_file_set_the_policy_does_not_own(tmp_path, monkeypatch):
    _mimic_chart(tmp_path)
    stray = tmp_path / "stray"
    stray.mkdir()
    _chart([(9, 10, 100, 1000, 1.0)]).to_parquet(stray / "1.parquet")
    source = _source("mimic", tmp_path)
    monkeypatch.setattr(source, "resolve_loader_from_disk", lambda table: stray)
    resolver = _resolver("value_probe", value_var="valuenum")
    with pytest.raises(StorageIntegrityError) as caught:
        resolver.load_concepts(["value_probe"], source, merge=False, verbose=False,
                               concept_workers=1, interval=pd.Timedelta(hours=1),
                               aggregate="median")
    assert caught.value.reason == "table_scan_loader_mismatch"
    assert _raised_at(caught, "loader=_wt_dir")


def test_hirid_end_time_refuses_a_missing_observation_chunk(tmp_path):
    from easyicu.concept import ConceptDictionary, ConceptResolver

    frames = [pd.DataFrame({"patientid": [1], "datetime": [T], "variableid": [200]}),
              pd.DataFrame({"patientid": [2], "datetime": [T], "variableid": [200]})]
    _chunked(tmp_path, "observations", frames)
    (tmp_path / "observations" / "2.parquet").unlink()
    with pytest.raises(StorageIntegrityError) as caught:
        ConceptResolver(ConceptDictionary({}))._synthesise_los_column(
            "end", _source("hirid", tmp_path), pd.DataFrame({"patientid": [1, 2]}),
        )
    assert caught.value.reason == "table_scan_manifest_mismatch"


def test_raw_column_synthesis_refuses_an_unregistered_id_table(tmp_path, monkeypatch):
    from easyicu.concept import ConceptDictionary, ConceptResolver

    monkeypatch.delenv("EASYICU_ALLOW_UNREGISTERED_SCAN", raising=False)
    _chunked(tmp_path, "chartevents", [_chart([(1, 10, 100, 1000, 80.0)])])
    _stays([1000]).to_parquet(tmp_path / "icustays.parquet")  # not in the manifest
    with pytest.raises(StorageIntegrityError) as caught:
        ConceptResolver(ConceptDictionary({}))._synthesise_los_column(
            "outtime", _source("mimic", tmp_path), pd.DataFrame({"icustay_id": [1000]}),
        )
    assert caught.value.reason == "table_scan_unregistered"


# --- datasource readers ------------------------------------------------------


def _vitals(ids):
    return pd.DataFrame({"patientunitstayid": ids, "observationoffset": [30] * len(ids),
                         "heartrate": [80.0] * len(ids)})


def test_the_wide_table_batch_reads_the_validated_set(tmp_path):
    from easyicu.datasource import load_wide_table_aggregated

    _chunked(tmp_path, "vitalperiodic", [_vitals([1]), _vitals([2])])
    frame = load_wide_table_aggregated(_source("eicu", tmp_path), "vitalperiodic", ["heartrate"])
    assert sorted(frame["patientunitstayid"]) == [1, 2]
    (tmp_path / "vitalperiodic" / "2.parquet").unlink()
    with pytest.raises(StorageIntegrityError):
        load_wide_table_aggregated(_source("eicu", tmp_path), "vitalperiodic", ["heartrate"])


def test_the_partitioned_fallback_reader_refuses_a_missing_chunk(tmp_path):
    _missing_chunk_stays(tmp_path)
    with pytest.raises(StorageIntegrityError):
        _source("mimic", tmp_path)._read_partitioned_data_fallback(
            tmp_path / "icustays", None, table_name="icustays")


def test_the_emar_lookup_refuses_a_missing_chunk(tmp_path):
    from easyicu.concept.emar_diuretics import _lookup

    detail = pd.DataFrame({"emar_id": ["a", "b"], "dose_given": [1.0, 2.0]})
    _chunked(tmp_path, "emar_detail", [detail.iloc[[0]], detail.iloc[[1]]])
    source = _source("miiv", tmp_path)
    found = _lookup(source, "emar_detail", ["emar_id", "dose_given"], "emar_id", pd.Series(["a", "b"]))
    assert sorted(found["emar_id"]) == ["a", "b"]
    (tmp_path / "emar_detail" / "2.parquet").unlink()
    with pytest.raises(StorageIntegrityError):
        _lookup(source, "emar_detail", ["emar_id", "dose_given"], "emar_id", pd.Series(["a"]))


# --- cohort and filter readers ------------------------------------------------


def test_the_patient_filter_refuses_a_missing_chunk(tmp_path):
    from easyicu.patient_filter import PatientFilter

    _missing_chunk_stays(tmp_path)
    with pytest.raises(StorageIntegrityError):
        PatientFilter("mimic", tmp_path)._read_table("icustays")


def test_the_patient_filter_records_a_legacy_csv_read(tmp_path):
    from easyicu.patient_filter import PatientFilter

    _stays([1]).to_csv(tmp_path / "icustays.csv", index=False)
    assert PatientFilter("mimic", tmp_path)._read_table("icustays")["icustay_id"].tolist() == [1]
    assert {"reason": "table_read_csv_unmanifested", "root": str(tmp_path),
            "table": "icustays"} in scan_fallbacks()


def test_patient_id_discovery_refuses_an_unregistered_stay_table(tmp_path, monkeypatch):
    from easyicu.api import get_all_patient_ids

    monkeypatch.delenv("EASYICU_ALLOW_UNREGISTERED_SCAN", raising=False)
    _chunked(tmp_path, "chartevents", [_chart([(1, 10, 100, 1000, 80.0)])])
    _stays([1, 2]).to_parquet(tmp_path / "icustays.parquet")
    with pytest.raises(StorageIntegrityError) as caught:
        get_all_patient_ids(tmp_path, database="mimic")
    assert caught.value.reason == "table_scan_unregistered"


def test_patient_id_discovery_refuses_a_missing_shard(tmp_path):
    from easyicu.api import get_all_patient_ids

    _missing_chunk_stays(tmp_path)
    with pytest.raises(StorageIntegrityError):
        get_all_patient_ids(tmp_path, database="mimic")


def test_the_fast_id_scan_refuses_a_missing_chunk(tmp_path):
    from easyicu.api.concepts import _build_fast_scan_expr

    _missing_chunk_stays(tmp_path)
    loader = SimpleNamespace(datasource=_source("mimic", tmp_path))
    with pytest.raises(StorageIntegrityError):
        _build_fast_scan_expr(loader, "icustays")


# --- CSV admission -------------------------------------------------------------


def test_a_listed_table_cannot_fall_back_to_csv_once_its_parquet_vanished(tmp_path):
    from easyicu.storage_layout import csv_table_scan

    _chunked(tmp_path, "icustays", [_stays([1])])
    (tmp_path / "icustays" / "1.parquet").unlink()
    _stays([1]).to_csv(tmp_path / "icustays.csv", index=False)
    with pytest.raises(StorageIntegrityError) as caught:
        csv_table_scan(_source("mimic", tmp_path), "icustays", tmp_path / "icustays.csv")
    assert caught.value.reason == "table_scan_files_missing"


def test_a_csv_read_beside_complete_parquet_is_recorded(tmp_path):
    from easyicu.storage_layout import csv_table_scan

    _chunked(tmp_path, "services", [pd.DataFrame({"hadm_id": [1]})])
    pd.DataFrame({"hadm_id": [1]}).to_csv(tmp_path / "SERVICES.csv", index=False)
    csv_table_scan(_source("mimic", tmp_path), "services", tmp_path / "SERVICES.csv")
    assert {"reason": "table_read_csv_beside_manifest", "root": str(tmp_path),
            "table": "services"} in scan_fallbacks()
