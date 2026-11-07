"""The storage owner decides a table is missing; readers never decide it alone.

Codex delta2 D2-STORAGE: ``PatientFilter._read_table`` checked the physical
files first, so a whole table the conversion manifest lists but whose files
are gone never reached the storage owner's validation and raised a plain
``FileNotFoundError``; the SICdb survival reader then turned that into
"survival unknown". The eMAR lookup had the same no-path branch, and a
conversion manifest that exists but cannot be stat'ed was treated as no
manifest. With a flat and a bucket installation side by side, PatientFilter
opened the flat one and the owner (which resolves the bucket) refused the
loader. All rows are invented.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from easyicu.datasource import ICUDataSource
from easyicu.patient_filter import PatientFilter, SourceTableNotInstalled
from easyicu.resources import load_data_sources
from easyicu.storage_layout import StorageIntegrityError, TableScanError, resolve_table_files
from tests.support.bucket_storage import bind_fixture_source, write_contract


def _manifest(root, tables):
    entries = [
        {"table": table, "row_count": rows, "outputs": [{"relative_path": path} for path in paths]}
        for table, (paths, rows) in tables.items()
    ]
    (root / "conversion_manifest.json").write_text(json.dumps({"tables": entries}))


def _source(database, root):
    return ICUDataSource(config=load_data_sources().get(database), base_path=root)


def _sic_cases(root):
    pd.DataFrame({
        "CaseID": [1, 2], "PatientID": [1, 2], "AgeOnAdmission": [60, 70],
        "HospitalDischargeType": [2028, 2028], "ICUOffset": [0, 0], "TimeOfStay": [7200, 7200],
    }).to_parquet(root / "cases.parquet")


def test_a_listed_table_whose_files_are_gone_is_an_integrity_error(tmp_path):
    _sic_cases(tmp_path)
    pd.DataFrame({"ReferenceGlobalID": [2028], "ReferenceValue": ["Survived"]}).to_parquet(
        tmp_path / "d_references.parquet"
    )
    _manifest(tmp_path, {"cases": (["cases.parquet"], 2), "d_references": (["d_references.parquet"], 1)})
    (tmp_path / "d_references.parquet").unlink()
    with pytest.raises(StorageIntegrityError) as caught:
        PatientFilter("sic", tmp_path)._read_table("d_references")
    assert caught.value.reason == "table_scan_files_missing"


def test_sic_survival_does_not_swallow_a_vanished_listed_reference_table(tmp_path):
    _sic_cases(tmp_path)
    pd.DataFrame({"ReferenceGlobalID": [2028], "ReferenceValue": ["Survived"]}).to_parquet(
        tmp_path / "d_references.parquet"
    )
    _manifest(tmp_path, {"cases": (["cases.parquet"], 2), "d_references": (["d_references.parquet"], 1)})
    (tmp_path / "d_references.parquet").unlink()
    with pytest.raises(StorageIntegrityError):
        PatientFilter("sic", tmp_path)._load_sic_demographics()


def test_a_table_the_installation_does_not_contain_is_typed_not_installed(tmp_path):
    _sic_cases(tmp_path)
    _manifest(tmp_path, {"cases": (["cases.parquet"], 2)})
    with pytest.raises(SourceTableNotInstalled) as caught:
        PatientFilter("sic", tmp_path)._read_table("d_references")
    assert isinstance(caught.value, FileNotFoundError)
    assert caught.value.reason == "source_table_not_installed"


def test_an_unreadable_manifest_is_not_treated_as_no_manifest(tmp_path):
    if os.name == "nt":
        pytest.skip("symlink loops need POSIX")
    _sic_cases(tmp_path)
    manifest = tmp_path / "conversion_manifest.json"
    manifest.symlink_to(manifest)  # stat raises ELOOP, not FileNotFoundError
    with pytest.raises(TableScanError) as caught:
        resolve_table_files(_source("sic", tmp_path), "cases")
    assert caught.value.reason == "conversion_manifest_unreadable"


def test_patient_filter_reads_the_installation_the_owner_resolves(tmp_path):
    # A flat cases.parquet and a valid single-key bucket install side by side:
    # the owner resolves the bucket; the reader must not open the flat file
    # and then fail as a loader mismatch.
    flat = pd.DataFrame({"CaseID": [1, 2], "PatientID": [1, 2], "Val": [1.0, 2.0]})
    flat.to_parquet(tmp_path / "data_float_h.parquet")
    bucket = tmp_path / "data_float_h_bucket" / "bucket_id=0"
    bucket.mkdir(parents=True)
    flat.to_parquet(bucket / "part.parquet")
    write_contract(tmp_path / "data_float_h_bucket", key="CaseID", modulus=1)
    bind_fixture_source(tmp_path / "data_float_h_bucket", "data_float_h")
    source = _source("sic", tmp_path)
    owned = resolve_table_files(source, "data_float_h")
    assert all("data_float_h_bucket" in str(p) for p in owned)
    frame = PatientFilter("sic", tmp_path)._read_table("data_float_h")
    assert sorted(frame["Val"]) == [1.0, 2.0]


def test_emar_lookup_asks_the_owner_before_declaring_a_table_absent(tmp_path):
    from easyicu.concept.emar_diuretics import EmarTableUnavailable, _lookup

    pd.DataFrame({"emar_id": ["a"], "dose_given": ["1"]}).to_parquet(tmp_path / "emar_detail.parquet")
    _manifest(tmp_path, {"emar_detail": (["emar_detail.parquet"], 1)})
    (tmp_path / "emar_detail.parquet").unlink()
    source = _source("miiv", tmp_path)
    with pytest.raises(StorageIntegrityError) as caught:
        _lookup(source, "emar_detail", ["emar_id"], "emar_id", pd.Series(["a"]))
    assert caught.value.reason == "table_scan_files_missing"

    (tmp_path / "conversion_manifest.json").unlink()
    with pytest.raises(EmarTableUnavailable) as absent:
        _lookup(source, "emar_detail", ["emar_id"], "emar_id", pd.Series(["a"]))
    assert absent.value.reason == "emar_diuretic_table_unavailable"
