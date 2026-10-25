"""The app's verified raw Demo remains public through the normal source picker."""

from __future__ import annotations

import json
import shutil

import pandas as pd
import pytest

from easyicu.webserver import dataio, demo_source_storage as storage
from easyicu.research_agent.providers import client_trust
from tests.support.official_demo_raw import official_raw as official_raw




def test_official_raw_picker_preserves_demo_without_preparing_a_package(official_raw):
    paths, source, _ = official_raw
    assert storage.verified_public_demo_raw(paths.raw) == source
    scan = dataio.scan_path(str(paths.raw))
    assert scan["ok"] is True
    assert scan["db_key"] == "eicu_demo"
    assert scan["source"] == "prepared"
    assert scan["privacy"]["raw_rows_read"] is False
    assert not (paths.raw / "_manifest.json").exists()
    assert storage.egress_source_database(str(paths.raw), "eicu_demo") == "eicu_demo"
    with storage.public_demo_egress_scope(str(paths.raw)):
        assert client_trust._PUBLIC_DEMO_DATABASE.get() == "eicu_demo"
    assert client_trust._PUBLIC_DEMO_DATABASE.get() == ""


@pytest.mark.parametrize("defect", [
    "outside_cache", "raw_symlink", "missing_archive", "wrong_archive_size",
    "missing_extracted", "missing_converted", "forged_digest", "wrong_release",
    "marker_symlink", "missing_conversion", "malformed_conversion", "conversion_symlink",
    "wrong_database", "wrong_root", "empty_inventory", "partial", "changed_input",
    "changed_output", "extra_parquet", "output_symlink", "escaping_member",
    "wrong_schema_family",
])
def test_unproved_raw_identity_cannot_claim_public_demo(official_raw, tmp_path, defect):
    paths, _, manifest = official_raw
    root = paths.raw
    conversion = root / "conversion_manifest.json"
    if defect == "outside_cache":
        copied = tmp_path / "copied-demo"
        shutil.copytree(root, copied)
        root = copied
        # Even a self-consistent copy is not the app-owned official cache.
        manifest["data_path"] = str(copied)
        for table in manifest["tables"]:
            for record in [table["input"], *table["outputs"]]:
                record["path"] = str(copied / record["relative_path"])
        (copied / "conversion_manifest.json").write_text(json.dumps(manifest))
    elif defect == "raw_symlink":
        moved = tmp_path / "raw-moved"
        root.rename(moved)
        root.symlink_to(moved, target_is_directory=True)
    elif defect == "missing_archive":
        paths.archive.unlink()
    elif defect == "wrong_archive_size":
        paths.archive.write_bytes(b"wrong")
    elif defect in {"missing_extracted", "missing_converted"}:
        (paths.extracted_marker if defect == "missing_extracted" else paths.converted_marker).unlink()
    elif defect in {"forged_digest", "wrong_release"}:
        marker = json.loads(paths.converted_marker.read_text())
        marker["archive_sha256" if defect == "forged_digest" else "version"] = "wrong"
        paths.converted_marker.write_text(json.dumps(marker))
    elif defect in {"marker_symlink", "conversion_symlink", "output_symlink"}:
        path = {"marker_symlink": paths.converted_marker,
                "conversion_symlink": conversion,
                "output_symlink": root / "patient.parquet"}[defect]
        moved = tmp_path / "moved-file"
        path.rename(moved)
        path.symlink_to(moved)
    elif defect == "missing_conversion":
        conversion.unlink()
    elif defect == "malformed_conversion":
        conversion.write_text("{")
    elif defect in {"changed_input", "changed_output"}:
        path = root / ("patient.csv" if defect == "changed_input" else "patient.parquet")
        with path.open("ab") as handle:
            handle.write(b"changed")
    elif defect == "extra_parquet":
        shutil.copy2(root / "patient.parquet", root / "private.parquet")
    else:
        if defect == "wrong_database":
            manifest["database"] = "eicu"
        elif defect == "wrong_root":
            manifest["data_path"] = str(tmp_path)
        elif defect == "empty_inventory":
            manifest["tables"] = []
        elif defect == "partial":
            manifest["tables"][0]["bad_rows_skipped"] = 1
        elif defect == "escaping_member":
            manifest["tables"][0]["input"]["relative_path"] = "../patient.csv"
        elif defect == "wrong_schema_family":
            # Refresh the receipts after a table from another database replaces it.
            (root / "patient.csv").write_text("subject_id,anchor_age,anchor_year\n")
            pd.DataFrame({"subject_id": pd.Series(dtype="int64"),
                          "anchor_age": pd.Series(dtype="int64"),
                          "anchor_year": pd.Series(dtype="int64")}).to_parquet(root / "patient.parquet")
            for record in [manifest["tables"][0]["input"], *manifest["tables"][0]["outputs"]]:
                info = (root / record["relative_path"]).stat()
                record.update(size_bytes=info.st_size, mtime_ns=info.st_mtime_ns)
        conversion.write_text(json.dumps(manifest))
    assert storage.verified_public_demo_raw(root) is None
    with pytest.raises(client_trust.ProviderConfigurationError) as caught:
        storage.egress_source_database(str(root), "eicu_demo")
    assert caught.value.reason_code == "public_demo_manifest_invalid"
    with storage.public_demo_egress_scope(str(root)):
        assert client_trust._PUBLIC_DEMO_DATABASE.get() == ""


def test_a_standalone_conversion_label_cannot_change_full_eicu_to_demo(tmp_path):
    (tmp_path / "patient.csv").write_text("patientunitstayid,uniquepid\n")
    (tmp_path / "conversion_manifest.json").write_text(json.dumps({
        "schema_version": "easyicu.conversion_manifest/1", "database": "eicu_demo",
        "data_path": str(tmp_path), "tables": [],
    }))
    assert dataio._detect_database(tmp_path) == "eicu"
