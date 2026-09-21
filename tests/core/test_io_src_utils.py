from __future__ import annotations

from easyicu.io.src_utils import (
    is_data_avail,
    is_tbl_avail,
    src_data_avail,
    src_prefix,
    src_tbl_avail,
)


def test_source_availability_uses_public_file_utilities(tmp_path) -> None:
    (tmp_path / "patient.parquet").write_bytes(b"fixture")

    assert src_data_avail("eicu", tmp_path)
    assert is_data_avail("eicu", tmp_path)
    assert src_tbl_avail("eicu", "patient", tmp_path)
    assert is_tbl_avail("eicu", "patient", tmp_path)


def test_source_metadata_uses_packaged_registry() -> None:
    assert src_prefix("mimic_demo") == ["mimic_demo", "mimic"]
