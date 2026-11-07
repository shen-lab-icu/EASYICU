"""R13 formal receipt chain on the shapes the seal tool and readers really write.

* R13-FARM-LINK: a farm receipt written by the seal tool (``lead_r13.seal_farm_receipt/2``)
  records each ``bucket_links`` entry as ``{kind, source, target, contract}``; formal acceptance
  and the install closure follow its ``target`` exactly as an older plain-path entry.
* R13-FARM-TABLES: a farm's bucket-only table is absent from its conversion manifest; the formal
  input snapshot still validates its bucket because the farm's PASS receipt binds it.
* R13-READ-LOG: every consumed flat/CSV file is also a logged table read, so the receipt's
  ``storage_reads`` names each consumed table (HiRID ``observation_tables_index``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu import formal_inputs
from easyicu.storage_layout import (
    FORMAT_CSV,
    FORMAT_PARQUET,
    SCAN_FALLBACK_LOG_ENV,
    read_consumed_log,
    read_table_read_log,
    record_consumed,
)
from easyicu.storage_seal_publication import (
    IN_ROOT_RECEIPT,
    SealPublicationError,
    assert_install_published,
    assert_seal_published,
    sealed_bucket_tables,
    sealed_link_target,
)
from tests.core.test_storage_seal_publication import _bucket, _receipt, _sealed


def _farm_entry(target: Path, source: Path) -> dict:
    return {"kind": "rebuilt", "source": str(source), "target": str(target),
            "contract": {"manifest_sha256": "0" * 64}}


def _farm(tmp_path: Path, *, entry=None, link_to=None):
    b6 = tmp_path / "b6_root"
    sealed = b6 / "mimic_cv_bucket_sX"
    bucket = _sealed(sealed)
    farm = b6 / "mimic_cv_bkt_sX"
    farm.mkdir()
    (farm / "chartevents_bucket").symlink_to(link_to or bucket, target_is_directory=True)
    _receipt(
        farm,
        schema="lead_r13.seal_farm_receipt/2",
        bucket_links={"chartevents_bucket": entry if entry is not None
                      else _farm_entry(bucket, b6 / "mimic_cv_dir" / "chartevents_bucket")},
        bucket_root=str(sealed),
    )
    return farm, bucket, sealed


def test_link_target_reads_both_receipt_shapes():
    assert sealed_link_target("/x/chartevents_bucket") == "/x/chartevents_bucket"
    assert sealed_link_target({"kind": "rebuilt", "target": "/x/c_bucket"}) == "/x/c_bucket"
    for bad in (None, "", {}, {"target": ""}, {"target": 3}, ["/x"], 7):
        assert sealed_link_target(bad) is None


def test_formal_accepts_a_seal_tool_farm_receipt(tmp_path):
    farm, _bucket_root, _sealed_root = _farm(tmp_path)
    assert_seal_published(farm / "chartevents_bucket", formal=True)
    assert_install_published(farm, formal=True)


def test_formal_refuses_a_dict_link_whose_target_is_elsewhere(tmp_path):
    b6 = tmp_path / "b6_root"
    stray = _sealed(b6 / "mimic_cv_old")
    farm, bucket, _ = _farm(tmp_path, link_to=None)
    (farm / "chartevents_bucket").unlink()
    (farm / "chartevents_bucket").symlink_to(stray, target_is_directory=True)
    with pytest.raises(SealPublicationError) as caught:
        assert_seal_published(farm / "chartevents_bucket", formal=True)
    assert caught.value.reason == "bucket_install_unaccepted"


@pytest.mark.parametrize("entry", [{"kind": "rebuilt"}, {"target": None}, 5])
def test_formal_refuses_a_link_entry_that_names_no_target(tmp_path, entry):
    farm, _, _ = _farm(tmp_path, entry=entry)
    with pytest.raises(SealPublicationError) as caught:
        assert_seal_published(farm / "chartevents_bucket", formal=True)
    assert caught.value.reason == "bucket_install_unaccepted"


def test_sealed_bucket_tables_follow_the_passing_receipt(tmp_path):
    farm, _, sealed = _farm(tmp_path)
    assert sealed_bucket_tables(farm) == ["chartevents"]
    assert sealed_bucket_tables(sealed) == ["chartevents"]
    assert sealed_bucket_tables(tmp_path / "nowhere") == []
    receipt = json.loads((farm / IN_ROOT_RECEIPT).read_text())
    receipt["verdict"] = "FAIL"
    (farm / IN_ROOT_RECEIPT).write_text(json.dumps(receipt))
    assert sealed_bucket_tables(farm) == []


def test_snapshot_registers_a_farm_bucket_only_table(tmp_path):
    farm, _, _ = _farm(tmp_path)
    (farm / "conversion_manifest.json").write_text(json.dumps(
        {"tables": [{"table": "d_items", "outputs": []}]}))
    assert "chartevents" in [t.lower() for t in formal_inputs._bucket_tables(farm)]
    receipt = json.loads((farm / IN_ROOT_RECEIPT).read_text())
    receipt["verdict"] = "FAIL"
    (farm / IN_ROOT_RECEIPT).write_text(json.dumps(receipt))
    assert formal_inputs._bucket_tables(farm) == ["d_items"]


@pytest.mark.parametrize("fmt, name, layout", [
    (FORMAT_CSV, "observation_tables_index.csv", "csv"),
    (FORMAT_PARQUET, "general.parquet", "flat"),
])
def test_every_consumed_flat_or_csv_file_is_a_logged_read(tmp_path, monkeypatch, fmt, name, layout):
    root = tmp_path / "hirid"
    (root / "observation_tables").mkdir(parents=True)
    path = root / "observation_tables" / name
    if fmt == FORMAT_CSV:
        path.write_text("patientid,part\n1,0\n")
    else:
        pd.DataFrame({"patientid": [1]}).to_parquet(path)
    log = tmp_path / "run.log"
    monkeypatch.setenv(SCAN_FALLBACK_LOG_ENV, str(log))
    table = name.split(".")[0]
    record_consumed(root, table, (path,), fmt)
    consumed = {e["table"] for e in read_consumed_log(log)}
    reads = read_table_read_log(log)
    assert consumed == {table}
    assert [(e["table"], e["layout"], e["layout_path"]) for e in reads] == [
        (table, layout, str(path.parent))
    ]


def test_bucket_formats_add_no_flat_read(tmp_path, monkeypatch):
    root = tmp_path / "miiv"
    bucket = _bucket(root)
    log = tmp_path / "run.log"
    monkeypatch.setenv(SCAN_FALLBACK_LOG_ENV, str(log))
    from easyicu.storage_layout import FORMAT_BUCKET_PART

    record_consumed(root, "chartevents", (bucket / "bucket_id=0" / "1.parquet",), FORMAT_BUCKET_PART)
    assert read_table_read_log(log) == ()
