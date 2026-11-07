"""Real builder and seal QC on synthetic no-float and float tables."""

import json
import subprocess

import pandas as pd
import pytest

from scripts import build_itemid_bucket_cache as builder
from tools.seal_buckets import seal_rebuild as seal


def _build(tmp, monkeypatch, *, floating):
    tmp.mkdir()
    full = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    # Model the clean seal clone's identity; leave the real builder and QC intact.
    monkeypatch.setattr(builder, "builder_code", lambda: {
        "source": "git", "commit": full, "dirty": False,
    })
    frame = pd.DataFrame({"patientunitstayid": range(100, 120),
                          "nursingchartcelltypevallabel": ["Heart Rate"] * 20,
                          "nursingchartvalue": ["60"] * 20})
    if floating:
        frame["float_value"] = [60.25] * 20
    frame.to_parquet(tmp / "nursecharting.parquet", index=False)
    builder.build_bucket_cache(
        tmp, table="nursecharting", key="patientunitstayid", bucket_count=2,
        memory_limit="256MB", threads=1,
    )
    return tmp / "nursecharting_bucket", full


def _healthy_nofloat(tmp, monkeypatch):
    bucket, full = _build(tmp, monkeypatch, floating=False)
    manifest = json.loads((bucket / "_BUCKET_MANIFEST.json").read_text())
    assert manifest["float_fidelity"]["columns"] == {}
    assert "rows" not in manifest["float_fidelity"]  # Old builder records remain valid.
    result = seal.qc_bucket(bucket, full)
    assert result["rows"] == 20
    assert result["qc_findings"] == []


def test_real_no_float_build_passes_seal_qc(tmp_path, monkeypatch):
    _healthy_nofloat(tmp_path / "no_float", monkeypatch)


@pytest.mark.parametrize("defect", ["mismatched_rows", "false_empty_columns"])
def test_float_fidelity_remains_strict_after_no_float_acceptance(tmp_path, monkeypatch, defect):
    _healthy_nofloat(tmp_path / "no_float", monkeypatch)
    bucket, full = _build(tmp_path / "floating", monkeypatch, floating=True)
    assert seal.qc_bucket(bucket, full)["qc_findings"] == []
    for name in ("_BUCKET_MANIFEST.json", "_BUCKET_BUILD_RECEIPT.json"):
        path = bucket / name
        payload = json.loads(path.read_text())
        if defect == "mismatched_rows":
            payload["float_fidelity"]["rows"] += 1
        else:
            payload["float_fidelity"]["columns"] = {}
            payload["float_fidelity"]["dictionary_encoding"] = "default"
            payload["float_fidelity"].pop("rows")
        path.write_text(json.dumps(payload))
    with pytest.raises(seal.SealError) as caught:
        seal.qc_bucket(bucket, full)
    if defect == "mismatched_rows":
        assert caught.value.reason == "float_fidelity_rows_vs_receipt"
    else:
        assert "float_fidelity_columns_mismatch" in caught.value.record["qc_findings"]
        assert "float_fidelity_rows_vs_receipt" in caught.value.record["qc_findings"]
