"""Seal-tool helpers (tools/seal_buckets): footer scan, bucket compare containment, -0.0 table.

Each finding must exit non-zero and be named in the JSON; totals are checked against the manifest and
the build receipt. Rows are synthetic.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tools.seal_buckets import bucket_compare, footer_scan, negzero_table

TOOLS = Path(footer_scan.__file__).parent


def _bucket(root: Path, rows: int = 40, receipt_rows: int | None = None) -> Path:
    bucket = root / "t_bucket"
    inventory = []
    for b in range(2):
        (bucket / f"bucket_id={b}").mkdir(parents=True)
        table = pa.table(
            {
                "k": pa.array(np.arange(rows // 2) * 2 + b, pa.int64()),
                "v": pa.array(
                    np.where(np.arange(rows // 2) % 3 == 0, -0.0, 1.5), pa.float64()
                ),
            }
        )
        path = bucket / f"bucket_id={b}" / "part.parquet"
        pq.write_table(table, path, use_dictionary=False)
        inventory.append(
            {
                "relative_path": f"bucket_id={b}/part.parquet",
                "row_count": rows // 2,
                "size": path.stat().st_size,
            }
        )
    derived = {"files": [{"path": "t/1.parquet", "sha256": "0" * 64}]}
    (bucket / "_BUCKET_MANIFEST.json").write_text(
        json.dumps(
            {"inventory": inventory, "float_fidelity": {}, "derived_from": derived}
        )
    )
    (bucket / "_BUCKET_BUILD_RECEIPT.json").write_text(
        json.dumps({"rows": rows if receipt_rows is None else receipt_rows})
    )
    (bucket / "_COMPLETE").write_text("complete\n")
    return bucket


def _tool(name: str, *args: str):
    return subprocess.run(
        [sys.executable, "-I", str(TOOLS / name), *args], capture_output=True, text=True
    )


def test_footer_scan_passes_a_consistent_root(tmp_path):
    _bucket(tmp_path)
    out = tmp_path / "scan.json"
    result = _tool("footer_scan.py", str(tmp_path), "--out", str(out))
    assert result.returncode == 0, result.stdout
    scan = json.loads(out.read_text())
    assert (
        scan["ok"]
        and scan["buckets"]["t_bucket"]["rows_footer"]
        == 40
        == scan["buckets"]["t_bucket"]["rows_receipt"]
    )


def test_footer_scan_flags_receipt_totals_truncated_footers_and_missing_markers(
    tmp_path,
):
    bucket = _bucket(tmp_path, receipt_rows=41)
    part = bucket / "bucket_id=0" / "part.parquet"
    part.write_bytes(part.read_bytes()[:-10])
    (bucket / "_COMPLETE").unlink()
    out = tmp_path / "scan.json"
    result = _tool("footer_scan.py", str(tmp_path), "--out", str(out))
    assert result.returncode == 1
    reasons = {f["reason"] for f in json.loads(out.read_text())["failures"]}
    assert {
        "parquet_magic_missing",
        "bucket_rows_footer_vs_manifest",
        "bucket_rows_manifest_vs_receipt",
        "bucket_complete_marker_missing",
    } <= reasons


def test_compare_identical_twins_exit_zero(tmp_path):
    a, b = _bucket(tmp_path / "a"), _bucket(tmp_path / "b")
    result = _tool(
        "bucket_compare.py", str(a), str(b), "--out", str(tmp_path / "c.json")
    )
    assert result.returncode == 0, result.stdout
    assert json.loads((tmp_path / "c.json").read_text())["verdict"] == "IDENTICAL"


def test_compare_refuses_a_part_that_leads_out_of_its_root(tmp_path):
    a, b = _bucket(tmp_path / "a"), _bucket(tmp_path / "b")
    outside = tmp_path / "outside.parquet"
    part = b / "bucket_id=1" / "part.parquet"
    outside.write_bytes(part.read_bytes())
    part.unlink()
    os.symlink(outside, part)
    result = _tool(
        "bucket_compare.py", str(a), str(b), "--out", str(tmp_path / "c.json")
    )
    assert result.returncode == 1
    out = json.loads((tmp_path / "c.json").read_text())
    assert out["verdict"] == "DIFFERS"
    assert {
        "root": str(b),
        "path": "bucket_id=1/part.parquet",
        "reason": "path_escapes_root",
    } in out["containment_findings"]


def test_compare_flags_a_symlinked_directory_inside_a_bucket(tmp_path):
    a, b = _bucket(tmp_path / "a"), _bucket(tmp_path / "b")
    elsewhere = tmp_path / "elsewhere"
    (b / "bucket_id=1").rename(elsewhere)
    os.symlink(elsewhere, b / "bucket_id=1")
    result = _tool(
        "bucket_compare.py", str(a), str(b), "--out", str(tmp_path / "c.json")
    )
    assert result.returncode == 1
    reasons = {
        f["reason"]
        for f in json.loads((tmp_path / "c.json").read_text())["containment_findings"]
    }
    assert "symlinked_dir_in_bucket" in reasons


def test_compare_exit_codes_distinguish_usage_from_findings(tmp_path):
    assert _tool("bucket_compare.py").returncode == 2
    a, b = _bucket(tmp_path / "a"), _bucket(tmp_path / "b")
    part = b / "bucket_id=0" / "part.parquet"
    table = pq.read_table(part)
    pq.write_table(
        table, part, use_dictionary=False, compression="gzip"
    )  # same values, other bytes
    result = _tool(
        "bucket_compare.py", str(a), str(b), "--out", str(tmp_path / "c.json")
    )
    assert result.returncode == 1  # LOGICALLY_IDENTICAL is not a pass for twins
    assert json.loads((tmp_path / "c.json").read_text())["verdict"] in (
        "LOGICALLY_IDENTICAL",
        "DIFFERS",
    )


def _receipt(
    path: Path, db: str, neg: int, verdict: str = "PASS", sha: str = "a" * 64
) -> Path:
    path.write_text(
        json.dumps(
            {
                "db": db,
                "verdict": verdict,
                "tables": {
                    "t": {
                        "float_fidelity": {
                            "v": {
                                "values": 10,
                                "nulls": 0,
                                "neg_zero": neg,
                                "pos_zero": 2,
                                "nan": 0,
                                "sha256": sha,
                            }
                        }
                    }
                },
            }
        )
    )
    return path


def test_negzero_table_checks_build2_profiles_and_pending_rows(
    tmp_path, capsys, monkeypatch
):
    r1 = _receipt(tmp_path / "r1.json", "aumc", 3)
    same = _receipt(tmp_path / "r2.json", "aumc", 3)
    other = _receipt(tmp_path / "r3.json", "aumc", 3, sha="b" * 64)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["negzero_table.py", *args])
        return negzero_table.main()

    assert run(str(r1), "--build2", str(same), "--allow-pending") == 0
    assert (
        "| AUMCdb | t | v | 10 | 3 | 3 | — | 2 | 0 | equal |" in capsys.readouterr().out
    )
    assert run(str(r1), "--build2", str(other), "--allow-pending") == 1
    assert run(str(r1)) == 1  # the other four DBs are pending
    failed = _receipt(tmp_path / "r4.json", "aumc", 3, verdict="FAILED")
    assert run(str(failed), "--allow-pending") == 1


def test_footer_scan_module_is_metadata_only():
    text = Path(footer_scan.__file__).read_text()
    assert "read_metadata" in text and "read_table" not in text
    assert "containment_findings" in Path(bucket_compare.__file__).read_text()
