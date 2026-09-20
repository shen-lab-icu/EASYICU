from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from scripts.build_itemid_bucket_cache import build_bucket_cache


def test_build_bucket_cache_preserves_rows_and_installs_atomically(
    tmp_path: Path,
) -> None:
    source = tmp_path / "numericitems"
    source.mkdir()
    expected = pd.DataFrame(
        {
            "admissionid": [1, 1, 2, 3, 3],
            "itemid": [10, 11, 10, 12, 13],
            "value": [1.0, 2.0, None, 4.0, 5.0],
        }
    )
    expected.iloc[:3].to_parquet(source / "1.parquet", index=False)
    expected.iloc[3:].to_parquet(source / "2.parquet", index=False)

    receipt = build_bucket_cache(
        tmp_path,
        table="numericitems",
        key="itemid",
        bucket_count=4,
        memory_limit="256MB",
        threads=1,
        verify_row_hash=True,
    )

    target = tmp_path / "numericitems_bucket"
    assert target.is_dir()
    assert not (tmp_path / ".numericitems_bucket_build").exists()
    assert (target / "_COMPLETE").read_text(encoding="utf-8") == "complete\n"
    stored_receipt = json.loads(
        (target / "_BUCKET_BUILD_RECEIPT.json").read_text(encoding="utf-8")
    )
    assert stored_receipt == receipt
    assert receipt["rows"] == len(expected)
    assert receipt["source_files"] == 2
    assert receipt["bucket_count"] == 4
    assert receipt["row_multiset_fingerprint"]["rows"] == len(expected)

    actual = duckdb.sql(
        f"""
        SELECT * EXCLUDE(bucket_id)
        FROM read_parquet(
          '{target.as_posix()}/bucket_id=*/*.parquet',
          hive_partitioning=true,
          union_by_name=true
        )
        ORDER BY admissionid, itemid
        """
    ).df()
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.sort_values(["admissionid", "itemid"]).reset_index(drop=True),
        check_dtype=False,
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_bucket_cache(
            tmp_path,
            table="numericitems",
            key="itemid",
            bucket_count=4,
        )
