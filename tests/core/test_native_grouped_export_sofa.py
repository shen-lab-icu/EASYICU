"""Native SOFA consolidation preserves the documented component rule."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from easyicu.api import extraction as api


def _completed_result(module: str = "demographics") -> dict[str, object]:
    return {"modules": {module: {"errors": []}}}


@pytest.mark.parametrize("force_duckdb", [False, True])
def test_native_sofa1_recomputes_total_after_hour_key_consolidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    force_duckdb: bool,
) -> None:
    components = [
        "sofa_resp",
        "sofa_coag",
        "sofa_liver",
        "sofa_cardio",
        "sofa_cns",
        "sofa_renal",
    ]
    requested = ["sofa", *components]
    monkeypatch.setattr(api, "EXTRACT_MODULES", {"sofa1_score": requested})
    if force_duckdb:
        monkeypatch.setattr(
            api,
            "_native_export_pandas_fallback_is_bounded",
            lambda _size: False,
        )
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "charttime": [0.0, 0.0, 0.0],
            "sofa": [4.0, 4.0, 0.0],
            "sofa_resp": [4.0, 0.0, 0.0],
            "sofa_coag": [0.0, 4.0, 0.0],
            "sofa_liver": [None, None, None],
            "sofa_cardio": [0.0, 0.0, 0.0],
            "sofa_cns": [0.0, 0.0, 0.0],
            "sofa_renal": [0.0, 0.0, 0.0],
        }
    )
    # These synthetic same-hour states are already callback-windowed organs.
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata({
        **(table.schema.metadata or {}),
        api._SOFA1_TIME_BASIS_KEY: api._SOFA1_TIME_BASIS,
    })
    pq.write_table(table, tmp_path / "sofa1_score.parquet")

    api._publish_native_export_v2(
        database="eicu",
        data_path="/raw/source-must-not-be-read",
        output_dir=str(tmp_path),
        modules=["sofa1_score"],
        max_patients=None,
        result=_completed_result("sofa1_score"),
    )

    published = pd.read_parquet(tmp_path / "sofa1_score.parquet")
    assert len(published) == 1
    assert published.loc[0, "sofa"] == 8.0
    assert published.loc[0, components].fillna(0).sum() == published.loc[0, "sofa"]
    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    policy = manifest["files"][0]["row_grain_audit"]["aggregation_policy"]
    assert policy["score_component"] == "max_non_null_worst_state"
    assert policy["derived_score_total"] == (
        "recomputed_after_component_consolidation"
    )
