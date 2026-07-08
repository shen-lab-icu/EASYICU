"""Unit tests for the deterministic source-data projection (WS1 / #53 A).

These lock the guarantee that a *primary/result* figure's source-data table,
built by ``project_source_data`` as a verbatim projection of the upstream
analysis output, passes the REAL ``FigureSourceDataValidator`` trace check by
construction — and that the projector fails loudly (never silently) when a key
is missing or non-traceable. A negative control confirms the projection does
not bypass the gate: a corrupted value is still caught.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.audits.validators import FigureSourceDataValidator
from easyicu.research_agent.projection import (
    ProjectionError,
    project_source_data,
)


def _ordinal_upstream() -> pd.DataFrame:
    # dose_response.csv shape: one row per KDIGO stage (E3), OR/CI per stage.
    return pd.DataFrame(
        {
            "stage": [0, 1, 2, 3],
            "odds_ratio": [1.0, 1.42, 2.05, 3.10],
            "ci_low": [1.0, 1.28, 1.80, 2.55],
            "ci_high": [1.0, 1.58, 2.34, 3.77],
            "point_estimate": [1.0, 1.42, 2.05, 3.10],
        }
    )


def _trace(source_csv: Path, upstream_csv: Path) -> dict:
    src_df = pd.read_csv(source_csv)
    return FigureSourceDataValidator._compare_source_to_upstream(
        source_df=src_df, source_path=source_csv, upstream_path=upstream_csv
    )


def test_project_source_data_is_verbatim_subset(tmp_path: Path):
    up = _ordinal_upstream()
    up_csv = tmp_path / "dose_response.csv"
    up.to_csv(up_csv, index=False)
    out_csv = tmp_path / "publication_figure_source_data.csv"

    res = project_source_data(
        upstream_frame=up,
        upstream_path=up_csv,
        key_columns=["stage"],
        value_columns=["odds_ratio", "ci_low", "ci_high"],
        out_csv=out_csv,
    )
    assert res.key_columns == ("stage",)
    assert res.n_rows == 4

    verdict = _trace(out_csv, up_csv)
    assert verdict.get("ok") is True, verdict


def test_project_carries_source_table_provenance(tmp_path: Path):
    up = _ordinal_upstream()
    up_csv = tmp_path / "dose_response.csv"
    up.to_csv(up_csv, index=False)
    out_csv = tmp_path / "src.csv"
    project_source_data(
        upstream_frame=up,
        upstream_path=up_csv,
        key_columns=["stage"],
        value_columns=["odds_ratio"],
        out_csv=out_csv,
    )
    got = pd.read_csv(out_csv)
    assert list(got["source_table"].unique()) == ["dose_response.csv"]


def test_project_raises_on_missing_key(tmp_path: Path):
    up = _ordinal_upstream().drop(columns=["stage"])
    with pytest.raises(ProjectionError, match="not present"):
        project_source_data(
            upstream_frame=up,
            upstream_path=tmp_path / "u.csv",
            key_columns=["stage"],
            value_columns=["odds_ratio"],
            out_csv=tmp_path / "o.csv",
        )


def test_project_raises_on_non_whitelisted_key(tmp_path: Path):
    up = _ordinal_upstream().rename(columns={"stage": "my_custom_id"})
    with pytest.raises(ProjectionError, match="not a recognised figure trace key"):
        project_source_data(
            upstream_frame=up,
            upstream_path=tmp_path / "u.csv",
            key_columns=["my_custom_id"],
            value_columns=["odds_ratio"],
            out_csv=tmp_path / "o.csv",
        )


def test_project_does_not_leak_earlier_ordered_key(tmp_path: Path):
    # upstream has BOTH "label" (earlier in _KEY_COLUMNS) and "stage"; if the
    # projection leaked "label" the validator would key on it instead of stage.
    up = _ordinal_upstream()
    up["label"] = ["s0", "s1", "s2", "s3"]
    up_csv = tmp_path / "dose_response.csv"
    up.to_csv(up_csv, index=False)
    out_csv = tmp_path / "src.csv"

    project_source_data(
        upstream_frame=up,
        upstream_path=up_csv,
        key_columns=["stage"],
        value_columns=["odds_ratio", "ci_low", "ci_high"],
        out_csv=out_csv,
    )
    got = pd.read_csv(out_csv)
    assert "label" not in got.columns
    # the intended key wins and the trace still passes
    assert _trace(out_csv, up_csv).get("ok") is True


def test_project_value_column_that_is_reserved_key_raises(tmp_path: Path):
    up = _ordinal_upstream()
    up["term"] = ["a", "b", "c", "d"]  # 'term' is a _KEY_COLUMNS member
    with pytest.raises(ProjectionError, match="reserved trace-key"):
        project_source_data(
            upstream_frame=up,
            upstream_path=tmp_path / "u.csv",
            key_columns=["stage"],
            value_columns=["odds_ratio", "term"],
            out_csv=tmp_path / "o.csv",
        )


def test_project_display_column_collision_raises(tmp_path: Path):
    up = _ordinal_upstream()
    with pytest.raises(ProjectionError, match="collides with an upstream column"):
        project_source_data(
            upstream_frame=up,
            upstream_path=tmp_path / "u.csv",
            key_columns=["stage"],
            value_columns=["odds_ratio"],
            out_csv=tmp_path / "o.csv",
            extra_display_columns={"odds_ratio": [9, 9, 9, 9]},
        )


def test_projection_does_not_bypass_gate_negative_control(tmp_path: Path):
    # Project correctly, then corrupt one value: the trace gate must still catch
    # it (proves the projection makes CORRECT figures pass, not ALL figures).
    up = _ordinal_upstream()
    up_csv = tmp_path / "dose_response.csv"
    up.to_csv(up_csv, index=False)
    out_csv = tmp_path / "src.csv"
    project_source_data(
        upstream_frame=up,
        upstream_path=up_csv,
        key_columns=["stage"],
        value_columns=["odds_ratio", "ci_low", "ci_high"],
        out_csv=out_csv,
    )
    tampered = pd.read_csv(out_csv)
    tampered.loc[2, "odds_ratio"] = 9.99  # wrong value for stage=2
    tampered.to_csv(out_csv, index=False)
    verdict = _trace(out_csv, up_csv)
    assert verdict.get("ok") is not True
