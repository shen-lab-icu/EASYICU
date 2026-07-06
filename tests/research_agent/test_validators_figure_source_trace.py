"""Regression: ``contrast_id`` is a valid figure→upstream trace key.

Root cause (2026-07-06, H2 fix3): a causal forest figure's
``publication_figure_source_data.csv`` and its upstream ``causal_effect.csv``
both key each estimated contrast by ``contrast_id``, but ``contrast_id`` was
absent from ``FigureSourceDataValidator._KEY_COLUMNS`` (only ``contrast`` was
present). The shared-key detection therefore fell through to ``no_shared_key``
and the faithfully-derived figure was rejected as "not a traceable subset",
blocking ``manuscript_ready``.

Fix: add ``contrast_id`` to the recognised key columns. This STRENGTHENS the
gate — it now verifies each contrast row traces to (and matches) an upstream
row — so a fabricated ``contrast_id`` must still be flagged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.research_agent.audits.validators import FigureSourceDataValidator


def _write_upstream(tmp_path: Path) -> Path:
    up = tmp_path / "causal_effect.csv"
    pd.DataFrame(
        {
            "contrast_id": ["primary_weighted_contrast"],
            "point_estimate": [2.79],
            "ci_low": [2.65],
            "ci_high": [2.95],
            "se": [0.03],
        }
    ).to_csv(up, index=False)
    return up


def test_contrast_id_is_a_recognized_trace_key(tmp_path: Path):
    up = _write_upstream(tmp_path)
    source = pd.DataFrame(
        {
            "contrast_id": ["primary_weighted_contrast"],
            "point_estimate": [2.79],
            "ci_low": [2.65],
            "ci_high": [2.95],
        }
    )
    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )
    assert res.get("ok") is True, res


def test_fabricated_contrast_id_still_flagged(tmp_path: Path):
    # Gate must NOT be weakened: a contrast_id absent upstream is a trace failure.
    up = _write_upstream(tmp_path)
    source = pd.DataFrame(
        {"contrast_id": ["ghost_contrast_not_estimated"], "point_estimate": [9.9]}
    )
    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )
    assert res.get("ok") is False
    assert res.get("reason") == "source_rows_not_in_upstream", res
