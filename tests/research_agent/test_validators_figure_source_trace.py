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


# --- ordinal dose-response: ``stage`` is a valid figure->upstream trace key ----
# Root cause (2026-07-07, E3): the ordinal runner writes dose_response.csv with a
# ``stage`` column (graded-exposure levels 0..K), and the figure renderer carries
# ``stage`` verbatim into publication_figure_source_data.csv. ``stage`` was absent
# from _KEY_COLUMNS, so shared-key detection fell through to ``no_shared_key`` and
# the faithfully-derived ordinal forest (odds_ratio per stage identical to
# upstream) was rejected, deadlocking the run. Adding ``stage`` STRENGTHENS the
# gate: it now verifies each stage row traces to and matches an upstream row.


def _write_dose_response(tmp_path: Path) -> Path:
    up = tmp_path / "dose_response.csv"
    pd.DataFrame(
        {
            "stage": [0, 1, 2, 3],
            "n": [37433, 14061, 5200, 2100],
            "event_rate": [0.0572, 0.0981, 0.150, 0.240],
            "odds_ratio": [1.0, 1.5871617453700098, 2.51, 4.02],
            "or_ci_low": [1.0, 1.4771, 2.30, 3.60],
            "or_ci_high": [1.0, 1.7054, 2.74, 4.49],
        }
    ).to_csv(up, index=False)
    return up


def test_stage_is_a_recognized_trace_key(tmp_path: Path):
    up = _write_dose_response(tmp_path)
    source = pd.DataFrame(
        {
            "stage": [0, 1, 2, 3],
            "display_label": ["0", "1", "2", "3"],
            "odds_ratio": [1.0, 1.5871617453700098, 2.51, 4.02],
            "ci_low": [1.0, 1.4771, 2.30, 3.60],
            "ci_high": [1.0, 1.7054, 2.74, 4.49],
            "source_table": ["dose_response.csv"] * 4,
        }
    )
    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )
    assert res.get("ok") is True, res
    assert res.get("key_column") == "stage", res


def test_stage_figure_with_wrong_odds_ratio_still_flagged(tmp_path: Path):
    # Gate must NOT be weakened: a stage row whose odds_ratio disagrees with the
    # upstream table is a value-trace failure, even though the key aligns.
    up = _write_dose_response(tmp_path)
    source = pd.DataFrame(
        {
            "stage": [0, 1, 2, 3],
            "odds_ratio": [1.0, 9.99, 2.51, 4.02],  # stage 1 fabricated
            "source_table": ["dose_response.csv"] * 4,
        }
    )
    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )
    assert res.get("ok") is False
    assert res.get("reason") == "source_values_disagree", res


def test_stage_figure_with_phantom_stage_still_flagged(tmp_path: Path):
    # A stage value absent upstream must fail the subset check.
    up = _write_dose_response(tmp_path)
    source = pd.DataFrame(
        {
            "stage": [0, 1, 7],  # stage 7 never estimated
            "odds_ratio": [1.0, 1.5871617453700098, 5.0],
            "source_table": ["dose_response.csv"] * 3,
        }
    )
    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )
    assert res.get("ok") is False
    assert res.get("reason") == "source_rows_not_in_upstream", res
