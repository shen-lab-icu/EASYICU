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


# --- cross-step declared source_table resolution --------------------------------
# Root cause (2026-07-08, E2): the primary association forest figure step
# (06_primary_association_model_figure) is built from a table produced by a
# DIFFERENT step (the per-level ORs live in 00_probe/lactate_group_odds_ratios.csv,
# not in the _figure-suffix sibling 06_primary_association_model). _upstream_step_ids
# only resolves the sibling step, so the true parent was never a comparison
# candidate and the (byte-identical) figure was rejected as "no shared key" against
# an unrelated sibling table — driving the run to replan exhaustion. The validator
# now also resolves the figure's self-declared ``source_table`` filenames across the
# whole run. This ADDS candidates only; the value-equality checks still guard
# fabrication.


def _write_cross_step_run(tmp_path: Path):
    """A run where the figure's declared parent lives in a non-sibling step."""
    steps = tmp_path / "steps"
    probe_out = steps / "00_probe" / "outputs"
    probe_out.mkdir(parents=True)
    pd.DataFrame(
        {
            "level": ["<2", ">=4"],
            "odds_ratio": [0.7610485449307591, 4.377726177511688],
            "ci_low": [0.7130845733975404, 4.105143972875267],
            "ci_high": [0.8122387011986698, 4.668407883353277],
        }
    ).to_csv(probe_out / "lactate_group_odds_ratios.csv", index=False)
    # The _figure-suffix sibling exists but holds only an UNRELATED table.
    sib_out = steps / "06_primary_association_model" / "outputs"
    sib_out.mkdir(parents=True)
    pd.DataFrame({"stage": ["measured", "unmeasured"], "n": [94458, 6034]}).to_csv(
        sib_out / "complete_case_attrition.csv", index=False
    )
    fig_out = steps / "06_primary_association_model_figure" / "outputs"
    fig_out.mkdir(parents=True)
    pd.DataFrame(
        {
            "level": ["<2", ">=4"],
            "odds_ratio": [0.7610485449307591, 4.377726177511688],
            "ci_low": [0.7130845733975404, 4.105143972875267],
            "ci_high": [0.8122387011986698, 4.668407883353277],
            "source_table": ["lactate_group_odds_ratios.csv"] * 2,
        }
    ).to_csv(fig_out / "publication_figure_source_data.csv", index=False)
    return fig_out


def _fig_step():
    from easyicu.research_agent.schema import AnalysisStep

    return AnalysisStep(
        step_id="06_primary_association_model_figure",
        intent="Render the primary association forest figure",
        method="figure",
    )


def test_declared_parent_in_other_step_is_resolved(tmp_path: Path):
    fig_out = _write_cross_step_run(tmp_path)
    findings = FigureSourceDataValidator().audit(
        step=_fig_step(), out_dir=fig_out, run_dir=tmp_path, step_summary={}
    )
    errors = [f for f in findings if f.severity == "error"]
    assert errors == [], [f.message for f in findings]


def test_cross_step_resolution_still_flags_fabrication(tmp_path: Path):
    # Gate must NOT be weakened: a figure that names a real parent but whose
    # values disagree with it is still a value-trace failure.
    fig_out = _write_cross_step_run(tmp_path)
    df = pd.read_csv(fig_out / "publication_figure_source_data.csv")
    df.loc[0, "odds_ratio"] = 99.9  # fabricated, not in the declared parent
    df.to_csv(fig_out / "publication_figure_source_data.csv", index=False)
    findings = FigureSourceDataValidator().audit(
        step=_fig_step(), out_dir=fig_out, run_dir=tmp_path, step_summary={}
    )
    errors = [f for f in findings if f.severity == "error"]
    assert errors, "tampered figure must still be flagged"


# --- fix #2 (2026-07-08): structural join fallback for unregistered key names ---
# A faithfully-derived figure often preserves the parent's OWN key column under a
# name absent from _KEY_COLUMNS (group / category_code / lactate_group). Rather
# than grow the allowlist per case (the gate-allowlist anti-pattern), the
# validator now accepts ANY shared, non-numeric, identifier-like column as the
# join key. Value-equality still runs, so fabrication is still caught.


def test_unregistered_identifier_column_resolves_join(tmp_path: Path):
    # `category_code` is NOT in _KEY_COLUMNS; the structural fallback must still
    # join on it because it is shared, non-numeric and per-row identifying.
    up = tmp_path / "lactate_group_odds_ratios.csv"
    pd.DataFrame(
        {
            "category_code": ["unmeasured", "lt2", "2to4", "ge4"],
            "odds_ratio": [1.0, 1.42, 2.05, 3.31],
            "ci_low": [1.0, 1.30, 1.90, 3.02],
            "ci_high": [1.0, 1.55, 2.21, 3.63],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "category_code": ["unmeasured", "lt2", "2to4", "ge4"],
            "group": ["Unmeasured", "<2", "2-<4", ">=4"],  # figure label, not a trace key
            "odds_ratio": [1.0, 1.42, 2.05, 3.31],
            "ci_low": [1.0, 1.30, 1.90, 3.02],
            "ci_high": [1.0, 1.55, 2.21, 3.63],
        }
    )
    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )
    assert res.get("ok") is True, res


def test_fallback_key_still_flags_fabricated_value(tmp_path: Path):
    # ADVERSARIAL: join resolves on the unregistered `category_code`, but a
    # tampered odds_ratio must STILL fail-close (value-equality runs post-join).
    up = tmp_path / "lactate_group_odds_ratios.csv"
    pd.DataFrame(
        {
            "category_code": ["unmeasured", "lt2", "2to4", "ge4"],
            "odds_ratio": [1.0, 1.42, 2.05, 3.31],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "category_code": ["unmeasured", "lt2", "2to4", "ge4"],
            "odds_ratio": [1.0, 1.42, 2.05, 99.9],  # tampered last row
        }
    )
    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )
    assert res.get("ok") is False
    assert res.get("reason") == "source_values_disagree", res


def test_numeric_only_shared_column_not_used_as_key(tmp_path: Path):
    # A shared column that is fully numeric in both frames is a VALUE, not a key;
    # the fallback must not join on it (that would fabricate a spurious match).
    up = tmp_path / "some_table.csv"
    pd.DataFrame({"odds_ratio": [1.0, 1.42, 2.05, 3.31]}).to_csv(up, index=False)
    source = pd.DataFrame({"odds_ratio": [1.0, 1.42, 2.05, 3.31]})
    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )
    assert res.get("ok") is False
    assert res.get("reason") == "no_shared_key", res
