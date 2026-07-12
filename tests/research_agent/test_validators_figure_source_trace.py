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
import pytest

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


def test_model_id_term_composite_key_disambiguates_shared_terms_and_flags_drift(
    tmp_path: Path,
):
    """Repeated terms trace within their model and still fail on local drift."""

    upstream = tmp_path / "coefficients.csv"
    pd.DataFrame(
        {
            "model_id": [
                "bili_source_aware_full",
                "bili_complete_case_measured",
            ],
            "term": ["bili_log1p", "bili_log1p"],
            "odds_ratio": [1.9259885602397633, 1.8824854001320446],
            "ci_low": [1.8555496206064033, 1.8134563251295070],
            "ci_high": [1.9991014484226914, 1.9541420615449500],
        }
    ).to_csv(upstream, index=False)
    source = pd.DataFrame(
        {
            "model_id": [
                "bili_source_aware_full",
                "bili_complete_case_measured",
            ],
            "term": ["bili_log1p", "bili_log1p"],
            "odds_ratio": [1.9259885602397633, 1.8824854001320446],
            "ci_low": [1.8555496206064033, 1.8134563251295070],
            "ci_high": [1.9991014484226914, 1.9541420615449500],
        }
    )

    clean = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=upstream,
    )
    assert clean.get("ok") is True, clean
    assert clean.get("key_column") == "model_id+term", clean

    tampered = source.copy()
    tampered.loc[
        tampered["model_id"].eq("bili_complete_case_measured"), "odds_ratio"
    ] = 9.99
    drift = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=tampered,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=upstream,
    )
    assert drift.get("ok") is False, drift
    assert drift.get("reason") == "source_values_disagree", drift
    assert drift.get("key_column") == "model_id+term", drift
    assert [item["key"] for item in drift["mismatches"]] == [
        "bili_complete_case_measured|bili_log1p"
    ]


def test_six_decimal_percentage_rounding_remains_traceable(tmp_path: Path):
    """A count-derived full-precision percentage matches its 6-dp parent."""

    upstream = tmp_path / "missingness_measurement_audit.csv"
    pd.DataFrame(
        {
            "concept": ["exposure_signal"],
            "missing_n": [40754],
            "measured_n": [34075],
            "n_total": [74829],
            "missing_pct": [54.462842],
            "measured_pct": [45.537158],
        }
    ).to_csv(upstream, index=False)
    source = pd.DataFrame(
        {
            "concept": ["exposure_signal"],
            "missing_n": [40754],
            "measured_n": [34075],
            "n_total": [74829],
            "missing_pct": [100.0 * 40754 / 74829],
            "measured_pct": [100.0 * 34075 / 74829],
        }
    )

    result = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "missingness_measurement_panel_source_data.csv",
        upstream_path=upstream,
    )

    assert result.get("ok") is True, result


def test_percentage_drift_beyond_rounding_tolerance_is_still_flagged(
    tmp_path: Path,
):
    upstream = tmp_path / "missingness_measurement_audit.csv"
    pd.DataFrame(
        {
            "concept": ["exposure_signal"],
            "missing_n": [40754],
            "n_total": [74829],
            "missing_pct": [54.462842],
        }
    ).to_csv(upstream, index=False)
    source = pd.DataFrame(
        {
            "concept": ["exposure_signal"],
            "missing_n": [40754],
            "n_total": [74829],
            "missing_pct": [54.462844],
        }
    )

    result = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "missingness_measurement_panel_source_data.csv",
        upstream_path=upstream,
    )

    assert result.get("ok") is False, result
    assert result.get("reason") == "source_values_disagree", result
    mismatch = result["mismatches"][0]
    assert mismatch["column"] == "missing_pct"
    assert mismatch["abs_diff"] > mismatch["abs_tolerance"] == 1e-6


def test_generic_figure_percentage_count_denominator_drift_is_error(
    tmp_path: Path,
):
    source = pd.DataFrame(
        {
            "stage": [0, 1],
            "count": [37433, 14061],
            # These percentages use a locked-cohort denominator of 74,829...
            "percentage": [100.0 * 37433 / 74829, 100.0 * 14061 / 74829],
            # ...but the renderer incorrectly paired them with valid-observed n.
            "denominator": [74708, 74708],
        }
    )
    findings = FigureSourceDataValidator._percentage_count_consistency_findings(
        source_df=source,
        source_path=tmp_path / "stage_distribution_source_data.csv",
        step_id="stage_distribution_figure",
    )

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail["pct_column"] == "percentage"
    assert findings[0].detail["count_column"] == "count"
    assert findings[0].detail["total_column"] == "denominator"


def test_generic_figure_percentage_count_denominator_consistency_passes(
    tmp_path: Path,
):
    source = pd.DataFrame(
        {
            "stage": [0, 1],
            "count": [37433, 14061],
            "percentage": [100.0 * 37433 / 74708, 100.0 * 14061 / 74708],
            "denominator": [74708, 74708],
        }
    )
    findings = FigureSourceDataValidator._percentage_count_consistency_findings(
        source_df=source,
        source_path=tmp_path / "stage_distribution_source_data.csv",
        step_id="stage_distribution_figure",
    )

    assert findings == []


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


def test_fallback_key_blocks_when_no_value_column_can_be_verified(tmp_path: Path):
    up = tmp_path / "outcome_by_group.csv"
    pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "mortality_rate": [0.10, 0.20, 0.30],
            "n": [100, 80, 60],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "estimate": [0.91, 0.92, 0.93],
            "source_table": [up.name] * 3,
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert res.get("reason") == "no_verifiable_values", res


def test_fallback_key_blocks_key_only_projection_without_any_value_check(
    tmp_path: Path,
):
    up = tmp_path / "outcome_by_group.csv"
    pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "mortality_rate": [0.10, 0.20, 0.30],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "source_table": [up.name] * 3,
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert res.get("reason") == "no_verifiable_values", res
    assert res.get("verified_value_mappings") == {}, res


def test_truthful_shared_count_cannot_launder_a_renamed_forged_estimate(
    tmp_path: Path,
):
    up = tmp_path / "outcome_by_group.csv"
    pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "mortality_rate": [0.10, 0.20, 0.30],
            "n": [100, 80, 60],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "estimate": [0.91, 0.92, 0.93],
            "n": [100, 80, 60],
            "source_table": [up.name] * 3,
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert res.get("reason") == "no_verifiable_values", res
    assert "estimate" in res.get("unverified_source_value_columns", []), res
    assert "n" in res.get("verified_source_value_columns", []), res


def test_fallback_key_still_accepts_truthful_shared_numeric_values(tmp_path: Path):
    up = tmp_path / "outcome_by_group.csv"
    pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "mortality_rate": [0.10, 0.20, 0.30],
            "n": [100, 80, 60],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "mortality_rate": [0.10, 0.20, 0.30],
            "source_table": [up.name] * 3,
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is True, res
    assert res.get("key_column") == "group", res


def test_named_key_blocks_renamed_unverified_value(tmp_path: Path):
    up = tmp_path / "outcome_by_stage.csv"
    pd.DataFrame(
        {
            "stage": [0, 1, 2],
            "mortality_rate": [0.10, 0.20, 0.30],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "stage": [0, 1, 2],
            "estimate": [0.91, 0.92, 0.93],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert res.get("reason") == "no_verifiable_values", res
    assert res.get("unverified_source_value_columns") == ["estimate"], res


@pytest.mark.parametrize("position_col", ["source_row_index", "_source_row_index"])
def test_source_row_index_blocks_renamed_unverified_value(
    tmp_path: Path, position_col: str
):
    up = tmp_path / "outcome_by_row.csv"
    pd.DataFrame({"mortality_rate": [0.10, 0.20, 0.30]}).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            position_col: [0, 1, 2],
            "estimate": [0.91, 0.92, 0.93],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert res.get("reason") == "no_verifiable_values", res


@pytest.mark.parametrize("position_col", ["source_row_index", "_source_row_index"])
def test_source_row_index_alias_accepts_truthful_parent_projection(
    tmp_path: Path, position_col: str
):
    up = tmp_path / "outcome_by_row.csv"
    upstream = pd.DataFrame(
        {
            "n": [100, 80, 60],
            "event_n": [10, 16, 18],
            "risk": [0.10, 0.20, 0.30],
        }
    )
    upstream.to_csv(up, index=False)
    source = upstream.copy()
    source[position_col] = [0, 1, 2]
    source["source_table"] = up.name
    source["plot_status"] = "plot"

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is True, res
    assert res.get("key_column") == position_col, res
    assert {"n", "event_n", "risk"} <= set(
        res.get("verified_value_mappings", {})
    )


def test_underscore_source_row_index_still_flags_tampered_value(tmp_path: Path):
    up = tmp_path / "outcome_by_row.csv"
    pd.DataFrame(
        {
            "n": [100, 80, 60],
            "risk": [0.10, 0.20, 0.30],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "_source_row_index": [0, 1, 2],
            "n": [100, 80, 60],
            "risk": [0.10, 0.20, 0.99],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert res.get("reason") == "source_values_disagree", res
    assert res.get("mismatches", [])[0].get("column") == "risk", res


@pytest.mark.parametrize("position_col", ["source_row_index", "_source_row_index"])
@pytest.mark.parametrize(
    ("positions", "reason"),
    [
        ([-1, 1, 2], "source_row_index_out_of_bounds"),
        ([0, 1.5, 2], "source_row_index_out_of_bounds"),
        ([0, "not-an-index", 2], "source_row_index_out_of_bounds"),
        ([0, 1, 3], "source_row_index_out_of_bounds"),
    ],
)
def test_source_row_index_alias_rejects_invalid_positions(
    tmp_path: Path,
    position_col: str,
    positions: list[object],
    reason: str,
):
    up = tmp_path / "outcome_by_row.csv"
    pd.DataFrame({"risk": [0.10, 0.20, 0.30]}).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            position_col: positions,
            "risk": [0.10, 0.20, 0.30],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert res.get("reason") == reason, res


@pytest.mark.parametrize("position_col", ["source_row_index", "_source_row_index"])
def test_source_row_index_alias_allows_truthful_long_form_projection(
    tmp_path: Path,
    position_col: str,
):
    """Multiple panels may legitimately project the same parent row."""

    up = tmp_path / "outcome_by_row.csv"
    pd.DataFrame({"risk": [0.10, 0.20]}).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            position_col: [0, 0, 1, 1],
            "risk": [0.10, 0.10, 0.20, 0.20],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is True, res
    assert res.get("n_source_rows") == 4, res


@pytest.mark.parametrize("position_col", ["source_row_index", "_source_row_index"])
def test_source_row_index_alias_rechecks_every_duplicate_projection_row(
    tmp_path: Path,
    position_col: str,
):
    up = tmp_path / "outcome_by_row.csv"
    pd.DataFrame({"risk": [0.10, 0.20]}).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            position_col: [0, 0, 1, 1],
            "risk": [0.10, 0.99, 0.20, 0.20],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert res.get("reason") == "source_values_disagree", res


def test_complete_multi_panel_long_form_projection_is_verified_per_panel(
    tmp_path: Path,
):
    up = tmp_path / "parent_results.csv"
    pd.DataFrame(
        {
            "risk": [0.10, 0.20],
            "risk_n": [10, 20],
            "median": [2.0, 4.0],
            "continuous_n": [9, 18],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "panel_id": ["A", "A", "B", "B"],
            "value_type": ["risk", "risk", "distribution", "distribution"],
            "source_row_index": [0, 1, 0, 1],
            "estimate": [0.10, 0.20, 2.0, 4.0],
            "count": [10, 20, 9, 18],
        }
    )

    clean = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )
    assert clean.get("ok") is True, clean
    assert clean.get("join_mode") == "panel_stratified_positional", clean
    assert set(clean.get("verified_panels", {})) == {"A", "B"}, clean

    source.loc[3, "estimate"] = 99.0
    forged = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )
    assert forged.get("ok") is False, forged
    assert forged.get("panel_id") == "B", forged
    assert forged.get("reason") == "no_verifiable_values", forged


def test_matching_source_row_index_aliases_are_accepted(tmp_path: Path):
    up = tmp_path / "outcome_by_row.csv"
    pd.DataFrame({"risk": [0.10, 0.20, 0.30]}).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "source_row_index": [0, 1, 2],
            "_source_row_index": [0, 1, 2],
            "risk": [0.10, 0.20, 0.30],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is True, res
    assert res.get("key_column") == "source_row_index", res


def test_conflicting_source_row_index_aliases_fail_closed(tmp_path: Path):
    up = tmp_path / "outcome_by_row.csv"
    pd.DataFrame({"risk": [0.10, 0.20, 0.30]}).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "source_row_index": [0, 1, 2],
            "_source_row_index": [0, 2, 1],
            "risk": [0.10, 0.20, 0.30],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert res.get("reason") == "conflicting_source_row_index_aliases", res


def test_structural_fallback_requires_actual_formatted_value_verification(
    tmp_path: Path,
):
    up = tmp_path / "outcome_by_group.csv"
    pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "mortality_rate": [0.91, 0.92, 0.93],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "display_estimate": ["91%", "forged", "93%"],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert res.get("reason") == "no_verifiable_values", res
    assert "display_estimate" in res.get("unverified_source_value_columns", []), res


def test_truthful_renamed_value_is_verified_by_row_aligned_vector(tmp_path: Path):
    up = tmp_path / "outcome_by_stage.csv"
    pd.DataFrame(
        {
            "stage": [0, 1, 2],
            "mortality_rate": [0.10, 0.20, 0.30],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "stage": [0, 1, 2],
            "estimate": [0.10, 0.20, 0.30],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is True, res
    assert res.get("verified_value_mappings") == {
        "estimate": "mortality_rate"
    }, res


def test_cross_name_estimate_cannot_be_laundered_by_equal_count_vector(
    tmp_path: Path,
):
    up = tmp_path / "outcome_by_group.csv"
    pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "mortality_rate": [0.10, 0.20, 0.30],
            "n": [100, 80, 60],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "estimate": [100, 80, 60],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert res.get("reason") == "no_verifiable_values", res
    assert res.get("unverified_source_value_columns") == ["estimate"], res


def test_cross_name_count_alias_can_match_parent_sample_size(tmp_path: Path):
    up = tmp_path / "outcome_by_group.csv"
    pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "n": [100, 80, 60],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "count": [100, 80, 60],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is True, res
    assert res.get("verified_value_mappings") == {"count": "n"}, res


def test_percentage_label_cannot_silently_inherit_zero_to_one_rate_scale(
    tmp_path: Path,
):
    up = tmp_path / "outcome_by_group.csv"
    pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "mortality_rate": [0.10, 0.20, 0.30],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "percentage": [0.10, 0.20, 0.30],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert res.get("reason") == "no_verifiable_values", res


def test_common_se_and_p_aliases_match_only_their_inferential_family(
    tmp_path: Path,
):
    up = tmp_path / "model_terms.csv"
    pd.DataFrame(
        {
            "term": ["a", "b", "c"],
            "se": [0.11, 0.12, 0.13],
            "p_value": [0.01, 0.02, 0.03],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "term": ["a", "b", "c"],
            "std_err": [0.11, 0.12, 0.13],
            "pval": [0.01, 0.02, 0.03],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is True, res
    assert res.get("verified_value_mappings") == {
        "pval": "p_value",
        "std_err": "se",
    }, res


@pytest.mark.parametrize("source_col", ["std_err", "std_error", "pval", "p_val"])
def test_se_and_p_aliases_cannot_launder_equal_point_estimate_vector(
    tmp_path: Path,
    source_col: str,
):
    up = tmp_path / "model_terms.csv"
    pd.DataFrame(
        {
            "term": ["a", "b", "c"],
            "estimate": [0.11, 0.12, 0.13],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "term": ["a", "b", "c"],
            source_col: [0.11, 0.12, 0.13],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert res.get("reason") == "no_verifiable_values", res


def test_declared_parent_cannot_be_laundered_by_unrelated_table(tmp_path: Path):
    from easyicu.research_agent.schema import AnalysisStep

    parent_out = tmp_path / "steps" / "01_parent" / "outputs"
    parent_out.mkdir(parents=True)
    pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "mortality_rate": [0.10, 0.20, 0.30],
        }
    ).to_csv(parent_out / "declared_parent.csv", index=False)
    pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "estimate": [0.91, 0.92, 0.93],
        }
    ).to_csv(parent_out / "unrelated_exact_match.csv", index=False)
    figure_out = tmp_path / "steps" / "01_parent_figure" / "outputs"
    figure_out.mkdir(parents=True)
    pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "estimate": [0.91, 0.92, 0.93],
            "source_table": ["declared_parent.csv"] * 3,
        }
    ).to_csv(figure_out / "publication_figure_source_data.csv", index=False)

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="01_parent_figure",
            intent="Render source-backed figure",
            method="figure",
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={},
    )

    errors = [finding for finding in findings if finding.severity == "error"]
    assert errors, "an unrelated table must not authenticate a declared parent"
    candidates = errors[0].detail.get("candidate_upstream_tables", [])
    assert candidates and all("declared_parent.csv" in item for item in candidates)


def test_shared_numeric_column_flags_opposite_infinities(tmp_path: Path):
    up = tmp_path / "outcome_by_group.csv"
    pd.DataFrame(
        {
            "group": ["low", "high"],
            "estimate": [0.10, float("-inf")],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "group": ["low", "high"],
            "estimate": [0.10, float("inf")],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert res.get("reason") == "source_values_disagree", res
    assert res["mismatches"][0]["column"] == "estimate"


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
