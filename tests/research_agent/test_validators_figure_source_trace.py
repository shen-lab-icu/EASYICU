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

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.audits.validators import FigureSourceDataValidator
from easyicu.research_agent.schema import AnalysisStep


def test_effect_source_inherits_primary_tier_only_from_matching_model_contract():
    source = pd.DataFrame(
        {
            "model_id": ["planned_model"],
            "term_role": ["exposure"],
            "source_variable": ["marker_max"],
            "odds_ratio": [1.25],
        }
    )
    completed = [
        {
            "step_id": "05_model",
            "status": "ok",
            "step_summary": {
                "model_contracts": [
                    {
                        "model_id": "planned_model",
                        "analysis_role": "primary",
                        "exposure_source": "marker_max",
                        "fit_status": "fitted",
                    }
                ]
            },
        }
    ]
    assert (
        FigureSourceDataValidator._contract_scoped_effect_product(
            product="table:adjusted_association_estimates",
            source_frame=source,
            upstream_step_id="05_model",
            completed_step_records=completed,
        )
        == "table:primary_adjusted_association_estimates"
    )

    forged = source.assign(source_variable="different_marker")
    assert (
        FigureSourceDataValidator._contract_scoped_effect_product(
            product="table:adjusted_association_estimates",
            source_frame=forged,
            upstream_step_id="05_model",
            completed_step_records=completed,
        )
        == "table:adjusted_association_estimates"
    )

    adjustment_row = source.assign(term_role="adjustment")
    assert (
        FigureSourceDataValidator._contract_scoped_effect_product(
            product="table:adjusted_association_estimates",
            source_frame=adjustment_row,
            upstream_step_id="05_model",
            completed_step_records=completed,
        )
        == "table:adjusted_association_estimates"
    )

    compact_source = source[["model_id", "odds_ratio"]].assign(
        exposure="marker_max"
    )
    assert (
        FigureSourceDataValidator._contract_scoped_effect_product(
            product="table:adjusted_association_estimates",
            source_frame=compact_source,
            upstream_step_id="05_model",
            completed_step_records=completed,
        )
        == "table:primary_adjusted_association_estimates"
    )

    forged_compact_source = compact_source.assign(exposure="different_marker")
    assert (
        FigureSourceDataValidator._contract_scoped_effect_product(
            product="table:adjusted_association_estimates",
            source_frame=forged_compact_source,
            upstream_step_id="05_model",
            completed_step_records=completed,
        )
        == "table:adjusted_association_estimates"
    )

    positional_source = pd.DataFrame({"source_row_index": [0, 1]})
    upstream = pd.DataFrame(
        {
            "model_id": ["planned_model", "planned_model"],
            "term_role": ["exposure", "adjustment"],
            "source_variable": ["marker_max", "age"],
            "odds_ratio": [1.25, 1.01],
        }
    )
    assert (
        FigureSourceDataValidator._contract_scoped_effect_product(
            product="table:adjusted_association_estimates",
            source_frame=positional_source,
            upstream_frame=upstream,
            upstream_step_id="05_model",
            completed_step_records=completed,
        )
        == "table:primary_adjusted_association_estimates"
    )


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


def test_nullable_boolean_metadata_does_not_break_exact_numeric_projection(
    tmp_path: Path,
):
    upstream = tmp_path / "robustness_summary.csv"
    pd.DataFrame(
        {
            "analysis": ["primary", "complete_case"],
            "odds_ratio": [1.25, 1.20],
            "estimate_identical_to_primary": [None, False],
        }
    ).to_csv(upstream, index=False)
    source = pd.DataFrame(
        {
            "source_row_index": [0, 1],
            "analysis": ["primary", "complete_case"],
            "odds_ratio": [1.25, 1.20],
            "estimate_identical_to_primary": [None, False],
        }
    )

    clean = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "robustness_source_data.csv",
        upstream_path=upstream,
    )
    assert clean.get("ok") is True, clean

    source.loc[1, "odds_ratio"] = 9.99
    drift = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "robustness_source_data.csv",
        upstream_path=upstream,
    )
    assert drift.get("ok") is False, drift
    assert drift.get("reason") == "source_values_disagree", drift


def test_estimate_unit_is_compared_as_text_not_parsed_as_a_number(
    tmp_path: Path,
):
    """A truthful unit label must not become a numeric parse failure."""

    upstream = tmp_path / "cluster_stability.csv"
    pd.DataFrame(
        {
            "metric": ["bootstrap_ari", "silhouette"],
            "estimate": [0.84, 0.31],
            "estimate_unit": ["adjusted_rand_index", "silhouette_coefficient"],
        }
    ).to_csv(upstream, index=False)
    source = pd.DataFrame(
        {
            "source_row_index": [0, 1],
            "metric": ["bootstrap_ari", "silhouette"],
            "estimate": [0.84, 0.31],
            "estimate_unit": ["adjusted_rand_index", "silhouette_coefficient"],
        }
    )

    clean = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "cluster_stability_source_data.csv",
        upstream_path=upstream,
    )
    assert clean.get("ok") is True, clean

    source.loc[1, "estimate_unit"] = "adjusted_rand_index"
    drift = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "cluster_stability_source_data.csv",
        upstream_path=upstream,
    )
    assert drift.get("ok") is False, drift
    assert drift.get("reason") == "source_values_disagree", drift
    assert drift.get("mismatches", [])[0]["column"] == "estimate_unit"


def test_numeric_candidate_with_equal_semantic_text_is_a_faithful_projection(
    tmp_path: Path,
):
    """Names containing risk/effect do not turn equal receipt text into NaN drift."""

    upstream = tmp_path / "distribution.csv"
    pd.DataFrame(
        {
            "source_row_index": [0, 1, 2],
            "row_role": ["exposure_level", "exposure_level", "overall"],
            "risk_difference_pct": [None, None, 4.25],
            "risk_difference_covariance": [None, None, "cluster_robust"],
            "risk_difference_effect_measure": [None, None, "risk_difference"],
        }
    ).to_csv(upstream, index=False)
    source = pd.read_csv(upstream)

    clean = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "distribution_source_data.csv",
        upstream_path=upstream,
    )
    assert clean.get("ok") is True, clean

    source.loc[2, "risk_difference_covariance"] = "model_based"
    drift = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "distribution_source_data.csv",
        upstream_path=upstream,
    )
    assert drift.get("ok") is False, drift
    assert drift.get("reason") == "source_values_disagree", drift


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
            "group": [
                "Unmeasured",
                "<2",
                "2-<4",
                ">=4",
            ],  # figure label, not a trace key
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
    assert {"n", "event_n", "risk"} <= set(res.get("verified_value_mappings", {}))


def test_source_row_index_disambiguates_repeated_named_keys_and_metadata_columns(
    tmp_path: Path,
):
    up = tmp_path / "table_one.csv"
    upstream = pd.DataFrame(
        {
            "variable": ["age", "age", "sex", "sex"],
            "group": ["overall", "exposed", "overall", "exposed"],
            "count_column": ["", "", "sex", "sex"],
            "count": [100, 40, 55, 21],
        }
    )
    upstream.to_csv(up, index=False)
    source = upstream.copy()
    source.insert(0, "source_table", up.name)
    source.insert(0, "source_row_index", range(len(source)))

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is True, res
    assert res.get("key_column") == "source_row_index", res
    assert res.get("verified_value_mappings", {}).get("count") == "count", res

    source.loc[1, "count"] = 41
    forged = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )
    assert forged.get("ok") is False, forged
    assert forged.get("reason") == "source_values_disagree", forged


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
    assert res.get("verified_value_mappings") == {"estimate": "mortality_rate"}, res


def test_truthful_plot_order_is_verified_from_parent_order_vector(tmp_path: Path):
    up = tmp_path / "outcome_by_stage.csv"
    pd.DataFrame(
        {
            "stage": ["low", "mid", "high"],
            "stage_order": [1, 2, 3],
            "mortality_rate": [0.10, 0.20, 0.30],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "stage": ["low", "mid", "high"],
            "stage_order": [1, 2, 3],
            "plot_order": [1, 2, 3],
            "estimate": [0.10, 0.20, 0.30],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is True, res
    assert res.get("verified_value_mappings", {}).get("plot_order") == (
        "derived:ordering(stage_order)"
    )


def test_forged_plot_order_cannot_borrow_parent_order_semantics(tmp_path: Path):
    up = tmp_path / "outcome_by_stage.csv"
    pd.DataFrame(
        {
            "stage": ["low", "mid", "high"],
            "stage_order": [1, 2, 3],
            "mortality_rate": [0.10, 0.20, 0.30],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "stage": ["low", "mid", "high"],
            "stage_order": [1, 2, 3],
            "plot_order": [3, 2, 1],
            "estimate": [0.10, 0.20, 0.30],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert "plot_order" in res.get("unverified_source_value_columns", []), res


def test_rank_estimate_cannot_borrow_verified_parent_order_vector(tmp_path: Path):
    up = tmp_path / "outcome_by_group.csv"
    pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "row_order": [1, 2, 3],
            "mortality_rate": [0.10, 0.20, 0.30],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "row_order": [1, 2, 3],
            "rank_estimate": [1, 2, 3],
            "estimate": [0.10, 0.20, 0.30],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert "rank_estimate" in res.get("unverified_source_value_columns", []), res


def test_declared_rate_target_cannot_be_laundered_by_sibling_rate(tmp_path: Path):
    up = tmp_path / "outcome_by_group.csv"
    pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "mortality_rate": [0.10, 0.20, 0.30],
            "readmission_rate": [0.70, 0.80, 0.90],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "estimate": [0.70, 0.80, 0.90],
            "value_type": ["mortality_rate"] * 3,
            "source_table": [up.name] * 3,
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert res.get("reason") == "source_values_disagree", res
    assert res.get("mismatches", [])[0]["upstream_column"] == "mortality_rate"


def test_declared_rate_target_accepts_truthful_named_parent_value(tmp_path: Path):
    up = tmp_path / "outcome_by_group.csv"
    pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "mortality_rate": [0.10, 0.20, 0.30],
            "readmission_rate": [0.70, 0.80, 0.90],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "estimate": [0.10, 0.20, 0.30],
            # Normalisation, rather than a case-specific spelling, binds this
            # declaration to the upstream mortality_rate column.
            "value_type": ["mortality rate"] * 3,
            "source_table": [up.name] * 3,
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is True, res
    assert res.get("verified_value_mappings") == {"estimate": "mortality_rate"}, res


def test_renamed_estimate_cannot_match_unrelated_location_summary(
    tmp_path: Path,
):
    up = tmp_path / "outcome_by_stage.csv"
    pd.DataFrame(
        {
            "stage": [0, 1, 2],
            "mortality_rate": [0.10, 0.20, 0.30],
            "mean_age": [55.0, 60.0, 65.0],
            "n": [100, 80, 60],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "stage": [0, 1, 2],
            # This vector came from mean_age, not from the claimed outcome.
            "estimate": [55.0, 60.0, 65.0],
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


def test_numeric_value_column_with_label_suffix_is_still_verified(
    tmp_path: Path,
):
    up = tmp_path / "outcome_by_stage.csv"
    pd.DataFrame(
        {
            "stage": [0, 1, 2],
            "estimate_label": [0.10, 0.20, 0.30],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "stage": [0, 1, 2],
            "estimate_label": [0.10, 9.90, 0.30],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert res.get("reason") == "source_values_disagree", res
    assert res.get("mismatches", [])[0]["column"] == "estimate_label"


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


def test_unknown_cross_name_numeric_alias_cannot_authenticate_unrelated_vector(
    tmp_path: Path,
):
    up = tmp_path / "outcome_by_group.csv"
    pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "age": [55.0, 60.0, 65.0],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "display_metric": [55.0, 60.0, 65.0],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert res.get("reason") == "no_verifiable_values", res
    assert res.get("unverified_source_value_columns") == ["display_metric"], res


def test_negated_semantic_label_cannot_authorize_location_summary_alias(
    tmp_path: Path,
):
    up = tmp_path / "outcome_by_group.csv"
    pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "mean_age": [55.0, 60.0, 65.0],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "value_type": ["not_mean", "not_mean", "not_mean"],
            "estimate": [55.0, 60.0, 65.0],
        }
    )

    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )

    assert res.get("ok") is False, res
    assert res.get("reason") == "no_verifiable_values", res


def test_errorbar_width_must_be_derived_from_verified_interval(tmp_path: Path):
    up = tmp_path / "effect_by_group.csv"
    pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "estimate": [1.1, 1.2, 1.3],
            "ci_low": [1.0, 1.0, 1.1],
            "ci_high": [1.2, 1.4, 1.5],
        }
    ).to_csv(up, index=False)
    source = pd.DataFrame(
        {
            "group": ["low", "mid", "high"],
            "estimate": [1.1, 1.2, 1.3],
            "ci_low": [1.0, 1.0, 1.1],
            "ci_high": [1.2, 1.4, 1.5],
            "errorbar_width": [99.0, 99.0, 99.0],
        }
    )

    forged = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )
    assert forged.get("ok") is False, forged
    assert "errorbar_width" in forged.get("unverified_source_value_columns", [])

    source["errorbar_width"] = source["ci_high"] - source["ci_low"]
    truthful = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "publication_figure_source_data.csv",
        upstream_path=up,
    )
    assert truthful.get("ok") is True, truthful
    assert truthful["verified_value_mappings"]["errorbar_width"] == (
        "derived:ci_high-ci_low"
    )


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


def test_duplicate_declared_basename_requires_exact_source_step(tmp_path: Path):
    parent_out = tmp_path / "steps" / "01_parent" / "outputs"
    unrelated_out = tmp_path / "steps" / "00_unrelated" / "outputs"
    figure_out = tmp_path / "steps" / "01_parent_figure" / "outputs"
    parent_out.mkdir(parents=True)
    unrelated_out.mkdir(parents=True)
    figure_out.mkdir(parents=True)
    shared_name = "outcome_by_group.csv"
    pd.DataFrame({"group": ["low", "high"], "mortality_rate": [0.1, 0.2]}).to_csv(
        parent_out / shared_name, index=False
    )
    pd.DataFrame({"group": ["low", "high"], "mortality_rate": [0.9, 0.8]}).to_csv(
        unrelated_out / shared_name, index=False
    )
    pd.DataFrame(
        {
            "group": ["low", "high"],
            "mortality_rate": [0.9, 0.8],
            "source_table": [shared_name] * 2,
        }
    ).to_csv(figure_out / "publication_figure_source_data.csv", index=False)

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="01_parent_figure",
            intent="Render the parent results.",
            method="figure",
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={},
    )

    errors = [item for item in findings if item.severity == "error"]
    assert errors
    assert errors[0].detail["best_mismatch"]["reason"] == (
        "ambiguous_declared_source_table_lineage"
    )


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


def _write_authoritative_figure_trace_run(
    tmp_path: Path,
) -> tuple[Path, Path, list[dict]]:
    parent_out = tmp_path / "steps" / "01_parent" / "outputs"
    parent_out.mkdir(parents=True)
    parent_path = parent_out / "outcome_by_group.csv"
    parent_bytes = b"group,mortality_rate\nlow,0.1\nhigh,0.2\n"
    parent_path.write_bytes(parent_bytes)

    figure_out = tmp_path / "steps" / "01_parent_figure" / "outputs"
    figure_out.mkdir(parents=True)
    pd.DataFrame(
        {
            "group": ["low", "high"],
            "mortality_rate": [0.1, 0.2],
            "source_table": [parent_path.name] * 2,
        }
    ).to_csv(figure_out / "publication_figure_source_data.csv", index=False)

    digest = hashlib.sha256(parent_bytes).hexdigest()
    evidence_id = f"table_outcome_by_group_{digest[:8]}"
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    evidence_path = evidence_dir / f"{evidence_id}__{parent_path.name}"
    evidence_path.write_bytes(parent_bytes)
    records = [
        {
            "step_id": "01_parent",
            "status": "ok",
            "evidence_ids": [evidence_id],
            "step_summary": {"output_files": [parent_path.name]},
        }
    ]
    manifest = {
        "per_step_records": records,
        "evidence": [
            {
                "evidence_id": evidence_id,
                "kind": "table",
                "relative_path": str(evidence_path.relative_to(tmp_path)),
                "sha256": digest,
                "produced_by_step": "01_parent",
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return parent_path, figure_out, records


def test_figure_source_uses_current_hash_verified_parent(tmp_path: Path):
    _parent_path, figure_out, records = _write_authoritative_figure_trace_run(tmp_path)

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="01_parent_figure",
            intent="Render a source-backed figure.",
            method="figure",
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={},
        completed_step_records=records,
    )

    assert [item for item in findings if item.severity == "error"] == []


def test_standalone_figure_uses_host_resolved_parent_not_step_id_suffix(
    tmp_path: Path,
):
    parent_path, figure_out, records = _write_authoritative_figure_trace_run(tmp_path)
    digest = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    bindings = {
        "table:outcome_by_group": {
            "declared_kind": "table",
            "product": "outcome_by_group",
            "produced_by_step": "01_parent",
            "evidence_id": f"table_outcome_by_group_{digest[:8]}",
            "sha256": digest,
        }
    }

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="05_publication_figure",
            intent="Render the host-resolved result table.",
            inputs=["table:outcome_by_group"],
            method="publication_figure",
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={"upstream_step_id": "01_parent"},
        completed_step_records=records,
        resolved_input_bindings=bindings,
    )

    assert [item for item in findings if item.severity == "error"] == []


def test_figure_source_accepts_hash_bound_evidence_copy_basename(tmp_path: Path):
    parent_path, figure_out, records = _write_authoritative_figure_trace_run(tmp_path)
    digest = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    evidence_id = f"table_outcome_by_group_{digest[:8]}"
    evidence_name = f"{evidence_id}__{parent_path.name}"
    pd.DataFrame(
        {
            "group": ["low", "high"],
            "mortality_rate": [0.1, 0.2],
            "source_table": [evidence_name, evidence_name],
        }
    ).to_csv(figure_out / "publication_figure_source_data.csv", index=False)

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="05_publication_figure",
            intent="Render the exact hash-bound table copy.",
            inputs=["table:outcome_by_group"],
            method="publication_figure",
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={},
        completed_step_records=records,
        resolved_input_bindings={
            "table:outcome_by_group": {
                "declared_kind": "table",
                "product": "outcome_by_group",
                "produced_by_step": "01_parent",
                "evidence_id": evidence_id,
                "sha256": digest,
            }
        },
    )

    assert [item for item in findings if item.severity == "error"] == []


def test_figure_source_rejects_summary_parent_that_conflicts_with_host_binding(
    tmp_path: Path,
):
    parent_path, figure_out, records = _write_authoritative_figure_trace_run(tmp_path)
    digest = hashlib.sha256(parent_path.read_bytes()).hexdigest()

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="05_publication_figure",
            intent="Render the host-resolved result table.",
            inputs=["table:outcome_by_group"],
            method="publication_figure",
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={"upstream_step_id": "02_stale_parent"},
        completed_step_records=records,
        resolved_input_bindings={
            "table:outcome_by_group": {
                "declared_kind": "table",
                "product": "outcome_by_group",
                "produced_by_step": "01_parent",
                "evidence_id": "table_outcome_by_group",
                "sha256": digest,
            }
        },
    )

    errors = [item for item in findings if item.severity == "error"]
    assert errors
    assert errors[0].detail["reason"] == "resolved_upstream_binding_mismatch"


def test_figure_source_cannot_use_unbound_table_from_same_producer(tmp_path: Path):
    parent_path, figure_out, records = _write_authoritative_figure_trace_run(tmp_path)
    unbound_path = parent_path.parent / "unbound_other.csv"
    unbound_bytes = b"group,mortality_rate\nlow,0.9\nhigh,0.8\n"
    unbound_path.write_bytes(unbound_bytes)
    unbound_digest = hashlib.sha256(unbound_bytes).hexdigest()
    unbound_id = f"table_unbound_other_{unbound_digest[:8]}"
    evidence_path = tmp_path / "evidence" / f"{unbound_id}__{unbound_path.name}"
    evidence_path.write_bytes(unbound_bytes)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evidence"].append(
        {
            "evidence_id": unbound_id,
            "kind": "table",
            "relative_path": str(evidence_path.relative_to(tmp_path)),
            "sha256": unbound_digest,
            "produced_by_step": "01_parent",
        }
    )
    manifest["per_step_records"][0]["evidence_ids"].append(unbound_id)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    records[0]["evidence_ids"].append(unbound_id)
    pd.DataFrame(
        {
            "group": ["low", "high"],
            "mortality_rate": [0.9, 0.8],
            "source_table": [unbound_path.name] * 2,
            "source_step_id": ["01_parent"] * 2,
        }
    ).to_csv(figure_out / "publication_figure_source_data.csv", index=False)
    bound_digest = hashlib.sha256(parent_path.read_bytes()).hexdigest()

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="05_publication_figure",
            intent="Render only the bound table.",
            inputs=["table:outcome_by_group"],
            method="publication_figure",
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={"upstream_step_id": "01_parent"},
        completed_step_records=records,
        resolved_input_bindings={
            "table:outcome_by_group": {
                "declared_kind": "table",
                "product": "outcome_by_group",
                "produced_by_step": "01_parent",
                "evidence_id": f"table_outcome_by_group_{bound_digest[:8]}",
                "sha256": bound_digest,
            }
        },
    )

    errors = [item for item in findings if item.severity == "error"]
    assert errors
    assert errors[0].detail["best_mismatch"]["reason"] == (
        "declared_source_table_not_found"
    )


def test_figure_source_rejects_parent_superseded_by_failure(tmp_path: Path):
    _parent_path, figure_out, records = _write_authoritative_figure_trace_run(tmp_path)
    records.append(
        {
            "step_id": "01_parent",
            "status": "contract_failed",
            "evidence_ids": [],
            "step_summary": {"status": "contract_failed"},
        }
    )

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="01_parent_figure",
            intent="Render a source-backed figure.",
            method="figure",
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={},
        completed_step_records=records,
    )

    errors = [item for item in findings if item.severity == "error"]
    assert errors
    assert errors[0].detail["noncurrent_upstream_step_ids"] == ["01_parent"]


def test_figure_source_rejects_tampered_parent_after_registration(
    tmp_path: Path,
):
    parent_path, figure_out, records = _write_authoritative_figure_trace_run(tmp_path)
    parent_path.write_text(
        "group,mortality_rate\nlow,0.9\nhigh,0.8\n", encoding="utf-8"
    )

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="01_parent_figure",
            intent="Render a source-backed figure.",
            method="figure",
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={},
        completed_step_records=records,
    )

    errors = [item for item in findings if item.severity == "error"]
    assert errors
    assert "hash-verified upstream" in errors[0].message


def test_figure_source_rejects_symlinked_parent_output(tmp_path: Path):
    parent_path, figure_out, records = _write_authoritative_figure_trace_run(tmp_path)
    target = tmp_path / "symlink_target.csv"
    target.write_bytes(parent_path.read_bytes())
    parent_path.unlink()
    parent_path.symlink_to(target)

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="01_parent_figure",
            intent="Render a source-backed figure.",
            method="figure",
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={},
        completed_step_records=records,
    )

    assert [item for item in findings if item.severity == "error"]


def test_figure_source_rejects_upstream_path_traversal(tmp_path: Path):
    _parent_path, figure_out, records = _write_authoritative_figure_trace_run(tmp_path)

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="figure_without_inferred_parent",
            intent="Render a source-backed figure.",
            method="figure",
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={"upstream_step_id": "../../outside"},
        completed_step_records=records,
    )

    errors = [item for item in findings if item.severity == "error"]
    assert errors
    assert errors[0].detail["unsafe_upstream_step_ids"] == ["../../outside"]


def test_figure_source_rejects_declared_table_path_traversal(tmp_path: Path):
    _parent_path, figure_out, records = _write_authoritative_figure_trace_run(tmp_path)
    source_path = figure_out / "publication_figure_source_data.csv"
    source = pd.read_csv(source_path)
    source["source_table"] = "../outcome_by_group.csv"
    source.to_csv(source_path, index=False)

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="01_parent_figure",
            intent="Render a source-backed figure.",
            method="figure",
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={},
        completed_step_records=records,
    )

    errors = [item for item in findings if item.severity == "error"]
    assert errors
    assert errors[0].detail["unsafe_declared_source_tables"] == [
        "../outcome_by_group.csv"
    ]


def test_figure_source_rejects_missing_upstream_binding(tmp_path: Path):
    _parent_path, figure_out, records = _write_authoritative_figure_trace_run(tmp_path)

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="standalone_visual",
            intent="Render a source-backed figure.",
            method="figure",
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={},
        completed_step_records=records,
    )

    errors = [item for item in findings if item.severity == "error"]
    assert errors
    assert errors[0].detail["reason"] == "missing_upstream_step_binding"


@pytest.mark.parametrize("mutation", ["malformed", "empty", "symlink"])
def test_figure_source_rejects_unverifiable_source_data_file(
    tmp_path: Path,
    mutation: str,
):
    _parent_path, figure_out, records = _write_authoritative_figure_trace_run(tmp_path)
    source_path = figure_out / "publication_figure_source_data.csv"
    if mutation == "malformed":
        source_path.write_bytes(b'"unterminated')
    elif mutation == "empty":
        source_path.write_text("group,mortality_rate,source_table\n", encoding="utf-8")
    else:
        outside = tmp_path / "outside_source.csv"
        outside.write_bytes(source_path.read_bytes())
        source_path.unlink()
        try:
            source_path.symlink_to(outside)
        except OSError:
            pytest.skip("symlinks unavailable")

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="01_parent_figure",
            intent="Render a source-backed figure.",
            method="figure",
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={},
        completed_step_records=records,
    )

    errors = [item for item in findings if item.severity == "error"]
    assert errors, findings
    reasons = {str(item.detail.get("reason") or "") for item in errors}
    expected = {
        "malformed": "source_data_read_failed",
        "empty": "source_data_empty",
        "symlink": "unsafe_source_data_path",
    }[mutation]
    assert expected in reasons, findings


def _effect_figure_bundle_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, list[dict], dict[str, dict[str, str]]]:
    parent_out = tmp_path / "steps" / "01_parent" / "outputs"
    figure_out = tmp_path / "steps" / "01_parent_figure" / "outputs"
    evidence_dir = tmp_path / "evidence"
    parent_out.mkdir(parents=True)
    figure_out.mkdir(parents=True)
    evidence_dir.mkdir()
    parent_path = parent_out / "primary_or.csv"
    parent_bytes = b"term,odds_ratio,ci_low,ci_high\nexposure,1.25,1.1,1.42\n"
    parent_path.write_bytes(parent_bytes)
    pd.DataFrame(
        {
            "term": ["exposure"],
            "odds_ratio": [1.25],
            "ci_low": [1.10],
            "ci_high": [1.42],
            "source_table": [parent_path.name],
        }
    ).to_csv(figure_out / "publication_figure_source_data.csv", index=False)
    digest = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    evidence_id = f"table_primary_or_{digest[:8]}"
    evidence_path = evidence_dir / f"{evidence_id}__{parent_path.name}"
    evidence_path.write_bytes(parent_bytes)
    records = [
        {
            "step_id": "01_parent",
            "status": "ok",
            "evidence_ids": [evidence_id],
            "step_summary": {"output_files": {"table:primary_or": parent_path.name}},
        }
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "per_step_records": records,
                "evidence": [
                    {
                        "evidence_id": evidence_id,
                        "kind": "table",
                        "relative_path": str(evidence_path.relative_to(tmp_path)),
                        "sha256": digest,
                        "produced_by_step": "01_parent",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    bindings = {
        "table:primary_or": {
            "declared_kind": "table",
            "product": "primary_or",
            "produced_by_step": "01_parent",
            "evidence_id": evidence_id,
            "sha256": digest,
        }
    }
    return parent_path, figure_out, records, bindings


def _effect_figure_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="05_plot",
        intent="Plot the host-bound effect result.",
        inputs=["table:primary_or"],
        expected_outputs=["figure:primary_or_forest"],
        method="visualization",
    )


def _write_two_panel_contract(
    path: Path,
    *,
    figure_id: str,
    source_data: list[str],
) -> None:
    path.write_text(
        json.dumps(
            {
                "figure_id": figure_id,
                "source_data": source_data,
                "panels": [
                    {
                        "panel_id": "A",
                        "title": "Primary estimate",
                        "role": "primary_estimand",
                        "claim": "The primary estimate is copied from the bound result table.",
                    },
                    {
                        "panel_id": "B",
                        "title": "Source audit",
                        "role": "audit",
                        "claim": "The audit panel exposes values from the same bound result table.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_effect_figure_requires_contract_declared_local_source_data(tmp_path: Path):
    _parent, figure_out, records, bindings = _effect_figure_bundle_fixture(tmp_path)
    (figure_out / "publication_figure_source_data.csv").unlink()
    (figure_out / "primary_or_forest.png").write_bytes(b"png")
    _write_two_panel_contract(
        figure_out / "primary_or_forest.figure_contract.json",
        figure_id="primary_or_forest",
        source_data=[],
    )

    findings = FigureSourceDataValidator().audit(
        step=_effect_figure_step(),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={
            "output_files": {"figure:primary_or_forest": "primary_or_forest.png"}
        },
        completed_step_records=records,
        resolved_input_bindings=bindings,
    )

    errors = [finding for finding in findings if finding.severity == "error"]
    assert errors
    assert errors[0].detail["reason"] == "missing_source_data"


def test_effect_figure_exact_bundle_with_truthful_source_data_passes(tmp_path: Path):
    _parent, figure_out, records, bindings = _effect_figure_bundle_fixture(tmp_path)
    (figure_out / "primary_or_forest.png").write_bytes(b"png")
    _write_two_panel_contract(
        figure_out / "primary_or_forest.figure_contract.json",
        figure_id="primary_or_forest",
        source_data=["publication_figure_source_data.csv"],
    )

    findings = FigureSourceDataValidator().audit(
        step=_effect_figure_step(),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={
            "output_files": {"figure:primary_or_forest": "primary_or_forest.png"}
        },
        completed_step_records=records,
        resolved_input_bindings=bindings,
    )

    assert [finding for finding in findings if finding.severity == "error"] == []


def test_effect_figure_validator_rejects_noncanonical_source_descriptors(
    tmp_path: Path,
):
    _parent, figure_out, records, bindings = _effect_figure_bundle_fixture(tmp_path)
    (figure_out / "primary_or_forest.png").write_bytes(b"png")
    contract_path = figure_out / "primary_or_forest.figure_contract.json"
    _write_two_panel_contract(
        contract_path,
        figure_id="primary_or_forest",
        source_data=["publication_figure_source_data.csv"],
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["source_data"] = [
        {
            "file": "publication_figure_source_data.csv",
            "path": "publication_figure_source_data.csv",
        }
    ]
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    findings = FigureSourceDataValidator().audit(
        step=_effect_figure_step(),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={
            "output_files": {"figure:primary_or_forest": "primary_or_forest.png"}
        },
        completed_step_records=records,
        resolved_input_bindings=bindings,
    )

    errors = [finding for finding in findings if finding.severity == "error"]
    assert errors
    assert errors[0].detail["reason"] == "invalid_contract_source_data"


def test_honest_decoy_bundle_cannot_authenticate_registered_forged_figure(
    tmp_path: Path,
):
    _parent, figure_out, records, bindings = _effect_figure_bundle_fixture(tmp_path)
    (figure_out / "forged.png").write_bytes(b"forged")
    (figure_out / "honest.png").write_bytes(b"honest")
    _write_two_panel_contract(
        figure_out / "honest.figure_contract.json",
        figure_id="honest",
        source_data=["publication_figure_source_data.csv"],
    )

    findings = FigureSourceDataValidator().audit(
        step=_effect_figure_step(),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={"output_files": {"figure:primary_or_forest": "forged.png"}},
        completed_step_records=records,
        resolved_input_bindings=bindings,
    )

    errors = [finding for finding in findings if finding.severity == "error"]
    assert errors
    assert errors[0].detail["reason"] == "missing_figure_contract"


def test_statistic_backed_figure_also_requires_same_stem_contract(tmp_path: Path):
    figure_out = tmp_path / "steps" / "05_plot" / "outputs"
    figure_out.mkdir(parents=True)
    (figure_out / "forged.png").write_bytes(b"forged")
    (figure_out / "honest.figure_contract.json").write_text(
        json.dumps({"figure_id": "honest", "panels": []}),
        encoding="utf-8",
    )

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="05_plot",
            intent="Plot model performance.",
            inputs=["statistic:auroc"],
            expected_outputs=["figure:model_performance"],
            method="visualization",
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={"output_files": {"figure:model_performance": "forged.png"}},
    )

    errors = [finding for finding in findings if finding.severity == "error"]
    assert errors
    assert errors[0].detail["reason"] == "missing_figure_contract"


def test_statistic_backed_result_cannot_self_label_as_supporting_to_skip_source(
    tmp_path: Path,
):
    figure_out = tmp_path / "steps" / "05_plot" / "outputs"
    figure_out.mkdir(parents=True)
    (figure_out / "model_performance.png").write_bytes(b"forged")
    (figure_out / "model_performance.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": "model_performance",
                "source_data": [],
                "panels": [
                    {
                        "panel_id": "A",
                        "title": "Audit",
                        "role": "audit",
                        "claim": "Coder-authored supporting label cannot weaken host lineage.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="05_plot",
            intent="Plot model performance.",
            inputs=["statistic:auroc"],
            expected_outputs=["figure:model_performance"],
            method="visualization",
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={
            "output_files": {"figure:model_performance": "model_performance.png"}
        },
    )

    errors = [finding for finding in findings if finding.severity == "error"]
    assert errors
    assert errors[0].detail["reason"] == "missing_source_data"


def _write_mixed_effect_figure_bundle(
    tmp_path: Path,
    *,
    source_estimate: float,
) -> tuple[Path, AnalysisStep, dict]:
    out_dir = tmp_path / "steps" / "04_primary_association" / "outputs"
    out_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "term": ["exposure"],
            "odds_ratio": [1.25],
            "ci_low": [1.10],
            "ci_high": [1.42],
        }
    ).to_csv(out_dir / "primary_association.csv", index=False)
    pd.DataFrame(
        {
            "term": ["exposure"],
            "odds_ratio": [source_estimate],
            "ci_low": [1.10],
            "ci_high": [1.42],
            "source_table": ["primary_association.csv"],
            "source_step_id": ["04_primary_association"],
        }
    ).to_csv(out_dir / "primary_or_forest_source_data.csv", index=False)
    (out_dir / "primary_or_forest.png").write_bytes(b"png")
    _write_two_panel_contract(
        out_dir / "primary_or_forest.figure_contract.json",
        figure_id="primary_or_forest",
        source_data=["primary_or_forest_source_data.csv"],
    )
    step = AnalysisStep(
        step_id="04_primary_association",
        intent="Estimate the association and render its planned forest plot.",
        expected_outputs=[
            "table:primary_association",
            "figure:primary_or_forest",
        ],
        method="logistic_regression",
    )
    summary = {
        "output_files": {
            "table:primary_association": "primary_association.csv",
            "figure:primary_or_forest": "primary_or_forest.png",
        }
    }
    return out_dir, step, summary


def test_mixed_effect_step_can_trace_figure_to_its_own_declared_table(
    tmp_path: Path,
):
    out_dir, step, summary = _write_mixed_effect_figure_bundle(
        tmp_path,
        source_estimate=1.25,
    )

    findings = FigureSourceDataValidator().audit(
        step=step,
        out_dir=out_dir,
        run_dir=tmp_path,
        step_summary=summary,
        completed_step_records=[],
        resolved_input_bindings={},
    )

    assert [finding for finding in findings if finding.severity == "error"] == []


def test_mixed_effect_step_still_rejects_forged_own_source_data(tmp_path: Path):
    out_dir, step, summary = _write_mixed_effect_figure_bundle(
        tmp_path,
        source_estimate=9.99,
    )

    findings = FigureSourceDataValidator().audit(
        step=step,
        out_dir=out_dir,
        run_dir=tmp_path,
        step_summary=summary,
        completed_step_records=[],
        resolved_input_bindings={},
    )

    errors = [finding for finding in findings if finding.severity == "error"]
    assert errors
    assert any(
        finding.detail.get("best_mismatch", {}).get("reason")
        == "source_values_disagree"
        for finding in errors
    )


def test_effect_figure_cannot_use_unrelated_same_step_cohort_table(
    tmp_path: Path,
):
    out_dir = tmp_path / "steps" / "04_primary" / "outputs"
    out_dir.mkdir(parents=True)
    pd.DataFrame({"group": ["low", "high"], "n": [60, 40]}).to_csv(
        out_dir / "cohort_summary.csv", index=False
    )
    pd.DataFrame(
        {
            "group": ["low", "high"],
            "n": [60, 40],
            "source_table": ["cohort_summary.csv"] * 2,
            "source_step_id": ["04_primary"] * 2,
        }
    ).to_csv(out_dir / "primary_or_forest_source_data.csv", index=False)
    (out_dir / "primary_or_forest.png").write_bytes(b"png")
    _write_two_panel_contract(
        out_dir / "primary_or_forest.figure_contract.json",
        figure_id="primary_or_forest",
        source_data=["primary_or_forest_source_data.csv"],
    )
    step = AnalysisStep(
        step_id="04_primary",
        intent="Estimate an adjusted association and render its forest plot.",
        method="logistic_regression",
        expected_outputs=["table:cohort_summary", "figure:primary_or_forest"],
    )

    findings = FigureSourceDataValidator().audit(
        step=step,
        out_dir=out_dir,
        run_dir=tmp_path,
        step_summary={
            "output_files": {
                "table:cohort_summary": "cohort_summary.csv",
                "figure:primary_or_forest": "primary_or_forest.png",
            }
        },
        completed_step_records=[],
        resolved_input_bindings={},
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_source_data_file_cannot_register_itself_as_same_step_parent(
    tmp_path: Path,
):
    out_dir = tmp_path / "steps" / "04_primary" / "outputs"
    out_dir.mkdir(parents=True)
    source_name = "primary_or_forest_source_data.csv"
    pd.DataFrame({"term": ["exposure"], "odds_ratio": [9.99]}).to_csv(
        out_dir / source_name, index=False
    )
    (out_dir / "primary_or_forest.png").write_bytes(b"png")
    _write_two_panel_contract(
        out_dir / "primary_or_forest.figure_contract.json",
        figure_id="primary_or_forest",
        source_data=[source_name],
    )
    step = AnalysisStep(
        step_id="04_primary",
        intent="Estimate an association and render its forest plot.",
        method="logistic_regression",
        expected_outputs=[
            "table:association_estimates",
            "figure:primary_or_forest",
        ],
    )

    findings = FigureSourceDataValidator().audit(
        step=step,
        out_dir=out_dir,
        run_dir=tmp_path,
        step_summary={
            "output_files": {
                "table:association_estimates": source_name,
                "figure:primary_or_forest": "primary_or_forest.png",
            }
        },
        completed_step_records=[],
        resolved_input_bindings={},
    )

    assert [finding for finding in findings if finding.severity == "error"]


def _statistic_figure_fixture(
    tmp_path: Path,
    *,
    source_value: float,
    statistic: str = "auroc",
) -> tuple[Path, AnalysisStep, dict, list[dict], dict]:
    parent_out = tmp_path / "steps" / "03_model" / "outputs"
    figure_out = tmp_path / "steps" / "04_plot" / "outputs"
    evidence_dir = tmp_path / "evidence"
    parent_out.mkdir(parents=True)
    figure_out.mkdir(parents=True)
    evidence_dir.mkdir()
    summary_path = evidence_dir / "model_summary__step_summary.json"
    summary_path.write_text(json.dumps({statistic: 0.81}), encoding="utf-8")
    digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    pd.DataFrame({"metric": [statistic], "value": [source_value]}).to_csv(
        figure_out / "model_performance_source_data.csv", index=False
    )
    (figure_out / "model_performance.png").write_bytes(b"png")
    _write_two_panel_contract(
        figure_out / "model_performance.figure_contract.json",
        figure_id="model_performance",
        source_data=["model_performance_source_data.csv"],
    )
    records = [
        {
            "step_id": "03_model",
            "status": "ok",
            "evidence_ids": ["model_summary"],
            "step_summary_evidence_id": "model_summary",
            "step_summary": {statistic: 0.81},
        }
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "per_step_records": records,
                "evidence": [
                    {
                        "evidence_id": "model_summary",
                        "kind": "log",
                        "relative_path": str(summary_path.relative_to(tmp_path)),
                        "sha256": digest,
                        "produced_by_step": "03_model",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    step = AnalysisStep(
        step_id="04_plot",
        intent="Render current model performance.",
        method="visualization",
        inputs=[f"statistic:{statistic}"],
        expected_outputs=["figure:model_performance"],
    )
    summary = {"output_files": {"figure:model_performance": "model_performance.png"}}
    bindings = {
        f"statistic:{statistic}": {
            "declared_kind": "statistic",
            "product": statistic,
            "produced_by_step": "03_model",
            "evidence_id": "model_summary",
            "sha256": digest,
            "absolute_path": str(summary_path),
        }
    }
    return figure_out, step, summary, records, bindings


def test_truthful_statistic_backed_figure_has_value_lineage(tmp_path: Path):
    figure_out, step, summary, records, bindings = _statistic_figure_fixture(
        tmp_path,
        source_value=0.81,
    )

    findings = FigureSourceDataValidator().audit(
        step=step,
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary=summary,
        completed_step_records=records,
        resolved_input_bindings=bindings,
    )

    assert [finding for finding in findings if finding.severity == "error"] == []


def test_standalone_statistic_artifact_authenticates_current_figure(
    tmp_path: Path,
):
    figure_out, step, summary, records, bindings = _statistic_figure_fixture(
        tmp_path,
        source_value=0.81,
    )
    statistic_path = tmp_path / "evidence" / "metric__auroc.json"
    statistic_path.write_text(
        json.dumps({"name": "auroc", "estimate": 0.81}),
        encoding="utf-8",
    )
    digest = hashlib.sha256(statistic_path.read_bytes()).hexdigest()
    records[0]["evidence_ids"].append("metric_artifact")
    bindings["statistic:auroc"] = {
        "declared_kind": "statistic",
        "product": "auroc",
        "produced_by_step": "03_model",
        "evidence_id": "metric_artifact",
        "sha256": digest,
        "absolute_path": str(statistic_path),
    }

    findings = FigureSourceDataValidator().audit(
        step=step,
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary=summary,
        completed_step_records=records,
        resolved_input_bindings=bindings,
    )

    assert [finding for finding in findings if finding.severity == "error"] == []

    records[0]["evidence_ids"].remove("metric_artifact")
    rejected = FigureSourceDataValidator().audit(
        step=step,
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary=summary,
        completed_step_records=records,
        resolved_input_bindings=bindings,
    )
    assert any(
        finding.detail.get("reason") == "resolved_input_evidence_mismatch"
        for finding in rejected
    )


def test_statistic_source_may_name_its_exact_bound_json_artifact(
    tmp_path: Path,
):
    figure_out, step, summary, records, bindings = _statistic_figure_fixture(
        tmp_path,
        source_value=0.81,
    )
    statistic_path = tmp_path / "evidence" / "metric__auroc.json"
    statistic_path.write_text(
        json.dumps({"name": "auroc", "estimate": 0.81}),
        encoding="utf-8",
    )
    digest = hashlib.sha256(statistic_path.read_bytes()).hexdigest()
    records[0]["evidence_ids"].append("metric_artifact")
    bindings["statistic:auroc"] = {
        "declared_kind": "statistic",
        "product": "auroc",
        "produced_by_step": "03_model",
        "evidence_id": "metric_artifact",
        "sha256": digest,
        "absolute_path": str(statistic_path),
    }
    source_path = figure_out / "model_performance_source_data.csv"
    source = pd.read_csv(source_path)
    source["source_table"] = statistic_path.name
    source.to_csv(source_path, index=False)

    findings = FigureSourceDataValidator().audit(
        step=step,
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary=summary,
        completed_step_records=records,
        resolved_input_bindings=bindings,
    )

    assert [finding for finding in findings if finding.severity == "error"] == []

    source["source_table"] = "foreign_statistic.json"
    source.to_csv(source_path, index=False)
    rejected = FigureSourceDataValidator().audit(
        step=step,
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary=summary,
        completed_step_records=records,
        resolved_input_bindings=bindings,
    )
    assert any(
        finding.detail.get("best_mismatch", {}).get("reason")
        == "declared_source_table_not_found"
        for finding in rejected
    )


def test_statistic_backed_figure_rejects_wrong_source_value(tmp_path: Path):
    figure_out, step, summary, records, bindings = _statistic_figure_fixture(
        tmp_path,
        source_value=0.99,
    )

    findings = FigureSourceDataValidator().audit(
        step=step,
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary=summary,
        completed_step_records=records,
        resolved_input_bindings=bindings,
    )

    errors = [finding for finding in findings if finding.severity == "error"]
    assert errors
    assert any(
        finding.detail.get("reason")
        in {"no_verifiable_figure_values", "incomplete_source_lineage_coverage"}
        for finding in errors
    )


def test_truthful_c_statistic_backed_figure_has_value_lineage(tmp_path: Path):
    figure_out, step, summary, records, bindings = _statistic_figure_fixture(
        tmp_path,
        source_value=0.81,
        statistic="c_statistic",
    )

    findings = FigureSourceDataValidator().audit(
        step=step,
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary=summary,
        completed_step_records=records,
        resolved_input_bindings=bindings,
    )

    assert [finding for finding in findings if finding.severity == "error"] == []


@pytest.mark.parametrize(
    ("extra_metric", "extra_value"),
    [("displayed_auroc", 0.99), ("calibration_slope", 99.0)],
)
def test_statistic_source_rejects_unbound_numeric_payload(
    tmp_path: Path,
    extra_metric: str,
    extra_value: float,
):
    figure_out, step, summary, records, bindings = _statistic_figure_fixture(
        tmp_path,
        source_value=0.81,
    )
    pd.DataFrame(
        {
            "metric": ["auroc", extra_metric],
            "value": [0.81, extra_value],
        }
    ).to_csv(figure_out / "model_performance_source_data.csv", index=False)

    findings = FigureSourceDataValidator().audit(
        step=step,
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary=summary,
        completed_step_records=records,
        resolved_input_bindings=bindings,
    )

    assert [finding for finding in findings if finding.severity == "error"]


@pytest.mark.parametrize(
    ("inputs", "method"),
    [
        (["artifact:predictions"], "visualization"),
        ([], "prediction_model"),
    ],
)
def test_prediction_result_cannot_self_label_audit_to_skip_source_data(
    tmp_path: Path,
    inputs: list[str],
    method: str,
):
    out_dir = tmp_path / "steps" / "04_plot" / "outputs"
    out_dir.mkdir(parents=True)
    (out_dir / "model_performance.png").write_bytes(b"png")
    (out_dir / "model_performance.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": "model_performance",
                "source_data": [],
                "panels": [
                    {
                        "panel_id": "A",
                        "title": "Audit",
                        "role": "audit",
                        "claim": "A coder label cannot downgrade a result figure.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="04_plot",
            intent="Compute and render model performance.",
            method=method,
            inputs=inputs,
            expected_outputs=["figure:model_performance"],
        ),
        out_dir=out_dir,
        run_dir=tmp_path,
        step_summary={
            "output_files": {"figure:model_performance": "model_performance.png"}
        },
    )

    errors = [finding for finding in findings if finding.severity == "error"]
    assert errors
    assert errors[0].detail["reason"] == "missing_source_data"


def test_model_file_alone_cannot_authenticate_plotted_metrics(tmp_path: Path):
    parent_out = tmp_path / "steps" / "03_model" / "outputs"
    figure_out = tmp_path / "steps" / "04_plot" / "outputs"
    evidence_dir = tmp_path / "evidence"
    parent_out.mkdir(parents=True)
    figure_out.mkdir(parents=True)
    evidence_dir.mkdir()
    model_path = evidence_dir / "prediction_model__model.json"
    model_path.write_text('{"model": "sealed"}', encoding="utf-8")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    pd.DataFrame({"metric": ["auroc"], "value": [0.81]}).to_csv(
        figure_out / "model_performance_source_data.csv", index=False
    )
    (figure_out / "model_performance.png").write_bytes(b"png")
    _write_two_panel_contract(
        figure_out / "model_performance.figure_contract.json",
        figure_id="model_performance",
        source_data=["model_performance_source_data.csv"],
    )
    records = [
        {
            "step_id": "03_model",
            "status": "ok",
            "evidence_ids": ["prediction_model"],
        }
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "per_step_records": records,
                "evidence": [
                    {
                        "evidence_id": "prediction_model",
                        "kind": "model",
                        "relative_path": str(model_path.relative_to(tmp_path)),
                        "sha256": digest,
                        "produced_by_step": "03_model",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="04_plot",
            intent="Render model performance.",
            method="visualization",
            inputs=["model:prediction_model"],
            expected_outputs=["figure:model_performance"],
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={
            "output_files": {"figure:model_performance": "model_performance.png"}
        },
        completed_step_records=records,
        resolved_input_bindings={
            "model:prediction_model": {
                "declared_kind": "model",
                "product": "prediction_model",
                "produced_by_step": "03_model",
                "evidence_id": "prediction_model",
                "sha256": digest,
                "absolute_path": str(model_path),
            }
        },
    )

    errors = [finding for finding in findings if finding.severity == "error"]
    assert errors
    assert errors[0].detail["reason"] == "non_replayable_figure_input"


def test_tabular_prediction_dataset_can_authenticate_result_figure(tmp_path: Path):
    figure_out = tmp_path / "steps" / "04_plot" / "outputs"
    evidence_dir = tmp_path / "evidence"
    figure_out.mkdir(parents=True)
    evidence_dir.mkdir()
    predictions_path = evidence_dir / "predictions__predictions.csv"
    frame = pd.DataFrame(
        {"row_id": [1, 2], "predicted_risk": [0.2, 0.8], "outcome": [0, 1]}
    )
    frame.to_csv(predictions_path, index=False)
    digest = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    frame.to_csv(figure_out / "model_performance_source_data.csv", index=False)
    (figure_out / "model_performance.png").write_bytes(b"png")
    _write_two_panel_contract(
        figure_out / "model_performance.figure_contract.json",
        figure_id="model_performance",
        source_data=["model_performance_source_data.csv"],
    )
    records = [
        {
            "step_id": "03_model",
            "status": "ok",
            "evidence_ids": ["predictions"],
        }
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "per_step_records": records,
                "evidence": [
                    {
                        "evidence_id": "predictions",
                        "kind": "table",
                        "relative_path": str(predictions_path.relative_to(tmp_path)),
                        "sha256": digest,
                        "produced_by_step": "03_model",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="04_plot",
            intent="Render model performance from current predictions.",
            method="visualization",
            inputs=["dataset:predictions"],
            expected_outputs=["figure:model_performance"],
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={
            "output_files": {"figure:model_performance": "model_performance.png"}
        },
        completed_step_records=records,
        resolved_input_bindings={
            "dataset:predictions": {
                "declared_kind": "dataset",
                "product": "predictions",
                "produced_by_step": "03_model",
                "evidence_id": "predictions",
                "sha256": digest,
                "absolute_path": str(predictions_path),
            }
        },
    )

    assert [finding for finding in findings if finding.severity == "error"] == []


def _audit_bound_tabular_figures(
    tmp_path: Path,
    *,
    declared_input: str,
    upstream: pd.DataFrame,
    figure_products: list[str],
    source: pd.DataFrame | None = None,
):
    declared_kind, product = declared_input.split(":", 1)
    parent_out = tmp_path / "steps" / "03_parent" / "outputs"
    figure_out = tmp_path / "steps" / "04_plot" / "outputs"
    evidence_dir = tmp_path / "evidence"
    parent_out.mkdir(parents=True)
    figure_out.mkdir(parents=True)
    evidence_dir.mkdir()

    parent_path = parent_out / f"{product}.csv"
    upstream.to_csv(parent_path, index=False)
    digest = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    evidence_id = f"{product}_{digest[:8]}"
    evidence_path = evidence_dir / f"{evidence_id}__{parent_path.name}"
    evidence_path.write_bytes(parent_path.read_bytes())

    source_name = "shared_figure_source_data.csv"
    (source if source is not None else upstream).to_csv(
        figure_out / source_name,
        index=False,
    )
    output_files = {}
    for figure_product in figure_products:
        figure_name = figure_product.split(":", 1)[1]
        figure_path = figure_out / f"{figure_name}.png"
        figure_path.write_bytes(b"png")
        _write_two_panel_contract(
            figure_out / f"{figure_name}.figure_contract.json",
            figure_id=figure_name,
            source_data=[source_name],
        )
        output_files[figure_product] = figure_path.name

    records = [
        {
            "step_id": "03_parent",
            "status": "ok",
            "evidence_ids": [evidence_id],
        }
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "per_step_records": records,
                "evidence": [
                    {
                        "evidence_id": evidence_id,
                        "kind": "table",
                        "relative_path": str(evidence_path.relative_to(tmp_path)),
                        "sha256": digest,
                        "produced_by_step": "03_parent",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    return FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="04_plot",
            intent="Render only the host-bound typed result.",
            method="visualization",
            inputs=[declared_input],
            expected_outputs=figure_products,
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={"output_files": output_files},
        completed_step_records=records,
        resolved_input_bindings={
            declared_input: {
                "declared_kind": declared_kind,
                "product": product,
                "produced_by_step": "03_parent",
                "evidence_id": evidence_id,
                "sha256": digest,
                "absolute_path": str(evidence_path),
            }
        },
    )


def test_cohort_score_columns_cannot_authenticate_model_performance(tmp_path: Path):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="dataset:cohort_summary",
        upstream=pd.DataFrame({"row_id": [1, 2], "severity_score": [3.0, 8.0]}),
        figure_products=["figure:model_performance"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_exact_typed_input_key_can_name_its_hash_verified_source_table(
    tmp_path: Path,
):
    upstream = pd.DataFrame(
        {"group": ["low", "high"], "n": [40, 60], "rate": [0.1, 0.2]}
    )
    source = upstream.assign(source_table="table:grouped_result")
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:grouped_result",
        upstream=upstream,
        source=source,
        figure_products=["figure:outcome_distribution"],
    )
    assert [finding for finding in findings if finding.severity == "error"] == []


def test_foreign_typed_input_key_cannot_name_a_bound_source_table(tmp_path: Path):
    upstream = pd.DataFrame(
        {"group": ["low", "high"], "n": [40, 60], "rate": [0.1, 0.2]}
    )
    source = upstream.assign(source_table="table:foreign_result")
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:grouped_result",
        upstream=upstream,
        source=source,
        figure_products=["figure:outcome_distribution"],
    )
    errors = [finding for finding in findings if finding.severity == "error"]
    assert errors
    assert any(
        (finding.detail or {}).get("reason")
        in {"declared_source_table_not_found", "incomplete_source_lineage_coverage"}
        or (finding.detail or {}).get("best_mismatch", {}).get("reason")
        == "declared_source_table_not_found"
        for finding in errors
    )


def test_value_name_is_not_shadowed_by_semantic_source_metadata(
    tmp_path: Path,
):
    source_path = tmp_path / "figure_source.csv"
    upstream_path = tmp_path / "model_estimates.csv"
    source = pd.DataFrame(
        {
            "term": ["const", "exposure"],
            "effect": [0.25, 1.5],
            "effect_source": ["odds_ratio", "odds_ratio"],
        }
    )
    source.to_csv(source_path, index=False)
    pd.DataFrame(
        {
            "term": ["const", "exposure"],
            "odds_ratio": [0.25, 1.5],
        }
    ).to_csv(upstream_path, index=False)

    comparison = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=source_path,
        upstream_path=upstream_path,
    )

    assert comparison["ok"] is True
    assert comparison["verified_value_mappings"]["effect"] == "odds_ratio"


def test_cohort_counts_cannot_authenticate_compound_prediction_figure(
    tmp_path: Path,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:cohort_summary",
        upstream=pd.DataFrame(
            {
                "group": ["development", "validation"],
                "n": [240, 160],
            }
        ),
        figure_products=["figure:discrimination_calibration"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


@pytest.mark.parametrize(
    "figure_product",
    ["figure:time_varying_discrimination", "figure:subgroup_forest"],
)
def test_cohort_counts_cannot_authenticate_registered_suite_result_figures(
    tmp_path: Path,
    figure_product: str,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:cohort_summary",
        upstream=pd.DataFrame({"group": ["a", "b"], "n": [120, 80]}),
        figure_products=[figure_product],
    )

    assert [finding for finding in findings if finding.severity == "error"]


@pytest.mark.parametrize(
    ("declared_input", "upstream", "figure_product"),
    [
        (
            "table:horizon_performance",
            pd.DataFrame(
                {
                    "row_id": [1, 2, 3],
                    "prediction_horizon_hours": [6, 12, 24],
                    "metric": ["auroc", "auroc", "auroc"],
                    "value": [0.74, 0.79, 0.81],
                }
            ),
            "figure:time_varying_discrimination",
        ),
        (
            "table:subgroup_effects",
            pd.DataFrame(
                {
                    "term": ["subgroup_a"],
                    "odds_ratio": [1.25],
                }
            ),
            "figure:subgroup_forest",
        ),
    ],
)
def test_registered_suite_result_figures_require_matching_typed_sources(
    tmp_path: Path,
    declared_input: str,
    upstream: pd.DataFrame,
    figure_product: str,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input=declared_input,
        upstream=upstream,
        figure_products=[figure_product],
    )

    assert [finding for finding in findings if finding.severity == "error"] == []


@pytest.mark.parametrize(
    "declared_input",
    ["table:model_performance", "table:horizon_performance"],
    ids=["static_product", "horizon_product_without_horizon_values"],
)
def test_static_performance_cannot_authenticate_time_varying_discrimination(
    tmp_path: Path,
    declared_input: str,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input=declared_input,
        upstream=pd.DataFrame({"metric": ["auroc"], "value": [0.81]}),
        figure_products=["figure:time_varying_discrimination"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_time_varying_performance_requires_metric_at_each_counted_horizon(
    tmp_path: Path,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:horizon_performance",
        upstream=pd.DataFrame(
            {
                "row_id": [1, 2],
                "prediction_horizon_hours": [6, 12],
                "metric": ["auroc", "sample_size"],
                "value": [0.81, 200.0],
            }
        ),
        figure_products=["figure:time_varying_discrimination"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_time_varying_discrimination_requires_same_metric_across_horizons(
    tmp_path: Path,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:horizon_performance",
        upstream=pd.DataFrame(
            {
                "row_id": [1, 2],
                "prediction_horizon_hours": [6, 12],
                "metric": ["auroc", "brier"],
                "value": [0.81, 0.12],
            }
        ),
        figure_products=["figure:time_varying_discrimination"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_row_paired_raw_predictions_authenticate_time_varying_performance(
    tmp_path: Path,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="dataset:predictions",
        upstream=pd.DataFrame(
            {
                "row_id": [1, 2, 3, 4],
                "prediction_horizon_hours": [6, 6, 12, 12],
                "prediction": [0.1, 0.8, 0.2, 0.9],
                "outcome": [0, 1, 0, 1],
            }
        ),
        figure_products=["figure:time_varying_discrimination"],
    )

    assert [finding for finding in findings if finding.severity == "error"] == []


@pytest.mark.parametrize(
    "figure_product",
    [
        "figure:roc_curve",
        "figure:calibration_curve",
        "figure:model_performance",
        "figure:discrimination_calibration",
    ],
    ids=["roc", "calibration", "performance", "compound"],
)
def test_prediction_values_without_observed_outcomes_cannot_authenticate_results(
    tmp_path: Path,
    figure_product: str,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="dataset:predictions",
        upstream=pd.DataFrame(
            {
                "row_id": [1, 2, 3],
                "predicted_risk": [0.1, 0.4, 0.8],
            }
        ),
        figure_products=[figure_product],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_disjoint_prediction_and_outcome_rows_cannot_authenticate_results(
    tmp_path: Path,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="dataset:predictions",
        upstream=pd.DataFrame(
            {
                "row_id": [1, 2, 3, 4],
                "predicted_risk": [0.1, 0.8, None, None],
                "outcome": [None, None, 0, 1],
            }
        ),
        figure_products=["figure:discrimination_calibration"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


@pytest.mark.parametrize(
    "upstream",
    [
        pd.DataFrame(
            {
                "row_id": [1, 2, 3],
                "predicted_risk": [-0.1, 0.4, 1.1],
                "outcome": [0, 0, 1],
            }
        ),
        pd.DataFrame(
            {
                "row_id": [1, 2, 3],
                "predicted_risk": [0.1, 0.4, 0.8],
                "outcome": [0, 2, 1],
            }
        ),
    ],
    ids=["risk_outside_probability_domain", "nonbinary_outcome"],
)
def test_raw_prediction_domains_must_support_replayable_binary_risk_figures(
    tmp_path: Path,
    upstream: pd.DataFrame,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="dataset:predictions",
        upstream=upstream,
        figure_products=["figure:discrimination_calibration"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


@pytest.mark.parametrize(
    "upstream",
    [
        pd.DataFrame(
            {
                "row_id": [1, 2, 3],
                "predicted_risk": [-9.0, 2.0, 8.0],
                "prediction": [0.1, 0.4, 0.8],
                "outcome": [0, 0, 1],
            }
        ),
        pd.DataFrame(
            {
                "row_id": [1, 2, 3],
                "predicted_risk": [0.1, 0.4, 0.8],
                "outcome": [9, 8, 7],
                "y_true": [0, 0, 1],
            }
        ),
    ],
    ids=["invalid_probability_with_valid_alias", "invalid_outcome_with_valid_alias"],
)
def test_valid_prediction_alias_cannot_launder_invalid_semantic_sibling(
    tmp_path: Path,
    upstream: pd.DataFrame,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="dataset:predictions",
        upstream=upstream,
        figure_products=["figure:discrimination_calibration"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_roc_table_cannot_authenticate_calibration_curve(tmp_path: Path):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:roc_curve",
        upstream=pd.DataFrame(
            {
                "row_id": [1, 2, 3],
                "threshold": [0.2, 0.5, 0.8],
                "fpr": [0.70, 0.25, 0.05],
                "tpr": [0.95, 0.72, 0.30],
            }
        ),
        figure_products=["figure:calibration_curve"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_calibration_table_cannot_authenticate_roc_curve(tmp_path: Path):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:calibration_curve",
        upstream=pd.DataFrame(
            {
                "row_id": [1, 2, 3],
                "predicted_risk": [0.1, 0.3, 0.7],
                "observed_risk": [0.08, 0.28, 0.74],
            }
        ),
        figure_products=["figure:roc_curve"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_disjoint_predicted_and_observed_risk_cannot_authenticate_calibration(
    tmp_path: Path,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:calibration_curve",
        upstream=pd.DataFrame(
            {
                "row_id": [1, 2, 3, 4],
                "predicted_risk": [0.1, 0.8, None, None],
                "observed_risk": [None, None, 0.2, 0.7],
            }
        ),
        figure_products=["figure:calibration_curve"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_disjoint_roc_coordinates_cannot_authenticate_roc_curve(tmp_path: Path):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:roc_curve",
        upstream=pd.DataFrame(
            {
                "row_id": [1, 2, 3, 4, 5, 6],
                "threshold": [0.1, 0.8, None, None, None, None],
                "fpr": [None, None, 0.2, 0.7, None, None],
                "tpr": [None, None, None, None, 0.3, 0.9],
            }
        ),
        figure_products=["figure:roc_curve"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_disjoint_threshold_and_net_benefit_cannot_authenticate_decision_curve(
    tmp_path: Path,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:decision_curve",
        upstream=pd.DataFrame(
            {
                "row_id": [1, 2, 3, 4],
                "threshold": [0.1, 0.8, None, None],
                "net_benefit": [None, None, 0.02, 0.04],
            }
        ),
        figure_products=["figure:decision_curve"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


@pytest.mark.parametrize(
    ("declared_input", "upstream", "figure_product"),
    [
        (
            "table:roc_curve",
            pd.DataFrame(
                {
                    "threshold": [0.5],
                    "fpr": [0.2],
                    "tpr": [0.8],
                }
            ),
            "figure:roc_curve",
        ),
        (
            "table:calibration_curve",
            pd.DataFrame(
                {
                    "predicted_risk": [0.5],
                    "observed_risk": [0.45],
                }
            ),
            "figure:calibration_curve",
        ),
        (
            "table:decision_curve",
            pd.DataFrame(
                {
                    "threshold": [0.5],
                    "net_benefit": [0.05],
                }
            ),
            "figure:decision_curve",
        ),
    ],
    ids=["roc", "calibration", "decision_curve"],
)
def test_single_point_table_cannot_authenticate_curve_figure(
    tmp_path: Path,
    declared_input: str,
    upstream: pd.DataFrame,
    figure_product: str,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input=declared_input,
        upstream=upstream,
        figure_products=[figure_product],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_valid_fpr_alias_cannot_launder_invalid_false_positive_rate(
    tmp_path: Path,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:roc_curve",
        upstream=pd.DataFrame(
            {
                "row_id": [1, 2],
                "threshold": [0.2, 0.8],
                "fpr": [0.70, 0.10],
                "false_positive_rate": [7.0, -2.0],
                "tpr": [0.90, 0.30],
            }
        ),
        figure_products=["figure:roc_curve"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_out_of_domain_auroc_cannot_authenticate_model_performance(tmp_path: Path):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:model_performance",
        upstream=pd.DataFrame({"metric": ["auroc"], "value": [999.0]}),
        figure_products=["figure:model_performance"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_valid_auroc_row_cannot_launder_out_of_domain_auroc_sibling(
    tmp_path: Path,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:model_performance",
        upstream=pd.DataFrame(
            {
                "row_id": [1, 2],
                "metric": ["auroc", "auroc"],
                "value": [0.81, 999.0],
            }
        ),
        figure_products=["figure:model_performance"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_wide_auroc_point_outside_confidence_interval_cannot_authenticate_figure(
    tmp_path: Path,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:model_performance",
        upstream=pd.DataFrame(
            {
                "auroc": [0.81],
                "auroc_ci_low": [0.82],
                "auroc_ci_high": [0.90],
            }
        ),
        figure_products=["figure:model_performance"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


@pytest.mark.parametrize(
    ("ci_low", "ci_high"),
    [
        (-0.10, 0.90),
        (0.82, 0.90),
        (0.70, 1.20),
    ],
    ids=["negative_lower", "point_below_interval", "upper_above_one"],
)
def test_long_form_auroc_requires_valid_confidence_interval(
    tmp_path: Path,
    ci_low: float,
    ci_high: float,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:model_performance",
        upstream=pd.DataFrame(
            {
                "metric": ["auroc"],
                "value": [0.81],
                "ci_low": [ci_low],
                "ci_high": [ci_high],
            }
        ),
        figure_products=["figure:model_performance"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_auroc_metadata_column_does_not_invalidate_numeric_performance(
    tmp_path: Path,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:model_performance",
        upstream=pd.DataFrame(
            {
                "auroc": [0.81],
                "auroc_ci_method": ["delong"],
            }
        ),
        figure_products=["figure:model_performance"],
    )

    assert [finding for finding in findings if finding.severity == "error"] == []


def test_risk_score_table_cannot_authenticate_model_performance(tmp_path: Path):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:risk_score",
        upstream=pd.DataFrame(
            {
                "row_id": [1, 2, 3],
                "risk_score": [2.0, 5.0, 9.0],
            }
        ),
        figure_products=["figure:model_performance"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


@pytest.mark.parametrize(
    "figure_product",
    [
        "figure:roc_curve",
        "figure:calibration_curve",
        "figure:model_performance",
        "figure:discrimination_calibration",
    ],
    ids=["roc", "calibration", "performance", "compound"],
)
def test_raw_predictions_with_observed_outcomes_authenticate_prediction_results(
    tmp_path: Path,
    figure_product: str,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="dataset:predictions",
        upstream=pd.DataFrame(
            {
                "row_id": [1, 2, 3, 4],
                "prediction": [0.05, 0.25, 0.65, 0.90],
                "outcome": [0, 0, 1, 1],
            }
        ),
        figure_products=[figure_product],
    )

    assert [finding for finding in findings if finding.severity == "error"] == []


def test_untyped_generic_subgroup_estimate_cannot_authenticate_subgroup_forest(
    tmp_path: Path,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:subgroup_effects",
        upstream=pd.DataFrame({"term": ["subgroup_a"], "estimate": [999.0]}),
        figure_products=["figure:subgroup_forest"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_negative_ratio_cannot_authenticate_subgroup_forest(tmp_path: Path):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:subgroup_effects",
        upstream=pd.DataFrame(
            {
                "row_id": [1],
                "term": ["subgroup_a"],
                "odds_ratio": [-1.25],
            }
        ),
        figure_products=["figure:subgroup_forest"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_negative_ratio_interval_bound_cannot_authenticate_subgroup_forest(
    tmp_path: Path,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:subgroup_effects",
        upstream=pd.DataFrame(
            {
                "row_id": [1],
                "term": ["subgroup_a"],
                "odds_ratio": [1.25],
                "ci_low": [-0.2],
                "ci_high": [1.6],
            }
        ),
        figure_products=["figure:subgroup_forest"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_bare_lower_upper_negative_ratio_interval_cannot_authenticate_forest(
    tmp_path: Path,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:subgroup_effects",
        upstream=pd.DataFrame(
            {
                "row_id": [1],
                "term": ["subgroup_a"],
                "odds_ratio": [1.25],
                "lower": [-0.2],
                "upper": [1.6],
            }
        ),
        figure_products=["figure:subgroup_forest"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_ratio_point_outside_positive_interval_cannot_authenticate_subgroup_forest(
    tmp_path: Path,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:subgroup_effects",
        upstream=pd.DataFrame(
            {
                "row_id": [1],
                "term": ["subgroup_a"],
                "odds_ratio": [1.25],
                "ci_low": [1.5],
                "ci_high": [2.0],
            }
        ),
        figure_products=["figure:subgroup_forest"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_ratio_interval_validation_ignores_unrelated_signed_effect_interval(
    tmp_path: Path,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:subgroup_effects",
        upstream=pd.DataFrame(
            {
                "row_id": [1],
                "term": ["subgroup_a"],
                "odds_ratio": [1.25],
                "or_ci_low": [0.8],
                "or_ci_high": [1.6],
                "mean_difference_ci_low": [-2.0],
                "mean_difference_ci_high": [1.0],
            }
        ),
        figure_products=["figure:subgroup_forest"],
    )

    assert [finding for finding in findings if finding.severity == "error"] == []


def test_explicit_ratio_interval_takes_precedence_over_generic_log_scale_interval(
    tmp_path: Path,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:primary_association",
        upstream=pd.DataFrame(
            {
                "term": ["exposure"],
                "coef": [0.18],
                "ci_lower": [-0.25],
                "ci_upper": [0.61],
                "odds_ratio": [1.20],
                "or_lower": [0.78],
                "or_upper": [1.84],
            }
        ),
        figure_products=["figure:primary_or_forest"],
    )

    assert [finding for finding in findings if finding.severity == "error"] == []


@pytest.mark.parametrize(
    ("declared_input", "upstream", "figure_product"),
    [
        (
            "table:roc_curve",
            pd.DataFrame(
                {
                    "row_id": [1, 2, 3],
                    "threshold": [0.2, 0.5, 0.8],
                    "fpr": [0.70, 0.25, 0.05],
                    "tpr": [0.95, 0.72, 0.30],
                }
            ),
            "figure:roc_curve",
        ),
        (
            "table:calibration_curve",
            pd.DataFrame(
                {
                    "row_id": [1, 2, 3],
                    "predicted_risk": [0.1, 0.3, 0.7],
                    "observed_risk": [0.08, 0.28, 0.74],
                }
            ),
            "figure:calibration_curve",
        ),
        (
            "table:model_performance",
            pd.DataFrame(
                {
                    "metric": ["auroc", "brier"],
                    "value": [0.81, 0.16],
                }
            ),
            "figure:model_performance",
        ),
    ],
    ids=["roc", "calibration", "performance"],
)
def test_typed_prediction_result_tables_authenticate_matching_figures(
    tmp_path: Path,
    declared_input: str,
    upstream: pd.DataFrame,
    figure_product: str,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input=declared_input,
        upstream=upstream,
        figure_products=[figure_product],
    )

    assert [finding for finding in findings if finding.severity == "error"] == []


@pytest.mark.parametrize(
    "upstream",
    [
        pd.DataFrame({"term": ["exposure"], "n": [120]}),
        pd.DataFrame({"term": ["exposure"], "n": [120], "odds_ratio": [float("nan")]}),
    ],
    ids=["no_effect_value", "non_finite_effect_value"],
)
def test_typed_effect_table_requires_finite_effect_value(
    tmp_path: Path,
    upstream: pd.DataFrame,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:primary_or",
        upstream=upstream,
        figure_products=["figure:primary_or_forest"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_false_adjusted_flag_cannot_authorize_adjusted_effect_figure(tmp_path: Path):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:primary_or",
        upstream=pd.DataFrame(
            {
                "term": ["exposure"],
                "odds_ratio": [1.25],
                "adjusted": [False],
            }
        ),
        figure_products=["figure:adjusted_or_forest"],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def test_uppercase_or_column_authenticates_typed_primary_or_figure(tmp_path: Path):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:primary_or",
        upstream=pd.DataFrame({"term": ["exposure"], "OR": [1.25]}),
        figure_products=["figure:primary_or_forest"],
    )

    assert [finding for finding in findings if finding.severity == "error"] == []


@pytest.mark.parametrize(
    "other_figure",
    ["figure:primary_hr_forest", "figure:model_performance"],
)
def test_one_or_source_cannot_authenticate_incompatible_sibling_figure(
    tmp_path: Path,
    other_figure: str,
):
    findings = _audit_bound_tabular_figures(
        tmp_path,
        declared_input="table:primary_or",
        upstream=pd.DataFrame({"term": ["exposure"], "odds_ratio": [1.25]}),
        figure_products=["figure:primary_or_forest", other_figure],
    )

    assert [finding for finding in findings if finding.severity == "error"]


def _write_bound_effect_parent(
    tmp_path: Path,
    *,
    step_id: str,
    product: str,
    odds_ratio: float,
) -> tuple[dict, dict]:
    out_dir = tmp_path / "steps" / step_id / "outputs"
    evidence_dir = tmp_path / "evidence"
    out_dir.mkdir(parents=True)
    evidence_dir.mkdir(exist_ok=True)
    path = out_dir / f"{product}.csv"
    pd.DataFrame({"term": ["exposure"], "odds_ratio": [odds_ratio]}).to_csv(
        path, index=False
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence_id = f"{product}_{digest[:8]}"
    evidence_path = evidence_dir / f"{evidence_id}__{path.name}"
    evidence_path.write_bytes(path.read_bytes())
    evidence = {
        "evidence_id": evidence_id,
        "kind": "table",
        "relative_path": str(evidence_path.relative_to(tmp_path)),
        "sha256": digest,
        "produced_by_step": step_id,
    }
    binding = {
        "declared_kind": "table",
        "product": product,
        "produced_by_step": step_id,
        "evidence_id": evidence_id,
        "sha256": digest,
    }
    return evidence, binding


def test_multi_parent_effect_figure_requires_source_coverage_for_every_parent(
    tmp_path: Path,
):
    primary_evidence, primary_binding = _write_bound_effect_parent(
        tmp_path,
        step_id="03_primary",
        product="primary_or",
        odds_ratio=1.25,
    )
    robust_evidence, robust_binding = _write_bound_effect_parent(
        tmp_path,
        step_id="04_robust",
        product="robust_or_estimates",
        odds_ratio=1.20,
    )
    figure_out = tmp_path / "steps" / "05_plot" / "outputs"
    figure_out.mkdir(parents=True)
    pd.DataFrame({"term": ["exposure"], "odds_ratio": [1.25]}).to_csv(
        figure_out / "primary_or_forest_source_data.csv", index=False
    )
    (figure_out / "primary_or_forest.png").write_bytes(b"png")
    _write_two_panel_contract(
        figure_out / "primary_or_forest.figure_contract.json",
        figure_id="primary_or_forest",
        source_data=["primary_or_forest_source_data.csv"],
    )
    records = [
        {
            "step_id": "03_primary",
            "status": "ok",
            "evidence_ids": [primary_evidence["evidence_id"]],
        },
        {
            "step_id": "04_robust",
            "status": "ok",
            "evidence_ids": [robust_evidence["evidence_id"]],
        },
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "per_step_records": records,
                "evidence": [primary_evidence, robust_evidence],
            }
        ),
        encoding="utf-8",
    )

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="05_plot",
            intent="Render primary and robustness estimates.",
            method="visualization",
            inputs=["table:primary_or", "table:robust_or_estimates"],
            expected_outputs=["figure:primary_or_forest"],
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={
            "output_files": {"figure:primary_or_forest": "primary_or_forest.png"}
        },
        completed_step_records=records,
        resolved_input_bindings={
            "table:primary_or": primary_binding,
            "table:robust_or_estimates": robust_binding,
        },
    )

    errors = [finding for finding in findings if finding.severity == "error"]
    assert errors
    assert any(
        finding.detail.get("reason") == "incomplete_source_lineage_coverage"
        and "robust_or_estimates.csv" in finding.detail.get("missing_bound_tables", [])
        for finding in errors
    )


@pytest.mark.parametrize("same_step", [False, True])
def test_truthful_tsv_parent_is_supported(tmp_path: Path, same_step: bool):
    if same_step:
        out_dir = tmp_path / "steps" / "04_plot" / "outputs"
        out_dir.mkdir(parents=True)
        pd.DataFrame({"group": ["low", "high"], "mortality_rate": [0.1, 0.2]}).to_csv(
            out_dir / "outcome_by_group.tsv", sep="\t", index=False
        )
        pd.DataFrame({"group": ["low", "high"], "mortality_rate": [0.1, 0.2]}).to_csv(
            out_dir / "outcome_distribution_source_data.csv", index=False
        )
        (out_dir / "outcome_distribution.png").write_bytes(b"png")
        _write_two_panel_contract(
            out_dir / "outcome_distribution.figure_contract.json",
            figure_id="outcome_distribution",
            source_data=["outcome_distribution_source_data.csv"],
        )
        findings = FigureSourceDataValidator().audit(
            step=AnalysisStep(
                step_id="04_plot",
                intent="Summarize and render current outcomes.",
                method="descriptive_analysis",
                expected_outputs=[
                    "table:outcome_by_group",
                    "figure:outcome_distribution",
                ],
            ),
            out_dir=out_dir,
            run_dir=tmp_path,
            step_summary={
                "output_files": {
                    "table:outcome_by_group": "outcome_by_group.tsv",
                    "figure:outcome_distribution": "outcome_distribution.png",
                }
            },
            completed_step_records=[],
            resolved_input_bindings={},
        )
    else:
        parent_out = tmp_path / "steps" / "03_parent" / "outputs"
        figure_out = tmp_path / "steps" / "04_plot" / "outputs"
        evidence_dir = tmp_path / "evidence"
        parent_out.mkdir(parents=True)
        figure_out.mkdir(parents=True)
        evidence_dir.mkdir()
        parent_path = parent_out / "outcome_by_group.tsv"
        pd.DataFrame({"group": ["low", "high"], "mortality_rate": [0.1, 0.2]}).to_csv(
            parent_path, sep="\t", index=False
        )
        digest = hashlib.sha256(parent_path.read_bytes()).hexdigest()
        evidence_id = f"outcome_by_group_{digest[:8]}"
        evidence_path = evidence_dir / f"{evidence_id}__{parent_path.name}"
        evidence_path.write_bytes(parent_path.read_bytes())
        pd.DataFrame({"group": ["low", "high"], "mortality_rate": [0.1, 0.2]}).to_csv(
            figure_out / "outcome_distribution_source_data.csv", index=False
        )
        (figure_out / "outcome_distribution.png").write_bytes(b"png")
        _write_two_panel_contract(
            figure_out / "outcome_distribution.figure_contract.json",
            figure_id="outcome_distribution",
            source_data=["outcome_distribution_source_data.csv"],
        )
        records = [
            {
                "step_id": "03_parent",
                "status": "ok",
                "evidence_ids": [evidence_id],
            }
        ]
        (tmp_path / "manifest.json").write_text(
            json.dumps(
                {
                    "per_step_records": records,
                    "evidence": [
                        {
                            "evidence_id": evidence_id,
                            "kind": "table",
                            "relative_path": str(evidence_path.relative_to(tmp_path)),
                            "sha256": digest,
                            "produced_by_step": "03_parent",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        findings = FigureSourceDataValidator().audit(
            step=AnalysisStep(
                step_id="04_plot",
                intent="Render the current outcome table.",
                method="visualization",
                inputs=["table:outcome_by_group"],
                expected_outputs=["figure:outcome_distribution"],
            ),
            out_dir=figure_out,
            run_dir=tmp_path,
            step_summary={
                "output_files": {
                    "figure:outcome_distribution": "outcome_distribution.png"
                }
            },
            completed_step_records=records,
            resolved_input_bindings={
                "table:outcome_by_group": {
                    "declared_kind": "table",
                    "product": "outcome_by_group",
                    "produced_by_step": "03_parent",
                    "evidence_id": evidence_id,
                    "sha256": digest,
                }
            },
        )

    assert [finding for finding in findings if finding.severity == "error"] == []
