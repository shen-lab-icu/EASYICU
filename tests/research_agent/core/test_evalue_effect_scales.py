"""Finalization preserves the declared effect scale and refuses missing conversions."""

import csv
import json
import math

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.contracts.post_analysis import EValueConversionSpec
from easyicu.research_agent.methods.sensitivity import compute_e_value
from easyicu.research_agent.orchestration.finalize import (
    _primary_association_evalue_rows,
    _write_primary_association_evalue_artifacts,
)


def _effect(tmp_path, scale, estimate=1.8, low=1.2, high=2.7, **extra):
    path = tmp_path / "synthetic_effect.csv"
    row = dict(estimate=estimate, ci_low=low, ci_high=high,
               effect_scale=scale, exposure="synthetic_marker", **extra)
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return path


@pytest.mark.parametrize("scale", ["risk_ratio", "hazard_ratio"])
def test_ratio_is_finalized_without_an_or_baseline_contract(tmp_path, scale):
    path = _effect(tmp_path, scale)
    evidence = EvidenceStore(tmp_path)
    record = evidence.register_file(
        kind="table", description="Synthetic primary effect", source_path=path,
        evidence_id="primary_association", produced_by_step="primary",
        producer="synthetic_owner", generation_mode="system",
    )
    artifacts = _write_primary_association_evalue_artifacts(
        evidence=evidence, run_dir=tmp_path, spec=None,
        per_step_records=[dict(step_id="primary", status="ok", evidence_ids=[record.evidence_id])],
    )
    assert artifacts is not None
    rows = list(csv.DictReader(artifacts.csv_path.open()))
    assert len(rows) == 1
    assert rows[0]["effect_scale"] == scale
    assert float(rows[0]["e_value"]) > 1
    assert rows[0]["odds_ratio"] == ""
    assert scale in artifacts.markdown_path.read_text()


@pytest.mark.parametrize("estimate,ci", [(2.0, (1.2, 3.0)), (0.5, (0.3, 0.8)), (1.4, (0.9, 1.8))])
def test_hr_uses_common_outcome_approximation_point_and_bounds(tmp_path, estimate, ci):
    def converted(hr):
        return (1 - 0.5 ** math.sqrt(hr)) / (1 - 0.5 ** math.sqrt(1 / hr))

    expected = compute_e_value(estimate=converted(estimate),
                               ci=tuple(converted(value) for value in ci), estimate_type="rr")
    path = _effect(tmp_path, "hazard_ratio", estimate, *ci)
    row, = _primary_association_evalue_rows(path, baseline_prevalence=0.4)
    assert row["e_value"] == pytest.approx(expected.e_value)
    assert row["e_value_lower_bound"] == pytest.approx(expected.e_value_lower_bound)
    assert "common" in row["note"].lower()


@pytest.mark.parametrize("estimate,low,high", [(0.1, 0.02, 0.2), (-0.1, -0.15, -0.02), (0, -0.05, 0.05)])
def test_rd_uses_only_its_explicit_reference_risk(tmp_path, estimate, low, high):
    path = _effect(tmp_path, "risk_difference", estimate, low, high, reference_risk=0.2)
    row, = _primary_association_evalue_rows(path, baseline_prevalence=0.7)
    expected = compute_e_value(estimate=1 + estimate / 0.2,
                               ci=(1 + low / 0.2, 1 + high / 0.2), estimate_type="rr")
    assert row["e_value"] == pytest.approx(expected.e_value)
    assert row["e_value_lower_bound"] == pytest.approx(expected.e_value_lower_bound)
    assert row["reference_risk"] == 0.2
    assert "fixed" in row["note"].lower()


@pytest.mark.parametrize("extra,reason", [
    ({}, "e_value_rd_reference_risk_missing"),
    ({"reference_risk": 0}, "e_value_rd_reference_risk_invalid"),
    ({"reference_risk": 0.95}, "e_value_rd_implied_risk_invalid"),
])
def test_rd_without_a_defined_conversion_has_a_reason(tmp_path, extra, reason):
    row, = _primary_association_evalue_rows(
        _effect(tmp_path, "risk_difference", 0.1, 0.02, 0.2, **extra), baseline_prevalence=0.2,
    )
    assert row["e_value"] is None
    assert row["reason_code"] == reason


def test_rr_keeps_its_scale_even_with_a_conflicting_legacy_or(tmp_path):
    row, = _primary_association_evalue_rows(
        _effect(tmp_path, "risk_ratio", odds_ratio=99), baseline_prevalence=0.2,
    )
    assert row["e_value"] == pytest.approx(compute_e_value(estimate=1.8, estimate_type="rr").e_value)


def test_legacy_or_still_uses_the_observed_baseline(tmp_path):
    path = tmp_path / "legacy.csv"
    path.write_text("odds_ratio,or_lower,or_upper\n1.8,1.2,2.7\n")
    row, = _primary_association_evalue_rows(path, baseline_prevalence=0.2)
    expected = compute_e_value(estimate=1.8, ci=(1.2, 2.7), estimate_type="or", baseline_prevalence=0.2)
    assert row["odds_ratio"] == 1.8
    assert row["e_value"] == expected.e_value
    rr = 1.8 / (0.8 + 0.2 * 1.8)
    bound_rr = 1.2 / (0.8 + 0.2 * 1.2)
    assert row["e_value"] == pytest.approx(rr + math.sqrt(rr * (rr - 1)))
    assert row["e_value_lower_bound"] == pytest.approx(bound_rr + math.sqrt(bound_rr * (bound_rr - 1)))


@pytest.mark.parametrize("scale,alias,estimate,low,high,kind", [
    ("odds_ratio", "odds_ratio", 1.8, 1.2, 2.7, "or"),
    ("odds_ratio", "or", 1.8, 1.2, 2.7, "or"),
    ("odds_ratio", "OR", 1.8, 1.2, 2.7, "or"),
    ("risk_ratio", "risk_ratio", 1.8, 1.2, 2.7, "rr"),
    ("hazard_ratio", "hazard_ratio", 1.8, 1.2, 2.7, "hr"),
    ("risk_difference", "risk_difference", -0.1, -0.15, -0.02, "rr"),
])
@pytest.mark.parametrize("empty_estimate", [None, ""])
def test_declared_scale_reads_its_named_point_estimate(
    tmp_path, scale, alias, estimate, low, high, kind, empty_estimate,
):
    # Association coefficients and their intervals also occur on the log scale.
    # Only the named effect and its effect-scale bounds are E-value inputs.
    row = dict(term="synthetic_marker", effect_scale=scale, coef=-0.5,
               ci_lower=-0.7, ci_upper=-0.3, ci_low=low, ci_high=high,
               reference_risk=0.2)
    row[alias] = estimate
    if empty_estimate is not None:
        row["estimate"] = empty_estimate
    path = tmp_path / "association.csv"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    result, = _primary_association_evalue_rows(path, baseline_prevalence=0.2)
    expected = compute_e_value(
        estimate=1 + estimate / 0.2 if scale == "risk_difference" else estimate,
        ci=(1 + low / 0.2, 1 + high / 0.2) if scale == "risk_difference" else (low, high),
        estimate_type=kind, baseline_prevalence=0.2,
    )
    assert result["estimate"] == estimate
    assert result["effect_scale"] == scale
    assert result["reason_code"] == ""
    assert result["e_value"] == pytest.approx(expected.e_value)
    assert result["e_value_lower_bound"] == pytest.approx(expected.e_value_lower_bound)


@pytest.mark.parametrize("scale,estimate,low,high", [
    ("odds_ratio", 1.8, 1.2, 2.7),
    ("risk_ratio", 1.8, 1.2, 2.7),
    ("hazard_ratio", 1.8, 1.2, 2.7),
    ("risk_difference", -0.1, -0.15, -0.02),
])
def test_named_effect_reaches_the_registered_evalue_artifact(tmp_path, scale, estimate, low, high):
    path = _effect(tmp_path, scale, "", low, high, **{scale: estimate, "reference_risk": 0.2})
    evidence = EvidenceStore(tmp_path)
    primary = evidence.register_file(
        kind="table", description="Synthetic named association effect", source_path=path,
        evidence_id="primary_association", produced_by_step="primary",
    )
    ledger = [dict(step_id="primary", status="ok", evidence_ids=[primary.evidence_id])]
    spec = None
    if scale == "odds_ratio":
        rate_path = tmp_path / "outcome_rate.csv"
        rate_path.write_text("population,outcome_rate\nanalysis_cohort,0.2\n")
        rate = evidence.register_file(
            kind="table", description="Synthetic declared baseline risk", source_path=rate_path,
            evidence_id="outcome_rate", produced_by_step="outcome",
        )
        ledger.append(dict(step_id="outcome", status="ok", evidence_ids=[rate.evidence_id]))
        spec = EValueConversionSpec(
            baseline_risk_column="outcome_rate", population_column="population", baseline_population="analysis_cohort",
        )
    artifacts = _write_primary_association_evalue_artifacts(
        evidence=evidence, per_step_records=ledger, run_dir=tmp_path, spec=spec,
    )
    assert artifacts is not None and artifacts.row_count == 1
    row, = list(csv.DictReader(artifacts.csv_path.open()))
    assert row["effect_scale"] == scale and float(row["estimate"]) == estimate
    assert float(row["e_value"]) > 1 and float(row["e_value_lower_bound"]) > 1
    assert row["reason_code"] == ""
    assert evidence.get("e_values") is not None


@pytest.mark.parametrize("estimate", [0, -1, "bad_effect", "nan", "inf"])
def test_an_invalid_declared_estimate_cannot_be_replaced_by_an_alias(tmp_path, estimate):
    row, = _primary_association_evalue_rows(
        _effect(tmp_path, "odds_ratio", estimate, odds_ratio=1.8), baseline_prevalence=0.2,
    )
    assert row["e_value"] is None
    assert row["reason_code"] == "e_value_effect_invalid"


def test_declared_estimate_takes_precedence_over_a_conflicting_same_scale_alias(tmp_path):
    row, = _primary_association_evalue_rows(
        _effect(tmp_path, "odds_ratio", odds_ratio=99), baseline_prevalence=0.2,
    )
    assert row["estimate"] == 1.8
    assert row["e_value"] == pytest.approx(compute_e_value(
        estimate=1.8, estimate_type="or", baseline_prevalence=0.2,
    ).e_value)


@pytest.mark.parametrize("scale", ["odds_ratio", "risk_ratio", "hazard_ratio", "risk_difference"])
def test_an_explicit_intercept_is_not_an_association_effect(tmp_path, scale):
    path = _effect(tmp_path, scale, term_role="intercept", reference_risk=0.2)
    assert _primary_association_evalue_rows(path, baseline_prevalence=0.2) == []


@pytest.mark.parametrize("extra", [{"coef": 0.5}, {"odds_ratio": 1.8}])
def test_a_missing_rr_estimate_does_not_borrow_another_scale(tmp_path, extra):
    row, = _primary_association_evalue_rows(
        _effect(tmp_path, "risk_ratio", "", **extra), baseline_prevalence=0.2,
    )
    assert row["e_value"] is None
    assert row["reason_code"] == "e_value_effect_invalid"


@pytest.mark.parametrize("scale,kind", [("hazard_ratio", "hr"), ("risk_ratio", "rr")])
def test_named_legacy_ratio_is_read_without_guessing_its_scale(tmp_path, scale, kind):
    path = tmp_path / "named_ratio.csv"
    path.write_text(f"term,{scale},ci_low,ci_high\nsynthetic_marker,1.8,1.2,2.7\n")
    row, = _primary_association_evalue_rows(path, baseline_prevalence=None)
    assert row["effect_scale"] == scale
    assert row["e_value"] == pytest.approx(compute_e_value(estimate=1.8, ci=(1.2, 2.7), estimate_type=kind).e_value)


@pytest.mark.parametrize("estimate", [0, -1, float("inf"), float("nan")])
def test_kernel_refuses_invalid_ratios(estimate):
    with pytest.raises(ValueError, match="finite positive"):
        compute_e_value(estimate=estimate, estimate_type="hr")


def test_invalid_interval_has_a_reason(tmp_path):
    row, = _primary_association_evalue_rows(_effect(tmp_path, "risk_ratio", 2, 3, 4), baseline_prevalence=None)
    assert row["reason_code"] == "e_value_interval_invalid" and row["e_value"] is None


@pytest.mark.parametrize("low,high", [("", 3), (1.2, "not_a_bound"), (float("inf"), 3)])
def test_malformed_interval_is_not_silently_removed(tmp_path, low, high):
    row, = _primary_association_evalue_rows(_effect(tmp_path, "risk_ratio", 2, low, high), baseline_prevalence=None)
    assert row["reason_code"] == "e_value_interval_invalid" and row["e_value"] is None


def test_missing_or_conversion_keeps_its_existing_reason(tmp_path):
    row, = _primary_association_evalue_rows(_effect(tmp_path, "odds_ratio"), baseline_prevalence=None)
    assert row["reason_code"] == "e_value_baseline_prevalence_unresolved" and row["e_value"] is None


def test_unconvertible_rd_is_an_audited_omission_not_a_computed_effect(tmp_path):
    path = _effect(tmp_path, "risk_difference", 0.1, 0.02, 0.2)
    evidence = EvidenceStore(tmp_path)
    record = evidence.register_file(kind="table", description="Synthetic risk difference", source_path=path,
                                    evidence_id="primary_association", produced_by_step="primary")
    artifacts = _write_primary_association_evalue_artifacts(
        evidence=evidence, per_step_records=[dict(step_id="primary", status="ok", evidence_ids=[record.evidence_id])],
        run_dir=tmp_path, spec=None,
    )
    assert artifacts is not None and artifacts.row_count == 0
    assert "e_value_rd_reference_risk_missing" in artifacts.markdown_path.read_text()


def test_or_finalization_preserves_the_unresolved_baseline_finding(tmp_path):
    from easyicu.research_agent.orchestration.finalize import _EValueBaselineUnresolved

    evidence = EvidenceStore(tmp_path)
    record = evidence.register_file(kind="table", description="Synthetic odds ratio", source_path=_effect(tmp_path, "odds_ratio"),
                                    evidence_id="primary_association", produced_by_step="primary")
    with pytest.raises(_EValueBaselineUnresolved) as stopped:
        _write_primary_association_evalue_artifacts(
            evidence=evidence, per_step_records=[dict(step_id="primary", status="ok", evidence_ids=[record.evidence_id])],
            run_dir=tmp_path, spec=None,
        )
    assert stopped.value.resolved.cause == "evalue_conversion_spec_required"
    assert not (tmp_path / "e_values.csv").exists()


def _survival_summary(*, continuous, rejected):
    ratio = dict(hazard_ratio=1.8, ci_low=1.2, ci_high=2.7)
    interval = dict(hazard_ratio=2.0, ci_low=1.3, ci_high=3.0,
                    start_days=0.0, end_days=10.0, p_value=0.01)
    report = dict(
        schema_version="easyicu.survival_reporting/2", execution_owner="landmark_survival_executor_v1",
        interpretation_ceiling="descriptive_prognostic_association_not_causal",
        exposure="synthetic_marker", outcome="synthetic_event", analysis_unit="stays", landmark_hours=12.0,
        adjustment_columns=[], constant_hazard_ratio_authorized=not rejected,
        proportional_hazards_status="violation_block_paper_authorization" if rejected else "not_rejected",
        proportional_hazards_test=dict(
            schema_version="easyicu.prespecified_rule_outcome/1", rule="proportional_hazards_test",
            diagnostic="schoenfeld_residual_test", alpha=0.05,
            global_p_value=0.01 if rejected else 0.5, exposure_p_value=0.01 if rejected else 0.5,
            disposition="assumption_rejected" if rejected else "assumption_not_rejected",
        ),
        time_varying_adjusted_association=dict(method="piecewise_time_varying_cox", adjustment_columns=[],
                                             intervals=[interval, dict(interval, start_days=10.0, end_days=20.0)]),
        manuscript_projection={},
    )
    if continuous:
        report.update(schema_version="easyicu.continuous_survival_reporting/2",
                      execution_owner="landmark_continuous_survival_executor_v1", exposure_increment=1.0,
                      primary_estimate="interval_per_step_hazard_ratios" if rejected else "per_step_hazard_ratio",
                      functional_form=dict(method="restricted_cubic_spline_likelihood_ratio_test", status="not_estimable",
                                           reason="tied_knots", knot_percentiles=[10.0, 50.0, 90.0],
                                           alpha=0.05, disposition="not_assessable"))
        report["time_varying_adjusted_association"]["status"] = "estimated"
        if not rejected:
            report["adjusted_hazard_ratio_per_unit"] = ratio
    else:
        report.update(contrast="present vs absent", rmst={})
        if not rejected:
            report["adjusted_hazard_ratio"] = ratio
    return dict(status="ok", reportable_survival_results=report)


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("rejected", [False, True])
def test_finalization_uses_selected_survival_claims_without_a_csv_alias(tmp_path, continuous, rejected):
    summary = _survival_summary(continuous=continuous, rejected=rejected)
    path = tmp_path / "step_summary.json"
    path.write_text(json.dumps(summary))
    evidence = EvidenceStore(tmp_path)
    record = evidence.register_file(kind="log", description="Synthetic selected survival results", source_path=path,
                                    evidence_id="suite_summary", produced_by_step="suite", producer="host",
                                    generation_mode="deterministic_standard")
    claims = evidence.register_step_summary_scientific_claims(step_id="suite", evidence_id=record.evidence_id, summary=summary)
    selected = [claim for claim in claims if claim.analysis_role == "primary" and claim.claim_type == "association"]
    ledger = [dict(step_id="suite", status="ok", evidence_ids=[record.evidence_id])]
    artifacts = _write_primary_association_evalue_artifacts(evidence=evidence, per_step_records=ledger, run_dir=tmp_path, spec=None)
    assert artifacts is not None
    rows = list(csv.DictReader(artifacts.csv_path.open()))
    assert len(rows) == artifacts.row_count == len(selected) == (2 if rejected else 1)
    assert rows[0]["term"] == selected[0].claim_ref
    assert rows[0]["effect_scale"] == "hazard_ratio"
    assert float(rows[0]["estimate"]) == (2.0 if rejected else 1.8)
    assert float(rows[0]["e_value"]) == pytest.approx(compute_e_value(
        estimate=selected[0].point_estimate, estimate_type="hr",
    ).e_value)
    result_record = next(item for item in evidence.records() if item.evidence_id == artifacts.evidence_id)
    assert record.evidence_id in result_record.inputs
    assert _write_primary_association_evalue_artifacts(
        evidence=evidence, per_step_records=[dict(step_id="suite", status="failed", evidence_ids=[])], run_dir=tmp_path, spec=None,
    ) is None
