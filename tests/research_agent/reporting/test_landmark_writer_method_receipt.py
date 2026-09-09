from __future__ import annotations

import copy

import pytest

from easyicu.research_agent.audits.envelope_consumers import (
    RegisteredOutputAuthorityError, RegisteredOutputEnvelopeConsumer,
)
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.contracts.result_envelope import normalize_step_result_shadow
from easyicu.research_agent.reporting.writer_evidence import _executed_method_boundary_rows
from ..execution.test_step_result_envelope import (
    _UPSTREAM_STEP, _commit_upstream_sidecar, _modern_upstream_record, _register_upstream_table,
)


def _receipt(clustered):
    common = {
        "comparison": "restricted_cubic_spline_vs_linear", "degrees_of_freedom": 1,
        "p_value": 0.2, "linear_aic": 80.0, "spline_aic": 82.0,
        "linear_bic": 84.0, "spline_bic": 88.0,
    }
    common.update({
        "method": "cluster_robust_nested_wald_chi2", "target_column": "oxygen_index",
        "information_criteria_basis": "working_independence_loglikelihood_descriptive_only",
        "statistic": 1.64,
    } if clustered else {"likelihood_ratio_statistic": 1.64})
    receipt = {
        "schema_version": f"easyicu.landmark_spline_runtime_receipt/{4 if clustered else 2}",
        "protocol_content_sha256": "a" * 64, "execution_contract_sha256": "b" * 64,
        "runtime_projection_sha256": "c" * 64,
        "landmark_hours": 24,
        "population_rule": "alive_and_under_observation_at_landmark_with_valid_exposure",
        "spline_knot_quantiles": [0.1, 0.5, 0.9], "observed_knots": [1.0, 2.0, 5.0],
        "adjustment_columns": ["age", "severity_index"], "primary_population_n": 100,
        "complete_case_n": 90, "events": 12,
        "functional_form_comparison": common,
        "population_flow": [
            {"stage": stage, "n": n, "excluded_from_previous": excluded, "population_rule": rule}
            for stage, n, excluded, rule in [
                ("source_cohort", 120, 0, "source"),
                ("alive_and_under_observation_at_landmark", 110, 10, "alive and observed"),
                ("valid_exposure_primary_population", 100, 10, "valid exposure"),
                ("complete_case_model_population", 90, 10, "complete model terms"),
            ]
        ],
        "adjusted_absolute_risk": {
            "method": "marginal_standardization_over_primary_complete_case_covariates",
            "interval": "delta_method_logit_scale_95_percent_confidence_interval", "grid_rows": 41,
        },
        "interpretation": "descriptive_prognostic_association_not_causal",
    }
    if clustered:
        receipt.update(variance_estimator="cluster_robust", cluster_unit="patient",
                       cluster_group_source="patient_id", cluster_group_derivation="identity", cluster_count=70)
    return receipt


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("corrupt_receipt", [False, True])
def test_sealed_method_receipt_survives_projection_and_rejects_drift(tmp_path, clustered, corrupt_receipt):
    table = tmp_path / "exposure_outcome_summary.csv"
    table.write_text("group,n\n0,40\n1,60\n")
    receipt = _receipt(clustered)
    if corrupt_receipt:
        del receipt["adjustment_columns"]
    summary = {"status": "ok", "analysis_family": "association",
               "scientific_runtime_receipt": receipt,
               "output_files": {"table:exposure_outcome_summary": table.name}}
    envelope = normalize_step_result_shadow(step_id=_UPSTREAM_STEP, step_summary=summary, output_dir=tmp_path, status="ok")
    store = EvidenceStore(tmp_path / "run")
    artifact = _register_upstream_table(store, table)
    sidecar = _commit_upstream_sidecar(store, envelope)
    record = _modern_upstream_record(sidecar_evidence_id=sidecar)
    record.update(step_summary=summary, evidence_ids=[artifact.evidence_id, sidecar],
                  deterministic_standard_analysis="signed_landmark_spline_association")
    consumer = RegisteredOutputEnvelopeConsumer()
    if corrupt_receipt:
        with pytest.raises(RegisteredOutputAuthorityError, match="method receipt"):
            consumer.authoritative_writer_records([record], evidence_store=store)
        return
    projected = consumer.authoritative_writer_records([record], evidence_store=store)
    (method,) = _executed_method_boundary_rows(projected, evidence=store)
    assert method["adjustment_columns"] == ["age", "severity_index"]
    assert method["landmark_hours"] == 24
    assert method["population_rule"] == receipt["population_rule"]
    assert method["population_flow"] == receipt["population_flow"]
    assert method["adjusted_absolute_risk"]["interval"] == receipt["adjusted_absolute_risk"]["interval"]
    assert (method.get("cluster_unit") == "patient") is clustered
    assert "p_value" not in method["functional_form_comparison"]
    forged = copy.deepcopy(record)
    forged["step_summary"]["scientific_runtime_receipt"]["adjustment_columns"] = ["age"]
    with pytest.raises(RegisteredOutputAuthorityError, match="canonical_source_digest_mismatch"):
        consumer.authoritative_writer_records([forged], evidence_store=store)
