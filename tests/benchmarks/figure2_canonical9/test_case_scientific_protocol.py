from __future__ import annotations

import json

import pytest

from benchmarks.figure2_canonical9.case_scientific_protocol import (
    ScientificCaseProtocolError,
    case_protocol_content_sha256,
    default_case_protocol_path,
    load_case_scientific_protocol,
    load_default_case_protocol,
)


def test_e2_h2_h3_protocols_are_strict_and_content_digestable() -> None:
    e2 = load_default_case_protocol("e2_lactate_mortality")
    h2 = load_default_case_protocol("h2_vasopressor_causal")
    h3 = load_default_case_protocol("h3_trajectory_clustering")

    assert e2.primary_population.startswith("Eligible ICU stays with at least one")
    assert "ssc_adult_2026" in {item.citation_id for item in e2.citations}
    assert h2.current_source_capture.reason_code == "H2_VERIFIED_NON_USE_UNAVAILABLE"
    assert h2.current_source_capture.verified_non_use_available is False
    assert h2.current_source_capture.binary_control_arm_authorized is False
    assert (
        h2.intended_target_trial.baseline_adjustment_timing
        == "at_or_before_icu_admission_time_zero"
    )
    assert h2.intended_target_trial.estimation_method == (
        "clone_censor_weight_with_stabilized_inverse_probability_censoring_weights"
    )
    assert "lactate" in h2.intended_target_trial.grace_period_time_varying_variables
    assert h3.supersedes_terminal_protocol.observed_mean_stability == 0.5357
    assert h3.selection_and_stability.candidate_cluster_counts == (2, 3, 4, 5, 6)
    assert h3.selection_and_stability.minimum_mean_stability == 0.7
    for protocol in (e2, h2, h3):
        assert len(case_protocol_content_sha256(protocol)) == 64


def test_h2_protocol_rejects_absence_as_verified_non_use(tmp_path) -> None:
    payload = json.loads(
        default_case_protocol_path("h2_vasopressor_causal").read_text("utf-8")
    )
    payload["current_source_capture"]["verified_non_use_available"] = True
    path = tmp_path / "h2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ScientificCaseProtocolError,
        match="SCIENTIFIC_CASE_PROTOCOL_INVALID",
    ):
        load_case_scientific_protocol(
            path,
            expected_task_id="h2_vasopressor_causal",
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("baseline_adjustment_timing", "before_recorded_initiation"),
        ("estimation_method", "stabilized_iptw"),
        (
            "post_time_zero_variable_role",
            "include_in_baseline_propensity_model_if_before_recorded_initiation",
        ),
    ),
)
def test_h2_protocol_rejects_time_zero_or_weighting_drift(
    tmp_path,
    field: str,
    invalid_value: str,
) -> None:
    payload = json.loads(
        default_case_protocol_path("h2_vasopressor_causal").read_text("utf-8")
    )
    payload["intended_target_trial"][field] = invalid_value
    path = tmp_path / "h2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ScientificCaseProtocolError,
        match="SCIENTIFIC_CASE_PROTOCOL_INVALID",
    ):
        load_case_scientific_protocol(
            path,
            expected_task_id="h2_vasopressor_causal",
        )


def test_h3_protocol_rejects_post_hoc_candidate_k_drift(tmp_path) -> None:
    payload = json.loads(
        default_case_protocol_path("h3_trajectory_clustering").read_text("utf-8")
    )
    payload["selection_and_stability"]["candidate_cluster_counts"] = [2, 3, 4]
    path = tmp_path / "h3.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ScientificCaseProtocolError,
        match="SCIENTIFIC_CASE_PROTOCOL_INVALID",
    ):
        load_case_scientific_protocol(
            path,
            expected_task_id="h3_trajectory_clustering",
        )
