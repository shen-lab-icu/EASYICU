from __future__ import annotations

import json

import pytest

from benchmarks.figure2_canonical9.case_scientific_protocol import (
    ScientificCaseProtocolError,
    build_runtime_scientific_projection,
    case_protocol_content_sha256,
    default_case_protocol_path,
    load_case_scientific_protocol,
    load_default_case_protocol,
    load_runtime_scientific_projection,
)


def test_e2_h2_h3_protocols_are_strict_and_content_digestable() -> None:
    e2 = load_default_case_protocol("e2_lactate_mortality")
    h2 = load_default_case_protocol("h2_vasopressor_causal")
    h3 = load_default_case_protocol("h3_trajectory_clustering")

    assert e2.primary_landmark.landmark_hours == 24
    assert e2.primary_model.exposure_form == "restricted_cubic_spline"
    assert e2.primary_model.knot_quantiles == (0.10, 0.50, 0.90)
    assert "ssc_adult_2026" in {item.citation_id for item in e2.citations}
    assert h2.current_source_capture.reason_code == "H2_VERIFIED_NON_USE_UNAVAILABLE"
    assert h2.current_source_capture.verified_non_use_available is False
    assert h2.current_source_capture.binary_control_arm_authorized is False
    assert h2.current_source_capture.pre_icu_treatment_history_authority is False
    assert h2.current_source_capture.initiator_status_authorized is False
    assert h2.future_unblock_contract.status.startswith("future_design_only")
    assert set(h2.future_unblock_contract.must_distinguish) == {
        "true_initiator",
        "prevalent_user",
        "verified_non_user",
    }
    assert h3.supersedes_terminal_protocol.observed_mean_stability == 0.5357
    assert h3.selection_and_stability.candidate_cluster_counts == (2, 3, 4, 5, 6)
    assert h3.selection_and_stability.minimum_mean_stability == 0.7
    assert "sofa2" not in h3.representation.features
    assert h3.representation.descriptive_only_features == ("sofa2",)
    h3_projection = build_runtime_scientific_projection(h3)
    h3_execution = h3_projection.deterministic_execution_contract
    assert h3_execution is not None
    assert tuple(h3_execution["coordinate_concepts"]) == h3.representation.features
    assert h3_execution["descriptive_only_concepts"] == ["sofa2"]
    assert len(h3_execution["representation_columns"]) == 7 * 6
    assert h3_execution["candidate_cluster_counts"] == [2, 3, 4, 5, 6]
    assert (
        h3_execution["upper_boundary_action"]
        == "fail_closed_if_selected_at_upper_boundary"
    )
    for protocol in (e2, h2, h3):
        assert len(case_protocol_content_sha256(protocol)) == 64
        projection = build_runtime_scientific_projection(protocol)
        assert projection.protocol_content_sha256 == case_protocol_content_sha256(
            protocol
        )
        assert len(projection.runtime_projection_sha256) == 64


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


def test_h2_protocol_rejects_underspecified_future_unblock(tmp_path) -> None:
    payload = json.loads(
        default_case_protocol_path("h2_vasopressor_causal").read_text("utf-8")
    )
    payload["future_unblock_contract"]["must_distinguish"] = [
        "true_initiator",
        "verified_non_user",
    ]
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


def test_runtime_projection_rejects_agent_visible_drift() -> None:
    protocol = load_default_case_protocol("h3_trajectory_clustering")
    projection = build_runtime_scientific_projection(protocol)
    payload = projection.model_dump(mode="json")
    payload["agent_visible_guardrails"][0] = "silently changed after review"

    with pytest.raises(ValueError, match="projection digest mismatch"):
        load_runtime_scientific_projection(payload)


def test_runtime_projection_rejects_deterministic_execution_drift() -> None:
    protocol = load_default_case_protocol("h3_trajectory_clustering")
    projection = build_runtime_scientific_projection(protocol)
    payload = projection.model_dump(mode="json")
    payload["deterministic_execution_contract"]["candidate_cluster_counts"] = [
        2,
        3,
        4,
    ]

    with pytest.raises(ValueError, match="projection digest mismatch"):
        load_runtime_scientific_projection(payload)


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
