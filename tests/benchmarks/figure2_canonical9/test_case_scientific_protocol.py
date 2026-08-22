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


def test_e2_h1_h2_h3_protocols_are_strict_and_content_digestable() -> None:
    e2 = load_default_case_protocol("e2_lactate_mortality")
    h1 = load_default_case_protocol("h1_ventilation_survival")
    h2 = load_default_case_protocol("h2_vasopressor_causal")
    h3 = load_default_case_protocol("h3_trajectory_clustering")

    assert e2.primary_landmark.landmark_hours == 24
    assert e2.primary_model.exposure_form == "restricted_cubic_spline"
    assert e2.primary_model.knot_quantiles == (0.10, 0.50, 0.90)
    assert "ssc_adult_2026" in {item.citation_id for item in e2.citations}
    assert h1.landmark_hours == 24
    assert h1.proportional_hazards_policy == "block_paper_authorization"
    assert h1.review_status == "ai_development_reviewed_human_attestation_pending"
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
    for protocol in (e2, h1, h2, h3):
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


def test_launcher_cannot_override_a_signed_runtime_execution_contract() -> None:
    from tools.run_research_agent_bench import (
        _bind_runtime_scientific_projection_options,
    )

    projection = build_runtime_scientific_projection(
        load_default_case_protocol("e2_lactate_mortality")
    ).model_dump(mode="json")
    bound = _bind_runtime_scientific_projection_options({}, projection)
    assert bound["current_case_scientific_runtime_authority"] == (
        projection["deterministic_execution_contract"]
    )
    assert bound["scientific_runtime_projection_sha256"] == (
        projection["runtime_projection_sha256"]
    )

    with pytest.raises(ValueError, match="AUTHORITY_OVERRIDE_FORBIDDEN"):
        _bind_runtime_scientific_projection_options(
            {"current_case_scientific_runtime_authority": {"forged": True}},
            projection,
        )
    with pytest.raises(ValueError, match="PROJECTION_OVERRIDE_FORBIDDEN"):
        _bind_runtime_scientific_projection_options(
            {"scientific_runtime_projection_sha256": "0" * 64},
            projection,
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


def test_h2_has_no_authority_for_a_causal_primary_result_contract() -> None:
    """H2 must never gain a causal headline by having its blanks filled in.

    ``validate_required_primary_result`` demands a
    ``family_primary_result_requirement`` from every ``causal_inference`` plan
    with a declared exposure and outcome. That requirement IS a treatment-effect
    contract: estimand, comparator, adjustment strategy, effect scale. The
    tempting way to make H2's preflight go green is to supply those values.

    The sealed H2 protocol forbids exactly that. Its reviewed capture decision
    is ``fail_closed`` with ``causal_contrast_authorized=False`` because
    verified non-use is unavailable, so a control arm cannot be identified from
    the recorded administrations at all -- ``construct_binary_control_arm`` and
    ``fit_psm_or_iptw_before_capture_authority`` are named forbidden actions.
    A contract invented to satisfy a validator would be a treatment effect the
    data cannot identify, which is the one failure mode the whole capture
    review exists to prevent.

    So this asserts the absence is authoritative, not an oversight: nothing in
    the sealed protocol supplies these coordinates, and H2's own recorded
    scientific result is that the contrast is not identifiable. H2's preflight
    failure is therefore the correct outcome, and any future host compilation
    of ``family_primary_result_requirement`` must keep excluding H2 until a new
    clinical-and-methods review changes the capture contract.
    """

    h2 = load_default_case_protocol("h2_vasopressor_causal")
    capture = h2.current_source_capture

    assert capture.decision == "fail_closed"
    assert capture.causal_contrast_authorized is False
    assert (
        h2.current_scientific_result
        == "treatment_contrast_not_identifiable_from_available_capture_contract"
    )
    for forbidden in (
        "construct_binary_control_arm",
        "fit_psm_or_iptw_before_capture_authority",
        "convert_missing_vasopressor_record_to_verified_non_use",
    ):
        assert forbidden in h2.forbidden_actions

    # The coordinates a causal family-primary contract needs are absent from
    # the sealed protocol, so there is no source to compile them from.
    payload = json.loads(
        default_case_protocol_path("h2_vasopressor_causal").read_text(encoding="utf-8")
    )
    flattened = json.dumps(payload).casefold()
    for coordinate in ("estimand", "comparator", "adjustment_strategy", "effect_scale"):
        assert coordinate not in flattened, (
            f"{coordinate!r} appeared in the sealed H2 protocol; re-review whether "
            "a causal contrast is now authorized instead of assuming it is"
        )

    # Unblocking is a review decision, not a default.
    assert h2.future_unblock_contract.new_clinical_and_methods_review_required is True


def test_h1_development_protocol_is_explicitly_not_human_attested() -> None:
    h1 = load_default_case_protocol("h1_ventilation_survival")
    projection = build_runtime_scientific_projection(h1)

    assert h1.review_status == "ai_development_reviewed_human_attestation_pending"
    assert projection.deterministic_execution_contract is not None
    assert (
        projection.deterministic_execution_contract["schema_version"]
        == "easyicu.landmark_survival_runtime_authority/1"
    )
    assert "human_attested" not in projection.model_dump_json()
