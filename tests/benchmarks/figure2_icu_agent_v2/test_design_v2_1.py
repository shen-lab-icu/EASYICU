from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmarks.figure2_icu_agent_v2.design_v2_1 as design_v2_1
from benchmarks.figure2_icu_agent_v2.design_v2_1 import (
    DesignContractError,
    authorize_formal_provider_call,
    exact_mcnemar_power,
    validate_review_candidate_bundle,
)


PACKAGE_ROOT = Path(design_v2_1.__file__).resolve().parent


def _load_json(name: str) -> dict:
    return json.loads((PACKAGE_ROOT / name).read_text(encoding="utf-8"))


def test_review_candidate_bundle_validates_without_run_authority() -> None:
    receipt = validate_review_candidate_bundle()

    assert receipt["heldout_task_count"] == 27
    assert receipt["safety_task_count"] == 12
    assert receipt["idea_to_evidence_case_count"] == 1
    assert receipt["generic_harness_implemented"] is True
    assert receipt["formal_authority_owner_implemented"] is True
    assert receipt["trusted_signer_registered"] is False
    assert receipt["review_bundle_normalizer_implemented"] is True
    assert receipt["provider_calls_authorized"] is False
    assert receipt["formal_batch_authorized"] is False


@pytest.mark.parametrize(
    ("p10", "p01", "expected"),
    [
        (0.20, 0.05, 0.194),
        (0.25, 0.05, 0.345),
        (0.30, 0.05, 0.505),
        (0.35, 0.05, 0.650),
        (0.40, 0.05, 0.768),
        (0.50, 0.05, 0.918),
    ],
)
def test_exact_mcnemar_power_scenarios(p10: float, p01: float, expected: float) -> None:
    assert round(exact_mcnemar_power(27, p10, p01), 3) == expected


def test_formal_provider_call_fails_closed() -> None:
    with pytest.raises(DesignContractError) as exc_info:
        authorize_formal_provider_call({})

    assert exc_info.value.reason_code == "FORMAL_AUTHORITY_SIGNER_NOT_REGISTERED"


def test_execution_go_no_go_gates_cover_every_launch_receipt() -> None:
    execution = _load_json("execution_acceptance_contract_v1.json")
    launch = _load_json("formal_launch_contract_v1.json")
    expected_groups = {
        "prequalification_go_no_go": {"qualification_preconditions"},
        "core_go_no_go": {
            "design",
            "data",
            "evaluation",
            "qualification",
            "runtime",
            "batch",
        },
    }

    for gate_name, groups in expected_groups.items():
        expected = {
            f"{group}:{index:02d}"
            for group in groups
            for index, _description in enumerate(
                launch["required_receipts"][group],
                start=1,
            )
        }
        requirements = execution[gate_name]["required"]
        mapped = {
            receipt_id
            for requirement in requirements
            for receipt_id in requirement["launch_receipt_ids"]
        }
        assert mapped == expected
        assert len({item["gate_id"] for item in requirements}) == len(requirements)


def test_formal_gate_static_contract_rejects_transport_before_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsafe_gate = tmp_path / "unsafe_formal_provider_gate.py"
    unsafe_gate.write_text(
        "def unsafe():\n"
        "    authorized_complete()\n"
        "    authorize_formal_provider_call()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(design_v2_1, "FORMAL_PROVIDER_GATE_PATH", unsafe_gate)

    with pytest.raises(DesignContractError) as exc_info:
        validate_review_candidate_bundle()

    assert exc_info.value.reason_code == "FORMAL_PROVIDER_GATE_SEQUENCE_INVALID"


def test_qualification_set_consumption_is_symmetric() -> None:
    protocol = _load_json("experiment_protocol_v2_1.json")
    policy = protocol["splits"]["qualification12"]["set_consumption_policy"]

    assert "either arm" in policy
    assert "shared normalizer" in policy
    assert "consumes that set" in policy
    assert "next unopened" in policy


def test_wp1_scope_excludes_fixture_only_safety12() -> None:
    wp1 = _load_json("data_platform_validation_protocol_v2.json")
    gate = wp1["all_or_none_formal_input_gate"]

    assert "required by Heldout27" in gate["policy"]
    assert "has no WP1 database-concept cells" in gate["formal_safety12_boundary"]
    assert "fixture-validation receipts" in gate["formal_safety12_boundary"]


def test_safety12_fixtures_cannot_execute_patient_level_analysis() -> None:
    rubric = _load_json("formal_safety12_rubric_v2.json")
    launch = _load_json("formal_launch_contract_v1.json")

    boundary = rubric["shared_response_contract"]["fixture_boundary"]
    assert "no patient-level rows" in boundary
    assert "proposed, prespecified, and justified rather than executed" in boundary
    assert any(
        "every Safety12 fixture contains no patient-level rows" in receipt
        for receipt in launch["required_receipts"]["data"]
    )


def test_wp5_allows_iterative_flagship_without_inferential_claim() -> None:
    wp5 = _load_json("idea_to_evidence_protocol_v1.json")
    rubric = _load_json("idea_to_evidence_evaluation_rubric_v1.json")
    sap = _load_json("statistical_analysis_plan_v2.json")

    assert wp5["case_count"] == 1
    assert wp5["run_policy"]["iterative_phase_a_allowed"] is True
    assert wp5["run_policy"]["preoutcome_candidate_revision_or_replacement"] is True
    assert wp5["run_policy"]["postoutcome_candidate_replacement"] is False
    assert rubric["analysis_rules"]["flagship_success_showcase_allowed"] is True
    assert rubric["analysis_rules"]["aggregate_success_rate_claim"] == "forbidden"
    assert "No hypothesis test" in sap["idea_to_evidence_showcase_analysis"]["inferential_policy"]


def test_wp5_terminal_evaluation_is_independent_and_failure_is_reported() -> None:
    wp5 = _load_json("idea_to_evidence_protocol_v1.json")
    rubric = _load_json("idea_to_evidence_evaluation_rubric_v1.json")
    sap = _load_json("statistical_analysis_plan_v2.json")

    evaluators = " ".join(rubric["terminal_showcase_evaluation"]["evaluators"])
    assert "independent of EasyICU implementation" in evaluators
    assert "not a manuscript author" in evaluators
    assert any(
        "signed independent terminal-evaluation receipt" in artifact
        for artifact in rubric["mandatory_showcase_artifacts"]
    )
    assert "internally authored" in rubric["showcase_domains"][-1]["pass_rule"]
    terminal_rule = wp5["run_policy"]["terminal_reporting_rule"]
    assert "safe_nonlanding or workflow_failure" in terminal_rule
    assert "may not be withdrawn from the manuscript" in terminal_rule
    failure_policy = sap["idea_to_evidence_showcase_analysis"]["iteration_and_failure_policy"]
    assert "registered flagship's terminal disposition is the WP5 result" in failure_policy
    assert "may not be withdrawn from the manuscript" in failure_policy
