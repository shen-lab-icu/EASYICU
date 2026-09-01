from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.figure2_icu_agent_v2.design_v2_1 import (
    DesignContractError,
    authorize_formal_provider_call,
    exact_mcnemar_power,
    validate_review_candidate_bundle,
)


PACKAGE_ROOT = Path("benchmarks/figure2_icu_agent_v2")


def _load_json(name: str) -> dict:
    return json.loads((PACKAGE_ROOT / name).read_text(encoding="utf-8"))


def test_review_candidate_bundle_validates_without_run_authority() -> None:
    receipt = validate_review_candidate_bundle()

    assert receipt["heldout_task_count"] == 27
    assert receipt["safety_task_count"] == 12
    assert receipt["idea_to_evidence_case_count"] == 1
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

    assert exc_info.value.reason_code == "FORMAL_PROVIDER_CALL_NOT_AUTHORIZED"


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
