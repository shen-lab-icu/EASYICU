from __future__ import annotations

import pytest

from benchmarks.figure2_icu_agent_v2.design_v2_1 import (
    DesignContractError,
    authorize_formal_provider_call,
    exact_mcnemar_power,
    validate_review_candidate_bundle,
)


def test_review_candidate_bundle_validates_without_run_authority() -> None:
    receipt = validate_review_candidate_bundle()

    assert receipt["heldout_task_count"] == 27
    assert receipt["safety_task_count"] == 12
    assert receipt["idea_to_evidence_case_count"] == 3
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
