from __future__ import annotations

import pytest

from easyicu.research_agent.reporting.manuscript_projection import (
    ManuscriptProjectionError,
    project_owner_issued_manuscript_claims,
)


_SCAFFOLD = """## Abstract

**Results:** The Writer omitted the complete owner result.

## Results

### Primary association

The Writer reported only a qualitative summary.

### Sensitivity and subgroup analyses

No additional prose.
"""


def _record(*, block_key: str = "reportable_model_results") -> dict:
    return {
        "step_id": "01_model",
        "status": "ok",
        "generation_mode": "deterministic_standard",
        "step_summary_evidence_id": "statistic_step_summary_model",
        "step_summary": {
            block_key: {
                "estimate": 1.234567,
                "interval": {"low": 1.1, "high": 1.4},
                "manuscript_projection": {
                    "schema_version": "easyicu.manuscript_projection/1",
                    "claims": [
                        {
                            "claim_id": "primary_estimate",
                            "targets": [
                                {"kind": "abstract_label", "label": "Results"},
                                {
                                    "kind": "markdown_heading",
                                    "label": "Primary association",
                                },
                            ],
                            "fragments": [
                                {"text": "The owner-issued estimate was "},
                                {
                                    "numeric_path": "estimate",
                                    "format_spec": ".3f",
                                },
                                {"text": " (95% CI, "},
                                {
                                    "numeric_path": "interval.low",
                                    "format_spec": ".3f",
                                },
                                {"text": " to "},
                                {
                                    "numeric_path": "interval.high",
                                    "format_spec": ".3f",
                                },
                                {"text": ")."},
                            ],
                        }
                    ],
                },
            }
        },
    }


def test_projects_any_typed_reportable_result_without_family_logic() -> None:
    repaired, repairs = project_owner_issued_manuscript_claims(
        _SCAFFOLD,
        per_step_records=[_record(block_key="reportable_calibration_results")],
    )

    assert repaired.count("The owner-issued estimate was 1.235") == 2
    assert repaired.count("95% CI, 1.100 to 1.400") == 2
    assert repaired.count("{evidence:statistic_step_summary_model}") == 2
    assert {item["target_label"] for item in repairs} == {
        "Results",
        "Primary association",
    }
    assert all(
        item["reason_code"] == "owner_manuscript_claim_projected" for item in repairs
    )


def test_projection_is_idempotent_when_every_declared_literal_is_present() -> None:
    once, first_repairs = project_owner_issued_manuscript_claims(
        _SCAFFOLD,
        per_step_records=[_record()],
    )
    twice, second_repairs = project_owner_issued_manuscript_claims(
        once,
        per_step_records=[_record()],
    )

    assert first_repairs
    assert twice == once
    assert second_repairs == []


def test_projection_rejects_non_deterministic_authority() -> None:
    record = _record()
    record["generation_mode"] = "llm"

    with pytest.raises(
        ManuscriptProjectionError,
        match="requires deterministic_standard authority",
    ):
        project_owner_issued_manuscript_claims(
            _SCAFFOLD,
            per_step_records=[record],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_path", "missing numeric path"),
        ("bad_format", "invalid numeric_path or format_spec"),
        ("missing_evidence", "requires step_summary_evidence_id"),
        ("missing_target", "target is absent"),
    ],
)
def test_projection_fails_closed_on_contract_drift(
    mutation: str,
    message: str,
) -> None:
    record = _record()
    block = record["step_summary"]["reportable_model_results"]
    claim = block["manuscript_projection"]["claims"][0]
    if mutation == "missing_path":
        claim["fragments"][1]["numeric_path"] = "unknown.value"
    elif mutation == "bad_format":
        claim["fragments"][1]["format_spec"] = "{unsafe}"
    elif mutation == "missing_evidence":
        record["step_summary_evidence_id"] = ""
    else:
        claim["targets"][1]["label"] = "Absent heading"

    with pytest.raises(ManuscriptProjectionError, match=message):
        project_owner_issued_manuscript_claims(
            _SCAFFOLD,
            per_step_records=[record],
        )
