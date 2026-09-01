"""Suffix slicing is scientific naming logic, not host-owned syntax repair."""

from __future__ import annotations

import pytest

from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.repair_registry import automatic_repair_allowed


def _step(ra):
    return ra.AnalysisStep(
        step_id="provenance_audit",
        intent="Validate declared provenance companions.",
        inputs=["artifact:analysis_cohort"],
        expected_outputs=["table:provenance_audit"],
        method="measurement_quality_control",
    )


def _findings(script: str, ra):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason") == "string_suffix_trim_length_mismatch"
    ]


@pytest.mark.parametrize(
    "script",
    [
        'names = {name[:-10] for name in columns if name.endswith("_measured")}\n',
        'names = {name[:-8] for name in columns if name.endswith("_max")}\n',
        'names = {name[:-1] for name in columns if name.endswith("_max")}\n',
        "names = {name[:-n] for name in columns if name.endswith(suffix)}\n",
    ],
)
def test_suffix_slice_intent_is_not_claimed_from_shape_alone(script: str, ra) -> None:
    assert _findings(script, ra) == []


def test_retired_suffix_rewrite_is_not_automatically_authorized() -> None:
    assert not automatic_repair_allowed("string_suffix_trim_length_v1")
