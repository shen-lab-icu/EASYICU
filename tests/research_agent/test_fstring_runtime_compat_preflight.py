from __future__ import annotations

import ast

import pytest

from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.repair_registry import (
    RepairClass,
    automatic_repair_allowed,
    repair_metadata_for,
)
from easyicu.research_agent.repairs.reasons import (
    RepairPromptAuthority,
    RepairReason,
    repair_reason_for_finding,
)
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair


def _step(ra):
    return ra.AnalysisStep(
        step_id="render",
        intent="Render an upstream typed table.",
        inputs=["table:upstream"],
        expected_outputs=["figure:panel"],
        method="visualization",
    )


def _findings(script: str, ra):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason") == "fstring_runtime_quote_incompatible"
    ]


def _repair(script: str, findings):
    return deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )


_DOUBLE_QUOTED = """
binding = {"identity_row": {"input_key": "table:upstream"}}
message = f"Loading {binding["identity_row"]["input_key"]}"
"""


def test_pep701_only_fstring_is_repaired_for_python311(ra) -> None:
    findings = _findings(_DOUBLE_QUOTED, ra)

    assert len(findings) == 1
    assert findings[0].detail == {
        "reason": "fstring_runtime_quote_incompatible",
        "occurrences": [
            {
                "line": 3,
                "column": 29,
                "end_line": 3,
                "end_column": 43,
                "outer_quote": "double",
            },
            {
                "line": 3,
                "column": 45,
                "end_line": 3,
                "end_column": 56,
                "outer_quote": "double",
            },
        ],
    }
    assert (
        repair_reason_for_finding(findings[0])
        is RepairReason.RUNTIME_SYNTAX_INCOMPATIBLE
    )
    repaired, names = _repair(_DOUBLE_QUOTED, findings)

    assert names == ["fstring_runtime_quote_compat_v1"]
    assert "binding['identity_row']['input_key']" in repaired
    ast.parse(repaired, feature_version=(3, 11))
    assert _findings(repaired, ra) == []


def test_single_quoted_outer_fstring_uses_double_quoted_subscript(ra) -> None:
    script = "binding = {'key': 'value'}\nmessage = f'{binding['key']}'\n"
    findings = _findings(script, ra)

    assert len(findings) == 1
    repaired, names = _repair(script, findings)

    assert names == ["fstring_runtime_quote_compat_v1"]
    assert 'binding["key"]' in repaired
    ast.parse(repaired, feature_version=(3, 11))


def test_multiple_occurrences_are_repaired_atomically(ra) -> None:
    script = (
        'binding = {"left": 1, "right": 2}\n'
        'message = f"{binding["left"]}:{binding["right"]}"\n'
    )
    findings = _findings(script, ra)

    assert len((findings[0].detail or {})["occurrences"]) == 2
    repaired, names = _repair(script, findings)

    assert names == ["fstring_runtime_quote_compat_v1"]
    assert "binding['left']" in repaired
    assert "binding['right']" in repaired
    ast.parse(repaired, feature_version=(3, 11))


def test_tampered_occurrence_coordinates_fail_closed(ra) -> None:
    finding = _findings(_DOUBLE_QUOTED, ra)[0]
    detail = dict(finding.detail or {})
    occurrences = [dict(item) for item in detail["occurrences"]]
    occurrences[0]["column"] += 1
    tampered = finding.model_copy(
        update={"detail": {**detail, "occurrences": occurrences}}
    )

    repaired, names = _repair(_DOUBLE_QUOTED, [tampered])

    assert repaired == _DOUBLE_QUOTED
    assert names == []


def test_runtime_quote_coordinates_survive_prompt_authority_projection(ra) -> None:
    finding = _findings(_DOUBLE_QUOTED, ra)[0]

    ticket = RepairPromptAuthority.create(findings=[finding]).payload()["typed_ticket"]
    detail = ticket[0]["occurrences"][0]["detail"]

    assert detail["reason"] == "fstring_runtime_quote_incompatible"
    assert detail["occurrences"] == (finding.detail or {})["occurrences"]


@pytest.mark.parametrize(
    "script",
    [
        "binding = {'key': 'value'}\nmessage = f\"{binding['key']}\"\n",
        "binding = {\"key\": \"value\"}\nmessage = f'''{binding[\"key\"]}'''\n",
        'binding = {"key": "value"}\nmessage = "binding[\\"key\\"]"\n',
        '# f"{binding["key"]}"\nmessage = "safe"\n',
    ],
)
def test_python311_compatible_or_inert_text_is_not_flagged(script: str, ra) -> None:
    assert _findings(script, ra) == []
    ast.parse(script, feature_version=(3, 11))


def test_quote_repair_preserves_string_literal_value(ra) -> None:
    script = """
binding = {"doctor's note": "present"}
message = f"{binding["doctor's note"]}"
"""
    findings = _findings(script, ra)

    repaired, names = _repair(script, findings)

    assert names == []
    assert repaired == script
    assert _findings(repaired, ra) == findings


def test_fstring_runtime_quote_repair_is_syntactic_and_automatic() -> None:
    metadata = repair_metadata_for("fstring_runtime_quote_compat_v1")

    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert metadata.introduces_numbers is False
    assert metadata.requires_disclosure is False
    assert automatic_repair_allowed(metadata.repair_id)
