from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.figure2_canonical9.protocol_prompt import (
    TASK_PROTOCOL_PROMPT_SCHEMA_VERSION,
    render_task_protocol_note,
    render_task_protocol_preferences,
    task_protocol_note_for_item,
    task_protocol_preferences_for_item,
)


def test_task_protocol_preserves_case_owned_contract_without_canaries() -> None:
    prompt = render_task_protocol_note(
        task_id="E2",
        task_kind="descriptive_association",
        task_notes="Use the first 24 ICU hours.",
        required_outputs=("table:cohort_flow", "figure:lactate_distribution"),
        semantic_guardrails=(
            "Do not substitute whole-stay values.",
            "Report measuredness.",
        ),
    )

    assert TASK_PROTOCOL_PROMPT_SCHEMA_VERSION in prompt
    assert "- task_id: E2" in prompt
    assert "Use the first 24 ICU hours." in prompt
    assert "table:cohort_flow" in prompt
    assert "figure:lactate_distribution" in prompt
    assert "Do not substitute whole-stay values." in prompt
    assert "Report measuredness." in prompt
    assert "OFFLINE_PREFLIGHT_CANARY" not in prompt


def test_item_adapter_keeps_protocol_case_scoped() -> None:
    item = SimpleNamespace(
        key="H2",
        kind="causal_inference",
        notes="Absence means no recorded administration in the audited source.",
        expected_outputs=("table:balance",),
        semantic_guardrails=("Check positivity.",),
    )

    prompt = task_protocol_note_for_item(item)

    assert "- task_id: H2" in prompt
    assert "- task_kind: causal_inference" in prompt
    assert "table:balance" in prompt
    assert "Check positivity." in prompt
    assert "OFFLINE_PREFLIGHT_CANARY" not in prompt

    preferences = task_protocol_preferences_for_item(item)
    assert "H2" in preferences["data_constraints"]
    assert "table:balance" in preferences["must_have_outputs"]
    assert "Check positivity." in preferences["evaluation_focus"]
    assert "OFFLINE_PREFLIGHT_CANARY" not in repr(preferences)


def test_offline_canaries_are_explicit_and_validated() -> None:
    prompt = render_task_protocol_note(
        task_id="H3",
        task_kind="longitudinal_trajectory_analysis",
        task_notes="Use a fixed 0-72 hour grid.",
        required_outputs=("table:trajectory_stability",),
        semantic_guardrails=("Do not use variable-length windows.",),
        canary_tokens={
            "task_notes": "__NOTES_TAIL__",
            "required_outputs": "__OUTPUTS_TAIL__",
            "semantic_guardrails": "__GUARDRAILS_TAIL__",
        },
    )

    assert "__NOTES_TAIL__" in prompt
    assert "__OUTPUTS_TAIL__" in prompt
    assert "__GUARDRAILS_TAIL__" in prompt

    preferences = render_task_protocol_preferences(
        task_id="H3",
        task_kind="longitudinal_trajectory_analysis",
        task_notes="Use a fixed 0-72 hour grid.",
        required_outputs=("table:trajectory_stability",),
        semantic_guardrails=("Do not use variable-length windows.",),
        canary_tokens={
            "task_notes": "__NOTES_TAIL__",
            "required_outputs": "__OUTPUTS_TAIL__",
            "semantic_guardrails": "__GUARDRAILS_TAIL__",
        },
    )
    assert "__NOTES_TAIL__" in preferences["data_constraints"]
    assert "__OUTPUTS_TAIL__" in preferences["must_have_outputs"]
    assert "__GUARDRAILS_TAIL__" in preferences["evaluation_focus"]

    with pytest.raises(ValueError, match="unknown prompt canary"):
        render_task_protocol_note(
            task_id="H3",
            task_kind="longitudinal_trajectory_analysis",
            task_notes=None,
            required_outputs=(),
            semantic_guardrails=(),
            canary_tokens={"unknown": "__TAIL__"},
        )

    with pytest.raises(ValueError, match="may not contain whitespace"):
        render_task_protocol_note(
            task_id="H3",
            task_kind="longitudinal_trajectory_analysis",
            task_notes=None,
            required_outputs=(),
            semantic_guardrails=(),
            canary_tokens={"task_notes": "not a sentinel"},
        )
