"""Case-scoped Canonical9 protocol text delivered to research-agent prompts.

The evaluation suite's required outputs and semantic guardrails are part of the
task contract, not hidden scorer answers.  Keep them in the benchmark item and
render them into ``ResearchContext.notes`` at launch time; never move one case's
variables or rules into a global agent prompt.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .case_scientific_protocol import load_runtime_scientific_projection

TASK_PROTOCOL_PROMPT_SCHEMA_VERSION = "easyicu.canonical9_task_prompt/1"

_SECTION_NAMES = ("task_notes", "required_outputs", "semantic_guardrails")


def _clean_items(values: Sequence[object] | None) -> list[str]:
    return [
        text
        for value in values or ()
        if (text := str(value or "").strip())
    ]


def _section_canary(
    section: str,
    canary_tokens: Mapping[str, str] | None,
) -> str:
    if not canary_tokens:
        return ""
    token = str(canary_tokens.get(section) or "").strip()
    if not token:
        return ""
    if any(character.isspace() for character in token):
        raise ValueError("offline prompt canary tokens may not contain whitespace")
    return f"\n  - OFFLINE_PREFLIGHT_CANARY: {token}"


def render_task_protocol_note(
    *,
    task_id: str,
    task_kind: str,
    task_notes: str | None,
    required_outputs: Sequence[object] | None,
    semantic_guardrails: Sequence[object] | None,
    canary_tokens: Mapping[str, str] | None = None,
) -> str:
    """Render the case-owned protocol block used by the benchmark launcher.

    ``canary_tokens`` is reserved for zero-Provider offline rendering.  The
    production launcher omits it, so canaries never become scientific content.
    """

    unknown = set(canary_tokens or {}) - set(_SECTION_NAMES)
    if unknown:
        raise ValueError(f"unknown prompt canary section(s): {sorted(unknown)!r}")
    clean_task_id = str(task_id or "").strip()
    clean_kind = str(task_kind or "").strip()
    notes = str(task_notes or "").strip() or "(none supplied)"
    outputs = _clean_items(required_outputs)
    guardrails = _clean_items(semantic_guardrails)
    output_lines = (
        "\n".join(f"  {index}. {value}" for index, value in enumerate(outputs, 1))
        or "  (none declared)"
    )
    guardrail_lines = (
        "\n".join(
            f"  {index}. {value}" for index, value in enumerate(guardrails, 1)
        )
        or "  (none declared)"
    )
    return (
        "BENCHMARK TASK PROTOCOL (case-scoped; binding for this task only)\n"
        f"- schema_version: {TASK_PROTOCOL_PROMPT_SCHEMA_VERSION}\n"
        f"- task_id: {clean_task_id}\n"
        f"- task_kind: {clean_kind}\n"
        "- operator/materialization notes:\n"
        f"  {notes}"
        f"{_section_canary('task_notes', canary_tokens)}\n"
        "- required outputs (preserve every item in the plan):\n"
        f"{output_lines}"
        f"{_section_canary('required_outputs', canary_tokens)}\n"
        "- semantic guardrails (must constrain planning, code, repair, and prose):\n"
        f"{guardrail_lines}"
        f"{_section_canary('semantic_guardrails', canary_tokens)}"
    )


def render_task_protocol_preferences(
    *,
    task_id: str,
    task_kind: str,
    task_notes: str | None,
    required_outputs: Sequence[object] | None,
    semantic_guardrails: Sequence[object] | None,
    canary_tokens: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Project the case contract into outbound-safe structured preferences.

    ``ResearchContext.notes`` is deliberately excluded from Provider-bound
    context.  These three existing preference fields are the typed, auditable
    channel for scientific constraints that every agent stage must receive.
    """

    unknown = set(canary_tokens or {}) - set(_SECTION_NAMES)
    if unknown:
        raise ValueError(f"unknown prompt canary section(s): {sorted(unknown)!r}")
    clean_task_id = str(task_id or "").strip()
    clean_kind = str(task_kind or "").strip()
    notes = str(task_notes or "").strip() or "(none supplied)"
    outputs = _clean_items(required_outputs)
    guardrails = _clean_items(semantic_guardrails)

    def _value_with_canary(value: str, section: str) -> str:
        suffix = _section_canary(section, canary_tokens).strip()
        return f"{value}\n{suffix}" if suffix else value

    return {
        "data_constraints": _value_with_canary(
            (
                "CANONICAL9 CASE-SCOPED DATA/DESIGN CONSTRAINTS "
                f"(task_id={clean_task_id}; task_kind={clean_kind}):\n{notes}"
            ),
            "task_notes",
        ),
        "must_have_outputs": _value_with_canary(
            "CANONICAL9 REQUIRED OUTPUTS (binding):\n"
            + (
                "\n".join(
                    f"{index}. {value}"
                    for index, value in enumerate(outputs, start=1)
                )
                or "(none declared)"
            ),
            "required_outputs",
        ),
        "evaluation_focus": _value_with_canary(
            "CANONICAL9 SEMANTIC GUARDRAILS (binding):\n"
            + (
                "\n".join(
                    f"{index}. {value}"
                    for index, value in enumerate(guardrails, start=1)
                )
                or "(none declared)"
            ),
            "semantic_guardrails",
        ),
    }


def task_protocol_note_for_item(item: object) -> str:
    """Render one imported benchmark item without exposing evaluator answers."""

    notes, outputs, guardrails = _prompt_fields_for_item(item)

    return render_task_protocol_note(
        task_id=str(getattr(item, "key", "") or ""),
        task_kind=str(getattr(item, "kind", "") or ""),
        task_notes=notes,
        required_outputs=outputs,
        semantic_guardrails=guardrails,
    )


def task_protocol_preferences_for_item(item: object) -> dict[str, str]:
    """Return outbound-safe structured preferences for one benchmark item."""

    notes, outputs, guardrails = _prompt_fields_for_item(item)

    return render_task_protocol_preferences(
        task_id=str(getattr(item, "key", "") or ""),
        task_kind=str(getattr(item, "kind", "") or ""),
        task_notes=notes,
        required_outputs=outputs,
        semantic_guardrails=guardrails,
    )


def _prompt_fields_for_item(
    item: object,
) -> tuple[str | None, Sequence[object] | None, Sequence[object] | None]:
    """Use the signed runtime projection when one is present on the item."""

    raw = getattr(item, "runtime_scientific_projection", None)
    if raw is None:
        return (
            getattr(item, "notes", None),
            getattr(item, "expected_outputs", None),
            getattr(item, "semantic_guardrails", None),
        )
    projection = load_runtime_scientific_projection(raw)
    task_id = str(getattr(item, "key", "") or "")
    declared_digest = str(
        getattr(item, "runtime_scientific_projection_sha256", "") or ""
    )
    if projection.task_id != task_id:
        raise ValueError("runtime scientific projection task mismatch")
    if declared_digest != projection.runtime_projection_sha256:
        raise ValueError("runtime scientific projection declared digest mismatch")
    if str(getattr(item, "case_scientific_protocol_sha256", "") or "") != (
        projection.protocol_content_sha256
    ):
        raise ValueError("runtime projection and case protocol digest mismatch")
    return (
        projection.canonical_protocol_json,
        projection.agent_visible_required_outputs,
        projection.agent_visible_guardrails,
    )


__all__ = [
    "TASK_PROTOCOL_PROMPT_SCHEMA_VERSION",
    "render_task_protocol_note",
    "render_task_protocol_preferences",
    "task_protocol_note_for_item",
    "task_protocol_preferences_for_item",
]
