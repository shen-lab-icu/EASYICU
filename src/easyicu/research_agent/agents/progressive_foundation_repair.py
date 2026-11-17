"""Bounded Planner repair of late foundation display-label requirements."""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence

from ..canonical_json import canonical_sha256, extract_json_payload
from ..planning.progressive_compiler import validate_progressive_foundation
from ..planning.progressive_contract import (
    ProgressiveFoundationMaterialization,
    ProgressivePlanCompileError,
    ProgressivePlanFoundation,
    ProgressiveStepMaterialization,
)
from ..providers.capabilities import llm_supports_strict_json_schema
from ..providers.protocol import LLMMessage, StructuredOutputRequest
from ..providers.structured_retry import (
    StructuredResponseFailure,
    automatic_retry_scope,
    call_llm_with_structured_retry,
)
from ..research_context.outbound import outbound_safe_context_payload
from ..schema import ResearchContext


def foundation_label_repair_keys(error: ProgressivePlanCompileError) -> tuple[str, ...]:
    """Only the label owner can request a foundation-label amendment."""
    if error.path != "display_labels":
        return ()
    if error.reason_code == "progressive_required_reader_display_labels_missing":
        expected = (error.details.get("required_key"),)
    elif error.reason_code == "progressive_required_binary_display_labels_missing":
        scope = error.details.get("required_scope")
        if not scope:
            return ()
        expected = (f"{scope}=0", f"{scope}=1")
    else:
        return ()
    keys = error.details.get("required_display_label_keys", expected)
    return tuple(key for key in keys if key and key in expected)


def repair_foundation_display_labels(
    llm: Any,
    *,
    finding: ProgressivePlanCompileError,
    materialization: ProgressiveFoundationMaterialization,
    context: ResearchContext,
    analysis_type: str,
    retained: Sequence[ProgressiveStepMaterialization],
    prompt_metrics: dict[str, Any],
    remaining_revisions: int,
    progress_callback: Callable[[Any], None] | None,
    resumed: bool,
) -> ProgressiveFoundationMaterialization:
    """Ask only for rejected keys; keep every other foundation coordinate.

    The caller supplies the remaining current-step compiler budget. Each
    answer, including a malformed one, spends one revision from that budget.
    """
    keys = foundation_label_repair_keys(finding)
    if not keys:
        raise finding
    foundation = materialization.foundation
    schema = StructuredOutputRequest.from_schema(
        name="progressive_foundation_label_repair",
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["display_labels"],
            "properties": {"display_labels": {
                "type": "object", "additionalProperties": False,
                "required": list(keys),
                "properties": {key: {"type": "string", "minLength": 1, "maxLength": 256}
                               for key in keys},
            }},
        },
    )
    names = tuple(dict.fromkeys(key.split("=", 1)[0] for key in keys))
    metadata = outbound_safe_context_payload(context, variable_names=names).get("variables", [])
    prompt = (
        "TARGETED FOUNDATION DISPLAY-LABEL REPAIR\n"
        "Return only {\"display_labels\": {<exact requested key>: <reader-facing meaning>}}. "
        "Use the sealed concept metadata below to author clinical labels. "
        "Do not mechanically humanize identifiers. Binary levels must have distinct meanings. "
        "All accepted steps, the current candidate, the outline, cohort, robustness intents, "
        "know-how decisions, and all other labels remain fixed.\n"
        + json.dumps({"required_display_label_keys": keys, "sealed_variables": metadata,
                      "compiler_observation": finding.details}, ensure_ascii=False)
    )

    def parse(raw: str) -> ProgressivePlanFoundation:
        payload = extract_json_payload(raw)
        if (not isinstance(payload, Mapping) or set(payload) != {"display_labels"}
                or not isinstance(payload["display_labels"], Mapping)
                or set(payload["display_labels"]) != set(keys)):
            raise ValueError("return display_labels with exactly the requested keys")
        proposed = payload["display_labels"]
        # Revalidate through the canonical model rather than model_copy's
        # unchecked update path. Only the requested keys can change.
        merged = {label.key: label.value for label in foundation.display_labels}
        merged.update(proposed)
        candidate = ProgressivePlanFoundation.model_validate({
            **foundation.model_dump(mode="python"), "display_labels": merged,
        })
        scope = finding.details.get("required_scope")
        reader_key = finding.details.get("required_key")
        validate_progressive_foundation(
            candidate, context=context, analysis_type=analysis_type,
            required_binary_display_label_scopes=(scope,) if scope else (),
            required_reader_display_label_keys=(reader_key,) if reader_key else (),
        )
        return candidate

    last_failure: Exception = finding
    messages = [LLMMessage(role="user", content=prompt)]
    for _ in range(remaining_revisions):
        prompt_metrics["compile_revision_count"] = prompt_metrics.get("compile_revision_count", 0) + 1
        if resumed:
            prompt_metrics["current_run_compile_revision_count"] = (
                prompt_metrics.get("current_run_compile_revision_count", 0) + 1
            )
        try:
            # The enclosing compiler budget owns every repair call, including
            # truncated JSON. Do not grant a hidden parse/resend retry here.
            with automatic_retry_scope(disabled=True):
                repaired = call_llm_with_structured_retry(
                    llm, messages, parser=parse,
                    role="progressive_planner_foundation",
                    max_retries=0, max_tokens=1_024, temperature=0.2,
                    structured_output=schema if llm_supports_strict_json_schema(llm) else None,
                    progress_callback=progress_callback, include_failed_response_on_retry=False,
                )
        except StructuredResponseFailure as exc:
            last_failure = exc
            messages = [LLMMessage(role="user", content=prompt + "\nValidator: " + str(exc.__cause__))]
            continue
        amended = materialization.model_copy(update={"foundation": repaired})
        prompt_metrics.setdefault("foundation_label_repairs", []).append({
            "reason_code": finding.reason_code,
            "required_display_label_keys": list(keys),
            "previous_foundation_sha256": canonical_sha256(materialization.model_dump(mode="json")),
            "foundation_sha256": canonical_sha256(amended.model_dump(mode="json")),
            "retained_step_count": len(retained),
            "retained_materializations_sha256": canonical_sha256(
                [item.model_dump(mode="json") for item in retained]
            ),
            "schema_authority_sha256": schema.authority_sha256,
        })
        return amended
    raise ProgressivePlanCompileError(
        "progressive_foundation_label_repair_exhausted",
        "The bounded foundation-label repair budget was exhausted.",
        path="display_labels", cause_code=finding.reason_code,
        findings=[{"required_display_label_keys": list(keys),
                   "repair_scope": "foundation_display_labels_only"}],
    ) from last_failure
