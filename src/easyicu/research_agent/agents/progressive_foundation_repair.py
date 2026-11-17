"""Bounded Planner repair of foundation display-label requirements."""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence

from ..canonical_json import canonical_sha256, extract_json_payload
from ..planning.progressive_compiler import (
    required_binary_display_label_scopes,
    required_reader_display_label_keys,
    validate_progressive_foundation,
)
from ..planning.progressive_contract import (
    ProgressiveFoundationMaterialization,
    ProgressivePlanCompileError,
    ProgressivePlanFoundation,
    ProgressivePlanOutline,
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


def defer_foundation_label_repair(check: Callable[[], Any]) -> None:
    """Keep projection parse retries from rewriting a label-only refusal.

    The caller must run the projected candidate through
    ``run_with_foundation_label_repair`` before publishing it. All other
    validation failures keep their original parse/host handling.
    """
    try:
        check()
    except ProgressivePlanCompileError as exc:
        if not foundation_label_repair_keys(exc):
            raise


def validate_projected_foundation(
    foundation: ProgressivePlanFoundation, *, context: ResearchContext,
    outline: ProgressivePlanOutline, require_robustness_intent: bool,
    retained: Sequence[ProgressiveStepMaterialization] = (),
) -> None:
    """Check both the outline's requirements and an accepted replay prefix."""
    validate_progressive_foundation(
        foundation, context=context, analysis_type=outline.analysis_type,
        require_robustness_intent=require_robustness_intent,
        robustness_replay_required=any(step.module_id == "robustness_replay" for step in outline.steps),
        required_binary_display_label_scopes=required_binary_display_label_scopes(
            context, [*outline.steps, *(item.step for item in retained)],
        ),
        required_reader_display_label_keys=required_reader_display_label_keys(context, outline.design_selection),
    )


def repair_projected_foundation(
    llm: Any, *, materialization: ProgressiveFoundationMaterialization,
    context: ResearchContext, outline: ProgressivePlanOutline,
    require_robustness_intent: bool, retained: Sequence[ProgressiveStepMaterialization],
    prompt_metrics: dict[str, Any], revision_limit: int,
    progress_callback: Callable[[Any], None] | None, resumed: bool,
    on_repaired: Callable[[ProgressiveFoundationMaterialization], None],
) -> ProgressiveFoundationMaterialization:
    """Validate a fresh or resumed foundation before any prefix is published."""
    def record_repaired(item: ProgressiveFoundationMaterialization) -> None:
        nonlocal materialization
        materialization = item
        on_repaired(item)

    run_with_foundation_label_repair(
        lambda current: validate_projected_foundation(
            current, context=context, outline=outline,
            require_robustness_intent=require_robustness_intent, retained=retained,
        ),
        llm, materialization=materialization, context=context, analysis_type=outline.analysis_type,
        retained=retained, prompt_metrics=prompt_metrics,
        revision_base=prompt_metrics["compile_revision_count"], revision_limit=revision_limit,
        progress_callback=progress_callback, resumed=resumed, on_repaired=record_repaired,
    )
    return materialization


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
    # Exhaustion does not replace the validator's typed refusal or coordinates.
    if last_failure is finding:
        raise finding
    raise finding from last_failure


def run_with_foundation_label_repair(
    check: Callable[[ProgressivePlanFoundation], Any],
    llm: Any,
    *,
    materialization: ProgressiveFoundationMaterialization,
    context: ResearchContext,
    analysis_type: str,
    retained: Sequence[ProgressiveStepMaterialization],
    prompt_metrics: dict[str, Any],
    revision_limit: int,
    revision_base: int,
    progress_callback: Callable[[Any], None] | None,
    resumed: bool,
    on_repaired: Callable[[ProgressiveFoundationMaterialization], None],
) -> Any:
    """Run a foundation-bound check; amend only the display labels it rejects.

    Each amendment spends the caller's current-step compiler budget
    (``revision_limit`` counted from ``revision_base``) and is reported through
    ``on_repaired`` before the check runs again on the amended foundation.
    """
    while True:
        try:
            return check(materialization.foundation)
        except ProgressivePlanCompileError as exc:
            if not foundation_label_repair_keys(exc):
                raise
            spent = prompt_metrics.get("compile_revision_count", 0) - revision_base
            materialization = repair_foundation_display_labels(
                llm, finding=exc, materialization=materialization, context=context,
                analysis_type=analysis_type, retained=retained, prompt_metrics=prompt_metrics,
                remaining_revisions=revision_limit - spent,
                progress_callback=progress_callback, resumed=resumed,
            )
            on_repaired(materialization)


def binding_repair_response_failure(
    observation: Mapping[str, Any] | None, *, step_id: str, step_index: int,
) -> ProgressivePlanCompileError | None:
    """Keep a prepared-input binding owner when its repair answer is malformed.

    The response budget is already spent; returning the binding finding stops a
    prepared run from being sent back to data preparation.
    """
    if not observation or observation.get("path") not in {
        "context_bindings", "raw_inputs", "product_inputs",
    }:
        return None
    owner_index = observation.get("step_index")
    return ProgressivePlanCompileError(
        str(observation["reason_code"]),
        str(observation.get("message") or "The prepared-input binding remains invalid."),
        step_id=observation.get("step_id") or step_id,
        step_index=owner_index if type(owner_index) is int else step_index,
        path=str(observation["path"]),
        cause_code="progressive_step_response_repair_exhausted",
        findings=observation.get("findings") or [],
    )


def spend_host_candidate_repair(
    error: ProgressivePlanCompileError,
    *,
    candidate: ProgressiveStepMaterialization,
    prompt_metrics: dict[str, Any],
    revision_base: int,
    revision_limit: int,
    resumed: bool,
    non_repairable: frozenset[str] | set[str],
) -> dict[str, Any] | None:
    """Admit a rejected template candidate to the current-step repair lane.

    Returns the compiler observation for the Planner's repair call and spends
    one revision, or ``None`` when the finding needs the outline or the shared
    current-step budget is spent (the caller then re-raises the finding).
    """
    if error.reason_code in non_repairable or error.details.get("repair_scope") == "outline_required":
        return None
    if prompt_metrics["compile_revision_count"] - revision_base >= revision_limit:
        return None
    prompt_metrics["compile_revision_count"] += 1
    if resumed:
        prompt_metrics["current_run_compile_revision_count"] += 1
    return {**error.details, "previous_materialization": candidate.model_dump(mode="json")}
