"""Lossless prompt views of sealed plans; never rewrite the source authority."""

from __future__ import annotations

import json
from typing import Any, Sequence

from ..canonical_json import canonical_json_bytes, sha256_bytes
from ..providers.prompt_budget import (
    CONSERVATIVE_BYTES_PER_TOKEN,
    PROMPT_TRANSPORT_BUDGETS,
    PromptConsumerBudget,
)
from ..providers.protocol import LLMMessage
from ..schema import AnalysisPlan


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def project_plan_revision_prompt(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Elide typed defaults only when full canonical JSON bytes round-trip.

    The host's source hashes and all non-plan prose stay verbatim. Sparse,
    unknown, malformed or coercing payloads stay verbatim too, with a receipt
    saying why. Receipts contain hashes and sizes, never the source content.
    """
    lines = text.splitlines(keepends=True)
    receipts: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        prefix = next((p for p in (
            "- source_plan_json: ", "- candidate_plan_seed_json: ",
        ) if line.startswith(p)), None)
        if prefix is None:
            continue
        body = line.rstrip("\r\n")
        raw = body[len(prefix):]
        original = raw.encode("utf-8")
        projected = original
        status = "retained"
        reason = "invalid_json_or_schema"
        try:
            payload = json.loads(raw, object_pairs_hook=_unique_object)
            canonical = canonical_json_bytes(payload)
            if prefix == "- source_plan_json: ":
                compact = AnalysisPlan.model_validate(payload).model_dump(
                    mode="json", exclude_defaults=True,
                )
                restored = AnalysisPlan.model_validate(compact).model_dump(mode="json")
                # dict equality would hide bool/int/float coercion.
                if canonical_json_bytes(restored) != canonical:
                    reason = "canonical_roundtrip_mismatch"
                else:
                    projected = canonical_json_bytes(compact)
                    status, reason = "compacted", "typed_defaults_exact_roundtrip"
            else:
                # A candidate seed is not a full AnalysisPlan. Only whitespace
                # and key order may change; it never undergoes schema coercion.
                projected = canonical
                status, reason = "compacted", "canonical_json_only"
        except (ValueError, TypeError):
            pass
        if len(projected) >= len(original):
            projected = original
            if status == "compacted":
                status, reason = "retained", "no_byte_saving"
        lines[index] = prefix + projected.decode("utf-8") + line[len(body):]
        receipts.append({
            "line": index + 1, "status": status, "reason": reason,
            "source_sha256": sha256_bytes(original), "source_bytes": len(original),
            "projected_sha256": sha256_bytes(projected),
            "projected_bytes": len(projected),
        })
    return "".join(lines), receipts


def planner_prompt_byte_limit(client: Any) -> int:
    """Use the transport's configured envelope and conservative estimator."""
    budget = getattr(client, "budget", None)
    if not isinstance(budget, PromptConsumerBudget):
        budget = PROMPT_TRANSPORT_BUDGETS["planner_plan_generation"]
    return int(budget.limit_tokens * CONSERVATIVE_BYTES_PER_TOKEN)


def retry_shape_reminder(messages: Sequence[LLMMessage], shape: str) -> str:
    """Reference a shape already kept in the immutable base, without copying it."""
    if shape and any(shape in message.content for message in messages):
        return "Follow the complete output shape and constraints in the initial request."
    return shape
