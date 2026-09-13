"""Lossless prompt views of sealed plans; never rewrite the source authority."""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Literal, Sequence

from ..canonical_json import canonical_json_bytes, canonical_sha256, sha256_bytes
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


PlanRevisionPromptStage = Literal["outline", "foundation", "step"]


def _stage_plan_projection(
    payload: dict[str, Any],
    *,
    stage: PlanRevisionPromptStage,
    step_id: str | None,
) -> dict[str, Any]:
    """Keep the exact source coordinates needed after outline acceptance.

    The complete source plan remains sealed by its digest and is shown to the
    outline call. Later calls already receive the validated outline, so they
    need the global plan choices, a lossless copy of the matching old step and
    a compact roster of the remaining steps. This prevents the same large plan
    from crowding every progressive request while preserving the old details
    at the stage that can act on them.
    """

    steps = list(payload.get("steps") or ())
    globals_ = {key: value for key, value in payload.items() if key != "steps"}
    roster_fields = (
        "step_id",
        "planned_analysis_role",
        "method",
        "expected_outputs",
        "sensitivity_spec_ids",
        "functional_form_spec",
    )
    roster = [
        {key: raw.get(key) for key in roster_fields if key in raw}
        for raw in steps
        if isinstance(raw, dict)
    ]
    selected = (
        [raw for raw in steps if isinstance(raw, dict) and raw.get("step_id") == step_id]
        if stage == "step" and step_id
        else []
    )
    return {
        "schema_version": "easyicu.plan_prompt_projection/1",
        "source_plan_sha256": sha256_bytes(canonical_json_bytes(payload)),
        "stage": stage,
        "plan_globals": globals_,
        "source_step_roster": roster,
        "selected_source_steps": selected,
    }


_PATH_SAFE_KEY = re.compile(r"^[^.\[\]]*$")
_PATH_TOKEN = re.compile(r"\.|([^.[\]]+)|\[(\d+)\]")


def _delta_path(path: tuple[Any, ...]) -> str:
    """Render a structural path; raises ``_DeltaUnsafe`` on ambiguous keys."""

    rendered = ""
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            key = str(part)
            if not _PATH_SAFE_KEY.match(key):
                raise _DeltaUnsafe(key)
            rendered += f".{key}" if rendered else key
    return rendered


class _DeltaUnsafe(Exception):
    """A key cannot be expressed unambiguously in a delta path."""


def _plan_delta_ops(
    base: Any, target: Any, path: tuple[Any, ...], ops: list[dict[str, Any]]
) -> None:
    """Append the leaf ops that turn ``base`` into ``target``.

    Dict keys diff recursively; equal-length lists diff per index so an edit
    inside one item stays one leaf op; a resized or retyped value is replaced
    wholesale.  The emitted ops are therefore an exhaustive change list --
    every difference is either named here or absent.
    """

    if isinstance(base, dict) and isinstance(target, dict):
        for key in base:
            if key not in target:
                ops.append({"path": _delta_path((*path, key)), "unset": True})
            else:
                _plan_delta_ops(base[key], target[key], (*path, key), ops)
        for key in target:
            if key not in base:
                ops.append(
                    {"path": _delta_path((*path, key)), "set": target[key]}
                )
        return
    if (
        isinstance(base, list)
        and isinstance(target, list)
        and len(base) == len(target)
    ):
        for index, (old, new) in enumerate(zip(base, target)):
            _plan_delta_ops(old, new, (*path, index), ops)
        return
    if base != target:
        ops.append({"path": _delta_path(path), "set": target})


def _parse_delta_path(path: str) -> tuple[Any, ...]:
    """Parse the ``key.sub[0]`` path grammar emitted by :func:`_delta_path`."""

    parts: list[Any] = []
    for match in _PATH_TOKEN.finditer(path):
        token, index = match.group(1), match.group(2)
        if token is None and index is None:
            continue
        parts.append(int(index) if index is not None else token)
    if _delta_path(tuple(parts)) != path:
        raise ValueError(f"ambiguous delta path: {path!r}")
    return tuple(parts)


def apply_plan_delta(base_view: dict[str, Any], delta: dict[str, Any]) -> Any:
    """Apply an ``easyicu.plan_revision_delta/1`` op list to a projected view.

    The host can restore a later source-plan block exactly: apply the ops to
    the earlier projected view, then compare canonical bytes or the declared
    ``source_plan_sha256``.  A malformed op raises instead of approximating.
    """

    if not isinstance(delta.get("ops"), list):
        raise ValueError("plan delta requires an ops list")
    result: Any = copy.deepcopy(base_view)
    for op in delta["ops"]:
        if not isinstance(op, dict) or "path" not in op:
            raise ValueError("plan delta op requires a path")
        parts = _parse_delta_path(str(op["path"]))
        if not parts:
            if "set" in op:
                result = op["set"]
                continue
            raise ValueError("plan delta op needs a non-root path to unset")
        node = result
        for part in parts[:-1]:
            node = node[part]
        last = parts[-1]
        if op.get("unset"):
            if not isinstance(node, dict):
                raise ValueError("unset targets a mapping key")
            node.pop(last)
        elif "set" in op:
            node[last] = op["set"]
        else:
            raise ValueError("plan delta op requires set or unset")
    return result


def _plan_delta_restores(
    base_view: dict[str, Any], ops: list[dict[str, Any]], target_view: Any
) -> bool:
    """Self-check that the emitted delta reproduces the target view exactly."""

    try:
        restored = apply_plan_delta(base_view, {"ops": ops})
    except (KeyError, IndexError, TypeError, ValueError, _DeltaUnsafe):
        return False
    return canonical_json_bytes(restored) == canonical_json_bytes(target_view)


_REPLAN_SECTION_HEADER = "DIGEST-BOUND FAILED EXECUTION REPLAN (host-derived):"
_SOURCE_PLAN_SHA_LINE = re.compile(r"^- source_plan_sha256: ([0-9a-f]{64})\s*$")
_SOURCE_PLAN_JSON_PREFIX = "- source_plan_json: "


def _elide_superseded_replan_sections(
    lines: list[str],
    *,
    stage: PlanRevisionPromptStage,
    step_id: str | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Replace verified ancestor plan JSON lines with digest pointers.

    A chained failed-execution replan accumulates one section per generation:
    every new source run appends its full plan to the contract it inherited.
    Only the LAST section is the current source plan; the earlier sections are
    ancestors whose requirements its own contract already required it to
    preserve ("Preserve the source question, cohort, all outcomes, methods,
    baseline variables, timing, sensitivity analyses and displays").  Sending
    each ancestor's full text again is what pushes a legitimate revision past
    the transport envelope, so each superseded plan JSON line is replaced by a
    digest-bound pointer that records its ``source_plan_sha256`` and the section
    that supersedes it.  Nothing about the current source plan, the candidate
    seed, or any surviving requirement changes.

    Fail-closed: unless every replan section has exactly one parseable source
    plan and one ``source_plan_sha256``, and no two sections share a digest,
    the text is returned untouched with no receipts. The digest and full typed
    round trip must also match. Non-plan text stays verbatim.
    """

    starts = [
        index
        for index, line in enumerate(lines)
        if line.strip() == _REPLAN_SECTION_HEADER
    ]
    if len(starts) < 2:
        return lines, []
    sections = [
        (start, starts[position + 1] if position + 1 < len(starts) else len(lines))
        for position, start in enumerate(starts)
    ]
    checked: list[tuple[int, int, str, int]] = []
    for start, end in sections:
        digests = [
            match.group(1) for line in lines[start:end]
            if (match := _SOURCE_PLAN_SHA_LINE.match(line.strip()))
        ]
        plan_lines = [
            index for index in range(start, end)
            if lines[index].startswith(_SOURCE_PLAN_JSON_PREFIX)
        ]
        if len(digests) != 1 or len(plan_lines) != 1:
            return lines, []
        digest, plan_at = digests[0], plan_lines[0]
        try:
            payload = json.loads(
                lines[plan_at].rstrip("\r\n")[len(_SOURCE_PLAN_JSON_PREFIX):],
                object_pairs_hook=_unique_object,
            )
            restored = AnalysisPlan.model_validate(payload).model_dump(mode="json")
            if (
                canonical_sha256(payload) != digest
                or canonical_json_bytes(restored) != canonical_json_bytes(payload)
            ):
                return lines, []
        except (ValueError, TypeError):
            return lines, []
        checked.append((start, end, digest, plan_at))
    digests = [item[2] for item in checked]
    if len(set(digests)) != len(digests):
        return lines, []
    kept_digest = digests[-1]
    output: list[str] = []
    receipts: list[dict[str, Any]] = []
    cursor = 0
    # Only the verified plan line is superseded. Text between replan headers
    # can also contain later review requirements or a candidate seed.
    for start, end, digest, plan_at in checked[:-1]:
        output.extend(lines[cursor:plan_at])
        newline = lines[plan_at][len(lines[plan_at].rstrip("\r\n")):]
        pointer = (
            f"- superseded_source_plan_sha256: {digest} "
            f"(historical plan; current source_plan_sha256: {kept_digest})."
            + newline
        )
        output.append(pointer)
        source_text = lines[plan_at].encode("utf-8")
        pointer_text = pointer.encode("utf-8")
        receipts.append({
            "line": plan_at + 1,
            "status": "superseded_elided",
            "reason": "ancestor_replan_superseded",
            "stage": stage,
            "step_id": step_id,
            "source_plan_sha256": digest,
            "superseded_by_plan_sha256": kept_digest,
            "source_bytes": len(source_text),
            "projected_bytes": len(pointer_text),
            "source_sha256": sha256_bytes(source_text),
            "projected_sha256": sha256_bytes(pointer_text),
        })
        cursor = plan_at + 1
    output.extend(lines[cursor:])
    return output, receipts


def project_plan_revision_prompt(
    text: str,
    *,
    stage: PlanRevisionPromptStage = "outline",
    step_id: str | None = None,
    byte_budget: int | None = None,
    elide_superseded_replans: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Elide typed defaults only when full canonical JSON bytes round-trip.

    The host's source hashes and all non-plan prose stay verbatim. Sparse,
    unknown, malformed or coercing payloads stay verbatim too, with a receipt
    saying why. Receipts contain hashes and sizes, never the source content.

    ``byte_budget`` optionally caps the outline source-plan line. The compact,
    semantically complete plan (every declared requirement, with only typed
    defaults elided) is used when it fits. When it does not, the line is kept
    at that same complete view and the receipt records
    ``declared_requirements_exceed_byte_budget``; no lossy "bounded outline
    view" is substituted, because dropping intent, inputs, scientific
    action/capability or typed specs would silently rewrite the research
    requirements. The caller decides whether to fail closed on the difference.
    Later stages keep their digest-bound staged projection (globals plus the
    matching source step) because the caller already holds the validated
    outline at that point.

    Repeated ``source_plan_json`` blocks -- chained failed-execution replans
    append the full previous plan each time -- are deduplicated as an
    ``easyicu.plan_revision_delta/1`` op list against the preceding source-plan
    block's projected view.  The first block keeps its complete view, so every
    declared requirement still appears verbatim once; later versions appear as
    an exhaustive change list bound to both plans' ``source_plan_sha256``.  The
    delta is emitted only when it is smaller than the block's own view and
    reproduces that view byte-exactly under :func:`apply_plan_delta`; otherwise
    the complete view stands.  The model therefore sees the full plan plus what
    each revision changed, never a requirements roster with fields removed.

    ``elide_superseded_replans`` is the budget-pressure rung above that: it
    collapses each verified ancestor plan JSON line to a digest
    pointer while the current (last) source plan keeps its complete view.  The
    planners enable it only after the exactly assembled request overflows their
    transport envelope, so ordinary requests still restore every version.
    """
    lines = text.splitlines(keepends=True)
    elision_receipts: list[dict[str, Any]] = []
    if elide_superseded_replans:
        lines, elision_receipts = _elide_superseded_replan_sections(
            lines,
            stage=stage,
            step_id=step_id,
        )
    receipts: list[dict[str, Any]] = list(elision_receipts)
    previous_plan_view: dict[str, Any] | None = None
    previous_plan_identity: dict[str, Any] = {}
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
        emitted_prefix = prefix
        delta_identity: dict[str, Any] = {}
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
                    if stage == "outline":
                        view: Any = compact
                        view_status = "compacted"
                        view_reason = "typed_defaults_exact_roundtrip"
                        view_prefix = prefix
                    else:
                        view = _stage_plan_projection(
                            payload,
                            stage=stage,
                            step_id=step_id,
                        )
                        view_status = "stage_projected"
                        view_reason = "digest_bound_stage_projection"
                        view_prefix = f"- source_plan_{stage}_projection_json: "
                    view_bytes = canonical_json_bytes(view)
                    # Choose the complete representation before building a
                    # delta. A stage view can be larger than the original
                    # plan; the next delta must use what was actually sent.
                    if len(view_prefix.encode("utf-8")) + len(view_bytes) >= (
                        len(prefix.encode("utf-8")) + len(original)
                    ):
                        view = payload
                        view_bytes = original
                        view_status, view_reason = "retained", "no_byte_saving"
                        view_prefix = prefix
                    projected = view_bytes
                    status, reason = view_status, view_reason
                    emitted_prefix = view_prefix
                    plan_sha256 = canonical_sha256(payload)
                    if previous_plan_view is not None:
                        ops: list[dict[str, Any]] | None = []
                        try:
                            _plan_delta_ops(previous_plan_view, view, (), ops)
                        except _DeltaUnsafe:
                            ops = None
                        if ops is not None:
                            delta_doc: dict[str, Any] = {
                                "delta_format": "easyicu.plan_revision_delta/1",
                                "stage": stage,
                                "base_line": previous_plan_identity.get("line"),
                                "base_source_sha256": previous_plan_identity.get(
                                    "source_sha256"
                                ),
                                "base_plan_sha256": previous_plan_identity.get(
                                    "plan_sha256"
                                ),
                                "source_sha256": sha256_bytes(original),
                                "source_plan_sha256": plan_sha256,
                                "semantics": (
                                    "the base_line block above carries the full "
                                    "projected plan; these ops name every leaf "
                                    "that changed, was added, or was removed"
                                ),
                                "ops": ops,
                            }
                            delta_bytes = canonical_json_bytes(delta_doc)
                            delta_prefix = "- source_plan_delta_json: "
                            if len(delta_prefix.encode("utf-8")) + len(delta_bytes) < (
                                len(view_prefix.encode("utf-8")) + len(view_bytes)
                            ) and (
                                _plan_delta_restores(previous_plan_view, ops, view)
                            ):
                                projected = delta_bytes
                                status = "delta"
                                reason = "chained_previous_source_plan"
                                emitted_prefix = delta_prefix
                                delta_identity = {
                                    "delta_base_line": previous_plan_identity.get(
                                        "line"
                                    ),
                                    "delta_base_plan_sha256": (
                                        previous_plan_identity.get("plan_sha256")
                                    ),
                                    "delta_op_count": len(ops),
                                }
                    if (
                        stage == "outline"
                        and byte_budget is not None
                        and len(projected) > byte_budget
                    ):
                        status = "budget_exceeded"
                        reason = "declared_requirements_exceed_byte_budget"
                    previous_plan_view = view
                    previous_plan_identity = {
                        "line": index + 1,
                        "source_sha256": sha256_bytes(original),
                        "plan_sha256": plan_sha256,
                    }
            else:
                # A candidate seed is not a full AnalysisPlan. Only whitespace
                # and key order may change; it never undergoes schema coercion.
                projected = canonical
                status, reason = "compacted", "canonical_json_only"
        except (ValueError, TypeError):
            pass
        if len(emitted_prefix.encode("utf-8")) + len(projected) >= (
            len(prefix.encode("utf-8")) + len(original)
        ):
            projected = original
            emitted_prefix = prefix
            delta_identity = {}
            if status in {"compacted", "delta"}:
                status, reason = "retained", "no_byte_saving"
        lines[index] = emitted_prefix + projected.decode("utf-8") + line[len(body):]
        receipts.append({
            "line": index + 1, "status": status, "reason": reason,
            "stage": stage,
            "step_id": step_id,
            "byte_budget": byte_budget,
            "over_budget_bytes": (
                max(0, len(projected) - byte_budget)
                if byte_budget is not None
                else 0
            ),
            "source_sha256": sha256_bytes(original), "source_bytes": len(original),
            "projected_sha256": sha256_bytes(projected),
            "projected_bytes": len(projected),
            **delta_identity,
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
