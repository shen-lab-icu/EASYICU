"""Receipted planner prompt views; never rewrite the source authority."""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Sequence

from ..canonical_json import canonical_json_bytes, canonical_sha256, sha256_bytes
from ..providers.prompt_budget import (
    CONSERVATIVE_BYTES_PER_TOKEN,
    PROMPT_TRANSPORT_BUDGETS,
    PromptConsumerBudget,
)
from ..providers.protocol import LLMMessage
from ..schema import AnalysisPlan, ResearchContext


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
            "block": "planning_contract",
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
            "block": "planning_contract",
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


# These are transport accounting coordinates, never input-derived labels.
PLANNER_PROMPT_BLOCK_NAMES = (
    "system_guide",
    "article_question_context",
    "variables",
    "action_rows",
    "design_evidence_cards",
    "know_how",
    "planning_contract",
    "response_shape",
    "schema",
)
OUTLINE_COMPACTION = "outline_authority_compacted"
_PLANNER_PROJECTION_LADDER = (
    "ancestor_replan_superseded",
    "optional_variable_descriptions_omitted",
    "advisory_guidance_projected",
    "literature_provenance_projected",
    OUTLINE_COMPACTION,
)


@dataclass(frozen=True)
class PlannerPromptBlock:
    """An assembly boundary and owner-supplied, semantically safe alternatives.

    Unannotated text is authority. Each alternative is a complete version of
    the block for this stage, supplied by the owner of the prompt content.
    """

    name: str
    text: str
    projections: tuple[tuple[str, str], ...] = ()


def outline_authority_block(name: str, text: str, compact: str) -> PlannerPromptBlock:
    """Register an alternate layout of a host-authored outline block."""
    return PlannerPromptBlock(name, text, ((OUTLINE_COMPACTION, compact),))


_ACTION_HEADER = "Retrieved scientific actions (only these may be selected):\n"
_ACTION_COLUMNS = (
    "action_id", "name", "purpose", "produces", "execution_mode",
    "notes", "required_inputs", "runtime_contract", "outline_position",
)
_RUNTIME_COLUMNS = (
    "outputs", "required_product_inputs", "article_roles", "execution_parameters",
)
_ACTION_TAIL_DEFAULTS = {"notes": "", "required_inputs": [], "runtime_contract": None,
                         "outline_position": None}
_ACTION_FIELDS = {"analysis_type", *_ACTION_COLUMNS}
_ACTION_REQUIRED_FIELDS = _ACTION_FIELDS - {"outline_position"}
_ACTION_MODE_CODES = {"host_owned": "H", "coder_generated": "C"}


def _prompt_json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _outline_action_table(rows: Sequence[Mapping[str, Any]], full: str) -> str:
    """Keep action choice, scientific constraints and dependencies in each row.

    The step request carries the executor identifier and product semantic
    roles. The outline can only select an action and name its dependencies;
    the host fills those two runtime fields during materialization.
    """
    if any(
        not _ACTION_REQUIRED_FIELDS <= row.keys() <= _ACTION_FIELDS
        or not isinstance(row["execution_mode"], str)
        or ("outline_position" in row and not isinstance(row["outline_position"], str))
        for row in rows
    ):
        return full
    positions = Counter(row["outline_position"] for row in rows if "outline_position" in row)
    shared_positions = [text for text, count in positions.items() if count > 1]
    position_indexes = {text: index for index, text in enumerate(shared_positions)}
    families: dict[str, list[list[Any]]] = {}
    for row in rows:
        mode = row["execution_mode"]
        if mode in _ACTION_MODE_CODES.values():
            return full
        position = row.get("outline_position")
        compact = {
            **row,
            "execution_mode": _ACTION_MODE_CODES.get(mode, mode),
            "outline_position": position_indexes.get(position, position),
        }
        runtime = row["runtime_contract"]
        if runtime is not None:
            if (
                not isinstance(runtime, Mapping)
                or set(runtime) != {*_RUNTIME_COLUMNS, "standard_executor"}
                or not isinstance(runtime["outputs"], (list, tuple))
                or any(
                    not isinstance(output, Mapping)
                    or set(output) != {"product_id", "semantic_role"}
                    for output in runtime["outputs"]
                )
            ):
                return full
            compact["runtime_contract"] = [
                [output["product_id"] for output in runtime["outputs"]],
                *(runtime[field] for field in _RUNTIME_COLUMNS[1:]),
            ]
        columns = list(_ACTION_COLUMNS)
        while (field := columns[-1]) in _ACTION_TAIL_DEFAULTS and (
            compact[field] == _ACTION_TAIL_DEFAULTS[field]
        ):
            columns.pop()
        values = [compact[field] for field in columns]
        families.setdefault(row["analysis_type"], []).append(values)
    return (
        _ACTION_HEADER
        + "Rows grouped by analysis_type. Read each row in this column order: "
        + _prompt_json(_ACTION_COLUMNS)
        + ". execution_mode H=host_owned, C=coder_generated. "
        "Omitted trailing cells mean empty notes, empty required_inputs, "
        "null runtime_contract or absent outline_position. Runtime contract columns: "
        + _prompt_json(_RUNTIME_COLUMNS)
        + ". The host binds executor and product role details during step materialization."
        + (
            " Integer outline_position cells index this list from zero: "
            + _prompt_json(shared_positions) + "."
            if shared_positions else ""
        )
        + "\n" + _prompt_json(families)
    )


def outline_action_blocks(
    rows: Sequence[Mapping[str, Any]],
    module_ids_by_analysis_type: Mapping[str, Sequence[str]],
) -> tuple[PlannerPromptBlock, PlannerPromptBlock]:
    """Keep the action roster and module compatibility rules in either layout."""
    menu = _ACTION_HEADER + _prompt_json(list(rows))
    matrix = [
        {
            "analysis_type": analysis_type,
            "available_modules": list(modules),
            "scientific_action_ids": [
                str(row.get("action_id") or "")
                for row in rows
                if row.get("analysis_type") == analysis_type
            ],
        }
        for analysis_type, modules in module_ids_by_analysis_type.items()
    ]
    compatibility = (
        "The adjusted_association module may use only "
        "association.adjusted_association (or null). The "
        "robustness_replay module must always set scientific_action_id to "
        "null because it replays the sealed sensitivity specifications; a "
        "separate scientific action needs its own custom_analysis step. "
        "Cohort-definition, measurement_audit, absolute_risk_context, raw-distribution, "
        "visualization, and report support steps must also set "
        "scientific_action_id to null. A table_one step must bind "
        "descriptive.table_one when that exact action appears in the "
        "selected analysis family's scientific_action_ids; otherwise it "
        "must use null."
    )
    return (
        outline_authority_block("action_rows", menu, _outline_action_table(rows, menu)),
        outline_authority_block(
            "action_rows",
            "Analysis-family action matrix (do not cross family boundaries):\n"
            + _prompt_json(matrix)
            + "\nAfter selecting analysis_type, use only that row's action ids. "
            + compatibility,
            "Use only the scientific action ids listed for the selected analysis_type. "
            + compatibility,
        ),
    )


def compact_outline_shape(
    template: dict[str, Any],
    analysis_types: Sequence[str],
    module_ids_by_analysis_type: Mapping[str, Sequence[str]],
) -> str:
    """Render shared candidate fields and shared family module rosters once."""
    design = template["design_selection"]
    selected, rejected = design["candidates"]
    overrides = {key: value for key, value in rejected.items() if selected[key] != value}
    example = {
        **template, "design_selection": {**design, "candidates": [selected]}
    }
    module_rosters: dict[tuple[str, ...], list[str]] = {}
    for analysis_type in analysis_types:
        modules = tuple(module_ids_by_analysis_type[analysis_type])
        module_rosters.setdefault(modules, []).append(analysis_type)
    return (
        _prompt_json(example)
        + "\nFor each rejected candidate use all the same fields, with these overrides: "
        + _prompt_json(overrides)
        + ". Fill placeholders separately for each candidate.\n"
        "Allowed module_id roster by analysis_type ([families, module_ids]): "
        + _prompt_json([(families, modules) for modules, families in module_rosters.items()])
    )


def outline_design_selection_block(full: str) -> PlannerPromptBlock:
    """Reference the complete candidate shape and retain the decision rules."""
    return outline_authority_block(
        "response_shape", full,
        "Pre-result design selection contract:\nCompare 2-4 scientifically "
        "distinct designs for this exact question. Select exactly one and "
        "give a scientific reason for each rejection. Never select using "
        "observed results, significance, AIC/BIC, or predictive performance. "
        "Use only retrieved variables and sealed citations. With reviewed "
        "comparator cards, resolve the seven dimensions for every candidate "
        "with the designated source keys and a case-specific rationale. "
        "Use the six reviewable_plan items in the question's language, each at "
        "least 8 characters. Recommend the choices before asking the researcher; "
        "label them recommended, never user-confirmed, and name the data checks "
        "that could trigger revision. For description, item 2 includes requested "
        "grouping; in items 3 and 4 explain any inapplicability in lieu of a bare N/A.",
    )


_ARTICLE_ROLE_COLUMNS = ("article_requirement_id", "role", "required")


def outline_article_roles_block(
    contracts: Sequence[Mapping[str, Any]],
    module_ids_by_analysis_type: Mapping[str, Sequence[str]],
    guidance: str,
) -> PlannerPromptBlock:
    """Share the module roster and label the article requirement columns once."""
    header = "Candidate-specific host article role contracts:\n"
    full = header + _prompt_json(list(contracts)) + guidance
    if any(
        contract["available_modules"] != list(module_ids_by_analysis_type[contract["analysis_type"]])
        or any(set(item) != set(_ARTICLE_ROLE_COLUMNS) for item in contract["requirements"])
        for contract in contracts
    ):
        return PlannerPromptBlock("article_question_context", full)
    compact = [
        {
            **{key: value for key, value in contract.items()
               if key not in {"available_modules", "requirements"}},
            "requirements": [
                [item[key] for key in _ARTICLE_ROLE_COLUMNS]
                for item in contract["requirements"]
            ],
        }
        for contract in contracts
    ]
    return outline_authority_block(
        "article_question_context",
        full,
        header + "Each requirements row is [article_requirement_id,role,required]. "
        "Use the allowed module_id roster for that analysis_type.\n"
        + _prompt_json(compact) + guidance,
    )


class PlannerPromptBlocks:
    """Keep prompt assembly and accounting on the same explicit boundaries."""

    def __init__(self, blocks: Sequence[str | PlannerPromptBlock] = ()) -> None:
        self.blocks: list[PlannerPromptBlock] = []
        for block in blocks:
            self.append(block)

    def append(
        self,
        text: str | PlannerPromptBlock,
        *,
        block: str = "response_shape",
        projections: tuple[tuple[str, str], ...] = (),
    ) -> None:
        if isinstance(text, str):
            if block == "know_how":
                projections += (
                    ("advisory_guidance_projected", project_know_how_guidance(text)),
                )
            elif block == "design_evidence_cards":
                projections += (
                    (
                        "literature_provenance_projected",
                        project_literature_card_provenance(text),
                    ),
                )
        item = (
            text
            if isinstance(text, PlannerPromptBlock)
            else PlannerPromptBlock(block, text, projections)
        )
        if item.name not in PLANNER_PROMPT_BLOCK_NAMES:
            raise ValueError("unknown planner prompt block")
        self.blocks.append(item)

    def render(self, capture: list[PlannerPromptBlock] | None = None) -> str:
        if capture is not None:
            capture.extend(self.blocks)
        return "\n\n".join(item.text for item in self.blocks if item.text)


def protected_planner_variable_names(
    context: ResearchContext, contract: str
) -> frozenset[str]:
    """Protect question/plan names, definitions, anchors and all typed choices.

    Matching is deliberately conservative: an alias anywhere in the binding
    question, preferences or reviewed contract makes its complete description
    authority. A false positive retains extra context; it cannot erase intent.
    The candidate roster itself is always authority, even for unnamed columns.
    """
    authority = (
        canonical_json_bytes(
            {
                "study": context.model_dump(mode="json", exclude={"variables"}),
                "contract": contract,
            }
        )
        .decode("utf-8")
        .casefold()
    )
    protected = {context.primary_exposure, context.target_outcome}
    for variable in context.variables:
        aliases = (
            variable.name,
            variable.source_concept,
            variable.description,
            *(variable.derived_from_concepts or ()),
        )
        if any(
            str(alias).strip().casefold() in authority
            for alias in aliases
            if str(alias or "").strip()
        ):
            protected.add(variable.name)
    return frozenset(name for name in protected if name)


def project_know_how_guidance(text: str) -> str:
    """Project only the known structured advisory format; unknown text stays.

    Keep every stop/confirmation claim byte-for-byte, card identity, readiness,
    concept/citation bindings and at least one complete claim per card. An
    optional claim is needed only for cards without mandatory claims; other
    optional claims may be withheld, as in the retrieval owner's projection.
    Never withhold a whole card: the compiler requires its claim disposition.
    """
    header, separator, raw = text.partition("\n{")
    if not separator:
        return text
    try:
        payload = json.loads("{" + raw, object_pairs_hook=_unique_object)
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != "easyicu.research_know_how_prompt/2"
        ):
            return text
        cards = payload["cards"]
        if not isinstance(cards, list):
            return text
        projected_cards = []
        for card in cards:
            claims = card["claims"]
            if (
                not isinstance(claims, list)
                or not claims
                or any(
                    not isinstance(claim, dict)
                    or not all(
                        key in claim
                        for key in ("claim_id", "field", "text", "citation_ids")
                    )
                    for claim in claims
                )
            ):
                return text
            optional_fields = {
                "population",
                "time_zero",
                "observation_window",
                "followup_window",
                "eligibility",
                "exposure",
                "outcome",
                "estimand",
                "method",
                "sensitivity",
            }
            # Future/unknown claim kinds cannot silently become optional.
            mandatory = [
                claim for claim in claims if claim["field"] not in optional_fields
            ]
            optional = [claim for claim in claims if claim["field"] in optional_fields]
            kept = (
                mandatory
                or sorted(
                    optional,
                    key=lambda claim: (
                        len(canonical_json_bytes(claim)),
                        canonical_json_bytes(claim),
                    ),
                )[:1]
            )
            projected_cards.append(
                {
                    **card,
                    "claims": sorted(kept, key=canonical_json_bytes),
                    "omitted_claim_count": int(card.get("omitted_claim_count", 0))
                    + len(claims)
                    - len(kept),
                }
            )
        projected = {
            **payload,
            "cards": sorted(projected_cards, key=canonical_json_bytes),
        }
        return header + "\n" + canonical_json_bytes(projected).decode("utf-8") + "\n"
    except (KeyError, TypeError, ValueError):
        return text


def project_literature_card_provenance(text: str) -> str:
    """The source evidence retains its digests; keep all prompt facts verbatim."""
    return re.sub(
        r"^(- \[[A-Za-z0-9_.:-]+\] role=[a-z_]+; supplement=[a-z_]+); full_text_sha256=[0-9a-f]{64}$",
        r"\1",
        text,
        flags=re.MULTILINE,
    )


@dataclass(frozen=True)
class PlannerPromptRequest:
    messages: list[LLMMessage]
    blocks: tuple[PlannerPromptBlock, ...]
    metrics: dict[str, Any]


def _planner_block_bytes(
    system: str,
    blocks: Sequence[PlannerPromptBlock],
    schema_bytes: int,
) -> dict[str, int]:
    counts = dict.fromkeys(PLANNER_PROMPT_BLOCK_NAMES, 0)
    counts["system_guide"] = len(system.encode("utf-8"))
    counts["schema"] = schema_bytes
    previous = False
    for block in blocks:
        if block.text:
            # Charge each separator to the following block, exactly once.
            counts[block.name] += len(block.text.encode("utf-8")) + (
                2 if previous else 0
            )
            previous = True
    return counts


_REPAIR_FEEDBACK: ContextVar[str] = ContextVar("planner_repair_feedback", default="")


@contextmanager
def planner_repair_feedback_scope(feedback: str):
    # Repair quotes belong to the prompt. They cannot be part of the
    # host-authored planning contract that is parsed to bind signed suites.
    token = _REPAIR_FEEDBACK.set(feedback)
    try:
        yield
    finally:
        _REPAIR_FEEDBACK.reset(token)


def project_planner_request(
    messages: Sequence[LLMMessage],
    blocks: Sequence[PlannerPromptBlock],
    *,
    byte_limit: int,
    schema_bytes: int = 0,
) -> PlannerPromptRequest:
    """Measure the complete request, then take ordered projections as needed.

    Neither limit nor schema is projected. Receipts expose sizes and hashes,
    never prompt text or input-derived names. Original sources and checkpoint
    authorities stay untouched. A request that still overflows is returned for
    the caller's typed refusal with the same complete accounting.
    """
    system, user = messages[0].content, messages[1].content
    current = tuple(blocks)
    if "\n\n".join(block.text for block in current if block.text) != user:
        # An unannotated/custom renderer has no proven advisory boundaries.
        current = (PlannerPromptBlock("response_shape", user),)
    if feedback := _REPAIR_FEEDBACK.get():
        current = (*current, PlannerPromptBlock("article_question_context", feedback))
    source_counts = _planner_block_bytes(system, current, schema_bytes)
    counts = source_counts
    tried = ["plan_revision_lossless", "complete"]
    receipts: list[dict[str, Any]] = []
    for reason in _PLANNER_PROJECTION_LADDER:
        if sum(counts.values()) <= byte_limit:
            break
        alternatives = [
            dict(block.projections).get(reason, block.text) for block in current
        ]
        atomic = reason == OUTLINE_COMPACTION
        # The outline alternatives share rosters. Apply all of them together
        # even if an early replacement already puts the request under budget.
        if atomic and sum(len(text.encode("utf-8")) for text in alternatives) >= sum(
            len(block.text.encode("utf-8")) for block in current
        ):
            continue
        tried.append(reason)
        for index, (block, candidate) in enumerate(zip(current, alternatives)):
            if candidate == block.text or (
                not atomic and len(candidate.encode("utf-8")) >= len(block.text.encode("utf-8"))
            ):
                continue
            source, projected = block.text.encode("utf-8"), candidate.encode("utf-8")
            receipts.append(
                {
                    "block": block.name,
                    "reason": reason,
                    "source_bytes": len(source),
                    "source_sha256": sha256_bytes(source),
                    "projected_bytes": len(projected),
                    "projected_sha256": sha256_bytes(projected),
                }
            )
            current = (
                *current[:index],
                replace(block, text=candidate),
                *current[index + 1 :],
            )
            counts = _planner_block_bytes(system, current, schema_bytes)
            if not atomic and sum(counts.values()) <= byte_limit:
                break
    total = sum(counts.values())
    return PlannerPromptRequest(
        messages=[
            LLMMessage(role="system", content=system),
            LLMMessage(
                role="user",
                content="\n\n".join(block.text for block in current if block.text),
            ),
        ],
        blocks=current,
        metrics={
            "request_bytes": total,
            "source_request_bytes": sum(source_counts.values()),
            "byte_limit": byte_limit,
            "message_bytes": total - schema_bytes,
            "schema_bytes": schema_bytes,
            "block_bytes": counts,
            "source_block_bytes": source_counts,
            # The compiler's public diagnostics permit host-authored integers
            # only. Flat coordinates retain the breakdown at that boundary.
            **{f"block_{name}_bytes": count for name, count in counts.items()},
            **{
                f"source_block_{name}_bytes": count
                for name, count in source_counts.items()
            },
            **{f"projection_{name}_attempts": 1 for name in tried},
            "projections_tried": tried,
            "prompt_projection_receipts": receipts,
            "without_know_how_total_bytes": sum(
                _planner_block_bytes(
                    system,
                    [block for block in current if block.name != "know_how"],
                    schema_bytes,
                ).values()
            ),
        },
    )
