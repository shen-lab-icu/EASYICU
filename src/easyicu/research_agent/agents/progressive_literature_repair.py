"""A focused literature repair keeps what the retry did not touch.

A compiler-directed retry of one Progressive step may return only the
literature bindings, or only the coordinates, it changed.  These rules carry
the previous attempt's untouched parts forward when that attempt already
covered the step's exact sealed roster, and otherwise leave the compiler to
fail closed.  ``agents.progressive_planner`` applies them between attempts.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..planning.progressive_contract import (
    ProgressiveOutlineStep,
    ProgressivePlanCompileError,
    ProgressiveStepMaterialization,
)


def preserve_literature_roster_across_targeted_repair(
    *,
    current: ProgressiveStepMaterialization,
    previous: ProgressiveStepMaterialization | None,
    outline_step: ProgressiveOutlineStep,
) -> ProgressiveStepMaterialization:
    """Carry forward valid bindings omitted by a focused compiler repair.

    A compiler-directed retry may need to expand one binding (for example,
    adding a dependence design element) without revisiting the other sealed
    sources.  Schema-imperfect providers sometimes return only the bindings
    they changed.  When the previous attempt already covered the exact
    outline-owned roster, retain its untouched model-authored applications and
    overlay the current replacements.  Extras, duplicates, or an incomplete
    previous roster still fail closed in the compiler.
    """

    if previous is None or previous.step.step_id != current.step.step_id:
        return current
    expected = tuple(outline_step.literature_citation_keys)
    previous_bindings = list(previous.step.literature_bindings)
    current_bindings = list(current.step.literature_bindings)
    previous_keys = tuple(item.citation_key for item in previous_bindings)
    current_keys = tuple(item.citation_key for item in current_bindings)
    if (
        not expected
        or len(previous_keys) != len(set(previous_keys))
        or set(previous_keys) != set(expected)
        or len(current_keys) != len(set(current_keys))
        or not set(current_keys).issubset(set(expected))
        or set(current_keys) == set(expected)
    ):
        return current
    previous_by_key = {item.citation_key: item for item in previous_bindings}
    current_by_key = {item.citation_key: item for item in current_bindings}
    merged = [current_by_key.get(key, previous_by_key[key]) for key in expected]
    return current.model_copy(
        update={"step": current.step.model_copy(update={"literature_bindings": merged})}
    )


def preserve_non_targeted_coordinates_across_literature_repair(
    *,
    current: ProgressiveStepMaterialization,
    previous: ProgressiveStepMaterialization | None,
    compiler_observation: Mapping[str, Any] | None,
) -> ProgressiveStepMaterialization:
    """Limit a literature-only retry to the compiler-owned repair path.

    Some providers answer a focused literature-roster finding by rebuilding the
    whole step and dropping already-valid product inputs.  The compiler finding
    is path-scoped, so a retry for ``literature_bindings`` must retain every
    other coordinate from the immediately preceding attempt.  The merged step
    is compiled again normally; this preserves fail-closed validation while
    preventing an unrelated regression from consuming the final repair turn.
    """

    if (
        previous is None
        or previous.step.step_id != current.step.step_id
        or previous.outline_step_sha256 != current.outline_step_sha256
    ):
        return current
    observation_path = str((compiler_observation or {}).get("path") or "").strip()
    if observation_path != "literature_bindings":
        return current
    bindings = list(current.step.literature_bindings)
    if (compiler_observation or {}).get("reason_code") in {
        "progressive_step_required_method_layer_unbound",
        "progressive_final_method_layer_unbound",
    }:
        # A coverage-only finding follows successful source-scope validation.
        # It authorizes adding a use, not revoking another use of the same
        # source. Preserve both model-authored explanations; never infer a new
        # design element or silently truncate an application/divergence.
        prior = {item.citation_key: item for item in previous.step.literature_bindings}
        keys = [item.citation_key for item in bindings]
        if len(prior) == len(previous.step.literature_bindings) and len(keys) == len(
            set(keys)
        ):
            merged = []
            for item in bindings:
                old = prior.get(item.citation_key)
                if old is None or old == item:
                    merged.append(item)
                    continue
                payload = item.model_dump(mode="python")
                payload["design_elements"] = list(
                    dict.fromkeys([*old.design_elements, *item.design_elements])
                )
                for field in ("application", "divergence"):
                    texts = list(
                        dict.fromkeys(
                            text
                            for text in (getattr(old, field), getattr(item, field))
                            if text
                        )
                    )
                    payload[field] = "\n\n".join(texts) or None
                try:
                    merged.append(type(item).model_validate(payload))
                except ValueError as exc:
                    raise ProgressivePlanCompileError(
                        "progressive_literature_repair_scope_conflict",
                        "coverage-only repair must retain prior design uses and "
                        "caveats within the bounded binding contract; provide a "
                        "complete combined binding with concise rationale",
                        step_id=current.step.step_id,
                        path="literature_bindings",
                    ) from exc
            bindings = merged
    repaired_step = previous.step.model_copy(update={"literature_bindings": bindings})
    return current.model_copy(update={"step": repaired_step})


__all__ = [
    "preserve_literature_roster_across_targeted_repair",
    "preserve_non_targeted_coordinates_across_literature_repair",
]
