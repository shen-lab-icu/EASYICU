"""Canonical task-selection contract for Figure 2 development tools.

This benchmark-owned module has one dependency-neutral responsibility: turn an
optional list of task ids into the canonical ordered subset. Materialization,
prompt preflight, and later offline canaries share this contract rather than
each interpreting subset order independently.
"""

from __future__ import annotations

from collections.abc import Sequence

from .evaluator.rubric_v1 import FIGURE2_TASK_IDS


class Canonical9TaskScopeError(ValueError):
    """A requested Canonical9 development scope is invalid."""


def canonical_task_scope(task_ids: Sequence[object] | None) -> tuple[str, ...]:
    """Return the full suite or one unique canonical ordered subset."""

    if task_ids is None or len(task_ids) == 0:
        return tuple(FIGURE2_TASK_IDS)
    requested = [str(value or "").strip() for value in task_ids]
    if any(not value for value in requested):
        raise Canonical9TaskScopeError("Canonical9 task ids must be non-empty")
    if len(requested) != len(set(requested)):
        raise Canonical9TaskScopeError("Canonical9 task ids must be unique")
    known = set(FIGURE2_TASK_IDS)
    unknown = sorted(set(requested) - known)
    if unknown:
        raise Canonical9TaskScopeError(f"unknown Canonical9 task id(s): {unknown}")
    selected = set(requested)
    return tuple(task_id for task_id in FIGURE2_TASK_IDS if task_id in selected)


__all__ = ["Canonical9TaskScopeError", "canonical_task_scope"]
