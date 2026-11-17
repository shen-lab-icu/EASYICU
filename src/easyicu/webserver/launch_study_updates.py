"""Keep launch defaults private until the study accepts its job.

Scientific preparation can infer design, timing and wording defaults. A
refused launch must not persist those defaults and invalidate its own host
decision. Stage only these owner-written patches, then merge them with the
job handoff in the study store's single revision-checked write. Concurrent
researcher edits still win; there is no rollback of someone else's state.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Mapping


@dataclass
class _Updates:
    study_id: str
    patch: dict[str, Any] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)


_CURRENT: ContextVar[_Updates | None] = ContextVar("launch_study_updates", default=None)


def stage_launch_updates(submit):
    """Give each synchronous submission its own disposable defaults."""
    @wraps(submit)
    def staged(request, *args, **kwargs):
        token = _CURRENT.set(_Updates(str(getattr(request, "study_context_id", "")).strip()))
        try:
            return submit(request, *args, **kwargs)
        finally:
            _CURRENT.reset(token)
    return staged


def upsert_launch_context(study: Mapping[str, Any], patch: dict[str, Any], **options) -> dict:
    """Stage a launch-owned patch; ordinary non-launch callers save as before."""
    from easyicu.webserver import study_contexts

    current = _CURRENT.get()
    if current is None:
        return study_contexts.upsert_context(patch, **options)
    if str(study.get("id") or "") != current.study_id or patch.get("id") != current.study_id:
        raise ValueError("launch_study_update_binding_mismatch")
    saved = study_contexts.get_context(current.study_id)
    revision = int(study.get("revision") or 0)
    if saved is not None and int(saved.get("revision") or 0) != revision:
        raise study_contexts.StudyContextError({
            "error": "study_context_revision_conflict", "study_context_id": current.study_id,
            "expected_revision": revision, "current_revision": int(saved.get("revision") or 0),
        })
    # Use the store owner's pure normalizer and scientific cross-field
    # validation; staging must not introduce a second interpretation of a patch.
    validation_flags = {key: value for key, value in options.items()
                        if key.startswith("_server_") and key != "_server_agent_landmark_decision"}
    normalized = study_contexts.validate_context_update(
        patch, current_context=study, lifecycle_write=False, **validation_flags,
    )
    current.patch.update({key: value for key, value in normalized.items() if key != "id"})
    current.flags.update({key: value for key, value in options.items() if key.startswith("_server_")})
    return {**study, **normalized}


def handoff_launch_context(context_id: str, **options) -> dict:
    """Commit staged defaults and the active job together, or write nothing."""
    from easyicu.webserver import study_contexts

    current = _CURRENT.get()
    if current is None or not current.patch:
        return study_contexts.handoff_context(context_id, **options)
    if context_id != current.study_id:
        raise ValueError("launch_study_update_binding_mismatch")
    expected_revision = options.pop("expected_revision")
    return study_contexts.upsert_context(
        {"id": context_id, **current.patch, **options},
        active=True, expected_revision=expected_revision, require_revision=True,
        lifecycle_write=True, **current.flags,
    )
