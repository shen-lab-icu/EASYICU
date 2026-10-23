"""Validate explicit run inputs before reading the profile's write policy."""

from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple

from .config import PipelineConfig


class RunEntryRuntime(Protocol):
    """Configuration read only after explicit resume coordinates are valid."""

    _config: PipelineConfig


def _validate_resume_session_key(value: Any) -> Optional[Tuple[str, str, str]]:
    """Validate an explicit resume session key early (fail fast on shape)."""

    if value is None:
        return None
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 3
        or not all(isinstance(part, str) and part.strip() for part in value)
    ):
        raise ValueError(
            "resume_session_key must be a (project_id, task_id, session_id) "
            "triple of non-empty strings"
        )
    return (value[0], value[1], value[2])


def analysis_only_after_input_validation(
    pipeline: RunEntryRuntime, requested: bool, resume_session_key: Any,
) -> bool:
    """Preserve checkpoint stops and enforce profiles that omit a manuscript."""
    _validate_resume_session_key(resume_session_key)
    return True if pipeline._config.generate_manuscript is False else requested
