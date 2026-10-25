"""Validate explicit run inputs before reading the profile's write policy."""

from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple

from .config import PipelineConfig
from ..contracts.claim_ceiling import bind_plan_tier, first_result


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
    return True if first_result(pipeline._config) or pipeline._config.generate_manuscript is False else requested


def seal_plan_tier(plan, config: PipelineConfig, *, resumed: bool):
    """Bind new plans; never rewrite a digest-verified approval on resume."""
    tier = config.execution_tier or "article"
    if resumed:
        if getattr(plan.design_selection, "execution_tier", "article") != tier:
            raise ValueError("An approved run's execution_tier cannot change; start a new run.")
        return plan
    return bind_plan_tier(plan, tier)
