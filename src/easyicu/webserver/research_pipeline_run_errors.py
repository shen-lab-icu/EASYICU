"""Stable Web-facing errors shared by Research Agent launch owners."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from easyicu.research_agent.orchestration.progress import ProgressControlSignal


class ResearchPipelineRunError(ProgressControlSignal):
    """Stable Web-facing failure from the true Research Agent bridge."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details or {})
