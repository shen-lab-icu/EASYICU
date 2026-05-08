"""Compatibility shim for generated scripts that import ``step_summary``.

Historically some agent-generated scripts imported a module-level
``step_summary`` object and then mutated or printed it. The current
runtime prefers writing a local ``step_summary.json`` file directly, but
keeping this shim avoids needless execution failures when older call
patterns reappear in free-model runs.
"""

from __future__ import annotations

step_summary: dict[str, object] = {}

__all__ = ["step_summary"]
