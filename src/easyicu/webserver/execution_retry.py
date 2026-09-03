"""Owner for approved execution-checkpoint retry policy.

Research Pipeline resolves the checkpoint while Copilot projects the next
action. Both adapters must ask this module whether a wrapper failure preserves
the immutable approved plan instead of maintaining separate reason-code sets.
"""

from __future__ import annotations

from typing import Any


_REPLAYABLE_GATE_REASONS = frozenset(
    {
        "research_agent_pipeline_failed_closed",
        "research_pipeline_execution_failed",
    }
)


def preserves_approved_execution_checkpoint(gate_reason: Any) -> bool:
    """Return whether a gate reason may reuse an approved execution checkpoint."""

    return str(gate_reason or "").strip() in _REPLAYABLE_GATE_REASONS


__all__ = ["preserves_approved_execution_checkpoint"]
