"""PHI-minimized research-context projection for repair transports."""

from __future__ import annotations

from ..schema import ResearchContext
from .outbound import format_outbound_safe_context


def format_repair_authority_context(
    ctx: ResearchContext,
    *,
    include_scientific_authority: bool,
    user_notes: str = "",
) -> str:
    """Render compact authority coordinates without observed cohort literals."""

    del include_scientific_authority, user_notes
    return format_outbound_safe_context(ctx)


__all__ = ["format_repair_authority_context"]
