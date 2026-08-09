"""Fail-closed policy for generated Matplotlib source-data repairs."""

from __future__ import annotations


def patch_matplotlib_patch_source_rows(code: str, run_log: str) -> str:
    """Decline artist-to-source-data projection.

    Matplotlib artists are rendering output, not scientific source evidence.
    Recovering bar heights after the fact would make a figure validate without
    table-level lineage, so the source-data gate must remain fail-closed.
    """

    del run_log
    return code


__all__ = ["patch_matplotlib_patch_source_rows"]
