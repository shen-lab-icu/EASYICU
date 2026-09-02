"""Shared deterministic precision rule for reader-facing numeric prose."""

from __future__ import annotations

import re


OVERPRECISE_READER_DECIMAL_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?\d+\.\d{5,}(?!\d)"
)


def round_reader_numeric_display(text: str) -> tuple[str, int]:
    """Round only over-precise standalone decimals to at most three places."""

    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"{float(match.group(0)):.3f}".rstrip("0").rstrip(".")

    return OVERPRECISE_READER_DECIMAL_RE.sub(replace, str(text or "")), count


__all__ = ["OVERPRECISE_READER_DECIMAL_RE", "round_reader_numeric_display"]
