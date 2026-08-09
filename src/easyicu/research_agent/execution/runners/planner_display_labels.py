"""One owner for Planner-declared binary level labels.

Several deterministic figure renderers must print the scientific name of a
binary level (``Sepsis-3 absent`` / ``Sepsis-3 present``) rather than a
placeholder such as ``Category 0`` or ``Level 1``.  The label authority is the
Planner's ``display_labels`` map, whose keys take the ``<column>=<0|1>`` form.

This module owns that parsing so no renderer re-implements it and so a renderer
can bind the resolved labels back to the exact column they describe.  It reads
no cohort, chooses no exposure, and never invents a label.
"""

from __future__ import annotations

import re
from typing import Mapping

__all__ = ["planner_binary_level_labels"]


_LEVEL_KEY = re.compile(r"\s*(.+?)\s*=\s*([01])\s*")


def planner_binary_level_labels(
    display_labels: Mapping[str, str] | None,
) -> tuple[str, str, str] | None:
    """Return ``(column, label_for_0, label_for_1)`` for the one complete pair.

    ``None`` means the Planner did not declare exactly one unambiguous binary
    label pair.  Callers must fail closed or fall back explicitly; this
    function never fabricates a placeholder.
    """

    pairs: dict[str, dict[int, str]] = {}
    for raw_key, raw_label in (display_labels or {}).items():
        match = _LEVEL_KEY.fullmatch(str(raw_key))
        label = " ".join(str(raw_label or "").split())
        if match is None or not label:
            continue
        pairs.setdefault(match.group(1).strip(), {})[int(match.group(2))] = label
    complete = [
        (column, levels[0], levels[1])
        for column, levels in pairs.items()
        if set(levels) == {0, 1} and levels[0] != levels[1] and column
    ]
    if len(complete) == 1:
        return complete[0]
    return None
