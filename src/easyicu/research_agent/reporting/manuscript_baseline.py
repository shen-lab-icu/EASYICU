"""Reader mention requirements from the existing accepted baseline contract.

This is a completeness check, not evidence that a baseline was computed or
that its definition is clinically valid. Execution and source authorities
remain separate; Writer must describe only the executed representation.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from ..planning.baseline_requirements import baseline_requirement_projection
from ..schema import ResearchContext


def missing_baseline_method_mentions(
    prose: str, expected: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """A data-availability result alone is not a description of a variable.

    This narrow completeness check does not verify that a stated representation
    matches execution; that remains the responsibility of source authority.
    """

    sentences = re.split(r"(?<=[.!?])\s+|\n\s*\n", prose)
    method_sentences = [sentence for sentence in sentences if not (
        re.search(r"\b(?:missing(?:ness)?|completeness|availability)\b", sentence, re.I)
        and not re.search(
            r"\b(?:defined|represented|derived|calculated|measured|collected|obtained)\b|"
            r"\b(?:first|last|mean|maximum|minimum|median)\s+(?:recorded\s+)?value\b",
            sentence, re.I,
        )
    )]
    return tuple(
        name for name, aliases in expected.items()
        if not any(
            re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", sentence, re.I)
            for sentence in method_sentences for alias in aliases if alias
        )
    )


def baseline_reporting_mentions(
    context: ResearchContext | None,
    reader_display_labels: Mapping[str, str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return source names and approved reader labels for every accepted row."""

    if not isinstance(context, ResearchContext):
        return {}
    labels = reader_display_labels or {}
    mentions: dict[str, tuple[str, ...]] = {}
    for table in baseline_requirement_projection(context)["tables"]:
        for row in table["variables"]:
            required = row["required"]
            aliases = [required, *row["available_columns"]]
            aliases.extend(labels.get(name, "") for name in row["available_columns"])
            mentions[required] = tuple(dict.fromkeys(
                " ".join(alias.split()) for alias in aliases if alias.strip()
            ))
    return mentions
