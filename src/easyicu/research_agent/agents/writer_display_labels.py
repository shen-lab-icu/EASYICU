"""Reader-facing label and output-language instructions for Writer prompts."""

from __future__ import annotations

import json
from typing import Mapping


def normalise_reader_display_labels(
    labels: Mapping[str, str] | None,
) -> dict[str, str]:
    """Keep non-empty authorized keys and whitespace-normalized labels."""

    return {
        str(key).strip(): " ".join(str(value).split())
        for key, value in dict(labels or {}).items()
        if str(key).strip() and " ".join(str(value).split())
    }


def reader_display_label_instruction(labels: Mapping[str, str]) -> str:
    """Render the closed reader-label policy plus its authorized mapping."""

    return (
        "READER-FACING LABEL RULE:\n"
        "- The mapping below is Planner-authorized presentation metadata. "
        "Use its clinical labels in manuscript prose and never copy its "
        "raw keys into reader-facing sentences. If a supplied label is in "
        "another language, render the same clinical meaning in the output "
        "language; do not invent or broaden its semantics.\n"
        "- Preserve raw identifiers only inside exact `{evidence:...}` and "
        "`{claim:...}` audit tokens.\n"
        + json.dumps(dict(labels), ensure_ascii=False, sort_keys=True)
        + "\n\n"
    )


def writer_language_instruction(language: str) -> str:
    """Return the Writer's output-language and audit-token contract."""

    if language == "zh":
        return (
            "OUTPUT LANGUAGE: zh / Simplified Chinese. Keep section headings "
            "as markdown headings. Preserve every `{evidence:<id>}` placeholder "
            "and `{claim:<step>.<claim>}` token exactly as ASCII; do not translate "
            "evidence ids or claim refs inside those tokens. Never expose filenames, "
            "raw variable names, or other code-like identifiers in reader prose."
        )
    return (
        "OUTPUT LANGUAGE: en / English. Preserve every `{evidence:<id>}` "
        "placeholder and `{claim:<step>.<claim>}` token exactly as ASCII."
    )


__all__ = [
    "normalise_reader_display_labels",
    "reader_display_label_instruction",
    "writer_language_instruction",
]
