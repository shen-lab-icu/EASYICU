"""Reader-facing label and output-language instructions for Writer prompts."""

from __future__ import annotations

import json
from typing import Mapping

from ..planning.exposure_group_compile import planned_exposure_contrasts
from ..contracts.exposure_group_rules import EXPOSURE_GROUP_CONTRASTS_KEY
from ..schema import AnalysisPlan, ResearchContext


def reader_contrast_instruction(
    plan: AnalysisPlan | None, labels: Mapping[str, str], *, context: ResearchContext,
) -> str:
    """Name the direction sealed in the executable plan, never a grouping default."""
    if plan is None:
        return ""
    variables = [
        item["variable"]
        for item in context.cohort.provenance.get(EXPOSURE_GROUP_CONTRASTS_KEY) or []
    ]
    pairs = planned_exposure_contrasts(plan, variables=variables)
    if not pairs:
        return ""
    lines = [
        f"- {labels.get(variable, variable)}: "
        f"{labels.get(f'{variable}={comparison}', f'level {comparison}')} vs "
        f"{labels.get(f'{variable}={reference}', f'level {reference}')} (reference)."
        for variable, (reference, comparison) in pairs.items()
    ]
    return (
        "COMPILED CONTRAST DIRECTION:\n"
        "Name each comparison in this direction, with the reference explicitly identified. "
        "Risk differences subtract the reference risk from the comparison risk.\n"
        + "\n".join(lines) + "\n\n"
    )


def normalise_reader_display_labels(
    labels: Mapping[str, str] | None,
) -> dict[str, str]:
    """Keep non-empty authorized keys and whitespace-normalized labels."""

    return {
        str(key).strip(): " ".join(str(value).split())
        for key, value in dict(labels or {}).items()
        if str(key).strip() and " ".join(str(value).split())
    }


def reader_display_label_instruction(
    labels: Mapping[str, str], *, plan: AnalysisPlan | None = None,
    context: ResearchContext | None = None,
) -> str:
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
        "- A key `<column>=<code>` names one level of that column. Name a "
        "level of a grouped exposure by its label; when a sentence needs its "
        "code, cite it as `level <code>`. Give a level no meaning, threshold "
        "or order that its label does not state.\n"
        + json.dumps(dict(labels), ensure_ascii=False, sort_keys=True)
        + "\n\n"
        + (reader_contrast_instruction(plan, labels, context=context) if context is not None else "")
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
    "reader_contrast_instruction",
    "writer_language_instruction",
]
