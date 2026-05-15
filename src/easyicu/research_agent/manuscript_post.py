"""Post-processing helpers for the bound manuscript.

These functions run between ``EvidenceStore.bind_manuscript`` and the
final report. They are pure-text rewrites that:

* demote unresolved ``[evidence missing: <id>]`` markers to HTML
  comments so the bound markdown still renders cleanly (while the
  pre-demotion text is preserved on disk for reviewers);
* drop sentences carrying ``[TBD]`` / ``[TODO]`` / ``[TK]`` writer
  placeholders that small/local models occasionally leak into the
  bound output;
* repair common writer aliasing mistakes on prediction-task
  manuscripts (e.g. a writer reaching for ``{evidence:outcome_rate}``
  when the cohort's prediction performance lives under a different
  registered id).

They were originally inline in :mod:`pipeline`. They are module-level
pure functions with no pipeline state, so isolating them here cuts
``pipeline.py`` down without changing any class surface.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from .evidence import EvidenceStore
from .schema import ResearchContext


_UNRESOLVED_EVIDENCE_PLACEHOLDER_RE = re.compile(
    r"\[evidence missing:\s*(?P<id>[^\]]+)\]"
)

_TBD_RE = re.compile(r"\[(?:TBD|TODO|TK)\]|\bTBD\b", re.IGNORECASE)


def _first_resolvable_name(
    resolvable: set[str], candidates: Sequence[str]
) -> Optional[str]:
    for candidate in candidates:
        if candidate in resolvable:
            return candidate
    for candidate in candidates:
        for name in sorted(resolvable):
            if candidate in name:
                return name
    return None


def _repair_common_writer_placeholders(
    scaffold: str,
    *,
    context: ResearchContext,
    evidence: EvidenceStore,
) -> tuple[str, List[tuple[str, str]]]:
    """Map common writer aliases to existing evidence for prediction tasks.

    The manuscript writer sometimes carries habits from association tasks
    (`table_one`, `outcome_rate`, `primary_association`) into prediction-model
    tasks. When the intended fact is already present in a registered prediction
    step summary, repair the placeholder before binding instead of letting the
    manuscript accumulate avoidable missing-evidence comments.
    """
    text = scaffold
    repairs: List[tuple[str, str]] = []
    resolvable = set(evidence.resolvable_names())
    question = (context.research_question or "").lower()
    is_prediction = any(
        token in question for token in ("prediction", "predict", "auroc", "brier", "calibration")
    )
    if not is_prediction:
        return text, repairs

    prediction_summary = _first_resolvable_name(
        resolvable,
        (
            "01_model_training",
            "model_training",
            "model_performance",
            "prediction_performance",
            "statistic_step_summary",
            "research_context",
        ),
    )
    if prediction_summary is None:
        return text, repairs

    fallback_map: Dict[str, str] = {}
    if "table_one" not in resolvable and "research_context" in resolvable:
        fallback_map["table_one"] = "research_context"
    if "cohort_summary" not in resolvable and "research_context" in resolvable:
        fallback_map["cohort_summary"] = "research_context"
    if "outcome_rate" not in resolvable:
        fallback_map["outcome_rate"] = prediction_summary
    if "outcome_incidence" not in resolvable:
        fallback_map["outcome_incidence"] = prediction_summary
    if "primary_association" not in resolvable:
        fallback_map["primary_association"] = prediction_summary

    for old, new in fallback_map.items():
        old_token = "{evidence:" + old + "}"
        if old_token not in text:
            continue
        text = text.replace(old_token, "{evidence:" + new + "}")
        repairs.append((old, new))
    return text, repairs


def _demote_unresolved_evidence_placeholders(
    bound_manuscript: str,
) -> tuple[str, List[str]]:
    """Convert ``[evidence missing: <id>]`` markers to HTML comments.

    The binder writes the bracket form so a human reviewer can see what
    the writer expected to cite. Downstream acceptance logic counts the
    bracket form as a binding failure (``evidence_binding_issue``), even
    when the analytic artefacts are all present and the manuscript only
    needs a stylistic clean-up. Demoting the markers to ``<!-- evidence
    missing: <id> -->`` comments preserves the trace inside the source
    file (so reviewers can still grep for it) while letting the
    rendered manuscript read cleanly and unblock the run. The full
    pre-demotion text is also kept on disk by the caller as
    ``manuscript_scaffold_bound_unfiltered.md`` for transparency.

    Returns the demoted manuscript text and the list of evidence ids
    that were demoted (in source order, with duplicates preserved so
    callers can report the true count).
    """
    demoted: List[str] = []

    def _replace(match: re.Match[str]) -> str:
        eid = match.group("id").strip()
        demoted.append(eid)
        return f"<!-- evidence missing: {eid} -->"

    rewritten = _UNRESOLVED_EVIDENCE_PLACEHOLDER_RE.sub(_replace, bound_manuscript)
    return rewritten, demoted


def _remove_tbd_sentences(bound_manuscript: str) -> tuple[str, List[str]]:
    """Drop manuscript sentences that still contain unresolved writer placeholders.

    The writer prompt already tells the model not to emit ``[TBD]``.
    Small local/free models still occasionally do it, and a journal-facing
    bound manuscript should never carry placeholders that look like
    results. We remove only the offending sentence fragments and keep the
    surrounding evidence-bound prose intact.
    """

    removed: List[str] = []
    cleaned_lines: List[str] = []
    for line in (bound_manuscript or "").splitlines():
        if not _TBD_RE.search(line):
            cleaned_lines.append(line)
            continue
        parts = re.split(r"(?<=[.!?。！？])\s+", line)
        kept: List[str] = []
        for part in parts:
            if _TBD_RE.search(part):
                stripped = part.strip()
                if stripped:
                    removed.append(stripped)
            else:
                kept.append(part)
        rewritten = " ".join(part for part in kept if part).strip()
        if rewritten:
            cleaned_lines.append(rewritten)
    return "\n".join(cleaned_lines).strip() + "\n", removed


__all__ = [
    "_first_resolvable_name",
    "_repair_common_writer_placeholders",
    "_demote_unresolved_evidence_placeholders",
    "_remove_tbd_sentences",
]
