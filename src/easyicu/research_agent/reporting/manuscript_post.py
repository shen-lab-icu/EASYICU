"""Post-processing helpers for the bound manuscript.

These functions run between ``EvidenceStore.bind_manuscript`` and the
final report. They are pure-text rewrites that:

* demote unresolved ``[evidence missing: <id>]`` markers to HTML
  comments so the bound markdown still renders cleanly (while the
  pre-demotion text is preserved on disk for reviewers);
* drop sentences carrying ``[TBD]`` / ``[TODO]`` / ``[TK]`` writer
  placeholders that small/local models occasionally leak into the
  bound output;
* repair an ordinally drifted step placeholder only when its semantic suffix
  names exactly one registered step, plus common prediction-task aliases while
  keeping outcome-rate aliases gated to binary/event-style targets.

They were originally inline in :mod:`pipeline`. They are module-level
pure functions with no pipeline state, so isolating them here cuts
``pipeline.py`` down without changing any class surface.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    EvidenceStore,
    NumericEffectScale,
    NumericEstimand,
    NumericClaim,
    _NUMERIC_IN_PROSE_RE,
)
from ..schema import ResearchContext
from .side_findings import (
    SideFinding,
    annotate_side_finding_leaks,
    side_finding_leaks,
)

_UNRESOLVED_EVIDENCE_PLACEHOLDER_RE = re.compile(
    r"\[evidence missing:\s*(?P<id>[^\]]+)\]"
)

_TBD_RE = re.compile(r"\[(?:TBD|TODO|TK)\]|\bTBD\b", re.IGNORECASE)

_EVIDENCE_REFERENCE_RE = re.compile(r"\{evidence:(?P<id>[^{}]+)\}")
_NUMBERED_STEP_ID_RE = re.compile(r"^(?P<ordinal>\d+)_(?P<suffix>.+)$")

_FORBIDDEN_INTERPRETIVE_TERMS = (
    "surprise",
    "surprising",
    "surprisingly",
    "unexpected",
    "unexpectedly",
    "interestingly",
    "notably",
    "striking",
    "strikingly",
)

_FORBIDDEN_INTERPRETIVE_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _FORBIDDEN_INTERPRETIVE_TERMS) + r")\b",
    re.IGNORECASE,
)
_MANUSCRIPT_METADATA_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?"
    r"(?:keywords?|key words|data\s+(?:and\s+code\s+)?availability|"
    r"code\s+availability|funding|conflicts?\s+of\s+interest|"
    r"acknowledg(?:e)?ments?|ethics\s+approval)"
    r"\s*(?:\*\*)?\s*[:：]?",
    re.I,
)


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


def _context_target_outcome_is_binary_like(context: ResearchContext) -> bool:
    target = str(getattr(context, "target_outcome", "") or "")
    variable = (
        context.variable(target) if target and hasattr(context, "variable") else None
    )
    source_concept = str(getattr(variable, "source_concept", "") or "").lower()
    description = str(getattr(variable, "description", "") or "").lower()
    dtype = str(getattr(variable, "dtype", "") or "").lower()
    question = (context.research_question or "").lower()
    haystack = " ".join([target.lower(), source_concept, description, question])
    non_binary_tokens = (
        "length of stay",
        "los",
        "time-to-event",
        "time to event",
        "survival",
        "cox",
        "hazard",
        "continuous",
        "mean difference",
    )
    if any(token in haystack for token in non_binary_tokens):
        return False
    if "float" in dtype and not any(
        token in haystack for token in ("binary", "event", "readmission")
    ):
        return False
    binary_tokens = (
        "binary",
        "event",
        "mortality",
        "death",
        "readmission",
        "readmit",
        "icu_death",
        "hospital_death",
    )
    return any(token in haystack for token in binary_tokens) or dtype in {
        "bool",
        "boolean",
    }


def _repair_unique_step_ordinal_placeholders(
    scaffold: str,
    *,
    resolvable: set[str],
) -> tuple[str, List[tuple[str, str]]]:
    """Repair only an off-by-ordinal citation with one exact semantic owner.

    Writer models sometimes preserve the full step name but invent the numeric
    prefix (for example ``03_feature_availability_flow`` when the registered
    step is ``02_feature_availability_flow``).  The suffix is the semantic step
    identity; it is safe to repair only when exactly one resolvable numbered
    step has that suffix.  No fuzzy spelling, substring, or nearest-neighbour
    match is permitted, and ambiguous suffixes remain unresolved.
    """

    by_suffix: Dict[str, set[str]] = {}
    for name in resolvable:
        match = _NUMBERED_STEP_ID_RE.fullmatch(str(name))
        if match is None:
            continue
        by_suffix.setdefault(match.group("suffix"), set()).add(str(name))

    text = scaffold
    repairs: List[tuple[str, str]] = []
    seen: set[str] = set()
    for match in _EVIDENCE_REFERENCE_RE.finditer(scaffold):
        old = match.group("id").strip()
        if old in seen or old in resolvable:
            continue
        seen.add(old)
        step_match = _NUMBERED_STEP_ID_RE.fullmatch(old)
        if step_match is None:
            continue
        candidates = sorted(by_suffix.get(step_match.group("suffix"), set()))
        if len(candidates) != 1:
            continue
        new = candidates[0]
        text = text.replace(
            "{evidence:" + old + "}",
            "{evidence:" + new + "}",
        )
        repairs.append((old, new))
    return text, repairs


def _repair_common_writer_placeholders(
    scaffold: str,
    *,
    context: ResearchContext,
    evidence: EvidenceStore,
    allowed_evidence_names: Optional[Sequence[str]] = None,
) -> tuple[str, List[tuple[str, str]]]:
    """Repair provable step ordinals and common prediction evidence aliases.

    The manuscript writer sometimes carries habits from association tasks
    (`table_one`, `outcome_rate`, `primary_association`) into prediction-model
    tasks. When the intended fact is already present in a registered prediction
    step summary, repair the placeholder before binding instead of letting the
    manuscript accumulate avoidable missing-evidence comments.
    """
    resolvable = set(
        evidence.resolvable_names()
        if allowed_evidence_names is None
        else allowed_evidence_names
    )
    text, repairs = _repair_unique_step_ordinal_placeholders(
        scaffold,
        resolvable=resolvable,
    )
    question = (context.research_question or "").lower()
    is_prediction = any(
        token in question
        for token in ("prediction", "predict", "auroc", "brier", "calibration")
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
    if _context_target_outcome_is_binary_like(context):
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


_METHOD_CITATION_REPAIR_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(
            r"\b(primary\s+(?:predictor|exposure)|exposure|predictor|"
            r"derived\s+from|variable[s]?)\b",
            re.I,
        ),
        (
            "01_define_cohort_and_derive",
            "define_cohort",
            "derive",
            "exposure",
            "clinical_semantics_resolution",
            "research_context",
            "00_probe",
        ),
    ),
    (
        re.compile(
            r"\b(primary\s+association|adjusted\s+association|"
            r"association\b.{0,100}\badjust(?:ment|ed)?\b|"
            r"logistic\s+regression|cox\s+regression|multivariable|"
            r"model\s+family|model(?:ed|ling|ing)?\b|regression)\b",
            re.I,
        ),
        (
            "04_primary_adjusted_association_model",
            "primary_association",
            "adjusted_association",
            "model",
            "00_probe",
        ),
    ),
    (
        re.compile(
            r"\b(sensitivity|robust(?:ness)?|alternative specification)\b", re.I
        ),
        (
            "05_sensitivity_comparison",
            "robustness_panel",
            "sensitivity",
        ),
    ),
    (
        re.compile(
            r"\b(missingness|data quality|imput(?:e|ation)|measurement)\b", re.I
        ),
        (
            "03_missingness_and_data_quality_audit",
            "missingness",
            "data_quality",
        ),
    ),
    (
        re.compile(
            r"\b(cohort|inclusion|exclusion|adult|stay-level|source table)\b", re.I
        ),
        (
            "01_define_cohort_and_derive",
            "table_one",
            "cohort",
            "00_probe",
            "research_context",
        ),
    ),
)


def _sentence_has_evidence_placeholder(sentence: str) -> bool:
    return "{evidence:" in sentence or "{{evidence:" in sentence


# Results-style conclusion markers. The methods-citation repair must not
# touch sentences that *claim findings* (performance, consistency,
# associations, effect conclusions) — those are the evidence filter's
# jurisdiction. Appending a Methods citation to an unsupported conclusion
# would launder it past the fail-closed result-sentence filter, and a
# warning-severity binding then blocks the manuscript text gate anyway.
_CONCLUSION_CLAIM_RE = re.compile(
    r"\b(indicat(?:e|es|ed|ing)|suggest(?:s|ed|ing)?|"
    r"demonstrat(?:e|es|ed|ing)|show(?:s|ed|ing)?|reveal(?:s|ed|ing)?|"
    r"consistent|robustly|we\s+(?:found|observed)|"
    r"was\s+associated|were\s+associated|performance\s+was|performed\s+well)\b",
    re.I,
)
# A sentence that REPORTS a numeric result (an effect estimate, CI, percentage,
# or p-value) must not be laundered past the fail-closed result-sentence filter
# with a Methods-step citation. This is intentionally narrow: it keys on
# reported *values* (a decimal, a percentage, a CI/p pattern), NOT on any bare
# digit. The broader `_looks_result_like_sentence` fires on any "\d", so it
# false-positives on numbered clinical concepts (Sepsis-3, SOFA-2, KDIGO stage
# 3, sep3_sofa2) that saturate ICU Methods prose — those are ordinary Methods
# sentences and must still receive their infrastructure citations.
_REPORTS_NUMERIC_RESULT_RE = re.compile(
    r"(\d+\.\d+"  # a decimal value (effect estimate, AUROC, median, ...)
    r"|\d+(?:\.\d+)?\s*%"  # a percentage
    r"|\bp\s*[<=>]\s*0?\.?\d"  # a p-value
    r"|95\s*%\s*ci"  # an explicit 95% CI
    r"|\bci\b\s*[:=]?\s*\d)",  # a CI reported with a value
    re.I,
)


def _append_evidence_citation(sentence: str, evidence_id: str) -> str:
    citation = f" {{evidence:{evidence_id}}}"
    match = re.search(r"([.!?。！？])(\s*)$", sentence)
    if match:
        return (
            sentence[: match.start(1)].rstrip()
            + citation
            + match.group(1)
            + match.group(2)
        )
    return sentence.rstrip() + citation


def _apply_writer_evidence_repair_decisions(
    scaffold: str,
    *,
    missing_sentences: Sequence[str],
    decisions: Sequence[Mapping[str, object]],
    allowed_claim_refs: Sequence[str] = (),
) -> tuple[str, List[Dict[str, object]]]:
    """Apply a validated writer citation/drop/claim decision without rewriting.

    The LLM chooses only among registered citations, one exact registered host
    claim, and deletion. This host function preserves every cited sentence
    byte-for-byte apart from appending the selected evidence placeholders. A
    claim decision replaces the whole sentence with one exact host-issued
    token; no model-authored direction, population, number, or interpretation
    survives that replacement.
    """

    sentences = [str(sentence).strip() for sentence in missing_sentences]
    allowed_claims = {
        str(claim_ref).strip()
        for claim_ref in allowed_claim_refs
        if str(claim_ref).strip()
    }
    if len(decisions) != len(sentences):
        raise ValueError("writer evidence repair must decide every missing sentence")
    rewritten = scaffold
    applied: List[Dict[str, object]] = []
    seen: set[int] = set()
    for decision in decisions:
        index = decision.get("index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(sentences)
            or index in seen
        ):
            raise ValueError("writer evidence repair index is invalid or duplicated")
        target = sentences[index]
        if target not in rewritten:
            raise ValueError(
                "writer evidence repair target is absent from the current scaffold"
            )
        action = str(decision.get("action") or "").strip().lower()
        raw_ids = decision.get("evidence_ids", [])
        if not isinstance(raw_ids, (list, tuple)):
            raise ValueError("writer evidence repair evidence_ids must be a sequence")
        evidence_ids = [
            str(evidence_id).strip()
            for evidence_id in raw_ids
            if str(evidence_id).strip()
        ]
        if action == "cite":
            if not evidence_ids:
                raise ValueError("cite decision requires at least one evidence id")
            replacement = target
            for evidence_id in evidence_ids:
                replacement = _append_evidence_citation(replacement, evidence_id)
        elif action == "claim":
            if evidence_ids:
                raise ValueError("claim decision cannot include evidence ids")
            claim_ref = str(decision.get("claim_ref") or "").strip()
            if claim_ref not in allowed_claims:
                raise ValueError("claim decision requires an allowed claim_ref")
            token = "{claim:" + claim_ref + "}"
            target_offset = rewritten.find(target)
            line_start = rewritten.rfind("\n", 0, target_offset) + 1
            before_target = rewritten[line_start:target_offset]
            if before_target.lstrip().startswith("#"):
                # Scientific findings do not belong in a manuscript title or
                # heading.  A model-selected claim is safely dropped here; the
                # unchanged strict gate still validates the remaining draft.
                replacement = ""
                action = "drop"
            else:
                labelled = re.match(
                    r"^(?P<label>\*\*[^*\n]{1,80}:\*\*)\s+.+$",
                    target,
                )
                replacement = (
                    f"{labelled.group('label')}\n\n{token}"
                    if labelled is not None
                    else token
                )
        elif action == "drop":
            if evidence_ids:
                raise ValueError("drop decision cannot include evidence ids")
            replacement = ""
        else:
            raise ValueError("writer evidence repair action must be cite, claim, or drop")
        rewritten = rewritten.replace(target, replacement, 1)
        seen.add(index)
        applied.append(
            {
                "index": index,
                "action": action,
                "evidence_ids": evidence_ids,
                "sentence": target[:500],
                **(
                    {"claim_ref": str(decision.get("claim_ref") or "").strip()}
                    if str(decision.get("claim_ref") or "").strip()
                    else {}
                ),
            }
        )
    return rewritten, sorted(applied, key=lambda item: int(item["index"]))


def _best_methods_citation(sentence: str, resolvable: set[str]) -> Optional[str]:
    for pattern, candidates in _METHOD_CITATION_REPAIR_RULES:
        if not pattern.search(sentence):
            continue
        resolved = _first_resolvable_name(resolvable, candidates)
        if resolved:
            return resolved
    return None


def _repair_common_writer_citation_omissions(
    scaffold: str,
    *,
    evidence: EvidenceStore,
    allowed_evidence_names: Optional[Sequence[str]] = None,
) -> tuple[str, List[Dict[str, str]]]:
    """Append evidence citations to common uncited Methods-style sentences.

    This is intentionally narrow. It repairs sentences that describe already
    registered analysis infrastructure — cohort/exposure derivation, model
    family, missingness/data-quality handling, or sensitivity design. It does
    not invent citations for free-form conclusions; if no matching registered
    evidence id is available, the strict evidence gate still blocks the draft.
    """
    resolvable = set(
        evidence.resolvable_names()
        if allowed_evidence_names is None
        else allowed_evidence_names
    )
    repairs: List[Dict[str, str]] = []
    out_lines: List[str] = []
    in_metadata_section = False
    for raw_line in scaffold.splitlines():
        stripped = raw_line.strip()
        if re.match(r"^#{1,6}\s+", stripped):
            in_metadata_section = bool(_MANUSCRIPT_METADATA_LINE_RE.match(stripped))
            out_lines.append(raw_line)
            continue
        if (
            not stripped
            or stripped.startswith("```")
            or _MANUSCRIPT_METADATA_LINE_RE.match(stripped)
            or in_metadata_section
        ):
            out_lines.append(raw_line)
            continue
        sentences = re.split(r"(?<=[.!?。！？])\s+", raw_line)
        changed = False
        fixed: List[str] = []
        for sentence in sentences:
            if not sentence.strip() or _sentence_has_evidence_placeholder(sentence):
                fixed.append(sentence)
                continue
            if _CONCLUSION_CLAIM_RE.search(
                sentence
            ) or _REPORTS_NUMERIC_RESULT_RE.search(sentence):
                # Leave uncited conclusions AND result-bearing claims (an effect
                # estimate, a reported statistic, etc.) to the fail-closed
                # result-sentence filter (enforce_evidence_bound_scaffold) instead
                # of laundering them past it with a Methods-step citation. A
                # result sentence with no conclusion verb — e.g. one that states
                # an odds ratio value directly — must stay under the filter's
                # jurisdiction, not be tagged to a Methods step. The numeric guard
                # keys on reported values, so a Methods sentence that merely names
                # a numbered concept (Sepsis-3, SOFA-2) is still repaired.
                fixed.append(sentence)
                continue
            evidence_id = _best_methods_citation(sentence, resolvable)
            if not evidence_id:
                fixed.append(sentence)
                continue
            repaired = _append_evidence_citation(sentence, evidence_id)
            repairs.append(
                {
                    "evidence_id": evidence_id,
                    "sentence": sentence.strip()[:500],
                }
            )
            fixed.append(repaired)
            changed = True
        out_lines.append(
            " ".join(part.strip() for part in fixed if part.strip())
            if changed
            else raw_line
        )
    return "\n".join(out_lines), repairs


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


_NUMERIC_BIND_SKIP_CONTEXTS = (
    re.compile(r"\{evidence:[^}]*\}"),
    re.compile(r"\[\^[A-Za-z0-9_]+\]"),
    # Skip only Markdown syntax and a conventional outline ordinal, never the
    # entire heading. Result-bearing heading text must pass the same NumericClaim
    # binding as body prose (e.g. ``## Primary outcome: 12.4%``).
    re.compile(r"(?m)^\s*#{1,6}\s+(?:\d+(?:\.\d+)*[.)]\s+)?"),
    # Common hash notation such as SHA-256 is provenance metadata, not a
    # manuscript number. Skip it so the numeric binder does not flag the
    # hash width as an untraced result value.
    re.compile(r"(?i)\bsha[- ]?256\b"),
    # D1 (pilot 20260515 fix): after sentence-level evidence binding,
    # every {evidence:foo} becomes ``[label](evidence/foo.json
    # "sha256=DEADBEEF")``. The sha256 prefix matches the numeric
    # regex's exponent branch (``273e4341`` reads as ``273 * 10^4341``).
    # Skip the whole Markdown link target — href + title attribute —
    # so binder never sees those characters.
    re.compile(r"\(evidence/[^)]*\)"),
)

_SEMANTIC_HINTS = (
    (
        (re.compile(r"\babsolute\s+risk\b", re.IGNORECASE),),
        (re.compile(r"outcome_absolute_risks", re.IGNORECASE),),
    ),
    (
        (
            re.compile(r"\bodds\s+ratio\b", re.IGNORECASE),
            re.compile(r"\baOR\b"),
            re.compile(r"\bOR\b"),
        ),
        (
            re.compile(r"odds[_:. -]*ratio", re.IGNORECASE),
            re.compile(r"primary[_:. -]*or", re.IGNORECASE),
            re.compile(r"adjusted[_:. -]*or", re.IGNORECASE),
            re.compile(r"(^|[_:. -])or($|[_:. -])", re.IGNORECASE),
        ),
    ),
    (
        (
            re.compile(r"\bhazard\s+ratio\b", re.IGNORECASE),
            re.compile(r"\bHR\b"),
        ),
        (
            re.compile(r"hazard[_:. -]*ratio", re.IGNORECASE),
            re.compile(r"primary[_:. -]*hr", re.IGNORECASE),
            re.compile(r"adjusted[_:. -]*hr", re.IGNORECASE),
            re.compile(r"(^|[_:. -])hr($|[_:. -])", re.IGNORECASE),
        ),
    ),
    (
        (
            re.compile(r"\baverage\s+treatment\s+effect\b", re.IGNORECASE),
            re.compile(r"\bATE\b"),
            re.compile(r"\brisk\s+difference\b", re.IGNORECASE),
            re.compile(r"\brisk\s+ratio\b", re.IGNORECASE),
            re.compile(r"\btreatment\s+effect\b", re.IGNORECASE),
        ),
        (
            re.compile(r"average[_:. -]*treatment[_:. -]*effect", re.IGNORECASE),
            re.compile(r"(^|[_:. -])ate($|[_:. -])", re.IGNORECASE),
            re.compile(r"risk[_:. -]*difference", re.IGNORECASE),
            re.compile(r"risk[_:. -]*ratio", re.IGNORECASE),
            re.compile(r"treatment[_:. -]*effect", re.IGNORECASE),
        ),
    ),
    (
        (
            re.compile(r"\bcoefficient\b", re.IGNORECASE),
            re.compile(r"\bcoef\b", re.IGNORECASE),
            re.compile(r"\bbeta\b", re.IGNORECASE),
            re.compile(r"\bmean\s+difference\b", re.IGNORECASE),
        ),
        (
            re.compile(r"coefficient", re.IGNORECASE),
            re.compile(r"(^|[_:. -])coef($|[_:. -])", re.IGNORECASE),
            re.compile(r"(^|[_:. -])beta($|[_:. -])", re.IGNORECASE),
            re.compile(r"mean[_:. -]*difference", re.IGNORECASE),
        ),
    ),
    (
        (
            re.compile(r"\blength\s+of\s+stay\b", re.IGNORECASE),
            re.compile(r"\bLOS\b"),
        ),
        (
            re.compile(r"length[_:. -]*of[_:. -]*stay", re.IGNORECASE),
            re.compile(r"(^|[_:. -])los($|[_:. -])", re.IGNORECASE),
            re.compile(r"median[_:. -]*los", re.IGNORECASE),
            re.compile(r"mean[_:. -]*los", re.IGNORECASE),
        ),
    ),
    (
        (
            re.compile(r"\bmortality(?:\s+rate)?\b", re.IGNORECASE),
            re.compile(r"死亡率"),
        ),
        (re.compile(r"mortality", re.IGNORECASE),),
    ),
    (
        (
            re.compile(r"\bAUROC\b"),
            re.compile(r"\bAUC\b"),
        ),
        (
            re.compile(r"auroc", re.IGNORECASE),
            re.compile(r"(^|[_:. -])auc($|[_:. -])", re.IGNORECASE),
        ),
    ),
    (
        (
            re.compile(r"\bCI\b"),
            re.compile(r"\bconfidence\s+interval\b", re.IGNORECASE),
        ),
        (
            re.compile(r"ci[_:. -]*(low|lower|high|higher|upper)", re.IGNORECASE),
            re.compile(r"(low|lower|high|higher|upper)[_:. -]*ci", re.IGNORECASE),
        ),
    ),
    (
        (
            re.compile(r"\bp\s*-?\s*value\b", re.IGNORECASE),
            re.compile(r"\bp\s*=", re.IGNORECASE),
        ),
        (re.compile(r"p[_:. -]*value", re.IGNORECASE),),
    ),
    (
        (re.compile(r"\bBrier\b", re.IGNORECASE),),
        (re.compile(r"brier", re.IGNORECASE),),
    ),
    (
        (
            re.compile(r"\b[nN]\s*="),
            re.compile(r"\b(?:patients|stays|cases)\b", re.IGNORECASE),
        ),
        (
            re.compile(r"(^|[_:. -])n($|[_:. -])", re.IGNORECASE),
            re.compile(r"count", re.IGNORECASE),
            re.compile(r"sample[_:. -]*size", re.IGNORECASE),
            re.compile(r"n[_:. -]*total", re.IGNORECASE),
        ),
    ),
)


def _spans_to_skip(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for pat in _NUMERIC_BIND_SKIP_CONTEXTS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    return spans


def _position_is_inside(pos: int, spans: Sequence[Tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


def _display_decimal_places(value_str: str) -> int:
    text = (value_str or "").strip().rstrip("%").replace(",", "")
    if not text or "e" in text.lower() or "." not in text:
        return 0
    frac = text.split(".", 1)[1]
    frac = re.sub(r"[^0-9].*$", "", frac)
    return len(frac)


def _parsed_numeric_literal(value_str: str) -> tuple[float, bool] | None:
    raw = value_str.strip()
    has_percent = raw.endswith("%")
    stripped = raw.rstrip("%").replace(",", "")
    try:
        return float(stripped), has_percent
    except ValueError:
        return None


def _lookup_literal_for_numeric_match(
    text: str,
    *,
    match_end: int,
    value: str,
) -> str:
    """Return the literal used for NumericClaim lookup.

    Manuscripts often render percentages with a thin space or ordinary space
    before the percent sign ("9.4 %"). The numeric regex intentionally matches
    only the number in that form, but the claim registry stores prevalence as a
    proportion (0.094). Preserve the manuscript text while letting the matcher
    use percent semantics whenever the next non-space character is "%".
    """

    trailer = text[match_end : min(len(text), match_end + 8)]
    return f"{value}%" if re.match(r"\s*%", trailer) else value


def _claim_numeric_distance(
    claim: NumericClaim,
    value_str: str,
    *,
    tolerance: Optional[float] = None,
) -> Optional[float]:
    parsed = _parsed_numeric_literal(value_str)
    if parsed is None:
        return None
    canonical, has_percent = parsed
    raw = value_str.strip()
    candidates = [claim.canonical]
    if has_percent:
        # Numeric summaries legitimately use both fraction fields (0.36)
        # and already-scaled percentage fields (36.0). Keep both candidates;
        # the evidence-aware selector resolves any collision from the cited
        # step or evidence record rather than guessing the source unit.
        candidates.append(claim.canonical * 100.0)
    if claim.value == raw or canonical in candidates:
        return 0.0
    display_places = _display_decimal_places(value_str)
    display_abs_tol = 0.0
    if display_places > 0:
        display_abs_tol = 0.5 * (10 ** (-display_places))
    elif has_percent:
        display_abs_tol = 0.5
    window = tolerance if tolerance is not None else claim.tolerance
    distances: List[float] = []
    for candidate in candidates:
        if abs(candidate) > 1e-9:
            rel = abs(candidate - canonical) / abs(candidate)
        else:
            rel = 0.0 if abs(canonical) <= 1e-12 else float("inf")
        abs_window = max(
            display_abs_tol,
            window * max(abs(candidate), abs(canonical)),
        )
        if rel <= window or abs(candidate - canonical) <= abs_window:
            distances.append(abs(candidate - canonical) / max(abs(candidate), 1e-9))
    return min(distances) if distances else None


def _candidate_claims_for_value(
    evidence: EvidenceStore,
    value_str: str,
    *,
    authoritative_claims: Optional[Sequence[NumericClaim]] = None,
) -> List[tuple[NumericClaim, float]]:
    candidates: List[tuple[NumericClaim, float]] = []
    claims = (
        evidence.numeric_claims()
        if authoritative_claims is None
        else authoritative_claims
    )
    for claim in claims:
        if claim.source_field == "__easyicu_numeric_claim_overflow__":
            continue
        distance = _claim_numeric_distance(claim, value_str)
        if distance is not None:
            candidates.append((claim, distance))
    return candidates


def _is_bibliographic_year_context(
    text: str,
    *,
    start: int,
    end: int,
    value: str,
) -> bool:
    """Return true for citation years that should not require NumericClaim.

    Evidence binding tracks study results, not bibliography metadata. Keep the
    check conservative so ordinary result years still surface as UNTRACED.
    """

    raw = (value or "").strip()
    if not re.fullmatch(r"(?:19|20)\d{2}", raw):
        return False
    year = int(raw)
    if year < 1900 or year > 2099:
        return False
    left = text[max(0, start - 80) : start]
    right = text[end : min(len(text), end + 80)]
    window = left + raw + right
    if re.search(r"\bet\s+al\.?,?\s*$", left, re.IGNORECASE):
        return True
    if re.search(r"\b[A-Z][A-Za-z'’-]+(?:\s+and\s+[A-Z][A-Za-z'’-]+)?\s*,\s*$", left):
        if re.search(r"^\s*(?:[);,\]]|and\b|;)", right):
            return True
    if re.search(
        r"\([^\)]*(?:et\s+al\.?|[A-Z][A-Za-z'’-]+(?:\s+and\s+[A-Z][A-Za-z'’-]+)?\s*,)\s*"
        + raw,
        window,
        re.IGNORECASE,
    ):
        return True
    return False


def _semantic_hint_score(context: str, claim: NumericClaim) -> int:
    source = claim.source_field
    score = 0
    for context_patterns, source_patterns in _SEMANTIC_HINTS:
        if not any(pattern.search(context) for pattern in context_patterns):
            continue
        if any(pattern.search(source) for pattern in source_patterns):
            score += 1
    score += _contextual_source_score(context, claim)
    return score


def _contextual_source_score(context: str, claim: NumericClaim) -> int:
    text = (context or "").lower()
    source = (claim.source_field or "").lower()
    step = (claim.step_id or "").lower()
    evidence_id = (claim.evidence_id or "").lower()
    score = 0

    if step and step in text:
        score += 20
    if evidence_id and evidence_id in text:
        score += 15

    if re.search(r"\bcomplete[-\s]?cases?\b", text):
        if "complete_case_flow.n_complete_case" in source:
            score += 8
        elif "n_final_model" in source:
            score += 5
        elif re.search(r"(?:primary_result\.n|modeled_n|primary_n)$", source):
            score += 3

    if "primary adjusted" in text or "primary specification" in text:
        if step.startswith("04_primary_adjusted"):
            score += 4
        if source.startswith("primary_or"):
            score += 3
        if "primary_result" in source:
            score += 1

    if "analysis cohort" in text or "analytic cohort" in text:
        if any(
            token in source
            for token in (
                "n_analysis_cohort",
                "analysis_cohort_n",
                "included_n",
                "n_final_model",
            )
        ):
            score += 5

    if re.search(r"\bsepsis[-\s]?3\b", text):
        if any(
            token in source
            for token in ("n_sepsis3", "sepsis3_positive", "sepsis3.events")
        ):
            score += 5
        if "sepsis3_prevalence" in source or "prevalence.sepsis3" in source:
            score += 4

    if re.search(r"\b(?:death|mortality)\b", text):
        if any(
            token in source
            for token in ("n_deaths", "death_positive", "death_n", "events")
        ):
            score += 5
        if "death_rate" in source or "death_prevalence" in source:
            score += 4

    if "risk ratio" in text:
        if "risk_ratio" in source:
            score += 6
    if "risk difference" in text:
        if "risk_difference" in source:
            score += 6

    if re.search(r"\b(?:confidence\s+interval|ci)\b", text):
        if re.search(r"primary_or_ci_(?:low|lower|high|upper)$", source):
            score += 6
        elif re.search(r"primary_ci_(?:low|lower|high|upper)$", source):
            score += 4
        elif re.search(r"ci_(?:low|lower|high|upper)$", source):
            score += 2
        if source.endswith("_from_summary"):
            score -= 1

    if re.search(r"\b(?:range|ranged|lower end|upper end|from)\b", text):
        if re.search(r"(?:^|[._-])range_(?:low|lower|high|upper)$", source):
            score += 7

    if re.search(r"\b(?:cohort[-\s]?restriction|los|length\s+of\s+stay)\b", text):
        if "worst_cohort_point_estimate" in source:
            score += 6
        elif "point_estimate" in source and "cohort" in step:
            score += 2

    context_source_pairs = (
        (
            r"\b(?:source\s+export|per-stay\s+export|source\s+cohort|source\s+population)\b",
            "cohort.n_stays",
        ),
        (r"\b(?:stays|icu\s+stays|stay-level)\b", "cohort.n_stays"),
        (r"\bpatients\b", "cohort.n_patients"),
        (r"\blactate\b", "variable_groups.lact."),
        (r"\b(?:temperature|temp)\b", "variable_groups.temp."),
        (r"\b(?:heart[-\s]?rate|hr)\b", "variable_groups.hr."),
        (r"\b(?:mean\s+arterial\s+pressure|map)\b", "variable_groups.map."),
    )
    for context_pattern, source_prefix in context_source_pairs:
        if not re.search(context_pattern, text):
            continue
        if source.startswith(source_prefix):
            score += 5
        elif (
            source_prefix in {"cohort.n_stays", "cohort.n_patients"}
            and source == "cohort.n_stays_and_patients"
        ):
            score += 5

    if "missingness" in text and ".missingness." in source:
        score += 3
    return score


def _same_numeric_fact(candidates: Sequence[tuple[NumericClaim, float]]) -> bool:
    if not candidates:
        return False
    first = candidates[0][0]
    for claim, _ in candidates[1:]:
        if claim.step_id != first.step_id:
            return False
        if abs(claim.canonical - first.canonical) > max(
            claim.tolerance, first.tolerance
        ):
            return False
    return True


def _source_field_tiebreak_score(context: str, claim: NumericClaim) -> int:
    text = (context or "").lower()
    source = (claim.source_field or "").lower()
    score = 0
    universe_context = bool(
        re.search(
            r"\b(?:universe|source\s+(?:export|population)|before\s+analysis)\b",
            text,
        )
    )
    analysis_count_context = bool(
        re.search(
            r"\b(?:analysis|analy[sz]ed|analytic|modeled|modelled|validated|"
            r"final|primary|included|retained|eligible)\b",
            text,
        )
    )
    if analysis_count_context:
        for token in (
            "n_final_model",
            "modeled_n",
            "primary_result.n",
            "primary_n",
            "n_complete_case",
            "n_analysis_cohort",
            "analysis_cohort_n",
            "cohort_definition.analysis_cohort_n",
            "attrition.n_analysis_cohort",
        ):
            if token in source:
                score += 6
                break
        if re.search(r"(?:^|[._-])n_universe$", source) and not universe_context:
            score -= 6
        if any(token in source for token in ("raw", "candidate", "source_population")):
            score -= 2
    if re.search(r"\b(?:input|source)\s+rows?\b", text):
        if source.endswith("n_input_rows"):
            score += 9
        elif source.endswith("n_at_start_rows"):
            score += 6
    if re.search(
        r"\b(?:final\s+cohort|cohort\s+(?:comprised|included|consisted)|"
        r"retained\s+as|final\s+rows?)\b",
        text,
    ):
        if source.endswith("n_final_rows"):
            score += 9
        elif source.endswith("n_remaining_rows"):
            score += 6
        elif source.endswith(("cohort_n", "cohort_rows", "row_count")):
            score += 4
    if re.search(
        r"\b(?:measurement[-\s]?provenance|paired\s+values?|"
        r"discordan(?:ce|t)|provenance\s+checks?)\b",
        text,
    ):
        if source.endswith("comparison_n"):
            score += 8
    if universe_context and re.search(r"(?:^|[._-])n_universe$", source):
        score += 4
    if "analysis cohort" in text or "analytic cohort" in text:
        for token in (
            "n_analysis_cohort",
            "analysis_cohort_n",
            "included_n",
            "n_final_model",
        ):
            if token in source:
                score += 5
                break
    if re.search(r"\b(?:stays|patients|cohort|denominator)\b", text):
        if re.search(r"(?:^|[._-])(?:n|count|included_n|n_total|n_universe)$", source):
            score += 2
    if re.search(r"\bsepsis[-\s]?3\b", text):
        if any(
            token in source
            for token in ("n_sepsis3", "sepsis3_positive", "sepsis3.events")
        ):
            score += 5
    if re.search(r"\b(?:death|mortality)\b", text):
        if any(
            token in source
            for token in ("n_deaths", "death_positive", "death_n", "events")
        ):
            score += 5
    if "prevalence" in text and "prevalence" in source:
        score += 3
    if "missingness" in source or "coercion" in source or "overlap" in source:
        score -= 3
    if "[" in source:
        score -= 2
    return score


def _step_order_key(step_id: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)", step_id or "")
    numeric = int(match.group(1)) if match else -1
    return (numeric, step_id or "")


def _claim_identity(claim: NumericClaim) -> str:
    return f"{claim.step_id}:{claim.source_field}@{claim.evidence_id}"


_CITED_EVIDENCE_RE = re.compile(r"\{evidence:(?P<id>[^}]+)\}")

#: The citation form the writer actually emits, once placeholders have been
#: rendered: ``[label](evidence/<evidence_id>__<filename>)``. The ``__`` join
#: is the EvidenceStore's own convention -- see
#: ``EvidenceStore._target_path``: ``base / f"{safe_id}__{safe_filename}"``.
#: The tail after the filename is ``[^)\n]*`` and not ``[^)\s]*`` because the
#: writer appends a title -- ``... "sha256=c962ff2e")`` -- so a space-stopping
#: tail matches nothing at all on a real manuscript.
_RENDERED_EVIDENCE_LINK_RE = re.compile(r"\]\(evidence/(?P<id>[^)\s/]+?)__[^)\n]*\)")


def _cited_evidence_ids(context: str) -> frozenset[str]:
    """Return the evidence ids this sentence explicitly cites.

    Both forms are read, because the placeholder form is not what survives to
    this point. Measured 2026-08-01 over all 115 recorded bound manuscripts:
    ``{evidence:<id>}`` appears **0** times and the rendered link form appears
    **541** times. So the caller's "the sentence names its source, restrict the
    candidates to it" rule -- and with it the only thing that could tell one
    step's estimate from another's when both registered the same number -- has
    never once fired on a real manuscript.

    canary37 is what that costs. Its Results sentence cited the primary model's
    own step summary, one line's worth of characters away from the value, and
    the primary estimate still went out as ambiguous across eleven candidate
    fields -- blocking a manuscript in which every other number had bound.
    """

    cited = {
        match.group("id").strip()
        for match in _CITED_EVIDENCE_RE.finditer(context or "")
    }
    cited.update(
        match.group("id").strip()
        for match in _RENDERED_EVIDENCE_LINK_RE.finditer(context or "")
    )
    return frozenset(value for value in cited if value)


def _evidence_lineage(evidence: EvidenceStore) -> Dict[str, frozenset[str]]:
    """Map each evidence id to itself plus its declared ancestor ids.

    A sentence citing a step's summary legitimately prints values the summary
    derived from that step's own table, so lineage — not identity — is the
    right scope. Lineage is *declared* at registration time, which is why it
    can be trusted here: an unrelated step cannot add itself to it after the
    fact.
    """

    direct: Dict[str, set[str]] = {}
    for record in evidence.records():
        parents = set(record.inputs or ())
        if record.script_evidence_id:
            parents.add(record.script_evidence_id)
        direct[record.evidence_id] = parents

    resolved: Dict[str, frozenset[str]] = {}

    def _walk(evidence_id: str, seen: frozenset[str]) -> frozenset[str]:
        if evidence_id in resolved:
            return resolved[evidence_id]
        if evidence_id in seen:  # defensive: a registration cycle
            return frozenset({evidence_id})
        ancestors = {evidence_id}
        for parent in direct.get(evidence_id, ()):  # noqa: SIM118
            ancestors |= _walk(parent, seen | {evidence_id})
        answer = frozenset(ancestors)
        resolved[evidence_id] = answer
        return answer

    for evidence_id in direct:
        _walk(evidence_id, frozenset())
    # A sentence may cite a registered alias rather than the canonical id.
    # Without the alias keys the citation would look unresolvable and scoping
    # would silently fall back to match-any.
    for alias, evidence_id in getattr(evidence, "_aliases", {}).items():
        if alias not in resolved and evidence_id in resolved:
            resolved[str(alias)] = resolved[evidence_id]
    return resolved


#: One citation in either form, used to walk the run that precedes a
#: sentence's own words.
_LEADING_CITATION_RE = re.compile(
    r"\s*(?:\{evidence:[^}\n]+\}|\[[^\]\n]*\]\(evidence/[^)\n]*\))"
)


def _sentence_cites_within_its_own_prose(context: str) -> bool:
    """Whether this sentence cites anything after its own words begin.

    The sentence window keeps the citation run on both sides of the prose,
    because the writer emits citations before a sentence as readily as after
    and position alone does not settle ownership.  For NARROWING that is right:
    an extra citation only costs recall.  For REFUSING it is not: a citation
    that merely terminates the previous sentence is not this sentence naming a
    source, and treating it as one blocks a number the sentence never
    misattributed.

    So strip the leading run and ask whether any citation remains.
    """

    text = context or ""
    cursor = 0
    while True:
        match = _LEADING_CITATION_RE.match(text, cursor)
        if match is None or match.end() <= cursor:
            break
        cursor = match.end()
    return bool(_cited_evidence_ids(text[cursor:]))


def _miscitation_detail(
    candidates: Sequence[tuple[NumericClaim, float]],
    *,
    context: str,
    lineage: Optional[Mapping[str, frozenset[str]]],
) -> Optional[Dict[str, List[str]]]:
    """Name the miscitation when a sentence cites a step that owns no such value.

    Three different failures currently leave the same mark on a manuscript:
    nobody registered this number, several registered claims tie for it, and
    the sentence cited a step that did not register it. Only the last is a
    writer error, and only the last can name its fix -- yet all three surfaced
    as "Manuscript numeric claims disagree with registered step_summary
    values", which names no sentence, no citation and no owner.

    MEASURED (e1 sepsis 10/10 and e3 KDIGO 11/11, both manuscripts written and
    both blocked by their own numeric audit): every remaining marker was the
    cohort size 94,458. The sentence

        The operational denominator comprised 94,458 ICU stays represented in
        the supplied cohort definition {evidence:00_probe}.

    cites its own source, that source resolves, and restricting the 45
    candidates to its lineage leaves zero -- ``00_probe``'s two evidence files
    contain no 94458 at all. The binder was right to refuse. The writer
    attributed the cohort denominator to a step that has nothing to do with it,
    and writer.txt already carries the rule with this very number as its worked
    example, so the gap is not instruction: the writer gets no repair, and the
    failure never told anyone which step it should have cited instead.

    Returns the cited ids and the ids that DO own the value, or ``None`` when
    this is not a miscitation.
    """

    if not candidates or not lineage:
        return None
    # No separate empty-`cited` guard: an empty set resolves to an empty
    # `resolvable` on the next line and returns there. A mutation deleting the
    # extra check changed nothing, which is what a redundant line looks like.
    cited = _cited_evidence_ids(context)
    resolvable = sorted(item for item in cited if item in lineage)
    if not resolvable:
        # An unresolved placeholder is its own failure and never scoped
        # anything, so it cannot have caused this refusal.
        return None
    if _restrict_to_cited_evidence(candidates, cited=cited, lineage=lineage):
        return None
    owners = sorted(
        {
            str(claim.step_id or claim.evidence_id or "")
            for claim, _ in candidates
            if (claim.step_id or claim.evidence_id)
        }
    )
    return {"cited": resolvable, "owned_by": owners}


def _restrict_to_cited_evidence(
    candidates: Sequence[tuple[NumericClaim, float]],
    *,
    cited: frozenset[str],
    lineage: Mapping[str, frozenset[str]],
) -> list[tuple[NumericClaim, float]]:
    """Keep only claims the sentence's own citation can vouch for.

    Without this the binder answers "does this number exist anywhere in the
    run?" A sentence reading ``the primary model achieved an AUROC of 0.85
    {evidence:primary_model}`` would happily bind 0.85 to the *sensitivity*
    step that actually produced it, and the numeric auditor — which also
    matched across all steps — would agree. The manuscript then carries a real,
    registered, correctly-hashed number attached to the wrong model.
    """

    # Only citations that resolve to a registered record can scope anything.
    # An unresolved placeholder is a different failure with its own finding;
    # treating it as a scope would untrace every number in the sentence and
    # bury the real cause.
    resolvable = frozenset(item for item in cited if item in lineage)
    if not resolvable:
        return list(candidates)
    # Union of each cited record with its declared ancestors. The direction
    # matters: citing a derived summary vouches for the table it was computed
    # from, but citing that table does not vouch for every later record that
    # happened to consume it.
    in_scope: frozenset[str] = frozenset().union(
        *(lineage.get(item, frozenset({item})) for item in resolvable)
    )
    return [
        (claim, distance)
        for claim, distance in candidates
        if claim.evidence_id in in_scope
    ]


def _select_numeric_claim(
    *,
    candidates: Sequence[tuple[NumericClaim, float]],
    context: str,
    previous_step_id: Optional[str],
    lineage: Optional[Mapping[str, frozenset[str]]] = None,
    prose_effect_scale: Optional[NumericEffectScale] = None,
    prose_estimand: Optional[NumericEstimand] = None,
) -> tuple[Optional[NumericClaim], bool]:
    """Return ``(claim, ambiguous)`` for one manuscript numeric literal."""

    candidates = [
        (claim, distance)
        for claim, distance in candidates
        if not (
            prose_effect_scale is not None
            and claim.effect_scale is not prose_effect_scale
        )
        and not (
            prose_estimand is not None
            and claim.estimand is not prose_estimand
        )
    ]
    if not candidates:
        return None, False

    cited = _cited_evidence_ids(context)
    if cited and lineage:
        scoped = _restrict_to_cited_evidence(candidates, cited=cited, lineage=lineage)
        if not scoped:
            # The sentence names its source and no candidate belongs to it.
            # Binding the value anyway would attach a foreign step's number to
            # this claim, so it stays untraced.
            #
            # Unless it named nothing. The window deliberately keeps the run of
            # citations on BOTH sides, because the writer puts them before a
            # sentence as readily as after and position alone does not identify
            # ownership. The comment defending that called an extra citation a
            # recall cost that "cannot by itself produce a wrong bind" -- true,
            # and beside the point: at this gate a lost bind is not a cost, it
            # is a blocked manuscript.
            #
            # MEASURED (e3 KDIGO, 11/11 steps, manuscript written): the Results
            # sentence carrying the primary estimate is, in the pre-binding
            # text, exactly
            #
            #   {evidence:03_stage_stratified_mortality_distribution} In the
            #   adjusted primary analysis, ... (odds ratio, 6.48; 95% CI,
            #   6.02-6.97).
            #
            # -- no citation of its own, only the previous sentence's trailing
            # one. All three of its numbers scoped to step 03's lineage, which
            # owns none of them, and went out ambiguous. The manuscript was
            # blocked on a premise that was false: this sentence never named a
            # source. So refuse only when the sentence actually cited something
            # itself; a purely inherited citation may narrow, never veto.
            if _sentence_cites_within_its_own_prose(context):
                return None, True
        else:
            candidates = scoped

    if len(candidates) == 1:
        return candidates[0][0], False

    remaining = list(candidates)
    semantic_scores = [
        (_semantic_hint_score(context, claim), claim, distance)
        for claim, distance in remaining
    ]
    best_semantic = max(score for score, _, _ in semantic_scores)
    if best_semantic > 0:
        remaining = [
            (claim, distance)
            for score, claim, distance in semantic_scores
            if score == best_semantic
        ]
        if len(remaining) == 1:
            return remaining[0][0], False

    best_distance = min(distance for _, distance in remaining)
    remaining = [
        (claim, distance)
        for claim, distance in remaining
        if abs(distance - best_distance) <= 1e-12
    ]
    if len(remaining) == 1:
        return remaining[0][0], False

    if previous_step_id:
        same_step = [
            (claim, distance)
            for claim, distance in remaining
            if claim.step_id == previous_step_id
        ]
        if same_step:
            remaining = same_step
            if len(remaining) == 1:
                return remaining[0][0], False

    best_step = max(_step_order_key(claim.step_id) for claim, _ in remaining)
    remaining = [
        (claim, distance)
        for claim, distance in remaining
        if _step_order_key(claim.step_id) == best_step
    ]
    if len(remaining) == 1:
        return remaining[0][0], False

    if _same_numeric_fact(remaining):
        ranked = sorted(
            remaining,
            key=lambda item: (
                -_source_field_tiebreak_score(context, item[0]),
                len(item[0].source_field or ""),
                item[0].source_field or "",
            ),
        )
        distinct_fields = {claim.source_field or "" for claim, _ in remaining}
        evidence_ids = {claim.evidence_id or "" for claim, _ in remaining}
        owner_is_cited = any(
            token and token.lower() in context.lower()
            for claim, _ in remaining
            for token in (claim.step_id, claim.evidence_id)
        )
        # Collapse same-step/same-value candidates only when the pick is
        # semantically defensible: either every candidate carries the same
        # source field (true duplicate registrations), the ranked winner has a
        # positive field-token score tying it to the prose context, or the
        # sentence explicitly cites the single immutable evidence record that
        # owns all remaining same-value fields. The latter covers one step
        # summary exposing the same denominator in its cohort, audit, and
        # input-binding sections without permitting a pick across records.
        # A zero-score tie between opaque, differently named fields must
        # stay ambiguous — a lexicographic pick is not provenance.
        if (
            len(distinct_fields) == 1
            or _source_field_tiebreak_score(context, ranked[0][0]) > 0
            or (len(evidence_ids) == 1 and owner_is_cited)
        ):
            return ranked[0][0], False

    return None, True


_NUMERIC_SENTENCE_BOUNDARY_RE = re.compile(r"(?:[.!?](?=\s|$)|\n{2,})")

_EFFECT_SCALE_PHRASE_PATTERNS = {
    NumericEffectScale.ODDS_RATIO: re.compile(r"\bodds[\s-]+ratios?\b", re.I),
    NumericEffectScale.HAZARD_RATIO: re.compile(r"\bhazard[\s-]+ratios?\b", re.I),
    NumericEffectScale.RISK_RATIO: re.compile(
        r"\b(?:risk[\s-]+ratios?|relative[\s-]+risks?)\b", re.I
    ),
}
_CI_MARKER = r"(?:95\s*%\s*(?:CI|confidence\s+interval)|confidence\s+interval)"
_PLAIN_PROSE_NUMBER = r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)%?"


def _prose_effect_scale(
    text: str, *, start: int, end: int
) -> Optional[NumericEffectScale]:
    """Resolve the scale label governing one numeric mention.

    Scale is mention-local rather than sentence-global: a results sentence may
    legitimately report OR, HR, and RR together. A following label is used only
    when it is directly postfix to the number; otherwise the latest preceding
    label governs the point estimate and any CI endpoints that follow it.
    """

    abbreviation_scales = {
        "OR": NumericEffectScale.ODDS_RATIO,
        "HR": NumericEffectScale.HAZARD_RATIO,
        "RR": NumericEffectScale.RISK_RATIO,
    }
    context_start = 0
    for boundary in _NUMERIC_SENTENCE_BOUNDARY_RE.finditer(text, 0, start):
        context_start = boundary.end()
    next_boundary = _NUMERIC_SENTENCE_BOUNDARY_RE.search(text, end)
    context_end = next_boundary.start() if next_boundary is not None else len(text)
    context = text[context_start:context_end]
    local_prefix = text[max(context_start, start - 24) : start]
    local_suffix = text[end : min(context_end, end + 32)]
    if re.search(
        r"\b(?:p|SE|standard\s+error)\s*[<=>:]?\s*$",
        local_prefix,
        re.I,
    ) or re.match(
        r"\s*(?:patients?|stays?|admissions?|observations?|rows?|groups?)\b",
        local_suffix,
        re.I,
    ):
        return None
    mentions: list[tuple[int, int, NumericEffectScale]] = []
    for scale, pattern in _EFFECT_SCALE_PHRASE_PATTERNS.items():
        mentions.extend(
            (context_start + match.start(), context_start + match.end(), scale)
            for match in pattern.finditer(context)
        )
    mentions.extend(
        (
            context_start + match.start(),
            context_start + match.end(),
            abbreviation_scales[match.group(0)],
        )
        for match in re.finditer(r"\b(?:OR|HR|RR)\b", context)
    )
    if not mentions:
        return None

    following = sorted((item for item in mentions if item[0] >= end), key=lambda x: x[0])
    if following:
        label_start, _label_end, scale = following[0]
        between = text[end:label_start]
        if len(between) <= 12 and re.fullmatch(r"[\s()\[\],:=-]*", between):
            return scale
    preceding = [item for item in mentions if item[1] <= start]
    if not preceding:
        return None
    latest_end = max(item[1] for item in preceding)
    scales = {item[2] for item in preceding if item[1] == latest_end}
    return next(iter(scales)) if len(scales) == 1 else None


def _prose_numeric_estimand(text: str, *, start: int, end: int) -> Optional[NumericEstimand]:
    """Classify an explicitly labelled point estimate or ordered CI endpoint."""

    prefix = text[max(0, start - 180) : start]
    suffix = text[end : min(len(text), end + 100)]
    separator = r"(?:-|–|—|\bto\b)"
    if re.search(_CI_MARKER + r".{0,50}$", prefix, flags=re.I | re.S):
        if re.match(r"\s*" + separator + r"\s*" + _PLAIN_PROSE_NUMBER, suffix):
            return NumericEstimand.CONFIDENCE_INTERVAL_LOWER
    if re.search(
        _CI_MARKER
        + r".{0,80}?"
        + _PLAIN_PROSE_NUMBER
        + r"\s*"
        + separator
        + r"\s*$",
        prefix,
        flags=re.I | re.S,
    ):
        return NumericEstimand.CONFIDENCE_INTERVAL_UPPER
    if re.search(
        r"(?:\b(?:OR|HR|RR)\b|\b(?:odds|hazard|risk)[\s-]+ratio)"
        r"\s*(?:=|:|,|\bof\b|\bwas\b)\s*$",
        prefix,
        flags=re.I,
    ):
        return NumericEstimand.POINT_ESTIMATE
    return None


def _numeric_sentence_context(text: str, *, start: int, end: int) -> str:
    """Return the current prose sentence for evidence-aware disambiguation.

    A fixed character window can include the citation from the previous
    sentence while truncating the current sentence's trailing evidence link.
    That makes a repeated denominator appear to belong to the wrong step.
    Sentence boundaries keep the exact local citation and exclude neighbouring
    claims. The cap only limits pathological generated run-on prose.

    The window then reaches PAST the terminal period through the evidence
    links that immediately follow it, because that is where the writer puts
    them. canary37 is the recorded cost: its Results sentence ended
    ``... from 1.02 to 2.39.`` and the citation naming the owning step sat
    just after the period, so the sentence window saw no citation at all, the
    cited-evidence restriction never ran, and the primary estimate stayed
    ambiguous across eleven candidate fields -- blocking a manuscript in which
    every other number had bound. Only links are absorbed, and only while they
    are unbroken by prose, so the next sentence's words can never enter.
    """

    context_start, context_end = _numeric_sentence_bounds(
        text,
        start=start,
        end=end,
    )
    return text[context_start:context_end]


def _numeric_sentence_bounds(text: str, *, start: int, end: int) -> Tuple[int, int]:
    """Return the exact sentence span used by numeric provenance binding."""

    context_start = 0
    for match in _NUMERIC_SENTENCE_BOUNDARY_RE.finditer(text, 0, start):
        context_start = match.end()
    # The leading run of citations is KEPT, not skipped. Skipping it was a
    # symmetry argument, and canary39 refuted it on a real manuscript: the
    # writer puts a citation before a sentence as readily as after, and there
    # the value's true owner sat BEFORE while the citation after it belonged to
    # the next sentence. Position does not identify ownership, so both
    # neighbouring runs stay in scope. That is safe because the restriction
    # below only ever NARROWS the candidate set: an extra citation costs recall,
    # it cannot by itself produce a wrong bind.
    next_boundary = _NUMERIC_SENTENCE_BOUNDARY_RE.search(text, end)
    context_end = next_boundary.end() if next_boundary is not None else len(text)
    context_end = _extend_through_trailing_citations(text, context_end)
    max_chars = 1600
    context_start = max(context_start, start - max_chars)
    context_end = min(context_end, end + max_chars)
    return context_start, context_end


#: A markdown link whose target is an evidence artefact, as the writer emits
#: it: ``[label](evidence/<file> "sha256=...")``. Anchored so only an unbroken
#: run of such links is absorbed.
_TRAILING_CITATION_RE = re.compile(r"\s*\[[^\]\n]*\]\(evidence/[^)\n]*\)")


def _extend_through_trailing_citations(text: str, context_end: int) -> int:
    """Extend a sentence window over the citations written after its period.

    Nothing but evidence links is absorbed: the first thing that is not one
    stops the walk, so a following sentence's prose -- and therefore its
    claims -- can never be pulled into this sentence's context.
    """

    cursor = context_end
    while True:
        match = _TRAILING_CITATION_RE.match(text, cursor)
        if match is None or match.end() <= cursor:
            # A pattern that can match the empty string would spin here
            # forever. The one above cannot -- it requires a bracketed label --
            # but a walk that trusts a regex to advance is one edit away from
            # hanging the writer phase, and a mutation of exactly that shape
            # did hang this test suite.
            return cursor
        cursor = match.end()


def repair_miscited_numeric_citations(
    scaffold: str,
    *,
    evidence: EvidenceStore,
) -> tuple[str, List[Dict[str, str]]]:
    """Add the owning step's citation to a number cited to the wrong step.

    MEASURED, and it is the whole remaining distance on both manuscripts the
    pipeline has produced. e1 sepsis (10/10 steps) and e3 KDIGO (11/11) each
    end with exactly two unbound numbers, all four the cohort size 94,458, all
    four a sentence citing a step that registered no such value::

        The operational denominator comprised 94,458 ICU stays represented in
        the supplied cohort definition {evidence:00_probe}.

    Replacing that one citation with an owning step takes e1 to 0 markers /
    13 bound and e3 to 0 markers / 11 bound. One citation per manuscript is
    the entire gap between "written" and "numerically verified".

    ``writer.txt`` already states the rule, with this very number as its worked
    example: a sentence printing values from different steps must cite EVERY
    step that owns one of them, and citing only one blocks the manuscript. The
    writer had the rule and did not follow it, and it gets no repair pass of
    its own, so the sentence is final the moment it is written.

    This repair is deliberately ADDITIVE. The writer's own citation stays --
    it is what the prose is about -- and the owner of the number is appended
    beside it, which is exactly the two-id form the rule asks for. Nothing is
    replaced and no citation is ever removed, so a genuine attribution error
    remains visible in the text rather than being quietly rewritten.

    A citation is added only when exactly one registered claim owns the value.
    ``NumericClaim`` does not yet carry the full estimand/exposure/outcome/
    population identity needed to prove that multiple same-valued claims are
    the same fact.  Therefore any multiplicity remains unmodified and the
    strict binder still refuses it; ordering alone is never semantic evidence.
    """

    lineage = _evidence_lineage(evidence)
    if not lineage:
        return scaffold, []
    resolvable = set(evidence.resolvable_names())
    skip_spans = _spans_to_skip(scaffold)
    repairs: List[Dict[str, str]] = []
    insertions: List[tuple[int, str]] = []
    for match in _NUMERIC_IN_PROSE_RE.finditer(scaffold):
        start, end = match.start("value"), match.end("value")
        if _position_is_inside(start, skip_spans):
            continue
        value = match.group("value")
        lookup_value = _lookup_literal_for_numeric_match(
            scaffold, match_end=end, value=value
        )
        if _is_bibliographic_year_context(scaffold, start=start, end=end, value=value):
            continue
        context = _numeric_sentence_context(scaffold, start=start, end=end)
        candidates = _candidate_claims_for_value(
            evidence, lookup_value, authoritative_claims=None
        )
        claim, _ambiguous = _select_numeric_claim(
            candidates=candidates,
            context=context,
            previous_step_id=None,
            lineage=lineage,
        )
        # Only a number that actually fails to bind is repaired. Keying on the
        # miscitation alone fired on 6 sentences in e3 where 2 were blocked:
        # the other 4 bound perfectly well and would have collected a citation
        # they did not need. A repair pass that edits text it was not asked to
        # fix is one that cannot be reviewed by its own diff.
        if claim is not None:
            continue
        detail = _miscitation_detail(candidates, context=context, lineage=lineage)
        if detail is None:
            continue
        distinct_candidates = {
            (claim.step_id, claim.evidence_id, claim.source_field)
            for claim, _distance in candidates
        }
        if len(distinct_candidates) != 1:
            continue
        citable_candidates = [
            claim
            for claim, _distance in candidates
            if claim.step_id in resolvable or claim.evidence_id in resolvable
        ]
        if len(citable_candidates) != 1:
            continue
        owner = next(
            (item for item in sorted(detail["owned_by"]) if item in resolvable),
            None,
        )
        if owner is None:
            continue
        token = "{evidence:" + owner + "}"
        if token in context:
            continue
        insertions.append((end, " " + token))
        repairs.append(
            {"value": value, "cited": ",".join(detail["cited"]), "added": owner}
        )
    if not insertions:
        return scaffold, []
    repaired = scaffold
    for position, token in sorted(insertions, key=lambda item: -item[0]):
        repaired = repaired[:position] + token + repaired[position:]
    return repaired, repairs


def bind_numeric_values(
    manuscript: str,
    *,
    evidence: EvidenceStore,
    enforcement_mode: Optional[EvidenceEnforcementMode] = None,
    footnote_prefix: str = "claim",
    per_step_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[str, Dict[str, NumericClaim], List[str]]:
    """Bind every numeric value in ``manuscript`` to a registered claim.

    **Engine-agnostic provenance invariant.** This binder takes only the
    manuscript *string* plus the :class:`EvidenceStore`; it has no knowledge of
    which brain produced the text. That is deliberate and load-bearing: whether
    the manuscript came from the offline mock, an API model, or a local
    coding-agent CLI (Codex / Claude Code), every printed number must still
    trace to a registered :class:`NumericClaim` or it is treated as untraced.
    No engine — however capable — can bypass this gate. As external agents get
    stronger they also get more confidently wrong, so this value-level check is
    the part of the framework that *gains* value as the brain improves, not the
    part that gets commoditised away. Do not add an engine-specific trust path
    that lets some provider's numbers skip binding.

    Each matched value gets a Markdown footnote ``[^claim_N]`` whose
    definition is appended at the bottom of the manuscript pointing to
    the source step, source field, and owning evidence id. Numbers
    that do not match any registered claim:

    * SOFT mode → emit a comment marker ``<!-- UNTRACED:<value> -->``
      next to the number so reviewers see them, and record the value
      in the returned ``untraced`` list.
    * STRICT mode → raise :class:`EvidenceEnforcementError` after
      scanning, with the offending values in ``detail['untraced']``.

    The implementation skips numbers that are already inside
    ``{evidence:<id>}`` placeholders, existing footnote markers
    ``[^xxx]``, and conventional Markdown outline ordinals. Numeric claims in
    heading text remain subject to the same provenance gate as body prose.

    Inspired by data-to-paper's ``\\hypertarget`` / ``\\hyperlink``
    binding (NEJM AI 2024) but emits Markdown footnotes so the output
    stays readable without a LaTeX pass.
    """
    if not manuscript:
        return manuscript, {}, []
    mode = enforcement_mode or evidence.enforcement_mode
    authoritative_claims = (
        evidence.authoritative_numeric_claims(per_step_records)
        if per_step_records is not None
        else None
    )

    lineage = _evidence_lineage(evidence)
    skip_spans = _spans_to_skip(manuscript)
    binding_map: Dict[str, NumericClaim] = {}
    untraced: List[str] = []
    miscited: List[Dict[str, Any]] = []
    used_ids: Dict[str, str] = {}  # claim identity -> footnote_id
    display_values: Dict[str, str] = {}
    previous_step_id: Optional[str] = None

    def _footnote_id_for(claim: NumericClaim, *, display_value: str) -> str:
        claim_key = _claim_identity(claim)
        existing = used_ids.get(claim_key)
        if existing is not None:
            display_values.setdefault(existing, display_value)
            return existing
        idx = len(used_ids) + 1
        fid = f"{footnote_prefix}_{idx}"
        used_ids[claim_key] = fid
        binding_map[fid] = claim
        display_values[fid] = display_value
        return fid

    out_parts: List[str] = []
    cursor = 0
    for match in _NUMERIC_IN_PROSE_RE.finditer(manuscript):
        start, end = match.start("value"), match.end("value")
        if _position_is_inside(start, skip_spans):
            continue
        value = match.group("value")
        lookup_value = _lookup_literal_for_numeric_match(
            manuscript,
            match_end=end,
            value=value,
        )
        if _is_bibliographic_year_context(
            manuscript,
            start=start,
            end=end,
            value=value,
        ):
            continue
        out_parts.append(manuscript[cursor:end])
        context = _numeric_sentence_context(manuscript, start=start, end=end)
        candidates = _candidate_claims_for_value(
            evidence,
            lookup_value,
            authoritative_claims=authoritative_claims,
        )
        claim, ambiguous = _select_numeric_claim(
            candidates=candidates,
            context=context,
            previous_step_id=previous_step_id,
            lineage=lineage,
            prose_effect_scale=_prose_effect_scale(
                manuscript,
                start=start,
                end=end,
            ),
            prose_estimand=_prose_numeric_estimand(
                manuscript,
                start=start,
                end=end,
            ),
        )
        if claim is None:
            untraced.append(value)
            miscitation = _miscitation_detail(
                candidates, context=context, lineage=lineage
            )
            if miscitation is not None:
                miscited.append({"value": value, **miscitation})
            if mode is EvidenceEnforcementMode.SOFT:
                if miscitation is not None:
                    out_parts.append(
                        f" <!-- MISCITED:{value}"
                        f":cited=[{','.join(miscitation['cited'])}]"
                        f":owned_by=[{','.join(miscitation['owned_by'])}] -->"
                    )
                elif ambiguous:
                    candidate_ids = ",".join(
                        _claim_identity(candidate) for candidate, _ in candidates
                    )
                    out_parts.append(
                        f" <!-- AMBIGUOUS:{value}:candidates=[{candidate_ids}] -->"
                    )
                else:
                    out_parts.append(f" <!-- UNTRACED:{value} -->")
        else:
            fid = _footnote_id_for(claim, display_value=value)
            out_parts.append(f"[^{fid}]")
            previous_step_id = claim.step_id
        cursor = end
    out_parts.append(manuscript[cursor:])
    bound = "".join(out_parts)

    if binding_map:
        defs = ["\n"]
        for fid, claim in binding_map.items():
            fields = [
                f"value={claim.value}",
                f"step={claim.step_id}",
                f"field={claim.source_field}",
                f"evidence={claim.evidence_id}",
            ]
            if claim.effect_scale is not None:
                fields.append(f"effect_scale={claim.effect_scale.value}")
            if claim.estimand is not None:
                fields.append(f"estimand={claim.estimand.value}")
            display_value = display_values.get(fid)
            if display_value and display_value != claim.value:
                fields.extend(
                    [
                        f"display={display_value}",
                        "match=rounded_or_transformed",
                    ]
                )
            if claim.is_derived:
                fields.append(f"formula={claim.formula}")
                if claim.explanation:
                    fields.append(f"explanation={claim.explanation}")
                if claim.derived_from:
                    sources = ", ".join(
                        f"{src_step}.{src_field}"
                        for src_step, src_field in claim.derived_from
                    )
                    fields.append(f"derived_from={sources}")
            defs.append(f"[^{fid}]: " + "; ".join(fields) + "\n")
        bound = bound.rstrip() + "\n" + "".join(defs)

    if mode is EvidenceEnforcementMode.STRICT and untraced:
        raise EvidenceEnforcementError(
            f"Manuscript contains {len(untraced)} numeric value(s) not "
            f"traceable to any registered claim (STRICT mode). "
            f"Examples: {untraced[:5]}"
            + (
                # The one refusal that names its own fix. Reported beside the
                # value list so the reader is not left to guess which sentence
                # cited what.
                f" Miscited: {miscited[:3]}"
                if miscited
                else ""
            ),
            detail={"untraced": untraced, "miscited": miscited} if miscited
            else {"untraced": untraced},
        )

    return bound, binding_map, untraced


def drop_untraceable_numeric_sentences(
    manuscript: str,
    *,
    evidence: EvidenceStore,
    per_step_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Remove only sentences that the unchanged STRICT numeric gate rejects.

    Writer prose is optional; numeric authority is not.  A model can still
    calculate a plausible number or attach a real value to the wrong evidence
    owner after being told not to.  Rather than guessing a replacement or
    weakening :func:`bind_numeric_values`, test each numeric sentence against
    that same gate, remove the complete rejected sentence, record why it was
    removed, and let the caller run the full-document STRICT binder again.
    """

    if not manuscript:
        return manuscript, []
    skip_spans = _spans_to_skip(manuscript)
    rejected_by_span: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for match in _NUMERIC_IN_PROSE_RE.finditer(manuscript):
        start, end = match.start("value"), match.end("value")
        if _position_is_inside(start, skip_spans):
            continue
        value = match.group("value")
        if _is_bibliographic_year_context(
            manuscript,
            start=start,
            end=end,
            value=value,
        ):
            continue
        span = _numeric_sentence_bounds(manuscript, start=start, end=end)
        if span in rejected_by_span:
            continue
        sentence = manuscript[span[0] : span[1]]
        try:
            bind_numeric_values(
                sentence,
                evidence=evidence,
                enforcement_mode=EvidenceEnforcementMode.STRICT,
                per_step_records=per_step_records,
            )
        except EvidenceEnforcementError as exc:
            detail = dict(exc.detail or {})
            rejected_by_span[span] = {
                "sentence": sentence.strip(),
                "untraced": [str(item) for item in detail.get("untraced", [])],
                "miscited": list(detail.get("miscited", [])),
            }

    if not rejected_by_span:
        return manuscript, []
    filtered = manuscript
    for (start, end), _detail in sorted(
        rejected_by_span.items(),
        key=lambda item: item[0][0],
        reverse=True,
    ):
        filtered = filtered[:start] + filtered[end:]
    removed = [
        detail
        for _span, detail in sorted(
            rejected_by_span.items(),
            key=lambda item: item[0][0],
        )
    ]
    return filtered, removed


def enforce_writer_claim_language(
    manuscript: str,
    *,
    enforcement_mode: EvidenceEnforcementMode,
    side_findings: Sequence[SideFinding] | None = None,
) -> Tuple[str, Dict[str, List[str]]]:
    """Block post-hoc rhetoric and side-finding leakage before final binding."""

    detail: Dict[str, List[str]] = {}
    forbidden_terms = sorted(
        {
            m.group(1).lower()
            for m in _FORBIDDEN_INTERPRETIVE_RE.finditer(manuscript or "")
        }
    )
    if forbidden_terms:
        detail["forbidden_terms"] = forbidden_terms

    leaks = side_finding_leaks(manuscript, side_findings or [])
    if leaks:
        detail["side_finding_leak"] = [
            finding.title or finding.finding_id for finding in leaks
        ]

    if enforcement_mode is EvidenceEnforcementMode.STRICT and detail:
        raise EvidenceEnforcementError(
            "Manuscript contains post-hoc claim language or side-finding leakage "
            "blocked by STRICT evidence mode.",
            detail=detail,
        )

    annotated = manuscript
    if forbidden_terms:
        for term in forbidden_terms:
            marker = f"<!-- LEXICON:{term} -->"
            if marker not in annotated:
                annotated = annotated.rstrip() + f"\n{marker}\n"
    if leaks:
        annotated = annotate_side_finding_leaks(annotated, leaks)
    return annotated, detail


__all__ = [
    "_first_resolvable_name",
    "_repair_common_writer_placeholders",
    "_demote_unresolved_evidence_placeholders",
    "_remove_tbd_sentences",
    "bind_numeric_values",
    "drop_untraceable_numeric_sentences",
    "enforce_writer_claim_language",
]
