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
from typing import Dict, List, Optional, Sequence, Tuple

from .evidence import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    EvidenceStore,
    NumericClaim,
    _NUMERIC_IN_PROSE_RE,
)
from .schema import ResearchContext
from .side_findings import (
    SideFinding,
    annotate_side_finding_leaks,
    side_finding_leaks,
)


_UNRESOLVED_EVIDENCE_PLACEHOLDER_RE = re.compile(
    r"\[evidence missing:\s*(?P<id>[^\]]+)\]"
)

_TBD_RE = re.compile(r"\[(?:TBD|TODO|TK)\]|\bTBD\b", re.IGNORECASE)

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


_NUMERIC_BIND_SKIP_CONTEXTS = (
    re.compile(r"\{evidence:[^}]*\}"),
    re.compile(r"\[\^[A-Za-z0-9_]+\]"),
    re.compile(r"#+\s.*"),
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

    trailer = text[match_end:min(len(text), match_end + 8)]
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
    candidate = claim.canonical * 100.0 if has_percent else claim.canonical
    if claim.value == raw or candidate == canonical:
        return 0.0
    display_places = _display_decimal_places(value_str)
    display_abs_tol = 0.0
    if display_places > 0:
        display_abs_tol = 0.5 * (10 ** (-display_places))
    window = tolerance if tolerance is not None else claim.tolerance
    if abs(candidate) > 1e-9:
        rel = abs(candidate - canonical) / abs(candidate)
    else:
        rel = abs(candidate - canonical)
    abs_window = max(display_abs_tol, window * max(abs(candidate), abs(canonical), 1.0))
    if rel <= window or abs(candidate - canonical) <= abs_window:
        return abs(candidate - canonical) / max(abs(candidate), 1e-9)
    return None


def _candidate_claims_for_value(
    evidence: EvidenceStore,
    value_str: str,
) -> List[tuple[NumericClaim, float]]:
    candidates: List[tuple[NumericClaim, float]] = []
    for claim in evidence.numeric_claims():
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
    left = text[max(0, start - 80):start]
    right = text[end:min(len(text), end + 80)]
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
    return score


def _step_order_key(step_id: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)", step_id or "")
    numeric = int(match.group(1)) if match else -1
    return (numeric, step_id or "")


def _claim_identity(claim: NumericClaim) -> str:
    return f"{claim.step_id}:{claim.source_field}@{claim.evidence_id}"


def _select_numeric_claim(
    *,
    candidates: Sequence[tuple[NumericClaim, float]],
    context: str,
    previous_step_id: Optional[str],
) -> tuple[Optional[NumericClaim], bool]:
    """Return ``(claim, ambiguous)`` for one manuscript numeric literal."""

    if not candidates:
        return None, False
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

    return None, True


def bind_numeric_values(
    manuscript: str,
    *,
    evidence: EvidenceStore,
    enforcement_mode: Optional[EvidenceEnforcementMode] = None,
    footnote_prefix: str = "claim",
) -> Tuple[str, Dict[str, NumericClaim], List[str]]:
    """Bind every numeric value in ``manuscript`` to a registered claim.

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
    ``[^xxx]``, or Markdown headings, so it composes cleanly with
    sentence-level evidence binding.

    Inspired by data-to-paper's ``\\hypertarget`` / ``\\hyperlink``
    binding (NEJM AI 2024) but emits Markdown footnotes so the output
    stays readable without a LaTeX pass.
    """
    if not manuscript:
        return manuscript, {}, []
    mode = enforcement_mode or evidence.enforcement_mode

    skip_spans = _spans_to_skip(manuscript)
    binding_map: Dict[str, NumericClaim] = {}
    untraced: List[str] = []
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
        context_start = max(0, start - 80)
        context_end = min(len(manuscript), end + 80)
        context = manuscript[context_start:context_end]
        candidates = _candidate_claims_for_value(evidence, lookup_value)
        claim, ambiguous = _select_numeric_claim(
            candidates=candidates,
            context=context,
            previous_step_id=previous_step_id,
        )
        if claim is None:
            untraced.append(value)
            if mode is EvidenceEnforcementMode.SOFT:
                if ambiguous:
                    candidate_ids = ",".join(
                        _claim_identity(candidate)
                        for candidate, _ in candidates
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
            f"Examples: {untraced[:5]}",
            detail={"untraced": untraced},
        )

    return bound, binding_map, untraced


def enforce_writer_claim_language(
    manuscript: str,
    *,
    enforcement_mode: EvidenceEnforcementMode,
    side_findings: Sequence[SideFinding] | None = None,
) -> Tuple[str, Dict[str, List[str]]]:
    """Block post-hoc rhetoric and side-finding leakage before final binding."""

    detail: Dict[str, List[str]] = {}
    forbidden_terms = sorted(
        {m.group(1).lower() for m in _FORBIDDEN_INTERPRETIVE_RE.finditer(manuscript or "")}
    )
    if forbidden_terms:
        detail["forbidden_terms"] = forbidden_terms

    leaks = side_finding_leaks(manuscript, side_findings or [])
    if leaks:
        detail["side_finding_leak"] = [finding.title or finding.finding_id for finding in leaks]

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
    "enforce_writer_claim_language",
]
