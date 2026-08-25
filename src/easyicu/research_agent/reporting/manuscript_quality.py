"""Deterministic reader-facing quality audit for bound manuscripts.

The evidence-bound manuscript remains the authoritative audit surface.  This
module derives a cleaner reader view without rewriting scientific prose and
checks a small set of structural and cross-section contracts that do not need
another model call.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Any, Mapping, Sequence


_REQUIRED_SECTIONS: Mapping[str, tuple[str, ...]] = {
    "Abstract": (),
    "Introduction": (),
    "Methods": (
        "Study design and cohort",
        "Variables",
        "Statistical analysis",
        "Software and reproducibility",
    ),
    "Results": (
        "Cohort characteristics",
        "Primary outcome",
        "Primary association",
        "Sensitivity and subgroup analyses",
    ),
    "Discussion": (),
    "Limitations": (),
    "Conclusion": (),
    "Data and code availability": (),
    "Funding": (),
    "Ethics approval": (),
    "Conflicts of interest": (),
    "Supplementary artifact release": (),
}
_ABSTRACT_LABELS = ("Background", "Methods", "Results", "Conclusions")
_READER_FACING_SECTIONS = frozenset(
    {
        "Abstract",
        "Introduction",
        "Methods",
        "Results",
        "Discussion",
        "Limitations",
        "Conclusion",
    }
)
_EVIDENCE_LINK_RE = re.compile(r"\[[^\]]+\]\(evidence/[^\n)]*(?:\"[^\"]*\")?\)")
_EVIDENCE_PLACEHOLDER_RE = re.compile(r"\{evidence:[^}\n]+\}")
_CLAIM_MARKER_RE = re.compile(r"\[\^claim_\d+\]")
_CLAIM_DEFINITION_RE = re.compile(r"^\[\^claim_\d+\]:.*$", flags=re.M)
_INTERNAL_PHRASES = (
    "analysis role:",
    "bound typed cohort",
    "host-bound",
    "host-materialized",
    "machine digest",
    "source aware analysis set",
)
_NAMED_METRIC_TERMS = (
    "adjusted rand index",
    "ari",
    "area under the receiver operating characteristic curve",
    "auroc",
    "auc",
    "bayesian information criterion",
    "bic",
    "brier score",
    "calibration intercept",
    "calibration slope",
    "c-index",
    "coefficient",
    "hazard ratio",
    "mean difference",
    "odds ratio",
    "risk difference",
    "risk ratio",
    "silhouette",
)
_OVERPRECISE_DECIMAL_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+\.\d{5,}(?!\d)")


@dataclass(frozen=True)
class ManuscriptQualityFinding:
    """One stable, owner-attributable writing-quality finding."""

    code: str
    severity: str
    section: str
    message: str
    excerpts: tuple[str, ...] = ()


@dataclass(frozen=True)
class ManuscriptQualityAudit:
    """Serializable result of the deterministic manuscript audit."""

    schema_version: str
    status: str
    source_sha256: str
    reader_sha256: str
    section_word_counts: Mapping[str, int]
    adjustment_sets: Mapping[str, tuple[str, ...]]
    internal_evidence_link_count: int
    numeric_claim_marker_count: int
    expected_display_labels: tuple[str, ...]
    observed_display_labels: tuple[str, ...]
    findings: tuple[ManuscriptQualityFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sections(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"^##\s+([^\n]+?)\s*$", text, flags=re.M))
    return {
        match.group(1).strip(): text[
            match.end() : matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        ].strip()
        for index, match in enumerate(matches)
    }


def _subsections(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"^###\s+([^\n]+?)\s*$", text, flags=re.M))
    return {
        match.group(1).strip(): text[
            match.end() : matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        ].strip()
        for index, match in enumerate(matches)
    }


def _has_prose(text: str) -> bool:
    cleaned = _CLAIM_DEFINITION_RE.sub("", text)
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.S)
    cleaned = re.sub(r"^#{1,6}\s+.*$", "", cleaned, flags=re.M)
    cleaned = _EVIDENCE_LINK_RE.sub("", cleaned)
    cleaned = _CLAIM_MARKER_RE.sub("", cleaned)
    return bool(re.search(r"[A-Za-z]{2,}", cleaned))


def _words(text: str) -> int:
    return len(re.findall(r"[A-Za-z][A-Za-z0-9'-]*", text))


def _strip_audit_markup(text: str) -> str:
    cleaned = _EVIDENCE_LINK_RE.sub("", text)
    cleaned = _EVIDENCE_PLACEHOLDER_RE.sub("", cleaned)
    cleaned = _CLAIM_DEFINITION_RE.sub("", cleaned)
    cleaned = _CLAIM_MARKER_RE.sub("", cleaned)
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.S)
    return cleaned


def render_reader_manuscript(bound_text: str) -> str:
    """Remove audit-only markup without changing claims, numbers, or citations."""

    cleaned = _strip_audit_markup(str(bound_text or ""))
    cleaned = re.sub(r"[ \t]+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + "\n"


def _replace_section_body(text: str, section: str, body: str) -> str:
    pattern = re.compile(
        rf"(^##\s+{re.escape(section)}\s*$)(.*?)(?=^##\s+|\Z)",
        flags=re.M | re.S,
    )
    return pattern.sub(
        lambda match: f"{match.group(1)}\n\n{body.strip()}\n\n",
        text,
        count=1,
    )


def _replace_subsection_body(
    text: str,
    section: str,
    subsection: str,
    body: str,
) -> str:
    section_body = _sections(text).get(section)
    if section_body is None:
        return text
    pattern = re.compile(
        rf"(^###\s+{re.escape(subsection)}\s*$)(.*?)(?=^###\s+|\Z)",
        flags=re.M | re.S,
    )
    replaced = pattern.sub(
        lambda match: f"{match.group(1)}\n\n{body.strip()}\n\n",
        section_body,
        count=1,
    )
    if replaced == section_body:
        return text
    return _replace_section_body(text, section, replaced)


def repair_registered_display_callouts(
    manuscript: str,
    *,
    expected_display_labels: Sequence[str],
) -> tuple[str, tuple[Mapping[str, str], ...]]:
    """Add neutral Results callouts only for host-registered displays."""

    repaired = str(manuscript or "")
    templates = {
        "Table 1": (
            "Cohort characteristics",
            "Cohort characteristics are summarized in Table 1 "
            "{evidence:table_one}.",
        ),
        "Figure 1": (
            "Primary association",
            "The principal study results are presented in Figure 1 "
            "{evidence:publication_figure_contract}.",
        ),
    }
    repairs: list[Mapping[str, str]] = []
    for raw_label in expected_display_labels:
        label = str(raw_label).strip()
        target = templates.get(label)
        results = _sections(repaired).get("Results")
        if target is None or results is None:
            continue
        if re.search(rf"\b{re.escape(label)}\b", results, flags=re.I):
            continue
        subsection, sentence = target
        subsection_body = _subsections(results).get(subsection)
        if subsection_body is None:
            continue
        repaired = _replace_subsection_body(
            repaired,
            "Results",
            subsection,
            "\n\n".join(part for part in (subsection_body.strip(), sentence) if part),
        )
        repairs.append(
            {
                "code": "MANUSCRIPT_DISPLAY_CALLOUT_RESTORED",
                "source": "registered_display_authority",
                "label": label,
            }
        )
    return repaired, tuple(repairs)


def repair_reader_structure_from_existing_prose(
    manuscript: str,
) -> tuple[str, tuple[Mapping[str, str], ...]]:
    """Restore required wrappers using only prose already in the draft."""

    repaired = str(manuscript or "")
    repairs: list[Mapping[str, str]] = []

    rounded_count = 0
    for section in _READER_FACING_SECTIONS:
        body = _sections(repaired).get(section)
        if body is None:
            continue

        def round_display(match: re.Match[str]) -> str:
            nonlocal rounded_count
            rounded_count += 1
            return f"{float(match.group(0)):.3f}".rstrip("0").rstrip(".")

        rounded = _OVERPRECISE_DECIMAL_RE.sub(round_display, body)
        if rounded != body:
            repaired = _replace_section_body(repaired, section, rounded)
    if rounded_count:
        repairs.append(
            {
                "code": "MANUSCRIPT_NUMERIC_DISPLAY_ROUNDED",
                "source": "existing_evidence_bound_numeric_prose",
                "count": str(rounded_count),
            }
        )

    section_map = _sections(repaired)
    abstract = section_map.get("Abstract")
    if abstract is not None and not re.search(
        r"\*\*Background:\*\*\s+\S+", abstract, flags=re.I
    ):
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", abstract)]
        candidate_index = next(
            (
                index
                for index, paragraph in enumerate(paragraphs)
                if _has_prose(paragraph)
                and not re.match(
                    r"\*\*(?:Methods|Results|Conclusions):\*\*",
                    paragraph,
                    flags=re.I,
                )
                and not paragraph.startswith("#")
            ),
            None,
        )
        if candidate_index is not None:
            paragraphs[candidate_index] = (
                "**Background:** " + paragraphs[candidate_index]
            )
            repaired = _replace_section_body(
                repaired,
                "Abstract",
                "\n\n".join(paragraphs),
            )
            repairs.append(
                {
                    "code": "MANUSCRIPT_ABSTRACT_LABEL_RESTORED",
                    "source": "existing_abstract_prose",
                }
            )

    section_map = _sections(repaired)
    abstract = section_map.get("Abstract")
    if abstract is not None and not re.search(
        r"\*\*Background:\*\*\s+\S+", abstract, flags=re.I
    ):
        introduction = section_map.get("Introduction", "")
        candidate = next(
            (
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+|\n\s*\n", introduction)
                if _has_prose(sentence)
                and (
                    "{evidence:" in sentence
                    or "{claim:" in sentence
                    or _EVIDENCE_LINK_RE.search(sentence) is not None
                )
            ),
            None,
        )
        if candidate is not None:
            populated = re.sub(
                r"(\*\*Background:\*\*)\s*(?=\n\s*\n|\Z)",
                lambda match: f"{match.group(1)} {candidate}",
                abstract,
                count=1,
                flags=re.I,
            )
            if populated == abstract:
                populated = f"**Background:** {candidate}\n\n{abstract.lstrip()}"
            repaired = _replace_section_body(repaired, "Abstract", populated)
            repairs.append(
                {
                    "code": "MANUSCRIPT_ABSTRACT_BACKGROUND_RESTORED",
                    "source": "existing_introduction_evidence_sentence",
                }
            )

    section_map = _sections(repaired)
    results = section_map.get("Results")
    if results is not None:
        subsections = _subsections(results)
        primary_outcome = subsections.get("Primary outcome")
        primary_association = subsections.get("Primary association", "")
        if primary_outcome is not None and not _has_prose(primary_outcome):
            candidate = next(
                (
                    sentence.strip()
                    for sentence in re.split(
                        r"(?<=[.!?])\s+|\n\s*\n", primary_association
                    )
                    if _has_prose(sentence)
                    and (
                        "{evidence:" in sentence
                        or "{claim:" in sentence
                        or _EVIDENCE_LINK_RE.search(sentence) is not None
                    )
                ),
                None,
            )
            if candidate is not None:
                repaired = _replace_subsection_body(
                    repaired,
                    "Results",
                    "Primary outcome",
                    candidate,
                )
                repairs.append(
                    {
                        "code": "MANUSCRIPT_PRIMARY_OUTCOME_RESTORED",
                        "source": "existing_primary_association_evidence_sentence",
                    }
                )

    section_map = _sections(repaired)
    abstract = section_map.get("Abstract")
    if abstract is not None and not re.search(
        r"\*\*Results:\*\*\s+\S+", abstract, flags=re.I
    ):
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", abstract)]
        methods_index = next(
            (
                index
                for index, paragraph in enumerate(paragraphs)
                if paragraph.casefold().startswith("**methods:**")
            ),
            None,
        )
        conclusions_index = next(
            (
                index
                for index, paragraph in enumerate(paragraphs)
                if paragraph.casefold().startswith("**conclusions:**")
            ),
            len(paragraphs),
        )
        unlabeled_index = next(
            (
                index
                for index, paragraph in enumerate(paragraphs)
                if methods_index is not None
                and methods_index < index < conclusions_index
                and _has_prose(paragraph)
                and not re.match(r"\*\*[A-Za-z ]+:\*\*", paragraph)
            ),
            None,
        )
        if unlabeled_index is not None:
            paragraphs[unlabeled_index] = "**Results:** " + paragraphs[unlabeled_index]
            repaired = _replace_section_body(
                repaired,
                "Abstract",
                "\n\n".join(paragraphs),
            )
            repairs.append(
                {
                    "code": "MANUSCRIPT_ABSTRACT_RESULTS_RELABELED",
                    "source": "existing_post_methods_abstract_prose",
                }
            )

    section_map = _sections(repaired)
    conclusion = section_map.get("Conclusion")
    if conclusion is not None and not _has_prose(conclusion):
        results = section_map.get("Results", "")
        primary = _subsections(results).get("Primary association", "")
        source = primary or results
        candidate = next(
            (
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", source)
                if _has_prose(sentence)
                and (
                    "{evidence:" in sentence
                    or "{claim:" in sentence
                    or _EVIDENCE_LINK_RE.search(sentence) is not None
                )
            ),
            None,
        )
        if candidate is not None:
            repaired = _replace_section_body(repaired, "Conclusion", candidate)
            repairs.append(
                {
                    "code": "MANUSCRIPT_CONCLUSION_RESTORED",
                    "source": "existing_results_evidence_sentence",
                }
            )

    section_map = _sections(repaired)
    abstract = section_map.get("Abstract")
    if abstract is not None and not re.search(
        r"\*\*Conclusions:\*\*\s+\S+", abstract, flags=re.I
    ):
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", abstract)]
        results_index = next(
            (
                index
                for index, paragraph in enumerate(paragraphs)
                if paragraph.casefold().startswith("**results:**")
            ),
            None,
        )
        unlabeled_index = next(
            (
                index
                for index in range(len(paragraphs) - 1, -1, -1)
                if results_index is not None
                and index > results_index
                and _has_prose(paragraphs[index])
                and not re.match(r"\*\*[A-Za-z ]+:\*\*", paragraphs[index])
            ),
            None,
        )
        if unlabeled_index is not None:
            paragraphs[unlabeled_index] = (
                "**Conclusions:** " + paragraphs[unlabeled_index]
            )
            repaired = _replace_section_body(
                repaired,
                "Abstract",
                "\n\n".join(paragraphs),
            )
            repairs.append(
                {
                    "code": "MANUSCRIPT_ABSTRACT_CONCLUSIONS_RELABELED",
                    "source": "existing_post_results_abstract_prose",
                }
            )

    section_map = _sections(repaired)
    abstract = section_map.get("Abstract")
    if abstract is not None and not re.search(
        r"\*\*Conclusions:\*\*\s+\S+", abstract, flags=re.I
    ):
        conclusion = section_map.get("Conclusion", "")
        results = section_map.get("Results", "")
        primary = _subsections(results).get("Primary association", "")
        source = conclusion or primary or results
        candidate = next(
            (
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+|\n\s*\n", source)
                if _has_prose(sentence)
                and (
                    "{evidence:" in sentence
                    or "{claim:" in sentence
                    or _EVIDENCE_LINK_RE.search(sentence) is not None
                )
            ),
            None,
        )
        if candidate is not None:
            populated = re.sub(
                r"(\*\*Conclusions:\*\*)\s*(?=\n\s*\n|\Z)",
                lambda match: f"{match.group(1)}\n\n{candidate}",
                abstract,
                count=1,
                flags=re.I,
            )
            if populated != abstract:
                repaired = _replace_section_body(repaired, "Abstract", populated)
                repairs.append(
                    {
                        "code": "MANUSCRIPT_ABSTRACT_CONCLUSIONS_RESTORED",
                        "source": "existing_conclusion_or_results_evidence_sentence",
                    }
                )
    return repaired, tuple(repairs)


def _normalise_adjustment_set(raw: str) -> tuple[str, ...]:
    cleaned = _strip_audit_markup(raw).replace("`", "")
    cleaned = re.sub(r"\[@[^\]]+\]", "", cleaned)
    cleaned = re.sub(r"\band\b", ",", cleaned, flags=re.I)
    values: list[str] = []
    for part in cleaned.split(","):
        value = re.sub(r"^(?:the\s+)?", "", part.strip(), flags=re.I)
        value = re.sub(r"\s+", " ", value).strip(" .;:").casefold()
        value = re.sub(r"_(?:first|last|max|mean|median|min)$", "", value)
        value = re.sub(
            r"\b(?:comorbidity|index|patient|score)\b",
            "",
            value,
        )
        value = re.sub(r"\s+", " ", value).strip()
        if value and value not in values:
            values.append(value)
    return tuple(sorted(values))


def _adjustment_sets(sections: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    found: dict[str, tuple[str, ...]] = {}
    methods = _strip_audit_markup(sections.get("Methods", ""))
    method_patterns = (
        r"adjustment covariates were\s+([^.;]+)",
        r"adjustment set (?:comprised|included|was)\s+([^.;]+)",
        r"model was adjusted for\s+([^.;]+)",
    )
    method_sets = {
        _normalise_adjustment_set(match.group(1))
        for pattern in method_patterns
        for match in re.finditer(pattern, methods, flags=re.I)
    }
    method_sets.discard(())
    if len(method_sets) == 1:
        found["Methods"] = next(iter(method_sets))

    results = _strip_audit_markup(sections.get("Results", ""))
    result_sets = {
        _normalise_adjustment_set(match.group(1))
        for match in re.finditer(
            r"after adjustment for\s+(.+),\s+[^,.\n]+?\s+"
            r"(?:was|were|had|showed)\b",
            results,
            flags=re.I,
        )
    }
    result_sets.update(
        _normalise_adjustment_set(match.group(1))
        for match in re.finditer(
            r",\s+after adjustment for\s+([^.;]+)",
            results,
            flags=re.I,
        )
    )
    result_sets.discard(())
    if len(result_sets) == 1:
        found["Results"] = next(iter(result_sets))
    return found


def _internal_excerpts(section_text: str) -> tuple[str, ...]:
    excerpts: list[str] = []
    for match in re.finditer(r"`([A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+)`", section_text):
        excerpts.append(match.group(0))
    for match in re.finditer(r"\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+){2,}\b", section_text):
        excerpts.append(match.group(0))
    lowered = section_text.casefold()
    for phrase in _INTERNAL_PHRASES:
        if phrase in lowered:
            excerpts.append(phrase)
    return tuple(dict.fromkeys(excerpts))


def _sentences(text: str) -> tuple[str, ...]:
    cleaned = _strip_audit_markup(text)
    return tuple(
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if sentence.strip()
    )


def _abstract_blocks(abstract: str) -> Mapping[str, str]:
    matches = list(
        re.finditer(
            r"\*\*(Background|Methods|Results|Conclusions):\*\*",
            abstract,
            flags=re.I,
        )
    )
    blocks: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(abstract)
        blocks[match.group(1).casefold()] = abstract[match.end() : end].strip()
    return blocks


def _unnamed_metric_excerpts(section_text: str) -> tuple[str, ...]:
    excerpts: list[str] = []
    for sentence in _sentences(section_text):
        lowered = re.sub(
            r"\[[^\[\]]*@[A-Za-z0-9_.:-]+[^\[\]]*\]",
            "",
            sentence,
        ).casefold()
        has_unlabelled_numeric_summary = (
            "point estimate" in lowered
            or (
                re.search(
                    r"\brobustness\b[^.!?]*"
                    r"(?:value|estimate|score|metric|range|ranged|reported)"
                    r"[^.!?]*[-+]?\d+(?:\.\d+)?",
                    lowered,
                )
                is not None
            )
        )
        if not has_unlabelled_numeric_summary:
            continue
        if any(
            re.search(rf"\b{re.escape(term)}\b", lowered) is not None
            for term in _NAMED_METRIC_TERMS
        ):
            continue
        excerpts.append(sentence[:300])
    return tuple(excerpts)


def _section_has_truncated_ending(section_text: str) -> bool:
    prose = _strip_audit_markup(section_text).rstrip()
    if not prose:
        return False
    return prose[-1] not in ".!?)]}"


def expected_manuscript_display_labels(
    evidence_ids: Sequence[str],
) -> tuple[str, ...]:
    """Map verified evidence identities to reader-facing display labels."""

    names = {str(item).strip() for item in evidence_ids if str(item).strip()}
    labels: list[str] = []
    if "table_one" in names:
        labels.append("Table 1")
    if "publication_figure_contract" in names:
        labels.append("Figure 1")
    return tuple(labels)


def audit_manuscript_quality(
    bound_text: str,
    *,
    expected_display_labels: Sequence[str] = (),
    require_administrative_sections: bool = True,
) -> ManuscriptQualityAudit:
    """Audit structure, terminology, and one high-confidence consistency rule."""

    text = str(bound_text or "")
    reader = render_reader_manuscript(text)
    section_map = _sections(text)
    findings: list[ManuscriptQualityFinding] = []

    if not re.search(r"^#\s+\S+", text, flags=re.M):
        findings.append(
            ManuscriptQualityFinding(
                code="MANUSCRIPT_TITLE_MISSING",
                severity="error",
                section="Title",
                message="The manuscript has no level-one title.",
            )
        )
    if not re.search(r"^\*\*Keywords:\*\*\s+\S+", text, flags=re.M | re.I):
        findings.append(
            ManuscriptQualityFinding(
                code="MANUSCRIPT_KEYWORDS_MISSING",
                severity="error",
                section="Title",
                message="The manuscript has no populated Keywords line.",
            )
        )

    required_sections = dict(_REQUIRED_SECTIONS)
    if not require_administrative_sections:
        for section in (
            "Data and code availability",
            "Funding",
            "Ethics approval",
            "Conflicts of interest",
            "Supplementary artifact release",
        ):
            required_sections.pop(section, None)
    for section, required_subsections in required_sections.items():
        body = section_map.get(section)
        if body is None:
            findings.append(
                ManuscriptQualityFinding(
                    code="MANUSCRIPT_SECTION_MISSING",
                    severity="error",
                    section=section,
                    message=f"Required section {section!r} is missing.",
                )
            )
            continue
        if not _has_prose(body):
            findings.append(
                ManuscriptQualityFinding(
                    code="MANUSCRIPT_SECTION_EMPTY",
                    severity="error",
                    section=section,
                    message=f"Required section {section!r} has no substantive prose.",
                )
            )
        subsection_map = _subsections(body)
        for subsection in required_subsections:
            subsection_body = subsection_map.get(subsection)
            if subsection_body is None or not _has_prose(subsection_body):
                findings.append(
                    ManuscriptQualityFinding(
                        code="MANUSCRIPT_SUBSECTION_MISSING_OR_EMPTY",
                        severity="error",
                        section=section,
                        message=(
                            f"Required subsection {subsection!r} is missing or empty."
                        ),
                        excerpts=(subsection,),
                    )
                )

    abstract = section_map.get("Abstract")
    if abstract is not None:
        for label in _ABSTRACT_LABELS:
            if not re.search(
                rf"\*\*{re.escape(label)}:\*\*\s+\S+", abstract, flags=re.I
            ):
                findings.append(
                    ManuscriptQualityFinding(
                        code="MANUSCRIPT_ABSTRACT_LABEL_MISSING_OR_EMPTY",
                        severity="error",
                        section="Abstract",
                        message=f"Abstract label {label!r} is missing or empty.",
                        excerpts=(label,),
                    )
                )
        abstract_blocks = _abstract_blocks(abstract)
        abstract_results = abstract_blocks.get("results", "")
        abstract_conclusions = abstract_blocks.get("conclusions", "")
        normalized_results = re.sub(
            r"\s+", " ", _strip_audit_markup(abstract_results)
        ).strip().casefold()
        normalized_conclusions = re.sub(
            r"\s+", " ", _strip_audit_markup(abstract_conclusions)
        ).strip().casefold()
        if normalized_results and normalized_results == normalized_conclusions:
            findings.append(
                ManuscriptQualityFinding(
                    code="MANUSCRIPT_ABSTRACT_CONCLUSION_DUPLICATES_RESULTS",
                    severity="error",
                    section="Abstract",
                    message=(
                        "Abstract Conclusions repeats Results verbatim instead of "
                        "stating the bounded interpretation."
                    ),
                )
            )

    adjustments = _adjustment_sets(section_map)
    if (
        "Methods" in adjustments
        and "Results" in adjustments
        and adjustments["Methods"] != adjustments["Results"]
    ):
        findings.append(
            ManuscriptQualityFinding(
                code="MANUSCRIPT_ADJUSTMENT_SET_CONFLICT",
                severity="error",
                section="Methods/Results",
                message="Methods and Results report different adjustment sets.",
                excerpts=(
                    "Methods: " + ", ".join(adjustments["Methods"]),
                    "Results: " + ", ".join(adjustments["Results"]),
                ),
            )
        )

    for section in _READER_FACING_SECTIONS:
        section_text = section_map.get(section, "")
        if section_text and _section_has_truncated_ending(section_text):
            findings.append(
                ManuscriptQualityFinding(
                    code="MANUSCRIPT_SECTION_TRUNCATED",
                    severity="error",
                    section=section,
                    message=(
                        "The section ends without terminal punctuation and may be "
                        "truncated. Regenerate the complete section."
                    ),
                    excerpts=(_strip_audit_markup(section_text).rstrip()[-160:],),
                )
            )
        excerpts = _internal_excerpts(section_text)
        if excerpts:
            findings.append(
                ManuscriptQualityFinding(
                    code="MANUSCRIPT_INTERNAL_TERM_EXPOSED",
                    severity="error",
                    section=section,
                    message=(
                        "Reader-facing prose exposes raw runtime identifiers or "
                        "engineering terminology. Methods may describe a variable "
                        "concept and analysis window, but not its implementation name."
                    ),
                    excerpts=excerpts[:12],
                )
            )
        unnamed_metrics = _unnamed_metric_excerpts(section_text)
        if unnamed_metrics:
            findings.append(
                ManuscriptQualityFinding(
                    code="MANUSCRIPT_METRIC_UNNAMED",
                    severity="error",
                    section=section,
                    message=(
                        "A reader-facing numeric result is reported without naming "
                        "its exact statistical metric."
                    ),
                    excerpts=unnamed_metrics[:8],
                )
            )
        overprecise = tuple(
            dict.fromkeys(
                _OVERPRECISE_DECIMAL_RE.findall(_strip_audit_markup(section_text))
            )
        )
        if overprecise:
            findings.append(
                ManuscriptQualityFinding(
                    code="MANUSCRIPT_NUMERIC_OVERPRECISION",
                    severity="error",
                    section=section,
                    message=(
                        "Reader-facing prose exposes machine precision rather than "
                        "a publication-scale display value."
                    ),
                    excerpts=overprecise[:12],
                )
            )

    raw_results_text = section_map.get("Results", "")
    normalized_expected_displays = tuple(
        dict.fromkeys(
            str(label).strip()
            for label in expected_display_labels
            if str(label).strip()
        )
    )
    observed_displays = tuple(
        label
        for label in normalized_expected_displays
        if re.search(rf"\b{re.escape(label)}\b", raw_results_text, flags=re.I)
    )
    for label in normalized_expected_displays:
        if label in observed_displays:
            continue
        findings.append(
            ManuscriptQualityFinding(
                code="MANUSCRIPT_DISPLAY_NOT_CALLED_OUT",
                severity="error",
                section="Results",
                message=(
                    f"A registered manuscript display ({label}) is not called out "
                    "in Results."
                ),
                excerpts=(label,),
            )
        )
    results_text = _strip_audit_markup(raw_results_text).casefold()
    discussion_text = _strip_audit_markup(section_map.get("Discussion", "")).casefold()
    reports_risk_difference = (
        "risk difference" in results_text
        or re.search(
            r"\{claim:[^}\n]*risk_difference[^}\n]*\}",
            raw_results_text,
            flags=re.I,
        )
        is not None
    )
    if reports_risk_difference and re.search(
        r"(?:no|not|does not|did not).{0,80}basis.{0,80}(?:absolute )?risk difference",
        discussion_text,
    ):
        findings.append(
            ManuscriptQualityFinding(
                code="MANUSCRIPT_REPORTED_RESULT_DISCLAIMED",
                severity="error",
                section="Discussion",
                message=(
                    "Discussion says a risk difference is unavailable although "
                    "Results reports one."
                ),
            )
        )

    return ManuscriptQualityAudit(
        schema_version="manuscript-quality-audit-v3",
        status="pass"
        if not any(item.severity == "error" for item in findings)
        else "changes_required",
        source_sha256=_sha256(text),
        reader_sha256=_sha256(reader),
        section_word_counts={name: _words(body) for name, body in section_map.items()},
        adjustment_sets=adjustments,
        internal_evidence_link_count=len(_EVIDENCE_LINK_RE.findall(text)),
        numeric_claim_marker_count=len(_CLAIM_MARKER_RE.findall(text)),
        expected_display_labels=normalized_expected_displays,
        observed_display_labels=observed_displays,
        findings=tuple(findings),
    )


__all__ = [
    "ManuscriptQualityAudit",
    "ManuscriptQualityFinding",
    "audit_manuscript_quality",
    "expected_manuscript_display_labels",
    "repair_registered_display_callouts",
    "repair_reader_structure_from_existing_prose",
    "render_reader_manuscript",
]
