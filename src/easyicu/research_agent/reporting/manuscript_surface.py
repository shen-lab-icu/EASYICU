"""Conservative manuscript-surface projections and shared display targets.

This leaf module never composes scientific prose.  It owns deterministic
reader cleanup and the advisory length targets already stated by the Writer's
section instructions so reporting layers can share them without importing one
another.
"""

import re
from typing import Any, Mapping


_REGION = re.compile(r"(?=^#{1,6} |^\*\*(?:Background|Methods|Results|Conclusions):\*\*)", re.M)
_CLAIM = re.compile(r"\{claim:[^{}\s]+\}[.!?]?")
_EVIDENCE_LINK_RE = re.compile(r'\[[^\]]+\]\(evidence/[^\n)]*(?:"[^"]*")?\)')
_EVIDENCE_PLACEHOLDER_RE = re.compile(r"\{evidence:[^}\n]+\}")
_CLAIM_MARKER_RE = re.compile(r"\[\^claim_\d+\]")
_CLAIM_PLACEHOLDER_RE = re.compile(
    r"\{claim:[A-Za-z0-9_-]+\.[a-z][a-z0-9_]*\}"
)
_CLAIM_DEFINITION_RE = re.compile(r"^\[\^claim_\d+\]:.*$", flags=re.M)


# Each value is ``(word_target, paragraph_target)``.  ``None`` means the
# Writer instruction states no numeric bound; no target is inferred.  These
# targets are advisory and never replace the scientific-maturity anti-stub
# floors.
MANUSCRIPT_SECTION_LENGTH_TARGETS: Mapping[
    str,
    tuple[tuple[int, int] | None, tuple[int, int] | None],
] = {
    "abstract": ((200, 300), (4, 4)),
    "introduction": ((300, 500), (3, 5)),
    "methods": ((400, 600), None),
    "results": ((400, 600), None),
    "discussion": ((400, 650), (4, 5)),
    "limitations": ((150, 250), (1, 1)),
}

_PROSE_SECTION_ALIASES = {
    "abstract": {"abstract"},
    "introduction": {"introduction", "background"},
    "methods": {"methods", "method", "materials and methods"},
    "results": {"results"},
    "discussion": {"discussion"},
    "limitations": {"limitations", "strengths and limitations"},
    "conclusion": {"conclusion", "conclusions"},
}


def _strip_audit_markup(text: str) -> str:
    cleaned = _EVIDENCE_LINK_RE.sub("", text)
    cleaned = _EVIDENCE_PLACEHOLDER_RE.sub("", cleaned)
    cleaned = _CLAIM_DEFINITION_RE.sub("", cleaned)
    cleaned = _CLAIM_MARKER_RE.sub("", cleaned)
    cleaned = _CLAIM_PLACEHOLDER_RE.sub("", cleaned)
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


def manuscript_section_prose_metrics(manuscript: str) -> dict[str, dict[str, int]]:
    """Measure reader-facing prose without crediting audit-only markup."""

    reader = render_reader_manuscript(manuscript)
    matches = list(
        re.finditer(r"^(?P<marks>#{1,3})\s+(?P<title>.+?)\s*$", reader, re.MULTILINE)
    )
    measured: dict[str, dict[str, int]] = {}
    for index, match in enumerate(matches):
        normalized = " ".join(
            re.sub(
                r"[^a-z0-9\u4e00-\u9fff]+",
                " ",
                match.group("title").casefold(),
            ).split()
        )
        section = next(
            (
                key
                for key, values in _PROSE_SECTION_ALIASES.items()
                if normalized in values
            ),
            None,
        )
        if section is None:
            continue
        level = len(match.group("marks"))
        end = len(reader)
        for candidate in matches[index + 1 :]:
            if len(candidate.group("marks")) <= level:
                end = candidate.start()
                break
        body = reader[match.end() : end]
        paragraphs = [block for block in re.split(r"\n\s*\n", body) if block.strip()]
        measured[section] = {
            "words": len(re.findall(r"\b[\w'-]+\b", body)),
            "paragraphs": len(paragraphs),
        }
    return measured


def manuscript_section_target_deviations(
    manuscript: str,
) -> list[dict[str, Any]]:
    """Compare reader prose with the Writer's non-gating length targets."""

    measured = manuscript_section_prose_metrics(manuscript)
    deviations: list[dict[str, Any]] = []
    for key, (word_target, paragraph_target) in MANUSCRIPT_SECTION_LENGTH_TARGETS.items():
        section = measured.get(key)
        if section is None:
            continue
        issues: list[str] = []
        if word_target:
            low, high = word_target
            if section["words"] < low:
                issues.append("below_word_target")
            elif section["words"] > high:
                issues.append("above_word_target")
        if paragraph_target:
            low, high = paragraph_target
            if section["paragraphs"] < low:
                issues.append("below_paragraph_target")
            elif section["paragraphs"] > high:
                issues.append("above_paragraph_target")
        if issues:
            deviations.append(
                {
                    "section": key,
                    "observed_words": section["words"],
                    "observed_paragraphs": section["paragraphs"],
                    "word_target": list(word_target) if word_target else None,
                    "paragraph_target": (
                        list(paragraph_target) if paragraph_target else None
                    ),
                    "issues": issues,
                    "gating": False,
                }
            )
    return deviations


def collapse_repeated_label_prefix(text: str, labels) -> tuple[str, tuple[dict[str, str], ...]]:
    """Remove a duplicated multiword prefix of a complete source-bound label.

    A raw field embedded in prose may already have part of its expanded label
    before it. Only exact alphabetic prefixes (at least two words) are handled;
    there is no synonym matching, numerical cleanup or scientific paraphrase.
    The caller keeps evidence/citation tokens out of these visible text pieces.
    """
    repairs = []
    for label in sorted(set(labels), key=len, reverse=True):
        if not re.fullmatch(r"[A-Za-z]+(?:[ -]+[A-Za-z]+){2,}", label):
            continue
        words = re.split(r"[ -]+", label)
        full = r"[ -]+".join(map(re.escape, words))
        for length in range(len(words) - 1, 1, -1):
            prefix = r"[ -]+".join(map(re.escape, words[:length]))
            pattern = re.compile(r"(?<![A-Za-z0-9_])" + prefix + r"[ -]+(?P<label>" + full + r")(?![A-Za-z0-9_])", re.I)
            text, count = pattern.subn(lambda match: match["label"], text)
            if count:
                repairs.append({"code": "MANUSCRIPT_REPEATED_LABEL_PREFIX_REMOVED",
                                "source": " ".join(words[:length]) + " " + label,
                                "replacement": label, "count": str(count)})
    return text, tuple(repairs)


def deduplicate_claim_paragraphs(text: str) -> str:
    """Repeat a claim across sections if needed, but once within each block."""
    regions = _REGION.split(text)
    for index, region in enumerate(regions):
        seen = set()
        removed = False
        parts = re.split(r"(\n\s*\n)", region)
        for i in range(0, len(parts), 2):
            token = parts[i].strip()
            if not _CLAIM.fullmatch(token):
                continue
            if token in seen:
                parts[i] = ""
                removed = True
            seen.add(token)
        if removed:
            regions[index] = re.sub(r"\n{3,}", "\n\n", "".join(parts))
    return "".join(regions)


def repair_filtered_section_openers(text: str, *, before_filter: str) -> str:
    """Remove a dangling connective only when filtering removed its antecedent.

    The surviving first sentence must have existed later in that same section;
    authored first sentences and all numerical/citation content are untouched.
    """
    headings = list(re.finditer(r"^## ([^\n]+)\n", text, re.M))
    for i, heading in reversed(list(enumerate(headings))):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[heading.end():end]
        sentence = re.split(r"(?<=[.!?])\s+", body.strip(), maxsplit=1)[0]
        prior = re.search(r"^## " + re.escape(heading.group(1)) + r"\n(.*?)(?=^## |\Z)",
                          before_filter, re.M | re.S)
        if not sentence or prior is None or prior.group(1).strip().startswith(sentence):
            continue
        if sentence not in prior.group(1):
            continue
        repaired = re.sub(r"\b(is|are|was|were|findings|results) therefore\b", r"\1", sentence, flags=re.I)
        if repaired != sentence:
            text = text[:heading.end()] + body.replace(sentence, repaired, 1) + text[end:]
    return text


def repeated_reader_paragraphs(section: str) -> tuple[str, ...]:
    """Exact long paragraph duplicates, scoped to one heading/abstract block."""
    duplicates = []
    for region in _REGION.split(section):
        seen = set()
        for paragraph in re.split(r"\n\s*\n", region):
            normalized = " ".join(paragraph.split())
            if len(normalized) < 80 or normalized.startswith("#"):
                continue
            if normalized in seen:
                duplicates.append(normalized)
            seen.add(normalized)
    return tuple(duplicates)
