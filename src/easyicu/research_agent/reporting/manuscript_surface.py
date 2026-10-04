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


#: Capitalized words a label keeps inside a sentence: eponyms, and population
#: and insurance categories that style guides capitalize.
_PROPER_LABEL_WORDS = frozenset({
    "African", "American", "Apgar", "Asian", "Berlin", "Black", "Braden", "Caprini",
    "Charlson", "Cox", "Elixhauser", "Glasgow", "Hispanic", "Horowitz", "Kaplan",
    "Killip", "Latino", "Medicaid", "Medicare", "Native", "Pacific", "Ramsay",
    "Richmond", "Wells", "White",
})


#: Words that may legitimately stand twice in a row ("in in-hospital death").
_FUNCTION_WORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "no", "not", "of", "on",
    "or", "per", "the", "to", "with", "within", "without",
})


def in_sentence_label(label: str) -> str:
    """A reader label as it reads inside a sentence.

    Labels are written like headings ("Death by 28 days"), so mid-sentence the
    heading capital drops ("associated with death by 28 days").  An acronym
    (ICU, SOFA score), a mixed-case term (pH, eGFR), a title-case name
    (Sequential Organ Failure Assessment) or an eponym (Charlson comorbidity
    index) keeps its capitals.  A sentence-initial use is the caller's to
    capitalize.
    """

    words = str(label).split(" ")
    first = words[0]
    if (
        not re.fullmatch(r"[A-Z][a-z]+(?:-[a-z]+)*", first)
        or first.split("-")[0] in _PROPER_LABEL_WORDS
        or (len(words) > 1 and re.match(r"[A-Z]", words[1]))
    ):
        return str(label)
    return first.lower() + str(label)[len(first):]


def collapse_repeated_label_prefix(text: str, labels) -> tuple[str, tuple[dict[str, str], ...]]:
    """Remove a duplicated prefix of a complete source-bound label.

    A raw field embedded in prose may already have part of its expanded label
    before it ("patient age" becoming "patient Patient age at baseline").  Only
    exact alphabetic prefixes of a label of at least three words are handled;
    there is no synonym matching, numerical cleanup or scientific paraphrase.
    The caller keeps evidence/citation tokens out of these visible text pieces.
    """
    repairs = []
    for label in sorted(set(labels), key=len, reverse=True):
        if not re.fullmatch(r"[A-Za-z]+(?:[ -]+[A-Za-z]+){2,}", label):
            continue
        words = re.split(r"[ -]+", label)
        full = r"[ -]+".join(map(re.escape, words))
        for length in range(len(words) - 1, 0, -1):
            if length == 1 and words[0].casefold() in _FUNCTION_WORDS:
                continue  # "a difference in in-hospital death" is English.
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


def claim_token_stands_in_block(text: str, start: int, end: int, token: str) -> bool:
    """Whether ``token`` already stands in the block holding ``text[start:end]``.

    Blocks are the ones ``deduplicate_claim_paragraphs`` keeps a claim once
    in: a heading or a structured-abstract label opens one.  The span itself
    is not searched, so the text a caller is about to replace never counts.
    """
    starts = [match.start() for match in _REGION.finditer(text)]
    block_start = max((index for index in starts if index <= start), default=0)
    block_end = min((index for index in starts if index > start), default=len(text))
    return token in text[block_start:start] or token in text[end:block_end]


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


_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?。！？])\s+")
#: A sentence this long said twice reads as a template, not as emphasis.
_REPEATED_SENTENCE_MIN_CHARS = 60


def repeated_reader_sentences(sections: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    """Long prose sentences said again, as ``(section, sentence)`` pairs.

    ``sections`` maps reader section names to their reader text in manuscript
    order.  Headings, tables, figures and footnotes are not prose.  The first
    occurrence stands; each later one is reported in its own section.
    """

    seen: set[str] = set()
    repeated: list[tuple[str, str]] = []
    for name, body in sections.items():
        for paragraph in re.split(r"\n\s*\n", body):
            lines = [
                line.strip() for line in paragraph.splitlines()
                if line.strip() and line.strip()[0] not in "#|!" and not line.strip().startswith("[^")
            ]
            for sentence in _SENTENCE_BREAK_RE.split(" ".join(" ".join(lines).split())):
                if len(sentence) < _REPEATED_SENTENCE_MIN_CHARS:
                    continue
                key = sentence.casefold()
                if key in seen:
                    repeated.append((name, sentence))
                seen.add(key)
    return tuple(repeated)


# A parenthesis that opens with an effect measure and closes its sentence
# promises that measure's value: "(adjusted hazard ratio for days 0 to 7
# after the landmark)." reads as an estimate whose number was lost.
_NAMED_ESTIMATE_RE = re.compile(
    r"\((?P<inner>(?:(?:adjusted|unadjusted|crude)\s+)?(?:"
    r"(?:subdistribution\s+)?(?:hazard|odds|risk|rate)\s+ratios?"
    r"|relative\s+risks?|(?:risk|mean)\s+differences?"
    r"|restricted\s+mean\s+survival\s+time\s+differences?"
    r")\b[^()]*)\)(?=\s*(?:[.;!?]|$))",
    re.I | re.M,
)
_ESTIMATE_VALUE_RE = re.compile(r"\d\.\d|\bCI\b|\d\s*%|\bp\s*[<=>]", re.I)


def estimates_without_values(text: str) -> tuple[str, ...]:
    """Sentence-closing parentheses that name an effect estimate without a value."""

    return tuple(dict.fromkeys(
        match.group(0) for match in _NAMED_ESTIMATE_RE.finditer(text)
        if _ESTIMATE_VALUE_RE.search(match["inner"]) is None
    ))
