"""Conservative cleanup after claim filtering; never compose new scientific prose."""

import re


_REGION = re.compile(r"(?=^#{1,6} |^\*\*(?:Background|Methods|Results|Conclusions):\*\*)", re.M)
_CLAIM = re.compile(r"\{claim:[^{}\s]+\}[.!?]?")


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
