"""Shared, dependency-free context rules for provenance-driven prose removal.

This owner never supplies a replacement antecedent or a scientific statement.
Only a leading dependent sentence in the *same paragraph* as a deleted opener
can be removed. Every additional removal is returned for the repair receipt.
Audit-only markup is not prose: it neither supplies an antecedent nor
separates a dependent sentence from the opener it depends on.  This owner
defines that markup once; the reader view strips the same markup.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


_DEPENDENT_OPENER = re.compile(
    r"^(?:It\b|Its\b|They\b|Their\b|These\s+(?:values|estimates|measures|representations)\b|"
    r"This\s+(?:value|estimate|measure|representation)\b|"
    r"The\s+(?:former|latter|clinical definition|maximum representation|"
    r"minimum representation)\b)",
    re.I,
)
_PARAGRAPH_BOUNDARY = re.compile(r"\r?\n[ \t]*\r?\n")
_SOFT_SPACE = r"(?:[ \t]*\r?\n[ \t]*|[ \t]+)"
_PROSE_SENTENCE_END = re.compile(
    rf"""(?x:
        (?:
            [!?。！？]
            | \. (?!
                (?<=(?i:\bvs\.)) {_SOFT_SPACE} \S
                | (?<=(?i:\bet[ \t]al\.)) {_SOFT_SPACE} [a-z(]
                | (?<=(?i:\bfig\.)) {_SOFT_SPACE} [Ss]?\d
            )
        )
        (?=\s|$)
    )"""
)
MARKDOWN_REGION_RE = re.compile(
    r"(?m)^##[ \t]+(?P<section>[^\r\n]+)"
    r"|(?P<structural>^[ \t]*(?:#{1,6}[ \t]+|\||!\[|\[\^|\*\*Keywords:\*\*)[^\r\n]*)"
    r"|^[ \t]*\*\*(?:Background|Methods|Results|Conclusions?):\*\*[ \t]*"
    r"|\r?\n[ \t]*\r?\n",
    re.I,
)
_OTHER_BLOCK_START = re.compile(r"^[ \t]*(?:```|~~~|>|(?:[-+*]|\d+[.)])[ \t]+)")
# Audit-only markup: a scaffold evidence token, its bound link, a numeric
# claim footnote marker, and an annotation comment (an unresolved evidence
# token is demoted to one).  Readers never see any of them, and a Writer often
# places a citation after the period, where sentence splitting assigns it to
# the next sentence.
EVIDENCE_TOKEN_PATTERN = r"\{evidence:[^}\n]+\}"
EVIDENCE_LINK_PATTERN = r'\[[^\]]+\]\(evidence/[^\n)]*(?:"[^"]*")?\)'
CLAIM_MARKER_PATTERN = r"\[\^claim_\d+\]"
ANNOTATION_COMMENT_PATTERN = r"<!--.*?-->"
_CITATION = "|".join(
    (EVIDENCE_TOKEN_PATTERN, EVIDENCE_LINK_PATTERN, CLAIM_MARKER_PATTERN, ANNOTATION_COMMENT_PATTERN)
)
#: Every piece of audit-only markup, for owners that strip it.
AUDIT_MARKUP_RE = re.compile(_CITATION, re.S)
SENTENCE_CITATIONS_RE = re.compile(rf"(?:{_CITATION})(?:\s*;\s*(?:{_CITATION}))*|\[@[^\]\n]+\]", re.S)
_LEADING_CITATION_MARKUP = re.compile(rf"(?:[ \t]*(?:{_CITATION}))*[ \t]*", re.S)


def _blank_sentence_citations(text: str) -> str:
    return SENTENCE_CITATIONS_RE.sub(lambda match: re.sub(r"[^\r\n]", " ", match[0]), text)


def prose_sentence_ends(text: str) -> tuple[int, ...]:
    """Return punctuation endpoints in source coordinates for one prose region."""
    return tuple(match.end() for match in _PROSE_SENTENCE_END.finditer(_blank_sentence_citations(text)))


def soft_wrapped_prose_lines(text: str) -> list[str]:
    """Keep line-oriented grammar, joining only recognized abbreviation continuations."""
    lines = text.splitlines(keepends=True)
    blocks = []
    start = 0
    for index, (line, following) in enumerate(zip(lines, lines[1:])):
        if following.strip() and not any(
            pattern.match(value)
            for pattern in (MARKDOWN_REGION_RE, _OTHER_BLOCK_START)
            for value in (lines[start], following)
        ):
            visible = _blank_sentence_citations(line).rstrip()
            if visible.endswith(".") and len(visible) not in prose_sentence_ends(line + following):
                # A period visible at the end of the first line was a
                # boundary in isolation. Only the known abbreviation rules
                # can suppress that boundary when the next line is present.
                continue
        blocks.append("".join(lines[start:index + 1]))
        start = index + 1
    if lines:
        blocks.append("".join(lines[start:]))
    return blocks


@dataclass(frozen=True)
class ContextualSentenceDeletion:
    start: int
    end: int
    dependent_sentences: tuple[str, ...] = ()


def has_dependent_opener(paragraph: str) -> bool:
    """Recognize narrow anaphoric openers, not arbitrary 'This study' prose."""

    text = paragraph.lstrip()
    return bool(_DEPENDENT_OPENER.match(text[_LEADING_CITATION_MARKUP.match(text).end():]))


def contextual_sentence_deletion(
    text: str,
    start: int,
    end: int,
) -> ContextualSentenceDeletion:
    """Extend deletion only past newly orphaned, same-paragraph sentences."""

    if not 0 <= start < end <= len(text):
        raise ValueError("manuscript deletion span is outside the source text")
    prefix = text[:start]
    boundaries = list(_PARAGRAPH_BOUNDARY.finditer(prefix))
    paragraph_start = boundaries[-1].end() if boundaries else 0
    # A heading with no blank line also starts a new paragraph. Any other
    # surviving prose before the target may still supply its antecedent.
    before = AUDIT_MARKUP_RE.sub("", text[paragraph_start:start]).strip()
    if before and not re.fullmatch(r"#{1,6}[^\n]*\n?", before):
        return ContextualSentenceDeletion(start, end)

    boundary = _PARAGRAPH_BOUNDARY.search(text, end)
    paragraph_end = boundary.start() if boundary else len(text)
    suffix = text[end:paragraph_end]
    # Do not traverse a hard line/heading/list boundary or a blank paragraph.
    suffix = suffix.split("\n", 1)[0]
    cursor = 0
    dependent: list[str] = []
    while cursor < len(suffix):
        whitespace = len(suffix[cursor:]) - len(suffix[cursor:].lstrip(" \t"))
        sentence_start = cursor + whitespace
        remaining = suffix[sentence_start:]
        if not has_dependent_opener(remaining):
            break
        terminal = next(iter(prose_sentence_ends(remaining)), None)
        if terminal is None:
            break  # Do not guess the end of an incomplete sentence.
        sentence_end = sentence_start + terminal
        dependent.append(suffix[sentence_start:sentence_end].strip())
        cursor = len(suffix) - len(suffix[sentence_end:].lstrip(" \t"))
    return ContextualSentenceDeletion(
        start,
        end + cursor,
        tuple(dependent),
    )
