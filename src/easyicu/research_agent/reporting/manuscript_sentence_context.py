"""Preserve local sentence coherence when a provenance gate removes prose.

This owner never supplies a replacement antecedent or a scientific statement.
Only a leading dependent sentence in the *same paragraph* as a deleted opener
can be removed. Every additional removal is returned for the repair receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


_DEPENDENT_OPENER = re.compile(
    r"^(?:It\b|They\b|These\s+(?:values|estimates|measures|representations)\b|"
    r"This\s+(?:value|estimate|measure|representation)\b|"
    r"The\s+(?:former|latter|clinical definition|maximum representation|"
    r"minimum representation)\b)",
    re.I,
)
_PARAGRAPH_BOUNDARY = re.compile(r"\n[ \t]*\n")
_SENTENCE_END = re.compile(r"[.!?。！？](?:[ \t]+|$)")


@dataclass(frozen=True)
class ContextualSentenceDeletion:
    start: int
    end: int
    dependent_sentences: tuple[str, ...] = ()


def has_dependent_opener(paragraph: str) -> bool:
    """Recognize narrow anaphoric openers, not arbitrary 'This study' prose."""

    return bool(_DEPENDENT_OPENER.match(paragraph.lstrip()))


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
    # surviving text before the target may still supply its antecedent.
    before = text[paragraph_start:start].strip()
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
        terminal = _SENTENCE_END.search(remaining)
        if terminal is None:
            break  # Do not guess the end of an incomplete sentence.
        sentence_end = sentence_start + terminal.end()
        dependent.append(suffix[sentence_start:sentence_end].strip())
        cursor = sentence_end
    return ContextualSentenceDeletion(
        start,
        end + cursor,
        tuple(dependent),
    )
