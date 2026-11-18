"""Shared, dependency-free context rules for provenance-driven prose removal.

This owner never supplies a replacement antecedent or a scientific statement.
Only a leading dependent sentence in the *same paragraph* as a deleted opener
can be removed. Every additional removal is returned for the repair receipt.
Audit-only markup is not prose: it neither supplies an antecedent nor
separates a dependent sentence from the opener it depends on.  This owner
defines that markup once; the reader view strips the same markup.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
import re


_DEPENDENT_OPENER = re.compile(
    r"^(?:It\b|Its\b|They\b|Their\b|These\s+(?:values|estimates|measures|representations)\b|"
    r"This\s+(?:value|estimate|measure|representation)\b|"
    r"The\s+(?:former|latter|clinical definition|maximum representation|"
    r"minimum representation)\b)",
    re.I,
)
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
_HEADING = re.compile(r"^[ \t]*(?P<level>#{1,6})(?:[ \t]+|$)(?P<title>.*)")
_LIST_ITEM = re.compile(r"^[ \t]*(?:[-+*]|[0-9]+[.)])[ \t]+")
_NUMBERED_CONTINUATION = re.compile(r"^[ \t]*[0-9]+\.[ \t]+")
# A standalone claim placeholder owns its line even without punctuation.
_STANDALONE_CLAIM = re.compile(r"[ \t]*\{claim:[^{}\r\n]+\}[.!?。！？]?[ \t]*")
_FENCE = re.compile(r"^[ \t]*(?P<marker>`{3,}|~{3,})")
_STRUCTURAL_LINE = re.compile(r"^[ \t]*(?:[>|]|!\[|\[\^[^\]\r\n]+\]:|\*\*Keywords:\*\*)", re.I)
_TABLE_DELIMITER = re.compile(r"^[ \t]*\|?[ \t]*:?-+:?(?:[ \t]*\|[ \t]*:?-+:?)*[ \t]*\|?[ \t]*$")
_TABLE_PIPE = re.compile(r"(?<!\\)\|")
_HTML_BLOCK_TAG = re.compile(
    r"^[ \t]*<(?P<closing>/)?(?P<tag>address|article|aside|blockquote|caption|dd|details|dialog|"
    r"div|dl|dt|fieldset|figcaption|figure|footer|form|h[1-6]|header|hr|li|main|nav|ol|p|"
    r"pre|script|section|style|summary|table|tbody|td|textarea|tfoot|th|thead|tr|ul)(?=[ \t/>]|$)",
    re.I,
)
_HTML_STANDALONE_TAG = re.compile(r"^[ \t]*<(?P<closing>/)?(?P<tag>[a-z][\w-]*)\s*/?>[ \t]*$", re.I)
_HTML_VOID_TAGS = frozenset("area base br col embed hr img input link meta param source track wbr".split())

# This label belongs to its prose line. Policy may preserve it when editing the body.
INLINE_SECTION_LABEL_RE = re.compile(
    r"\*\*(?P<label>Background|Methods|Results|Conclusions?|背景|方法|结果|结论)[:：]\*\*[ \t]*",
    re.I,
)

EVIDENCE_TOKEN_PATTERN = r"\{evidence:[^}\n]+\}"
EVIDENCE_LINK_PATTERN = r'\[[^\]]+\]\(evidence/[^\n)]*(?:"[^"]*")?\)'
CLAIM_MARKER_PATTERN = r"\[\^claim_\d+\]"
ANNOTATION_COMMENT_PATTERN = r"<!--.*?-->"
_AUDIT_MARKUP = "|".join(
    (EVIDENCE_TOKEN_PATTERN, EVIDENCE_LINK_PATTERN, CLAIM_MARKER_PATTERN, ANNOTATION_COMMENT_PATTERN)
)
AUDIT_MARKUP_RE = re.compile(_AUDIT_MARKUP, re.S)
# These references normalize across binding for sentence keys and headline checks.
AUDIT_SENTENCE_CITATIONS_RE = re.compile(rf"(?:{_AUDIT_MARKUP})(?:\s*;\s*(?:{_AUDIT_MARKUP}))*|\[@[^\]\n]+\]", re.S)
_COMMENT = re.compile(ANNOTATION_COMMENT_PATTERN, re.S)
_FOOTNOTE = r"\[\^[^\]\r\n]+\](?!:)"
_LINK = r'(?<!!)\[[^\]\r\n]+\]\((?:[^()\r\n]|\([^()\r\n]*\))*\)'
_CITATION = rf"(?:{_AUDIT_MARKUP}|{_FOOTNOTE}|{_LINK}|\[@[^\]\r\n]+\])"
SENTENCE_CITATIONS_RE = re.compile(rf"{_CITATION}(?:[ \t]*;[ \t]*{_CITATION})*", re.S)
_SPACE = re.compile(r"\s*")


def _blank(matches: re.Pattern[str], text: str) -> str:
    return matches.sub(lambda match: re.sub(r"[^\r\n]", " ", match[0]), text)


def _trailing_citation_end(text: str, end: int) -> int:
    while match := SENTENCE_CITATIONS_RE.match(text, _SPACE.match(text, end).end()):
        end = match.end()
    return end


@dataclass(frozen=True)
class ProseSentence:
    start: int
    end: int
    terminal: int | None


def prose_sentences(text: str) -> Iterator[ProseSentence]:
    """Locate prose in one region using source offsets, including attached citations."""
    visible = _blank(_COMMENT, text)
    punctuation = _blank(SENTENCE_CITATIONS_RE, text)
    start = 0
    for match in _PROSE_SENTENCE_END.finditer(punctuation):
        terminal = match.end()
        if terminal <= start:
            continue
        left = _SPACE.match(visible, start, terminal).end()
        end = _trailing_citation_end(text, terminal)
        if left < terminal:
            yield ProseSentence(left, end, terminal)
        start = end
    tail = visible[start:].rstrip()
    if tail:
        yield ProseSentence(_SPACE.match(tail).end() + start, start + len(tail), None)


@dataclass(frozen=True)
class ManuscriptRegion:
    start: int
    end: int
    kind: str
    heading: str | None = None
    complete: bool = True


def _html_block_stop(lines: Sequence[str], index: int) -> tuple[int, bool] | None:
    line = lines[index]
    if line.lstrip().startswith("<!--"):
        for last in range(index, len(lines)):
            if "-->" in lines[last]:
                # An inline annotation preceding prose stays in its sentence.
                if last == index and lines[last].partition("-->")[2].strip():
                    return None
                return last + 1, True
        return len(lines), False

    tag = _HTML_BLOCK_TAG.match(line) or _HTML_STANDALONE_TAG.match(line)
    if not tag:
        return None
    name = tag["tag"].lower()
    if tag["closing"] or line.rstrip().endswith("/>") or name in _HTML_VOID_TAGS:
        return index + 1, True
    closing = re.compile(rf"</{re.escape(tag['tag'])}[ \t]*>", re.I)
    raw_text = name in {"pre", "script", "style", "textarea"}
    for last in range(index, len(lines)):
        if closing.search(lines[last]):
            return last + 1, True
        if not raw_text and not lines[last].strip():
            return last, True
    return len(lines), False


def _table_rows(lines: Sequence[str]) -> set[int]:
    """Recognize pipe tables even when the Writer omits an outer pipe."""
    rows: set[int] = set()
    for index in range(1, len(lines)):
        if not (_TABLE_PIPE.search(lines[index - 1]) and _TABLE_PIPE.search(lines[index])
                and _TABLE_DELIMITER.fullmatch(lines[index])):
            continue
        last = index + 1
        while last < len(lines) and _TABLE_PIPE.search(lines[last]):
            last += 1
        rows.update(range(index - 1, last))
    return rows


def _continues_prose(line: str, following: str, *, numbered: bool) -> bool:
    if not following.strip() or line.endswith(("  ", "\\")):
        return False
    visible = _blank(SENTENCE_CITATIONS_RE, line).rstrip()
    if visible.endswith("."):
        joined = _blank(SENTENCE_CITATIONS_RE, line + "\n" + following)
        if not _PROSE_SENTENCE_END.match(joined, len(visible) - 1):
            return True
    if numbered or _STANDALONE_CLAIM.fullmatch(line) or _STANDALONE_CLAIM.fullmatch(following):
        return False
    return (
        not visible.endswith(tuple(".!?。！？"))
        or SENTENCE_CITATIONS_RE.match(following.lstrip()) is not None
    )


def manuscript_regions(text: str) -> Iterator[ManuscriptRegion]:
    """Return logical prose lines and atomic Markdown regions in source coordinates."""
    lines = [match[0] for match in re.finditer(r"[^\n]*\n|[^\n]+\Z", text)]
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    content = [line.rstrip("\r\n") for line in lines]
    table_rows = _table_rows(content)
    index = 0
    while index < len(lines):
        line = content[index]
        stop = index + 1
        heading = None
        complete = True
        if not line.strip():
            kind = "blank"
        elif fence := _FENCE.match(line):
            kind = "fence"
            marker = fence["marker"]
            close = re.compile(rf"[ \t]*{marker[0]}{{{len(marker)},}}[ \t]*")
            while stop < len(lines) and not close.fullmatch(content[stop]):
                stop += 1
            complete = stop < len(lines)
            stop += complete
        elif (html_stop := _html_block_stop(content, index)) is not None:
            kind = "html"
            stop, complete = html_stop
        elif match := _HEADING.match(line):
            kind = "heading"
            if match["level"] == "##":
                heading = match["title"]
        elif index in table_rows or _STRUCTURAL_LINE.match(line):
            kind = "structural"
        elif marker := _LIST_ITEM.match(line):
            kind = "structural"
            while stop < len(lines) and content[stop].strip():
                continuation = content[stop]
                if len(continuation) - len(continuation.lstrip(" \t")) < marker.end():
                    break
                stop += 1
        else:
            kind = "prose"
            while stop < len(lines):
                following = content[stop]
                listed = _LIST_ITEM.match(following)
                if (
                    stop in table_rows
                    or _HEADING.match(following)
                    or _STRUCTURAL_LINE.match(following)
                    or prose_label_end(following)
                    or _FENCE.match(following)
                    or _html_block_stop(content, stop) is not None
                    or (listed and not _NUMBERED_CONTINUATION.match(following))
                    or not _continues_prose(content[stop - 1], following, numbered=bool(listed))
                ):
                    break
                stop += 1
        yield ManuscriptRegion(offsets[index], offsets[stop - 1] + len(content[stop - 1]), kind, heading, complete)
        index = stop


def prose_label_end(text: str) -> int:
    start = len(text) - len(text.lstrip(" \t"))
    label = INLINE_SECTION_LABEL_RE.match(text, start)
    return label.end() if label else 0


@dataclass(frozen=True)
class ContextualSentenceDeletion:
    start: int
    end: int
    dependent_sentences: tuple[str, ...] = ()


def has_dependent_opener(paragraph: str) -> bool:
    """Recognize narrow anaphoric openers."""
    text = paragraph.lstrip()
    return bool(_DEPENDENT_OPENER.match(text[_trailing_citation_end(text, 0):].lstrip()))


def contextual_sentence_deletion(text: str, start: int, end: int) -> ContextualSentenceDeletion:
    """Extend a deletion through its citations and newly orphaned same-line prose."""
    if not 0 <= start < end <= len(text):
        raise ValueError("manuscript deletion span is outside the source text")

    paragraph_start = 0
    for region in manuscript_regions(text):
        if region.end <= start:
            if region.kind != "prose":
                paragraph_start = region.end
            continue
        if region.kind != "prose" or region.start > start or region.end < end:
            return ContextualSentenceDeletion(start, end)
        body = text[region.start:region.end]
        prefix = text[paragraph_start:region.start] + body[prose_label_end(body):start - region.start]
        cursor = region.start + _trailing_citation_end(body, end - region.start)
        if not _blank(SENTENCE_CITATIONS_RE, prefix).strip():
            dependent = []
            for sentence in prose_sentences(text[cursor:region.end]):
                sentence_start = cursor + sentence.start
                # An unpunctuated fragment or a new physical line needs a judge.
                if sentence.terminal is None or "\n" in text[end:sentence_start]:
                    break
                raw = text[sentence_start:cursor + sentence.end]
                if not has_dependent_opener(raw):
                    break
                dependent.append(raw)
                end = cursor + sentence.end
            if dependent:
                tail = text[end:region.end]
                end += len(tail) - len(tail.lstrip(" \t"))
                return ContextualSentenceDeletion(start, end, tuple(dependent))
        return ContextualSentenceDeletion(start, cursor)
    return ContextualSentenceDeletion(start, end)


def coalesced_sentence_deletions(
    text: str, ranges: Iterable[tuple[int, int]],
) -> Iterator[tuple[int, int, slice]]:
    """Merge adjacent sorted targets before considering whole-line whitespace."""
    indexed = enumerate(ranges)
    initial = next(indexed, None)
    if initial is None:
        return
    first, (group_start, group_end) = initial
    last = first
    for last, (start, end) in indexed:
        if start > group_end and text[group_end:start].strip(" \t"):
            yield group_start, group_end, slice(first, last)
            group_start, group_end, first = start, end, last
        else:
            group_end = max(group_end, end)
    yield group_start, group_end, slice(first, last + 1)


def sentence_deletion_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Consume the line ending only when the deletion owns the whole source line."""
    end = start + len(text[start:end].rstrip(" \t"))
    line_start = text.rfind("\n", 0, start) + 1
    newline = text.find("\n", end)
    line_end = newline if newline != -1 else len(text)
    if text[line_start:start].strip():
        return start, end
    tail = text[end:line_end]
    if tail.strip():
        return start, end + len(tail) - len(tail.lstrip(" \t"))
    return line_start, newline + 1 if newline != -1 else line_end
