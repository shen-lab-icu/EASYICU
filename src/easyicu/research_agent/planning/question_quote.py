"""Find contiguous requirement quotes and recover their original question offsets."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from string import ascii_letters, digits
from typing import Iterator


QUOTE_NOT_IN_QUESTION = "question_requirement_quote_not_in_question"
QUESTION_QUOTE_RULE = (
    "Copy one contiguous span from the research question. Matching folds Unicode "
    "NFKC (including width), case, typographic quote marks and the ideographic full "
    "stop. Whitespace runs collapse and spaces next to CJK text or comparison "
    "operators are optional. Keep ASCII letter and digit runs and combining marks "
    "intact. Keep the wording, order and punctuation."
)
_PUNCTUATION = str.maketrans({
    **dict.fromkeys("“”「」『』", '"'),
    **dict.fromkeys("‘’", "'"),
    "。": ".",
})
_QUOTE_PARTS = re.compile(r"\s+|\S")
_CJK_NAMES = ("CJK ", "HIRAGANA ", "KATAKANA ", "HANGUL ", "BOPOMOFO ")


def _optional_space_next_to(char: str) -> bool:
    return char in "<>=≤≥" or unicodedata.name(char, "").startswith(_CJK_NAMES)


def _quote_parts(text: str) -> Iterator[tuple[str, int, int]]:
    for match in _QUOTE_PARTS.finditer(text):
        start, end = match.span()
        part = match[0]
        if part.isspace():
            if (
                start == 0 or end == len(text)
                or _optional_space_next_to(text[start - 1])
                or _optional_space_next_to(text[end])
            ):
                continue
            part = " "
        yield part.casefold().translate(_PUNCTUATION), start, end


def normalize_question_quote(text: str) -> str:
    """Apply the comparison in QUESTION_QUOTE_RULE without dropping punctuation."""
    return "".join(part for part, _, _ in _quote_parts(unicodedata.normalize("NFKC", text)))


def _is_source_boundary(text: str, offset: int) -> bool:
    """Keep combining marks with their bases and match whole ASCII runs."""
    if offset == len(text):
        return True
    after = text[offset]
    if unicodedata.category(after).startswith("M"):
        return False
    if offset == 0:
        return True
    before = text[offset - 1]
    return not (
        (before in ascii_letters and after in ascii_letters)
        or (before in digits and after in digits)
    )


def _decomposed_offsets(text: str) -> Iterator[int]:
    """Offsets in NFKD code points, whose count is invariant under NFKC."""
    offset = 0
    yield offset
    for char in text:
        offset += len(unicodedata.normalize("NFKD", char))
        yield offset


@dataclass(frozen=True)
class QuestionQuoteSpan:
    """Exact source text and its zero-based, half-open Python string offsets."""

    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if (
            type(self.start) is not int or type(self.end) is not int
            or not isinstance(self.text, str)
            or self.start < 0 or self.end <= self.start
            or self.end - self.start != len(self.text)
        ):
            raise ValueError("invalid question quote span")


class QuestionQuoteMatcher:
    """Normalize a question once and locate each quote at original boundaries."""

    def __init__(self, question: str) -> None:
        self.question = question
        self.starts: dict[int, int] = {}
        self.ends: dict[int, int] = {}
        folded = unicodedata.normalize("NFKC", question)
        original = {
            offset: index for index, offset in enumerate(_decomposed_offsets(question))
            if _is_source_boundary(question, index)
        }
        offsets = [original.get(offset) for offset in _decomposed_offsets(folded)]
        parts = []
        cursor = 0
        for part, start, end in _quote_parts(folded):
            if (source_start := offsets[start]) is not None:
                self.starts[cursor] = source_start
            cursor += len(part)
            if (source_end := offsets[end]) is not None:
                self.ends[cursor] = source_end
            parts.append(part)
        self.normalized = "".join(parts)

    def find(self, quote: str, *, offset: int = 0) -> QuestionQuoteSpan | None:
        """Return the first matching original span starting at or after offset."""
        needle = normalize_question_quote(quote)
        if not needle:
            return None
        position = self.normalized.find(needle)
        while position >= 0:
            start = self.starts.get(position)
            end = self.ends.get(position + len(needle))
            if start is not None and start >= offset and end is not None:
                text = self.question[start:end]
                # Canonical reordering can preserve lengths while moving marks.
                if normalize_question_quote(text) == needle:
                    return QuestionQuoteSpan(start, end, text)
            position = self.normalized.find(needle, position + 1)
        return None
