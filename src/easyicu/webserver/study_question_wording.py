"""Read-only matching of a proposed question to host-captured user wording."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Mapping, Sequence

from easyicu.webserver.study_intent import contains_negation_or_exclusion

QUESTION_WORDING_FIELD = "question_wording"
WORDING_SOURCES = frozenset({"user_message", "clicked_option", "confirmed_proposal"})
_QUANTITATIVE_PUNCTUATION = frozenset("%‰‱-−/\\*")
_CLAUSE_BOUNDARY = re.compile(r"[。！？；，,.!?;:\r\n]")
_QUOTED_BLOCK = re.compile(
    r'''"([^"]*)"|「([^」]*)」|『([^』]*)』|“([^”]*)”|‘([^’]*)’|(?<!\w)'([^']*)'|`([^`]*)`''',
    re.DOTALL,
)


def normalize_question_text(question: str) -> str:
    """Use the study store's whitespace normalization for wording digests."""
    return " ".join(question.split())


def question_sha256(question: str) -> str:
    return hashlib.sha256(normalize_question_text(str(question)).encode("utf-8")).hexdigest()


def _key(text: str) -> tuple[str, list[int]]:
    indices = [
        index for index, char in enumerate(text)
        if not char.isspace() and (
            not unicodedata.category(char).startswith("P")
            or char in _QUANTITATIVE_PUNCTUATION
            # Decimal points, numeric separators and ratio/time colons carry
            # quantities rather than sentence punctuation.
            or (index > 0 and index + 1 < len(text)
                and text[index - 1].isdigit() and text[index + 1].isdigit())
        )
    ]
    return "".join(text[index] for index in indices), indices


def _safe_cut(text: str, start: int, stop: int, *, whole_clause: bool) -> bool:
    prefix = _CLAUSE_BOUNDARY.split(text[:start])[-1]
    suffix = _CLAUSE_BOUNDARY.split(text[stop:], maxsplit=1)[0]
    if contains_negation_or_exclusion(prefix) or contains_negation_or_exclusion(suffix):
        return False
    if whole_clause:
        if not _key(prefix)[0] and not _key(suffix)[0]:
            return True
        # Only paired quotes around the complete wording count. An apostrophe
        # or an adjacent quoted phrase must not authorize a sentence fragment.
        for match in _QUOTED_BLOCK.finditer(text):
            for index, value in enumerate(match.groups(), 1):
                if value is None:
                    continue
                left, right = match.span(index)
                if (
                    left <= start and stop <= right
                    and not _key(text[left:start])[0] and not _key(text[stop:right])[0]
                ):
                    return True
        return False
    return True


def _match(proposal: str, texts: Sequence[str], *, whole_clause: bool = False) -> str | None:
    needle, _ = _key(proposal)
    if len(needle) < 8:
        return None
    rows = [(text, *_key(text)) for text in texts]
    haystack = "".join(key for _, key, _ in rows)
    begin = haystack.find(needle)
    while begin >= 0:
        end, offset, pieces = begin + len(needle), 0, []
        for text, key, indices in rows:
            left, right = max(0, begin - offset), min(len(key), end - offset)
            if left < right:
                if not _safe_cut(text, indices[left], indices[right - 1] + 1, whole_clause=whole_clause):
                    break
                start = 0 if left == 0 else indices[left]
                stop = len(text) if right == len(key) else indices[right - 1] + 1
                pieces.append(text[start:stop].strip())
            offset += len(key)
        else:
            return "\n".join(pieces)
        begin = haystack.find(needle, begin + 1)
    return None


def _explicitly_confirms_wording(message: str) -> bool:
    for clause in re.split(r"(?<=[。.!！?？;；,，])", str(message or "")):
        # Require a direct selection rather than two words found in advice,
        # quoted history, a question, a denial, or an assertion about validity.
        text = clause.strip()
        if re.fullmatch(
            r"(?:(?:我(?:们)?(?:想|要)?|请|现在)\s*)*"
            r"(?:确认|采用|就用|使用)(?:上述|这个|该|这份|刚才的)?(?:研究)?"
            r"(?:问题(?:的?措辞|的?表述)?|题目|措辞|定义|表述)"
            r"(?:即可|就这样)?[。！.!;；,，]*",
            text,
        ) or re.fullmatch(
            r"(?:(?:please|i|we|let['’]s|(?:i|we)\s+(?:will|want to|would like to))\s+)*"
            r"(?:confirm|use|adopt)\s+(?:(?:this|that|the|proposed|above|revised)\s+)*"
            r"(?:question(?:\s+wording)?|wording|definition)"
            r"(?:\s+as\s+(?:written|shown|proposed))?[.!;,]*",
            text, re.IGNORECASE,
        ):
            return True
    return False


def resolve_question_wording(
    proposal: str, *, user_turns: Sequence[Mapping[str, Any]], previous_reply: str = "",
) -> tuple[str, dict[str, str]] | None:
    """Return original text plus provenance, without interpreting its science.

    Unknown, superseded and host-generated turns break adjacency. A clicked
    model option confirms only wording inside that option; a plain affirmative
    cannot authorize a reformulation from an assistant reply.
    """

    turns = list(user_turns)
    latest = turns[-1] if turns else {}
    source = "user_message"
    wording = None
    if latest.get("message_origin") == "model_option" and not latest.get("superseded"):
        wording = _match(proposal, [str(latest.get("text") or "")])
        source = "clicked_option"
    if wording is None:
        block: list[str] = []
        for turn in turns:
            if turn.get("message_origin") != "typed" or turn.get("superseded"):
                block = []
                continue
            block.append(str(turn.get("text") or ""))
            found = _match(proposal, block)
            if found is not None:
                wording = found
                source = "user_message"
    if (
        wording is None and latest.get("message_origin") in {"typed", "model_option"}
        and not latest.get("superseded")
        and _explicitly_confirms_wording(str(latest.get("text") or ""))
    ):
        wording = _match(proposal, [previous_reply], whole_clause=True)
        source = "confirmed_proposal"
    if not wording:
        return None
    return wording, {"source": source, "question_sha256": question_sha256(wording)}


def normalize_question_wording(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if (
        not isinstance(value, Mapping)
        or set(value) != {"source", "question_sha256"}
        or value.get("source") not in WORDING_SOURCES
        or re.fullmatch(r"[a-f0-9]{64}", str(value.get("question_sha256") or "")) is None
    ):
        raise ValueError("question wording requires a host source and question digest")
    return {"source": str(value["source"]), "question_sha256": str(value["question_sha256"])}
