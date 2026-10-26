"""Path-free search documents; these are UI coordinates, never run authority."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

MAX_TEXT = 12_000
MAX_QUERY = 200
MAX_HITS = 50
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\U00020000-\U0003134f]")


def normalized(text: str) -> str:
    # Per-character normalization also permits an exact map back to UI offsets.
    return "".join(unicodedata.normalize("NFKC", ch).casefold() for ch in text)


def terms(text: str) -> tuple[str, ...]:
    """Latin words and CJK runs, with punctuation treated as separators."""
    return tuple(_WORDS.findall(normalized(text)))


def tokens(text: str) -> set[str]:
    """Index CJK unigrams/bigrams without a segmenter or model dependency."""
    result = set()
    for term in terms(text):
        for word in re.findall(r"[\u3400-\u9fff\uf900-\ufaff\U00020000-\U0003134f]+|[^\u3400-\u9fff\uf900-\ufaff\U00020000-\U0003134f]+", term):
            if _CJK.match(word):
                result.update(word)
                result.update(word[i:i + 2] for i in range(len(word) - 1))
            else:
                result.add(word)
    return result


def text_value(value: object) -> str:
    return value[:MAX_TEXT] if isinstance(value, str) else ""


@dataclass(frozen=True, slots=True)
class SearchDocument:
    project_id: str
    kind: str
    title: str
    text: str
    target: dict

    @property
    def scope(self) -> str:
        return "projects" if self.kind == "project" else "files" if self.kind == "file" else "tasks"

    @property
    def identity(self) -> str:
        import json

        return hashlib.sha256(json.dumps([self.kind, self.target], sort_keys=True).encode()).hexdigest()

    @property
    def searchable(self) -> str:
        return f"{self.title[:240]}\n{self.text[:MAX_TEXT]}"

    def hit(self, query: str) -> dict:
        original = self.searchable
        folded = normalized(original)
        positions = []
        for index, ch in enumerate(original):
            positions.extend([index] * len(normalized(ch)))
        spans = []
        for term in terms(query):
            start = folded.find(term)
            if start >= 0:
                spans.append((positions[start], positions[start + len(term) - 1] + 1))
        start = max(0, min((a for a, _ in spans), default=0) - 60)
        snippet = original[start:start + 240]

        def utf16(value):
            return len(value.encode("utf-16-le")) // 2

        matches = [
            {"start": utf16(original[start:a]), "end": utf16(original[start:b])}
            for a, b in sorted(set(spans)) if start <= a < b <= start + len(snippet)
        ]
        return {
            "id": self.identity, "kind": self.kind, "scope": self.scope,
            "project_id": self.project_id, "title": self.title[:240],
            "conversation_id": self.target.get("conversation_id"),
            "message_id": self.target.get("message_id"), "file_id": self.target.get("file_id"),
            "snippet": snippet, "snippet_offset": utf16(original[:start]),
            "matches": matches, "offset_unit": "utf16", "open": self.target,
        }
