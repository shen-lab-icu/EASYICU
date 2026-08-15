"""Extractive, axis-covering source excerpt selection for literature screening."""

from __future__ import annotations

import re
from typing import Iterable


def select_source_backed_excerpt(
    text: str,
    *,
    focus_terms: Iterable[str] = (),
    design_terms: Iterable[str] = (),
    max_sentences: int = 5,
    max_chars: int = 1_200,
) -> str:
    """Select source sentences that cover declared axes before design context.

    Selection remains wholly extractive.  One earliest matching sentence is
    retained for each distinct focus term in caller-supplied priority order;
    remaining capacity is filled with design-bearing source sentences.  This
    prevents several exposure synonyms near the start of an abstract from
    crowding the outcome sentence out of the bounded receipt.
    """

    normalized = " ".join(str(text or "").split())
    if not normalized:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    sentence_axes = [_clinical_text(sentence) for sentence in sentences]
    normalized_focus = list(
        dict.fromkeys(
            term
            for term in (_clinical_text(value) for value in focus_terms)
            if term
        )
    )
    selected: list[str] = []
    for focus in normalized_focus:
        for sentence, sentence_axis in zip(sentences, sentence_axes):
            if focus in sentence_axis and sentence not in selected:
                selected.append(sentence)
                break
    folded_design_terms = tuple(
        value.casefold().strip() for value in design_terms if value.strip()
    )
    for sentence in sentences:
        if sentence in selected:
            continue
        folded = sentence.casefold()
        if any(term in folded for term in folded_design_terms):
            selected.append(sentence)
    excerpt = " ".join(selected[: max(1, int(max_sentences))]) or " ".join(
        sentences[:2]
    )
    return excerpt[: max(1, int(max_chars))].rstrip()


def _clinical_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


__all__ = ["select_source_backed_excerpt"]
