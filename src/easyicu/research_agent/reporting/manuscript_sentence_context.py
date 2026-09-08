"""Compatibility exports for the shared sentence-context contract."""

from ..contracts.manuscript_sentence_context import (
    ContextualSentenceDeletion,
    contextual_sentence_deletion,
    has_dependent_opener,
)

__all__ = [
    "ContextualSentenceDeletion",
    "contextual_sentence_deletion",
    "has_dependent_opener",
]
