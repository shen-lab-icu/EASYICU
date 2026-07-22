"""Opt-in cross-run learning components.

Canonical submission profiles keep both memory systems disabled.  The package
exists for explicit exploratory use and does not import either implementation
eagerly.
"""

from __future__ import annotations

from .store import (
    FileSystemMemoryStore,
    LangGraphMemoryStoreAdapter,
    MemoryAccessPolicy,
    MemoryObject,
    MemoryPromotionReceipt,
    MemoryReviewAttestation,
    promote_quarantined_memory,
    quarantine_run_lesson,
    select_memory,
)
from .runtime import (
    REVIEWED_MEMORY_PROMPT_LIMIT_BYTES,
    ReviewedMemoryBudgetError,
    ReviewedMemoryBundle,
    ReviewedMemoryIntegrityError,
    ReviewedMemoryRuntime,
    attach_step_reviewed_memory,
    build_reviewed_memory_bundle,
)

__all__ = [
    "FileSystemMemoryStore",
    "LangGraphMemoryStoreAdapter",
    "MemoryAccessPolicy",
    "MemoryObject",
    "MemoryPromotionReceipt",
    "MemoryReviewAttestation",
    "REVIEWED_MEMORY_PROMPT_LIMIT_BYTES",
    "ReviewedMemoryBudgetError",
    "ReviewedMemoryBundle",
    "ReviewedMemoryIntegrityError",
    "ReviewedMemoryRuntime",
    "attach_step_reviewed_memory",
    "build_reviewed_memory_bundle",
    "promote_quarantined_memory",
    "quarantine_run_lesson",
    "select_memory",
]
