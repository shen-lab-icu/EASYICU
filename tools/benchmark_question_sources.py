"""Path-only allow-list for benchmark question sources."""

from __future__ import annotations

from os import PathLike
from pathlib import Path, PurePath

ALLOWED_QUESTION_SOURCES = frozenset(
    {
        "tests/",
        "benchmarks/figure2_icu_agent_v2/formal_safety12_taskbank_v1.jsonl",
        "benchmarks/meta_generalization/meta_benchmark.jsonl",
        "benchmarks/figure2_canonical9/canonical_input_freeze_v1.json",
    }
)

_REPO_ROOT = PurePath(__file__).parents[1]
_RESOLVED_REPO_ROOT = Path(__file__).resolve().parents[1]
_HELD_OUT_MARKERS = ("heldout", "ho30")


class HeldOutSourceRefused(ValueError):
    """A benchmark question source refused with a stable reason code."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def require_allowed_question_source(path: str | PathLike[str]) -> None:
    """Check a repository-relative or absolute source path without filesystem I/O."""
    source = PurePath(path)
    for root in (_REPO_ROOT, _RESOLVED_REPO_ROOT):
        if source.is_relative_to(root):
            source = source.relative_to(root)
            break

    if any(
        marker in part.casefold()
        for part in source.parts
        for marker in _HELD_OUT_MARKERS
    ):
        raise HeldOutSourceRefused("held_out_source_refused")

    if ".." not in source.parts:
        relative = source.as_posix()
        for allowed in ALLOWED_QUESTION_SOURCES:
            if relative == allowed or (
                allowed.endswith("/") and relative.startswith(allowed)
            ):
                return

    raise HeldOutSourceRefused("question_source_not_allow_listed")
