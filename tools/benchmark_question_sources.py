"""Path-only allow-list for benchmark question sources."""

from __future__ import annotations

from os import PathLike
from pathlib import PurePath

ALLOWED_QUESTION_SOURCES = frozenset(
    {
        "tests/",
        "benchmarks/figure2_icu_agent_v2/formal_safety12_taskbank_v1.jsonl",
        "benchmarks/meta_generalization/meta_benchmark.jsonl",
        "benchmarks/figure2_canonical9/canonical_input_freeze_v1.json",
    }
)

_REPO_ROOT = PurePath(__file__).parents[1]
_HELD_OUT_PREFIXES = ("heldout", "ho30")


class HeldOutSourceRefused(ValueError):
    """A benchmark question source refused with a stable reason code."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def require_allowed_question_source(path: str | PathLike[str]) -> None:
    """Check a repository-relative or absolute source path without filesystem I/O."""
    source = PurePath(path)
    if any(part.casefold().startswith(_HELD_OUT_PREFIXES) for part in source.parts):
        raise HeldOutSourceRefused("held_out_source_refused")

    if source.is_absolute() and source.is_relative_to(_REPO_ROOT):
        source = source.relative_to(_REPO_ROOT)

    if ".." not in source.parts:
        relative = source.as_posix()
        for allowed in ALLOWED_QUESTION_SOURCES:
            if relative == allowed or (allowed.endswith("/") and relative.startswith(allowed)):
                return

    raise HeldOutSourceRefused("question_source_not_allow_listed")
