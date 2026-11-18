"""Scientific product formats and the host run delivery/export contract.

The fixed download boundary and planning resolve the same typed outputs here;
delivery promises no scientific result, patient export or publication readiness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Sequence


FIGURE_SUFFIXES = frozenset({".png", ".svg", ".pdf", ".tif", ".tiff"})
KNOWN_FILE_SUFFIXES = frozenset(
    {
        *FIGURE_SUFFIXES,
        ".csv",
        ".tsv",
        ".parquet",
        ".feather",
        ".json",
        ".jsonl",
        ".md",
        ".txt",
        ".log",
        ".pkl",
        ".pickle",
        ".joblib",
        ".npy",
        ".npz",
    }
)

_AUDIT_PHYSICAL_FILE_KINDS = frozenset({"log", "table"})
_REPORT_PHYSICAL_FILE_KINDS = frozenset({"log"})


def file_kinds(value: object) -> frozenset[str]:
    """Project a physical suffix to the closed product-kind vocabulary."""

    suffix = Path(str(value or "").strip()).suffix.lower()
    if suffix in FIGURE_SUFFIXES:
        return frozenset({"figure"})
    if suffix in {".csv", ".tsv"}:
        return frozenset({"table", "artifact", "dataset", "test"})
    if suffix in {".parquet", ".feather"}:
        return frozenset({"artifact", "dataset", "table"})
    if suffix in {".pkl", ".pickle", ".joblib"}:
        return frozenset({"model", "artifact"})
    if suffix in {".npy", ".npz"}:
        return frozenset({"artifact", "dataset", "model"})
    if suffix in {".md", ".txt", ".log", ".jsonl"}:
        return frozenset({"log", "artifact"})
    if suffix == ".json":
        return frozenset({"artifact", "manifest", "log", "model", "test"})
    return frozenset()


def descriptor_path_is_compatible(*, kind: str, path: str) -> bool:
    """Return whether an exact typed descriptor can use ``path``.

    ``audit`` and ``report`` are semantic roles rather than suffixes.  They are
    authorized only by explicit typed descriptors while their physical payloads
    use ordinary table/log formats; suffix inference alone never creates them.
    """

    physical_kinds = file_kinds(path)
    if kind == "audit":
        return bool(physical_kinds & _AUDIT_PHYSICAL_FILE_KINDS)
    if kind == "report":
        return bool(physical_kinds & _REPORT_PHYSICAL_FILE_KINDS)
    return kind in physical_kinds


def contained_regular_output_file(out_dir: Path, value: object) -> Path:
    """Return one link-free regular file lexically contained in ``out_dir``.

    This helper runs inside the macOS execution sandbox.  Calling
    :meth:`Path.resolve` there asks the kernel to inspect ancestors such as
    ``/private`` that are intentionally outside the sandbox allowlist.  The
    host already supplies the output root, so containment is enforced without
    ancestor traversal. Output-local absolute aliases are first reduced
    lexically to a relative path; outside paths and parent components remain
    invalid. Every output-local component is then checked for symlinks, and
    hard-linked files are rejected.
    """

    root = Path(out_dir)
    raw = str(value or "").strip()
    supplied = Path(raw)
    if supplied.is_absolute():
        try:
            relative = supplied.relative_to(root.absolute())
        except ValueError as exc:
            raise ValueError("output path must be contained") from exc
    else:
        relative = supplied
    if not raw or ".." in relative.parts:
        raise ValueError("output path must be contained")
    try:
        if root.is_symlink() or not root.is_dir():
            raise ValueError("output root is not a regular directory")
        candidate = root
        for part in relative.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError("output path traverses a symlink")
        stat_result = candidate.stat()
    except OSError as exc:
        raise ValueError("output file is unavailable") from exc
    if not candidate.is_file() or stat_result.st_nlink != 1:
        raise ValueError("output path is not a single-link regular file")
    return candidate


RUN_DELIVERY_SCHEMA_VERSION = "easyicu.run_delivery/1"
HOST_DELIVERY_REASON = "question_requirement_covered_by_host_delivery"
DeliverableKind = Literal[
    "article_preview", "manuscript_pdf", "result_tables", "figures", "references",
]

# The fixed browser document boundary, previously owned by agent_runs. Keeping
# it here lets planning read the same capabilities without importing webserver.
RUN_DOCUMENT_SPECS = MappingProxyType({
    "manuscript_revision.pdf": {"media_type": "application/pdf", "max_bytes": 16 * 1024 * 1024},
    "manuscript_scaffold.pdf": {"media_type": "application/pdf", "max_bytes": 16 * 1024 * 1024},
    "manuscript_scaffold.tex": {"media_type": "text/x-tex; charset=utf-8", "max_bytes": 2 * 1024 * 1024},
    "manuscript_scaffold.bib": {"media_type": "application/x-bibtex; charset=utf-8", "max_bytes": 2 * 1024 * 1024},
    "system_validation_report.html": {"media_type": "text/html; charset=utf-8", "max_bytes": 8 * 1024 * 1024},
    "system_validation_report.pdf": {"media_type": "application/pdf", "max_bytes": 16 * 1024 * 1024},
})


@dataclass(frozen=True)
class RunDeliverable:
    artifact_names: tuple[str, ...]
    # Compatibility reader for stored requirements predating typed kinds.
    # Only a whole delivery-only request is accepted, never a substring of a
    # scientific requirement. The Planner's prose about feasibility is ignored.
    request_names: str


RUN_DELIVERABLES: Mapping[DeliverableKind, RunDeliverable] = MappingProxyType({
    "article_preview": RunDeliverable(
        ("manuscript_draft.json", "manuscript_provenance.json"),
        r"(?:web|html|browser|online|网页|在线|浏览器)\s*"
        r"(?:(?:article|manuscript|report|文章|论文|报告)\s*)?(?:preview|预览)"
        r"|(?:article|manuscript|report)\s+text|正文",
    ),
    "manuscript_pdf": RunDeliverable(
        ("manuscript_scaffold.pdf", "manuscript_revision.pdf", "manuscript_pdf_receipt.json"),
        r"pdf",
    ),
    "result_tables": RunDeliverable(
        ("result_tables.json",), r"(?:result\s+)?tables?|结果表(?:格)?|表格",
    ),
    "figures": RunDeliverable(
        ("figure_gallery.json",), r"figures?|图表|图件|图形|图片",
    ),
    "references": RunDeliverable(
        ("manuscript_scaffold.bib",),
        r"reference\s+list|references|bibliography|citations?"
        r"|参考文献(?:清单|列表)?|引用(?:清单|列表)?|文献列表",
    ),
})
RUN_DELIVERY_ARTIFACT_NAMES = tuple(dict.fromkeys(
    name for spec in RUN_DELIVERABLES.values() for name in spec.artifact_names
    if name not in RUN_DOCUMENT_SPECS
))

# Neutral delivery grammar only. Unknown words (methods, data requests,
# destinations, formats, etc.) leave a residue and keep ordinary gap handling.
_DELIVERY_GRAMMAR = re.compile(
    r"\b(?:please|provide|include|deliver|a|an|the|and|or|with|downloadable|"
    r"download|manuscript|article|report|files?|readable)\b"
    r"|(?:请|提供|包含|交付|以及|并|和|及|与|或|可下载|下载|文件|信息充分|清晰|且|可读|的)",
    re.IGNORECASE,
)
_SEPARATORS = re.compile(r"[\s,，、;；。.：:()（）/\-]+")


def requested_deliverable_kinds(quote: str) -> tuple[DeliverableKind, ...]:
    """Translate a complete delivery request to typed kinds, including old records.

    Scientific/mixed clauses and unknown deliveries deliberately resolve to
    nothing, so mentioning a PDF cannot hide an unavailable method or variable.
    """
    remainder = quote
    kinds: list[DeliverableKind] = []
    for kind, spec in RUN_DELIVERABLES.items():
        pattern = re.compile(r"(?<![a-z])(?:" + spec.request_names + r")(?![a-z])", re.IGNORECASE)
        remainder, count = pattern.subn(" ", remainder)
        if count:
            kinds.append(kind)
    remainder = _DELIVERY_GRAMMAR.sub(" ", remainder)
    return tuple(kinds) if kinds and not _SEPARATORS.sub("", remainder) else ()


def host_delivery_coverage(owners: Mapping[DeliverableKind, Sequence[str]]) -> dict[str, object] | None:
    """Describe host delivery backed by a plan's typed product owners."""
    if not owners or not all(steps and kind in RUN_DELIVERABLES for kind, steps in owners.items()):
        return None
    return {
        "source": "host_verified",
        "contract": RUN_DELIVERY_SCHEMA_VERSION,
        "deliverable_kinds": list(owners),
        "artifact_names": list(dict.fromkeys(
            name for kind in owners for name in RUN_DELIVERABLES[kind].artifact_names
        )),
        "owner_step_ids_by_kind": {kind: list(steps) for kind, steps in owners.items()},
        "reason": "The host verified a typed owner for each requested deliverable in this plan.",
    }


__all__ = [
    "DeliverableKind",
    "HOST_DELIVERY_REASON",
    "RUN_DELIVERABLES",
    "RUN_DELIVERY_ARTIFACT_NAMES",
    "RUN_DELIVERY_SCHEMA_VERSION",
    "RUN_DOCUMENT_SPECS",
    "host_delivery_coverage",
    "requested_deliverable_kinds",
    "FIGURE_SUFFIXES",
    "KNOWN_FILE_SUFFIXES",
    "contained_regular_output_file",
    "descriptor_path_is_compatible",
    "file_kinds",
]
