"""Physical file-format contract for declared scientific products."""

from __future__ import annotations

from pathlib import Path


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


__all__ = [
    "FIGURE_SUFFIXES",
    "KNOWN_FILE_SUFFIXES",
    "descriptor_path_is_compatible",
    "file_kinds",
]
