"""Filesystem-only output helpers shared by pipeline and execute layers."""

from __future__ import annotations

import shutil
from pathlib import Path


def _has_figure_exports(out_dir: Path) -> bool:
    figure_suffixes = {".png", ".svg", ".pdf", ".tiff", ".tif", ".pptx"}
    return any(
        path.is_file() and path.suffix.lower() in figure_suffixes
        for path in out_dir.iterdir()
    )


def _clear_output_dir(out_dir: Path) -> None:
    """Recreate a step output directory without following untrusted symlinks."""

    # Generated code may replace the output leaf itself with a symlink.  Using
    # ``exists``/``iterdir`` first would follow that link and could delete an
    # arbitrary host directory during repair.  Remove any non-directory leaf
    # lexically, then create the expected directory in its place.
    if out_dir.is_symlink() or (out_dir.exists() and not out_dir.is_dir()):
        out_dir.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in out_dir.iterdir():
        if child.is_symlink() or not child.is_dir():
            child.unlink(missing_ok=True)
        else:
            shutil.rmtree(child)


__all__ = ["_clear_output_dir", "_has_figure_exports"]
