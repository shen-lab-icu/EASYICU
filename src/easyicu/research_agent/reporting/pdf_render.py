"""Render the manuscript LaTeX scaffold into a PDF (optional).

The pipeline already writes ``manuscript_scaffold.tex`` plus a
``manuscript_scaffold.bib`` next to it. PDF rendering is opt-in
because not every CI environment has TeXLive; users who want a PDF
call :func:`render_pdf_for_run` (or pass
``enable_pdf_render=True`` to ``ResearchAgentPipeline``).

Engine resolution order:

1. ``latexmk`` — best, runs bibtex automatically.
2. ``tectonic`` — single-binary, downloads packages on demand.
3. ``xelatex`` then ``pdflatex`` — falls back to a hand-driven 3-pass
   sequence (latex → bibtex → latex → latex) for environments that
   only have the basic engines.

When no engine is available the function returns ``None`` and emits
a finding so the pipeline can record it without raising.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class PDFRenderResult:
    pdf_path: Optional[Path]
    log_path: Optional[Path]
    engine: Optional[str]
    success: bool
    notes: List[str]


def _which_first(*candidates: str) -> Optional[str]:
    for cand in candidates:
        if shutil.which(cand):
            return cand
    return None


def _run_with_log(
    cmd: List[str], *, cwd: Path, timeout: float = 240.0
) -> Tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        timeout=timeout,
    )
    stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
    stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
    return proc.returncode, stdout + "\n" + stderr


def render_pdf_for_run(
    *,
    tex_path: Path,
    bib_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    timeout: float = 240.0,
) -> PDFRenderResult:
    """Compile a LaTeX scaffold to PDF.

    Args:
        tex_path: ``manuscript_scaffold.tex`` path.
        bib_path: ``manuscript_scaffold.bib`` path (sibling).
        output_dir: where to write the PDF. Defaults to ``tex_path.parent``.
        timeout: per-engine wall clock seconds.

    Returns a :class:`PDFRenderResult`. ``success=False`` means no
    engine was available or the run failed; ``pdf_path`` is None in
    that case.
    """
    tex_path = Path(tex_path)
    if not tex_path.exists():
        return PDFRenderResult(
            pdf_path=None,
            log_path=None,
            engine=None,
            success=False,
            notes=[f"tex file not found: {tex_path}"],
        )
    out_dir = Path(output_dir) if output_dir is not None else tex_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: List[str] = []

    # Copy bib + tex into out_dir if they live elsewhere; the LaTeX
    # engines all read inputs relative to their cwd, so we run them
    # from out_dir with the file's basename.
    tex_path = tex_path.resolve()
    out_dir = out_dir.resolve()
    if tex_path.parent != out_dir:
        target_tex = out_dir / tex_path.name
        target_tex.write_bytes(tex_path.read_bytes())
        tex_in_out = target_tex
    else:
        tex_in_out = tex_path
    tex_basename = tex_in_out.name

    # Copy bib alongside the tex if missing in out_dir.
    if bib_path is not None and bib_path.exists():
        target_bib = out_dir / bib_path.name
        if target_bib.resolve() != bib_path.resolve():
            target_bib.write_bytes(bib_path.read_bytes())

    # Engine 1: latexmk — handles bibtex automatically.
    latexmk = shutil.which("latexmk")
    if latexmk:
        rc, log = _run_with_log(
            [
                latexmk,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_basename,
            ],
            cwd=out_dir,
            timeout=timeout,
        )
        log_path = out_dir / (tex_in_out.stem + ".pdfrender.log")
        log_path.write_text(log, encoding="utf-8")
        pdf_path = out_dir / (tex_in_out.stem + ".pdf")
        if rc == 0 and pdf_path.exists():
            return PDFRenderResult(
                pdf_path=pdf_path,
                log_path=log_path,
                engine="latexmk",
                success=True,
                notes=notes,
            )
        notes.append(f"latexmk failed (rc={rc}); see {log_path}")

    # Engine 2: tectonic
    tectonic = shutil.which("tectonic")
    if tectonic:
        rc, log = _run_with_log(
            [tectonic, "-X", "compile", "--keep-logs", tex_basename],
            cwd=out_dir,
            timeout=timeout,
        )
        log_path = out_dir / (tex_in_out.stem + ".pdfrender.log")
        log_path.write_text(log, encoding="utf-8")
        pdf_path = out_dir / (tex_in_out.stem + ".pdf")
        if rc == 0 and pdf_path.exists():
            return PDFRenderResult(
                pdf_path=pdf_path,
                log_path=log_path,
                engine="tectonic",
                success=True,
                notes=notes,
            )
        notes.append(f"tectonic failed (rc={rc}); see {log_path}")

    # Engine 3: hand-driven xelatex/pdflatex + bibtex.
    engine = _which_first("xelatex", "pdflatex")
    if engine is None:
        return PDFRenderResult(
            pdf_path=None,
            log_path=None,
            engine=None,
            success=False,
            notes=notes
            + [
                "No LaTeX engine available. Install MacTeX, TeXLive, "
                "MikTeX, or `pip install pylatex` + tectonic."
            ],
        )
    bibtex = shutil.which("bibtex")
    log_lines: List[str] = []
    for pass_idx in range(2):
        rc, log = _run_with_log(
            [
                engine,
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_basename,
            ],
            cwd=out_dir,
            timeout=timeout,
        )
        log_lines.append(f"=== {engine} pass {pass_idx + 1} (rc={rc}) ===")
        log_lines.append(log)
        if rc != 0:
            break
        if bibtex and pass_idx == 0:
            aux = out_dir / (tex_in_out.stem + ".aux")
            if aux.exists():
                rc_b, log_b = _run_with_log(
                    [bibtex, tex_in_out.stem],
                    cwd=out_dir,
                    timeout=timeout,
                )
                log_lines.append(f"=== bibtex (rc={rc_b}) ===")
                log_lines.append(log_b)
    log_path = out_dir / (tex_in_out.stem + ".pdfrender.log")
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    pdf_path = out_dir / (tex_in_out.stem + ".pdf")
    if pdf_path.exists():
        return PDFRenderResult(
            pdf_path=pdf_path,
            log_path=log_path,
            engine=engine,
            success=True,
            notes=notes,
        )
    return PDFRenderResult(
        pdf_path=None,
        log_path=log_path,
        engine=engine,
        success=False,
        notes=notes + [f"{engine} failed; see {log_path}"],
    )


__all__ = ["PDFRenderResult", "render_pdf_for_run"]
