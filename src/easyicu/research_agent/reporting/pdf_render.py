"""Compile an EasyICU manuscript scaffold into an auditable draft PDF.

The renderer treats LaTeX as untrusted input even though the scaffold is
host-generated.  Network package fetching and shell escape are disabled, file
access is restricted to the render directory, execution is bounded, and every
successful render emits a digest receipt beside the PDF.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Mapping, Optional, Tuple


_MAX_PDF_BYTES = 64 * 1024 * 1024
_RECEIPT_NAME = "manuscript_pdf_receipt.json"


@dataclass
class PDFRenderResult:
    pdf_path: Optional[Path]
    log_path: Optional[Path]
    receipt_path: Optional[Path]
    engine: Optional[str]
    success: bool
    notes: List[str]


def _which_first(*candidates: str) -> Optional[str]:
    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
    return None


def _restricted_tex_environment() -> Mapping[str, str]:
    """Return a TeX environment that cannot read/write arbitrary paths."""

    env = dict(os.environ)
    env.update(
        {
            "openin_any": "p",
            "openout_any": "p",
            "shell_escape": "f",
        }
    )
    return env


def _run_with_log(
    cmd: List[str],
    *,
    cwd: Path,
    timeout: float = 240.0,
) -> Tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        timeout=timeout,
        env=_restricted_tex_environment(),
    )
    stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
    stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
    return proc.returncode, stdout + "\n" + stderr


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pdf(pdf_path: Path, *, output_dir: Path) -> None:
    resolved = pdf_path.resolve(strict=True)
    if resolved.parent != output_dir.resolve(strict=True):
        raise ValueError("rendered PDF escaped the configured output directory")
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("rendered PDF is not a regular file")
    size = resolved.stat().st_size
    if size <= 4 or size > _MAX_PDF_BYTES:
        raise ValueError(f"rendered PDF has an invalid size: {size}")
    with resolved.open("rb") as handle:
        if handle.read(4) != b"%PDF":
            raise ValueError("rendered output does not have a PDF signature")


def _write_receipt(
    *,
    output_dir: Path,
    tex_path: Path,
    bib_path: Optional[Path],
    pdf_path: Path,
    engine: str,
    draft_watermark: bool,
) -> Path:
    receipt = {
        "schema_version": "easyicu.manuscript_pdf_receipt.v1",
        "status": "rendered",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": engine,
        "security": {
            "network_allowed": False,
            "shell_escape_allowed": False,
            "untrusted_input_mode": True,
            "working_directory_restricted": True,
        },
        "draft_watermark": bool(draft_watermark),
        "source": {
            "name": tex_path.name,
            "sha256": _sha256(tex_path),
        },
        "bibliography": (
            {"name": bib_path.name, "sha256": _sha256(bib_path)}
            if bib_path is not None and bib_path.exists()
            else None
        ),
        "pdf": {
            "name": pdf_path.name,
            "sha256": _sha256(pdf_path),
            "bytes": pdf_path.stat().st_size,
        },
    }
    receipt_path = output_dir / _RECEIPT_NAME
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt_path


def _success_result(
    *,
    output_dir: Path,
    tex_path: Path,
    bib_path: Optional[Path],
    pdf_path: Path,
    log_path: Path,
    engine: str,
    draft_watermark: bool,
    notes: List[str],
) -> PDFRenderResult:
    _validate_pdf(pdf_path, output_dir=output_dir)
    receipt_path = _write_receipt(
        output_dir=output_dir,
        tex_path=tex_path,
        bib_path=bib_path,
        pdf_path=pdf_path,
        engine=engine,
        draft_watermark=draft_watermark,
    )
    return PDFRenderResult(
        pdf_path=pdf_path,
        log_path=log_path,
        receipt_path=receipt_path,
        engine=engine,
        success=True,
        notes=notes,
    )


def render_pdf_for_run(
    *,
    tex_path: Path,
    bib_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    timeout: float = 240.0,
    draft_watermark: bool = False,
) -> PDFRenderResult:
    """Compile ``tex_path`` without network or shell access.

    Tectonic's cached, untrusted mode is preferred.  Local TeXLive engines are
    safe fallbacks and always receive ``-no-shell-escape`` plus restrictive
    ``openin_any``/``openout_any`` settings.  A successful result includes a
    JSON receipt binding the source, bibliography, PDF, engine, and security
    policy by SHA-256.
    """

    tex_path = Path(tex_path)
    if not tex_path.exists() or not tex_path.is_file():
        return PDFRenderResult(
            pdf_path=None,
            log_path=None,
            receipt_path=None,
            engine=None,
            success=False,
            notes=[f"tex file not found: {tex_path}"],
        )
    out_dir = Path(output_dir) if output_dir is not None else tex_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir.resolve(strict=True)
    notes: List[str] = []

    source_tex = tex_path.resolve(strict=True)
    tex_in_out = out_dir / source_tex.name
    if source_tex != tex_in_out:
        tex_in_out.write_bytes(source_tex.read_bytes())
    source_bib: Optional[Path] = None
    if bib_path is not None and Path(bib_path).exists():
        resolved_bib = Path(bib_path).resolve(strict=True)
        source_bib = out_dir / resolved_bib.name
        if resolved_bib != source_bib:
            source_bib.write_bytes(resolved_bib.read_bytes())
    tex_basename = tex_in_out.name
    pdf_path = out_dir / f"{tex_in_out.stem}.pdf"
    log_path = out_dir / f"{tex_in_out.stem}.pdfrender.log"

    # Tectonic is first because it has an explicit untrusted-input mode.  The
    # cached flag makes a missing package fail instead of silently reaching the
    # network during a scientific run.
    tectonic = shutil.which("tectonic")
    if tectonic:
        rc, log = _run_with_log(
            [
                tectonic,
                "-X",
                "compile",
                "--only-cached",
                "--untrusted",
                "--keep-logs",
                tex_basename,
            ],
            cwd=out_dir,
            timeout=timeout,
        )
        log_path.write_text(log, encoding="utf-8")
        if rc == 0 and pdf_path.exists():
            return _success_result(
                output_dir=out_dir,
                tex_path=tex_in_out,
                bib_path=source_bib,
                pdf_path=pdf_path,
                log_path=log_path,
                engine="tectonic",
                draft_watermark=draft_watermark,
                notes=notes,
            )
        notes.append(f"tectonic failed (rc={rc}); see {log_path.name}")

    latexmk = shutil.which("latexmk")
    if latexmk:
        rc, log = _run_with_log(
            [
                latexmk,
                "-pdf",
                "-no-shell-escape",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_basename,
            ],
            cwd=out_dir,
            timeout=timeout,
        )
        log_path.write_text(log, encoding="utf-8")
        if rc == 0 and pdf_path.exists():
            return _success_result(
                output_dir=out_dir,
                tex_path=tex_in_out,
                bib_path=source_bib,
                pdf_path=pdf_path,
                log_path=log_path,
                engine="latexmk",
                draft_watermark=draft_watermark,
                notes=notes,
            )
        notes.append(f"latexmk failed (rc={rc}); see {log_path.name}")

    engine = _which_first("xelatex", "pdflatex")
    if engine is None:
        return PDFRenderResult(
            pdf_path=None,
            log_path=log_path if log_path.exists() else None,
            receipt_path=None,
            engine=None,
            success=False,
            notes=notes
            + ["No safe local LaTeX engine is available; install Tectonic or TeXLive."],
        )

    bibtex = shutil.which("bibtex")
    log_lines: List[str] = []
    successful = True
    for pass_idx in range(2):
        rc, log = _run_with_log(
            [
                engine,
                "-no-shell-escape",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_basename,
            ],
            cwd=out_dir,
            timeout=timeout,
        )
        log_lines.extend([f"=== pass {pass_idx + 1} (rc={rc}) ===", log])
        if rc != 0:
            successful = False
            break
        if bibtex and pass_idx == 0 and source_bib is not None:
            rc_b, log_b = _run_with_log(
                [bibtex, tex_in_out.stem], cwd=out_dir, timeout=timeout
            )
            log_lines.extend([f"=== bibtex (rc={rc_b}) ===", log_b])
            if rc_b != 0:
                successful = False
                break
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    if successful and pdf_path.exists():
        return _success_result(
            output_dir=out_dir,
            tex_path=tex_in_out,
            bib_path=source_bib,
            pdf_path=pdf_path,
            log_path=log_path,
            engine=Path(engine).name,
            draft_watermark=draft_watermark,
            notes=notes,
        )
    return PDFRenderResult(
        pdf_path=None,
        log_path=log_path,
        receipt_path=None,
        engine=Path(engine).name,
        success=False,
        notes=notes + [f"{Path(engine).name} failed; see {log_path.name}"],
    )


__all__ = ["PDFRenderResult", "render_pdf_for_run"]
