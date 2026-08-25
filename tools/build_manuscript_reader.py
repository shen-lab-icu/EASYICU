#!/usr/bin/env python3
"""Build a provider-free evidence-bound manuscript reader bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.literature import LiteratureBundle
from easyicu.research_agent.reporting.bibtex import render_bibtex
from easyicu.research_agent.reporting.latex import scaffold_to_latex
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.reporting.manuscript_provenance import (
    build_manuscript_provenance,
    strip_numeric_provenance,
)
from easyicu.research_agent.reporting.pdf_render import render_pdf_for_run


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_literature(run_dir: Path) -> LiteratureBundle | None:
    path = run_dir / "preplan_literature_bundle.json"
    try:
        return LiteratureBundle.model_validate_json(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, ValueError):
        return None


def _copy_figures(
    *, run_dir: Path, output_dir: Path, evidence: EvidenceStore
) -> list[tuple[str, str]]:
    target_dir = output_dir / "figures"
    copied: list[tuple[str, str]] = []
    seen_sha256: set[str] = set()
    for record in evidence.records():
        if record.kind != "figure" or len(copied) >= 12:
            continue
        source = run_dir / record.relative_path
        # Use the browser/LaTeX-compatible raster projection only.  The same
        # figure is commonly registered as PNG, PDF and TIFF; embedding every
        # format duplicates panels, and TIFF is not portable across engines.
        if source.suffix.lower() != ".png":
            continue
        try:
            source.resolve().relative_to(run_dir.resolve())
        except (OSError, ValueError):
            continue
        if not source.is_file() or _sha256(source) != record.sha256:
            continue
        if record.sha256 in seen_sha256:
            continue
        seen_sha256.add(record.sha256)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{record.evidence_id}{source.suffix.lower()}"
        shutil.copy2(source, target)
        step_label = str(record.produced_by_step or record.evidence_id)
        step_label = step_label.lstrip("0123456789_").replace("_", " ").strip()
        copied.append(
            (step_label[:1].upper() + step_label[1:], f"figures/{target.name}")
        )
    return copied


def build_bundle(
    *, run_dir: Path, output_dir: Path, claim_base_url: str | None = None
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = run_dir / "manuscript_scaffold_bound.md"
    source_bound = source_path.read_text(encoding="utf-8")
    evidence = EvidenceStore(run_dir, enforcement_mode=EvidenceEnforcementMode.STRICT)

    unbound = strip_numeric_provenance(source_bound)
    corrected, binding_map, untraced = bind_numeric_values(
        unbound,
        evidence=evidence,
        enforcement_mode=EvidenceEnforcementMode.STRICT,
    )
    if untraced:
        raise ValueError(f"unexpected untraced numeric values: {untraced[:8]}")
    provenance = build_manuscript_provenance(
        manuscript=corrected,
        evidence=evidence,
        binding_map=binding_map,
    )

    markdown_path = output_dir / "manuscript_scaffold_bound.md"
    provenance_path = output_dir / "manuscript_provenance.json"
    markdown_path.write_text(corrected, encoding="utf-8")
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    literature = _load_literature(run_dir)
    figure_paths = _copy_figures(
        run_dir=run_dir, output_dir=output_dir, evidence=evidence
    )
    tex = scaffold_to_latex(
        markdown=corrected,
        bibliography=literature,
        bibliography_basename="manuscript_scaffold",
        figure_paths=figure_paths or None,
        draft_watermark=True,
        claim_base_url=claim_base_url,
    )
    tex_path = output_dir / "manuscript_scaffold.tex"
    tex_path.write_text(tex, encoding="utf-8")
    bib_path: Path | None = None
    if literature is not None and literature.citations:
        bib_path = output_dir / "manuscript_scaffold.bib"
        bib_path.write_text(render_bibtex(literature), encoding="utf-8")

    pdf_result = render_pdf_for_run(
        tex_path=tex_path,
        bib_path=bib_path,
        output_dir=output_dir,
        draft_watermark=True,
    )
    if not pdf_result.success or pdf_result.pdf_path is None:
        raise RuntimeError("PDF render failed: " + "; ".join(pdf_result.notes))

    receipt = {
        "schema_version": "easyicu.manuscript-reader-build/1",
        "source_run_id": run_dir.name,
        "source_manuscript_sha256": _sha256(source_path),
        "corrected_manuscript_sha256": _sha256(markdown_path),
        "provenance_sha256": _sha256(provenance_path),
        "pdf_sha256": _sha256(pdf_result.pdf_path),
        "claim_count": provenance["claim_count"],
        "figure_count": len(figure_paths),
        "provider_calls": 0,
        "claim_ceiling": "analysis_only",
        "publication_authorized": False,
        "semantic_rebinding_changed_source": corrected != source_bound,
        "outputs": sorted(
            [
                *(path.name for path in output_dir.iterdir()),
                "manuscript_reader_build_receipt.json",
            ]
        ),
    }
    (output_dir / "manuscript_reader_build_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--claim-base-url")
    args = parser.parse_args()
    receipt = build_bundle(
        run_dir=args.run_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        claim_base_url=args.claim_base_url,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
