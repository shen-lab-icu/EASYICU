"""Export an identified report revision from sealed, read-only analysis inputs."""

from dataclasses import replace
import base64
import hashlib
import json
from pathlib import Path

from easyicu.research_agent.reporting.latex import scaffold_to_latex
from easyicu.research_agent.reporting.manuscript_figures import build_manuscript_figures
from easyicu.research_agent.reporting.manuscript_tables import build_manuscript_tables
from easyicu.research_agent.reporting.pdf_render import render_pdf_for_run
from easyicu.research_agent.reporting.manuscript_quality import render_reader_manuscript
from easyicu.research_agent.reporting.manuscript_labels import source_bound_manuscript_labels
from easyicu.research_agent.reporting.registered_report_inputs import ReadOnlyReportEvidence
from easyicu.research_agent.reporting.writer_only_migration import WriterOnlyMigrationError


def build_revision_figure_gallery(prepared) -> dict:
    """Use the same verified figure selection for the Web and PDF revision."""
    evidence = ReadOnlyReportEvidence(prepared.source_run_dir)
    execution = json.loads((evidence.root / "manifest.json").read_text())["per_step_records"]
    records = evidence.current_verified_records(execution)
    projection = build_manuscript_figures(evidence_records=records, run_dir=evidence.root, prefer_png=True)
    if projection.findings or projection.omitted_evidence_ids:
        raise WriterOnlyMigrationError(code="REPORT_EXPORT_FIGURE_INVALID", detail="Registered figures cannot be projected.")
    figures = []
    total = 0
    for figure in projection.figures:
        path = evidence.root / figure.relative_path
        content = path.read_bytes()
        total += len(content)
        if path.suffix.lower() != ".png" or total > 4_000_000 or hashlib.sha256(content).hexdigest() != figure.figure_sha256:
            raise WriterOnlyMigrationError(code="REPORT_EXPORT_FIGURE_CHANGED", detail="Web figure cannot be bound to its source.")
        figures.append({
            "label": "Result figure" if figure.placement == "main" else "Supplementary figure",
            "caption": figure.caption, "placement": figure.placement,
            "name": path.name, "sha256": figure.figure_sha256,
            "contract_sha256": figure.contract_sha256,
            "source_evidence_id": figure.evidence_id,
            "tier": "primary_publication" if figure.placement == "main" else "supporting_step",
            "data_url": "data:image/png;base64," + base64.b64encode(content).decode("ascii"),
        })
    return {
        "kind": "figure_gallery", "schema_version": "easyicu.web-pipeline-figure-gallery/1",
        "status": "available", "figures": figures, "embedded_count": len(figures),
        "primary_count": sum(f["placement"] == "main" for f in figures),
        "supporting_count": sum(f["placement"] == "supplementary" for f in figures),
        "authority_ceiling": "analysis_only", "original_run_figures_preserved": True,
        "context_notes": list(projection.context_notes),
    }


def export_revision_pdf(*, prepared, output: Path, reader: str, revision: dict) -> dict:
    """Keep the original run/PDF untouched; bind the export to this revision."""
    bound = output / "manuscript_bound.md"
    if bound.is_symlink() or hashlib.sha256(bound.read_bytes()).hexdigest() != revision["output_sha256"]:
        raise WriterOnlyMigrationError(code="REPORT_EXPORT_REVISION_CHANGED", detail="Revision content changed.")
    if reader != render_reader_manuscript(bound.read_text(encoding="utf-8")):
        raise WriterOnlyMigrationError(code="REPORT_EXPORT_READER_CHANGED", detail="Reader does not match the bound revision.")
    evidence = ReadOnlyReportEvidence(prepared.source_run_dir)
    execution = json.loads((evidence.root / "manifest.json").read_text())["per_step_records"]
    records = evidence.current_verified_records(execution)
    gallery = build_manuscript_figures(evidence_records=records, run_dir=evidence.root)
    if gallery.findings or gallery.omitted_evidence_ids:
        raise WriterOnlyMigrationError(code="REPORT_EXPORT_FIGURE_INVALID", detail="Registered figures cannot be exported.")
    export = output / "pdf"
    export.mkdir(exist_ok=False)
    figures = []
    for index, figure in enumerate(gallery.figures):
        source = evidence.root / figure.relative_path
        content = source.read_bytes()
        if source.is_symlink() or hashlib.sha256(content).hexdigest() != figure.figure_sha256:
            raise WriterOnlyMigrationError(code="REPORT_EXPORT_FIGURE_CHANGED", detail="Figure content changed.")
        name = f"figure_{index + 1}{source.suffix}"
        (export / name).write_bytes(content)
        figures.append(replace(figure, relative_path=name))
    display_plan = prepared.plan.model_copy(update={"display_labels": source_bound_manuscript_labels(
        prepared.context, prepared.plan.display_labels, include_unlabeled=True,
    )})
    tables = build_manuscript_tables(plan=display_plan, evidence_records=records, run_dir=evidence.root)
    title = next((line[2:].strip() for line in reader.splitlines() if line.startswith("# ")), "Research report")
    tex = scaffold_to_latex(
        markdown=reader, title=title,
        authors=[f"EasyICU development report | Revision {revision['revision_id']}",
                 f"Source run {revision['source_run_id']}"],
        bibliography=prepared.literature, inline_bibliography=True,
        figures=figures, tables=tables, draft_watermark=True,
        figure_context=[note["text"] for note in gallery.context_notes],
    )
    tex_path = export / "manuscript_revision.tex"
    tex_path.write_text(tex, encoding="utf-8")
    receipt = render_pdf_for_run(tex_path=tex_path, output_dir=export, draft_watermark=True)
    pdf = export / "manuscript_revision.pdf"
    if not receipt.success or not pdf.is_file() or pdf.stat().st_size > 16 * 1024 * 1024:
        raise WriterOnlyMigrationError(code="REPORT_EXPORT_FAILED", detail="PDF rendering did not complete.")
    if any("Missing character:" in log.read_text(errors="replace") for log in export.glob("*.log")):
        raise WriterOnlyMigrationError(code="REPORT_EXPORT_MISSING_GLYPH", detail="PDF lost one or more text characters.")
    return {
        "name": "manuscript_revision.pdf",
        "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "revision_id": revision["revision_id"],
        "manuscript_sha256": revision["output_sha256"],
        "receipt_sha256": hashlib.sha256((export / "manuscript_pdf_receipt.json").read_bytes()).hexdigest(),
    }
