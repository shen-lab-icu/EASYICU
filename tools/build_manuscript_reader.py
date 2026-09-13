#!/usr/bin/env python3
"""Build a provider-free evidence-bound manuscript reader bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.authority.runtime_artifacts import (
    current_step_records,
    verified_run_evidence_path,
)
from easyicu.research_agent.execution.runners.landmark_survival_executor import (
    build_survival_manuscript_projection,
)
from easyicu.research_agent.literature import LiteratureBundle
from easyicu.research_agent.reporting.article_display_package import (
    inspect_article_display_package,
    reader_figure_rows,
)
from easyicu.research_agent.reporting.bibtex import render_bibtex
from easyicu.research_agent.reporting.latex import scaffold_to_latex
from easyicu.research_agent.reporting.manuscript_post import (
    bind_numeric_values,
    drop_untraceable_numeric_sentences,
)
from easyicu.research_agent.reporting.manuscript_projection import (
    project_owner_issued_manuscript_claims,
)
from easyicu.research_agent.reporting.manuscript_quality import (
    repair_reader_structure_from_existing_prose,
)
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


def _load_reader_title(run_dir: Path) -> str:
    """Use the host-owned manuscript packet when Writer's title was filtered."""

    path = run_dir / "manuscript_packet.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, ValueError):
        return "EasyICU analysis-only manuscript draft"
    title = str(payload.get("title") or "").strip().lstrip("#").strip()
    return title or "EasyICU analysis-only manuscript draft"


def _publication_figure_exclusion_reason(run_dir: Path) -> str | None:
    """Fail closed when the source run has not cleared publication figure QA."""

    path = run_dir / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, ValueError):
        return None
    readiness = payload.get("readiness")
    if not isinstance(readiness, dict):
        return None
    if readiness.get("publication_figure_visual_qa_passed") is False:
        return "source_run_publication_figure_visual_qa_failed"
    return None


def _display_index(
    display_inventory: Mapping[str, Any] | None,
) -> dict[str, dict[str, str]]:
    """Map verified artifact digests to their registered display identity."""

    index: dict[str, dict[str, str]] = {}
    for row in (display_inventory or {}).get("displays") or ():
        if not isinstance(row, Mapping):
            continue
        display_id = str(row.get("display_id") or "").strip()
        if not display_id:
            continue
        contract_sha256 = str(row.get("contract_sha256") or "").strip().lower()
        keys = [contract_sha256, str(row.get("source_sha256") or "").strip().lower(),
                str(row.get("preferred_preview_sha256") or "").strip().lower()]
        for export in (row.get("exports") or {}).values():
            if isinstance(export, Mapping):
                keys.append(str(export.get("sha256") or "").strip().lower())
        for key in keys:
            if re.fullmatch(r"[a-f0-9]{64}", key):
                index.setdefault(
                    key,
                    {"display_id": display_id, "contract_sha256": contract_sha256},
                )
    return index


def _step_method_summaries(
    records: Sequence[Mapping[str, Any]] | None,
) -> dict[str, dict[str, str]]:
    """Project bounded scalar step coordinates for the reader claim panel."""

    summaries: dict[str, dict[str, str]] = {}
    for record in records or []:
        if not isinstance(record, Mapping):
            continue
        step_id = str(record.get("step_id") or "").strip()
        if not step_id:
            continue
        summary: dict[str, str] = {}
        for key in ("intent", "planned_analysis_role"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                summary[key] = value.strip()[:400]
        if summary:
            summaries[step_id] = summary
    return summaries


_READER_NOTE_COPY = {
    "strict_untraceable_numeric_sentence_removed": (
        "A numeric sentence without a registered evidence source was removed "
        "from the reader text."
    ),
    "source_run_publication_figure_visual_qa_failed": (
        "Publication figure quality assurance did not pass for the source run, "
        "so publication figures are withheld from this reader package."
    ),
}


def _reader_verification_notes(
    deterministic_repairs: Sequence[Mapping[str, Any]] | None,
    *,
    figure_exclusion_reason: str | None,
) -> list[dict[str, Any]]:
    """Turn bounded deterministic repair receipts into typed reader notes."""

    notes: list[dict[str, Any]] = []
    for repair in deterministic_repairs or []:
        if not isinstance(repair, Mapping):
            continue
        code = str(
            repair.get("reason_code") or repair.get("kind") or ""
        ).strip()[:120]
        if not code:
            continue
        notes.append(
            {
                "code": code,
                "severity": "warning",
                "text": _READER_NOTE_COPY.get(
                    code, "Reader preparation applied a deterministic repair."
                ),
            }
        )
        if len(notes) >= 12:
            break
    if figure_exclusion_reason:
        notes.append(
            {
                "code": str(figure_exclusion_reason)[:120],
                "severity": "warning",
                "text": _READER_NOTE_COPY.get(
                    str(figure_exclusion_reason),
                    "A registered display was withheld from this reader package.",
                ),
            }
        )
    return notes


def _prepare_reader_manuscript(
    source_bound: str,
    *,
    per_step_records: list[dict[str, Any]] | None = None,
) -> tuple[str, tuple[dict[str, Any], ...]]:
    """Apply provider-free reader repairs before provenance is projected."""

    migrated_records = copy.deepcopy(per_step_records or [])
    migration_repairs: list[dict[str, Any]] = []
    for record in migrated_records:
        summary = record.get("step_summary")
        if not isinstance(summary, dict):
            continue
        reporting = summary.get("reportable_survival_results")
        if not isinstance(reporting, dict) or "manuscript_projection" in reporting:
            continue
        association = reporting.get("time_varying_adjusted_association")
        intervals = (
            association.get("intervals") if isinstance(association, dict) else None
        )
        if (
            record.get("generation_mode") != "deterministic_standard"
            or record.get("deterministic_standard_analysis")
            != "signed_landmark_survival_suite"
            or reporting.get("schema_version") != "easyicu.survival_reporting/1"
            or reporting.get("execution_owner") != "landmark_survival_executor_v1"
            or not isinstance(intervals, list)
            or not intervals
        ):
            continue
        reporting["manuscript_projection"] = build_survival_manuscript_projection(
            interval_count=len(intervals)
        )
        migration_repairs.append(
            {
                "reason_code": "legacy_owner_projection_contract_migrated",
                "step_id": str(record.get("step_id") or ""),
                "from_schema_version": "easyicu.survival_reporting/1",
                "projection_schema_version": "easyicu.manuscript_projection/1",
            }
        )
    repaired, repairs = repair_reader_structure_from_existing_prose(source_bound)
    repaired, projection_repairs = project_owner_issued_manuscript_claims(
        repaired,
        per_step_records=migrated_records,
    )
    return repaired, tuple(
        [
            *(dict(item) for item in repairs),
            *migration_repairs,
            *(dict(item) for item in projection_repairs),
        ]
    )


def _load_verified_current_step_records(
    run_dir: Path, evidence: EvidenceStore
) -> list[dict[str, Any]]:
    """Rehydrate current step summaries only from digest-verified evidence."""

    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, ValueError):
        return []
    records = manifest.get("per_step_records")
    if not isinstance(records, list):
        return []
    verified: list[dict[str, Any]] = []
    for raw in current_step_records(records):
        if not isinstance(raw, dict):
            continue
        evidence_id = str(raw.get("step_summary_evidence_id") or "").strip()
        evidence_record = evidence.get(evidence_id) if evidence_id else None
        if evidence_record is None:
            continue
        source = verified_run_evidence_path(run_dir, evidence_record)
        if source is None:
            continue
        try:
            summary = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if not isinstance(summary, dict):
            continue
        item = dict(raw)
        item["step_summary"] = summary
        verified.append(item)
    return verified


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


def _copy_article_display_figures(
    *, package_dir: Path, output_dir: Path
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], dict[str, Any]]:
    """Copy a typed article package into separate reader figure groups."""

    inventory = inspect_article_display_package(package_dir)
    copied: dict[str, list[tuple[str, str]]] = {"main": [], "supplementary": []}
    for placement in ("main", "supplementary"):
        target_dir = output_dir / "figures" / placement
        for index, row in enumerate(
            reader_figure_rows(inventory, placement=placement), start=1
        ):
            source = package_dir / str(row["preferred_preview_path"])
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{index:02d}_{source.name}"
            shutil.copy2(source, target)
            copied[placement].append(
                (
                    str(row.get("label") or row.get("display_id") or target.stem),
                    str(target.relative_to(output_dir)),
                )
            )
    (output_dir / "article_display_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return copied["main"], copied["supplementary"], inventory


def build_bundle(
    *,
    run_dir: Path,
    output_dir: Path,
    claim_base_url: str | None = None,
    article_display_package: Path | None = None,
    manuscript_source: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = (
        manuscript_source.expanduser().resolve(strict=True)
        if manuscript_source is not None
        else run_dir / "manuscript_scaffold_bound.md"
    )
    if not source_path.is_file():
        raise ValueError(f"manuscript source is not a file: {source_path}")
    source_bound = source_path.read_text(encoding="utf-8")
    evidence = EvidenceStore(run_dir, enforcement_mode=EvidenceEnforcementMode.STRICT)
    verified_step_records = _load_verified_current_step_records(run_dir, evidence)
    prepared_bound, deterministic_repairs = _prepare_reader_manuscript(
        source_bound,
        per_step_records=verified_step_records,
    )
    prepared_bound = evidence.bind_manuscript(
        prepared_bound,
        per_step_records=verified_step_records,
    )

    unbound = strip_numeric_provenance(prepared_bound)
    unbound, removed_numeric_sentences = drop_untraceable_numeric_sentences(
        unbound,
        evidence=evidence,
        per_step_records=verified_step_records,
    )
    deterministic_repairs = tuple(
        [
            *deterministic_repairs,
            *(
                {
                    "reason_code": "strict_untraceable_numeric_sentence_removed",
                    **dict(item),
                }
                for item in removed_numeric_sentences
            ),
        ]
    )
    corrected, binding_map, untraced = bind_numeric_values(
        unbound,
        evidence=evidence,
        enforcement_mode=EvidenceEnforcementMode.STRICT,
        per_step_records=verified_step_records,
    )
    if untraced:
        raise ValueError(f"unexpected untraced numeric values: {untraced[:8]}")
    literature = _load_literature(run_dir)
    figure_exclusion_reason: str | None = None
    supplementary_figure_paths: list[tuple[str, str]] = []
    display_inventory: dict[str, Any] | None = None
    if article_display_package is not None:
        package_dir = article_display_package.expanduser().resolve(strict=True)
        figure_paths, supplementary_figure_paths, display_inventory = (
            _copy_article_display_figures(
                package_dir=package_dir,
                output_dir=output_dir,
            )
        )
    else:
        figure_exclusion_reason = _publication_figure_exclusion_reason(run_dir)
        figure_paths = (
            []
            if figure_exclusion_reason
            else _copy_figures(
                run_dir=run_dir, output_dir=output_dir, evidence=evidence
            )
        )
    provenance = build_manuscript_provenance(
        manuscript=corrected,
        evidence=evidence,
        binding_map=binding_map,
        verify="mark",
        display_index=_display_index(display_inventory),
        method_summaries=_step_method_summaries(verified_step_records),
        reader_notes=_reader_verification_notes(
            deterministic_repairs,
            figure_exclusion_reason=figure_exclusion_reason,
        ),
    )

    markdown_path = output_dir / "manuscript_scaffold_bound.md"
    provenance_path = output_dir / "manuscript_provenance.json"
    markdown_path.write_text(corrected, encoding="utf-8")
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tex = scaffold_to_latex(
        markdown=corrected,
        title=_load_reader_title(run_dir),
        bibliography=literature,
        bibliography_basename="manuscript_scaffold",
        figure_paths=figure_paths or None,
        draft_watermark=True,
        claim_base_url=claim_base_url,
        supplementary_figure_paths=supplementary_figure_paths or None,
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
        "source_manuscript_path": str(source_path),
        "source_manuscript_sha256": _sha256(source_path),
        "corrected_manuscript_sha256": _sha256(markdown_path),
        "provenance_sha256": _sha256(provenance_path),
        "pdf_sha256": _sha256(pdf_result.pdf_path),
        "claim_count": provenance["claim_count"],
        "figure_count": len(figure_paths) + len(supplementary_figure_paths),
        "main_figure_count": len(figure_paths),
        "supplementary_figure_count": len(supplementary_figure_paths),
        "article_display_inventory": (
            "article_display_inventory.json" if display_inventory is not None else None
        ),
        "figure_exclusion_reason": figure_exclusion_reason,
        "provider_calls": 0,
        "claim_ceiling": "analysis_only",
        "publication_authorized": False,
        "deterministic_repairs": list(deterministic_repairs),
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
    parser.add_argument(
        "--article-display-package",
        type=Path,
        help="Optional digest-inventoried article figure/table package.",
    )
    parser.add_argument(
        "--manuscript-source",
        type=Path,
        help="Optional evidence-bound manuscript projection to rebind against the run.",
    )
    args = parser.parse_args()
    receipt = build_bundle(
        run_dir=args.run_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        claim_base_url=args.claim_base_url,
        article_display_package=args.article_display_package,
        manuscript_source=args.manuscript_source,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
