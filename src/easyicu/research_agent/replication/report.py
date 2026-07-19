"""Cross-database / replication reporting helpers extracted from pipeline.py.

Contains the per-database run-summary extractor, the two cross-database
markdown renderers (comparison + readiness summary), the wrapper
validation report, the literature-provenance one-liner, and the paper
replication note builder. Behaviour is unchanged from the prior in-place
definitions; this module exists so pipeline.py can stay focused on
orchestration rather than report formatting.

Moved out on 2026-05-27 as part of the pipeline.py size-reduction effort.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..schema import PaperProfile, PaperReplicationSpec, PipelineResult

__all__ = [
    "_literature_provenance_note",
    "_build_replication_notes",
    "_extract_cross_database_run_summary",
    "_render_cross_database_comparison_markdown",
    "_render_cross_database_summary_markdown",
    "_render_cross_database_validation_report",
]


def _literature_provenance_note(
    *,
    enable_literature: bool,
    enable_pubmed: bool,
    enable_tavily: bool,
) -> str:
    if not enable_literature:
        return "literature_provenance: literature agent disabled for this run."
    sources = ["curated registry"]
    if enable_pubmed:
        sources.append("PubMed")
    if enable_tavily:
        sources.append("Tavily")
    return "literature_provenance: references sourced from " + ", ".join(sources) + "."


def _build_replication_notes(
    *,
    paper_profile: PaperProfile,
    replication_spec: PaperReplicationSpec,
    mode: str,
) -> str:
    lines = [
        "Paper replication mode is active.",
        f"Source paper: {paper_profile.paper_title or paper_profile.paper_source}.",
        f"Replication goal: {replication_spec.replication_goal}.",
        f"Mode: {mode}.",
        "Use only EasyICU-observed numbers in the manuscript.",
        "If the original paper is referenced, phrase it as 'original paper reported ...'.",
        "Treat unmappable design elements as explicit deviations, not silent substitutions.",
    ]
    if replication_spec.mapped_concepts:
        lines.append(
            "Mapped concepts: "
            + ", ".join(
                f"{k}->{v}" for k, v in sorted(replication_spec.mapped_concepts.items())
            )
            + "."
        )
    if replication_spec.unmappable_items:
        lines.append(
            "Unmappable items: " + "; ".join(replication_spec.unmappable_items) + "."
        )
    return "\n".join(lines)


def _extract_cross_database_run_summary(
    *,
    database: str,
    result: PipelineResult,
) -> Dict[str, Any]:
    run_dir = Path(result.workdir)
    run_status_path = run_dir / "run_status.json"
    gates: Dict[str, Any] = {}
    status = "missing_run_status"
    if run_status_path.exists():
        try:
            payload = json.loads(run_status_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                status = str(payload.get("status") or status)
                raw_gates = payload.get("gates")
                if isinstance(raw_gates, dict):
                    gates = raw_gates
        except Exception:
            status = "invalid_run_status"
    return {
        "database": database,
        "run_id": result.run_id,
        "status": status,
        "execution_complete": bool(gates.get("execution_complete")),
        "evidence_complete": bool(gates.get("evidence_complete")),
        "numeric_verified": bool(gates.get("numeric_verified")),
        "analysis_validated": bool(gates.get("analysis_validated")),
        "manuscript_ready": bool(gates.get("manuscript_ready")),
        "publication_ready": bool(gates.get("publication_ready")),
        "missing_evidence_count": int(gates.get("missing_evidence_count") or 0),
        "numeric_error_count": int(gates.get("numeric_error_count") or 0),
        "manifest_path": str(result.manifest_path),
        "report_path": str(result.report_path),
        "manuscript_path": str(result.manuscript_path),
    }


def _render_cross_database_comparison_markdown(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# Cross-database effect comparison",
        "",
        "| database | run_id | predictor | primary_or | ci_low | ci_high | status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {database} | {run_id} | {predictor} | {primary_or} | {primary_ci_low} | {primary_ci_high} | {status} |".format(
                database=row.get("database", ""),
                run_id=row.get("run_id", ""),
                predictor=row.get("predictor", "") or "",
                primary_or=(
                    row.get("primary_or", "")
                    if row.get("primary_or") is not None
                    else ""
                ),
                primary_ci_low=(
                    row.get("primary_ci_low", "")
                    if row.get("primary_ci_low") is not None
                    else ""
                ),
                primary_ci_high=(
                    row.get("primary_ci_high", "")
                    if row.get("primary_ci_high") is not None
                    else ""
                ),
                status=row.get("status", ""),
            )
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_cross_database_summary_markdown(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# Cross-database readiness summary",
        "",
        "| database | run_id | status | execution | evidence | numeric | validated | manuscript | publication | missing evidence | numeric errors |",
        "|---|---|---|---|---|---|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {database} | {run_id} | {status} | {execution_complete} | {evidence_complete} | {numeric_verified} | {analysis_validated} | {manuscript_ready} | {publication_ready} | {missing_evidence_count} | {numeric_error_count} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_cross_database_validation_report(
    *,
    question: Optional[str],
    target_outcome: Optional[str],
    rows: Sequence[Dict[str, Any]],
    run_summaries: Sequence[Dict[str, Any]],
) -> str:
    successful = sum(1 for row in run_summaries if row.get("execution_complete"))
    manuscript_ready = sum(1 for row in run_summaries if row.get("manuscript_ready"))
    publication_ready = sum(1 for row in run_summaries if row.get("publication_ready"))
    lines = [
        "# Cross-database validation report",
        "",
        f"- Research question: {question or 'n/a'}",
        f"- Target outcome: {target_outcome or 'n/a'}",
        f"- Databases run: {len(run_summaries)}",
        f"- Execution-complete runs: {successful}/{len(run_summaries)}",
        f"- Manuscript-ready runs: {manuscript_ready}/{len(run_summaries)}",
        f"- Publication-ready runs: {publication_ready}/{len(run_summaries)}",
        "",
        "## Effect comparison",
        "",
    ]
    lines.extend(_render_cross_database_comparison_markdown(rows).splitlines())
    lines.extend(
        [
            "",
            "## Readiness summary",
            "",
        ]
    )
    lines.extend(_render_cross_database_summary_markdown(run_summaries).splitlines())
    lines.append("")
    return "\n".join(lines) + "\n"
