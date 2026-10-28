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
import math
from html import escape
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
        "| database | run_id | predictor | primary_or | ci_low | ci_high | ci_source | status |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {database} | {run_id} | {predictor} | {primary_or} | {primary_ci_low} | {primary_ci_high} | {ci_source} | {status} |".format(
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
                ci_source=row.get("ci_source", "") or "",
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
    run_results: Optional[Dict[str, PipelineResult]] = None,
    report_dir: Optional[Path] = None,
    hartung_knapp: bool = False,
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
    if run_results is not None:
        lines.extend(_replication_meta_analysis(run_results, report_dir, hartung_knapp=hartung_knapp).splitlines())
    lines.append("")
    return "\n".join(lines) + "\n"


def _replication_meta_analysis(run_results: Dict[str, PipelineResult], report_dir: Optional[Path], *, hartung_knapp: bool = False) -> str:
    from .metrics import pool_database_estimates, read_replication_estimate

    inputs = []
    for database, run in run_results.items():
        try:
            inputs.append(read_replication_estimate(run.workdir, database))
        except (ValueError, OSError, KeyError, AttributeError):
            inputs.append(dict(database=database, gate_passed=False))
    # The first requested database is the prespecified reference, even when
    # it failed. Never choose the largest compatible subset after seeing fits.
    reference = inputs[0].get("estimand_sha256", "") if inputs else ""
    calculation = pool_database_estimates(inputs, reference_estimand=reference, hartung_knapp=hartung_knapp)
    rendered = render_database_meta_analysis(calculation)
    if report_dir is not None:
        (report_dir / "meta_analysis.json").write_text(json.dumps(calculation, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (report_dir / "meta_analysis_report.md").write_text(rendered["markdown"], encoding="utf-8")
        if rendered["forest_svg"]:
            (report_dir / "meta_analysis_forest.svg").write_text(rendered["forest_svg"], encoding="utf-8")
            rendered["markdown"] += "\n![Per-database forest plot](meta_analysis_forest.svg)\n"
    return rendered["markdown"]


def render_database_meta_analysis(result: dict) -> dict[str, str]:
    """Render aggregate-only pooling tables and a self-contained SVG forest.

    This renderer consumes the replication owner's governed calculation. The
    SVG uses log coordinates for ratios and labels them on their original
    scale. It contains no host paths, scripts, external images or patient rows.
    """
    lines = ["## Per-database meta-analysis", "", "Claim ceiling: pooled observational estimate (analysis only).", ""]
    if result["status"] != "pooled":
        lines += ["Pooling unavailable: " + result["reason_code"], ""]
        lines += [f"- {row['database']}: {row['reason_code']}" for row in result["excluded"]]
        return {"markdown": "\n".join(lines) + "\n", "forest_svg": ""}
    random = result["random"]
    lines += [f"Effect measure: {result['unit']}; analysis scale: {result['analysis_scale']}. SE and tau² use that analysis scale.",
              f"Confidence level: {100 * random['confidence']:.6g}%. " + result["assumptions"], "", "| Database | Estimate | CI low | CI high | RE weight (%) |",
              "|---|---:|---:|---:|---:|"]
    forest = []
    for row, weight in zip(result["included"], random["weights"]):
        lines.append(f"| {row['database']} | {row['estimate']:.6g} | {row['ci_low']:.6g} | {row['ci_high']:.6g} | {100 * weight:.4g} |")
        forest.append((str(row["database"]), row["estimate"], row["ci_low"], row["ci_high"], weight))
    lines += ["", "### Heterogeneity and pooled estimates", "",
              "| Model | Estimate | CI low | CI high | Q | df | p | I² (%) | tau² | Interval |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for key in ("fixed", "random", "dl_sensitivity"):
        item = result[key]
        label = "Fixed effect" if key == "fixed" else item["tau_method"]
        lines.append(f"| {label} | {item['estimate']:.6g} | {item['ci_low']:.6g} | {item['ci_high']:.6g} | {item['q']:.6g} | {item['q_df']} | {item['q_p']:.6g} | {item['i_squared']:.6g} | {item['tau_squared']:.6g} | {item['interval_method']} |")
    if random["prediction_low"] is not None:
        lines += ["", f"REML prediction interval: [{random['prediction_low']:.6g}, {random['prediction_high']:.6g}] (t, df={random['prediction_df']})."]
    else:
        lines += ["", "Prediction interval unavailable: " + random["prediction_reason"]]
    if random["interval_method"] == "Hartung-Knapp":
        lines += ["", "Unmodified Hartung-Knapp: residual scale Q_RE/(k-1), without a floor at one; t(k-1) confidence interval."]
    lines += ["", "### Leave-one-database-out", "", "| Omitted database | REML estimate | CI low | CI high | tau² |",
              "|---|---:|---:|---:|---:|"]
    for row in result["leave_one_out"]:
        item = row["random"]
        lines.append(f"| {row['omitted_database']} | {item['estimate']:.6g} | {item['ci_low']:.6g} | {item['ci_high']:.6g} | {item['tau_squared']:.6g} |"
                     if item else f"| {row['omitted_database']} | unavailable: {row['reason_code']} | | | |")
    lines += ["", "### Excluded databases", ""]
    lines += [f"- {row['database']}: {row['reason_code']}" for row in result["excluded"]] or ["None."]
    return {"markdown": "\n".join(lines) + "\n", "forest_svg": _database_forest_svg(forest, result)}


def _database_forest_svg(forest: list[tuple], result: dict) -> str:
    random = result["random"]
    transform = math.log if result["analysis_scale"] == "log" else float
    endpoints = [value for _, point, low, high, _ in forest for value in (point, low, high)]
    endpoints += [random[key] for key in ("estimate", "ci_low", "ci_high", "prediction_low", "prediction_high")
                  if random[key] is not None]
    null = 1.0 if result["analysis_scale"] == "log" else 0.0
    low, high = min(map(transform, [*endpoints, null])), max(map(transform, [*endpoints, null]))
    span = max(high - low, 1e-9)

    def x(value):
        return 280 + 400 * (transform(value) - low) / span

    height = 155 + 40 * len(forest)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="940" height="{height}" viewBox="0 0 940 {height}" role="img">',
             '<title>REML forest plot: pooled observational estimate</title>',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" font-size="13" fill="#172b3a">',
             '<text x="16" y="24">Database</text><text x="700" y="24">Estimate [CI]; RE weight</text>',
             f'<line x1="{x(null):.3f}" x2="{x(null):.3f}" y1="35" y2="{height - 45}" stroke="#85939b" stroke-dasharray="4 4"/>']
    for index, (database, point, lo, hi, weight) in enumerate(forest):
        y = 55 + 40 * index
        size = 4 + 10 * math.sqrt(weight)
        parts += [f'<g id="database-{escape(database, quote=True)}"><text x="16" y="{y + 4}">{escape(database)}</text>',
                  f'<line x1="{x(lo):.3f}" x2="{x(hi):.3f}" y1="{y}" y2="{y}" stroke="#245b78"/>',
                  f'<rect x="{x(point) - size / 2:.3f}" y="{y - size / 2:.3f}" width="{size:.3f}" height="{size:.3f}" fill="#245b78"/>',
                  f'<text x="700" y="{y + 4}">{point:.4g} [{lo:.4g}, {hi:.4g}]; {100 * weight:.3g}%</text></g>']
    y = 55 + 40 * len(forest)
    parts += [f'<text x="16" y="{y + 4}">Pooled REML</text>',
              f'<polygon id="pooled-diamond" points="{x(random["ci_low"]):.3f},{y} {x(random["estimate"]):.3f},{y - 7} {x(random["ci_high"]):.3f},{y} {x(random["estimate"]):.3f},{y + 7}" fill="#8a3c42"/>',
              f'<text x="700" y="{y + 4}">{random["estimate"]:.4g} [{random["ci_low"]:.4g}, {random["ci_high"]:.4g}]</text>']
    if random["prediction_low"] is not None:
        parts += [f'<text x="16" y="{y + 34}">Prediction interval</text>',
                  f'<line id="prediction-interval" x1="{x(random["prediction_low"]):.3f}" x2="{x(random["prediction_high"]):.3f}" y1="{y + 30}" y2="{y + 30}" stroke="#8a3c42" stroke-width="3"/>',
                  f'<text x="700" y="{y + 34}">[{random["prediction_low"]:.4g}, {random["prediction_high"]:.4g}]</text>']
    parts += [f'<text x="280" y="{height - 15}">{min([*endpoints, null]):.4g}</text>',
              f'<text x="610" y="{height - 15}">{max([*endpoints, null]):.4g}</text>',
              f'<text x="360" y="{height - 15}">{escape(result["unit"])} ({result["analysis_scale"]} axis)</text>', '</g></svg>']
    return "".join(parts)
