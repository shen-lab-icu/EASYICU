"""Typed engineering-validation report for one governed Research Agent run.

This is deliberately not a manuscript renderer.  It summarizes existing,
browser-safe run projections to demonstrate orchestration, provenance, and
fail-closed behavior without upgrading any clinical or publication authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from html import escape
from typing import Any, List, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from easyicu.research_agent.orchestration.human_review_checkpoint import (
    HumanReviewCheckpoint,
)


def _text(value: Any, limit: int = 1_200) -> str:
    return re.sub(r"\s+", " ", str("" if value is None else value)).strip()[:limit]


def projection_payload_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ValidationMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    value: str
    detail: str
    evidence_refs: List[str] = Field(default_factory=list, max_length=12)


class ValidationStage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str
    label: str
    status: Literal["verified", "withheld", "blocked", "not_assessed"]
    summary: str
    evidence_refs: List[str] = Field(default_factory=list, max_length=12)


class ValidationFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    severity: str
    domain: str
    message: str
    remediation: str
    evidence_refs: List[str] = Field(default_factory=list, max_length=12)


class ValidationSourceBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact: str
    sha256: str
    binding_scope: Literal["browser_projection_payload", "run_private_receipt"]


class ValidationFigure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    label: str
    status: str


class ValidationCaseTable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    label: str
    evidence_id: str
    columns: List[str] = Field(default_factory=list, max_length=10)
    rows: List[List[str]] = Field(default_factory=list, max_length=8)


class ValidationCaseStudy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["bounded_demonstration_case"] = "bounded_demonstration_case"
    question: str
    analysis_type: str
    scientific_claim_ceiling: str
    generated_numbers: Literal[False] = False
    primary_table: Optional[ValidationCaseTable] = None
    figures: List[ValidationFigure] = Field(default_factory=list, max_length=8)


class ProviderUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    calls: int = Field(ge=0)
    accounted_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
    ledger_sha256: Optional[str] = None


class SystemValidationReport(BaseModel):
    """One non-paper, source-bound engineering validation dossier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.system-validation-report/1"] = (
        "easyicu.system-validation-report/1"
    )
    artifact_kind: Literal["system_validation_report"] = "system_validation_report"
    authority_class: Literal["engineering_validation_only"] = (
        "engineering_validation_only"
    )
    claim_ceiling: Literal["engineering_validation_only"] = (
        "engineering_validation_only"
    )
    reportable: Literal[False] = False
    publication_authorized: Literal[False] = False
    run_id: str
    title: str
    subtitle: str
    status: Literal[
        "engineering_validation_complete", "engineering_validation_incomplete"
    ]
    executive_summary: str
    thesis: str
    metrics: List[ValidationMetric] = Field(default_factory=list, max_length=12)
    lifecycle: List[ValidationStage] = Field(default_factory=list, max_length=12)
    demonstrated: List[str] = Field(default_factory=list, max_length=20)
    not_demonstrated: List[str] = Field(default_factory=list, max_length=20)
    case_study: ValidationCaseStudy
    provider_usage: Optional[ProviderUsage] = None
    scientific_findings: List[ValidationFinding] = Field(
        default_factory=list, max_length=12
    )
    source_bindings: List[ValidationSourceBinding] = Field(
        default_factory=list, max_length=24
    )
    next_validation_work: List[str] = Field(default_factory=list, max_length=20)


_CASE_COLUMNS = (
    "exposure_level",
    "n_rows",
    "exposure_denominator",
    "exposure_pct",
    "outcome_events",
    "outcome_denominator",
    "outcome_rate_pct",
    "interval_method",
)


def _case_table(result_tables: Mapping[str, Any]) -> Optional[ValidationCaseTable]:
    fallback: Optional[ValidationCaseTable] = None
    for raw in list(result_tables.get("tables") or [])[:40]:
        if not isinstance(raw, Mapping):
            continue
        headers = [_text(value, 120) for value in list(raw.get("headers") or [])]
        header_set = set(headers)
        if not {"exposure_level", "n_rows", "exposure_denominator", "exposure_pct"}.issubset(header_set):
            continue
        selected = [column for column in _CASE_COLUMNS if column in headers]
        indices = [headers.index(column) for column in selected]
        rows = []
        for source_row in list(raw.get("rows") or [])[:8]:
            if not isinstance(source_row, (list, tuple)):
                continue
            rows.append(
                [
                    _text(source_row[index], 160) if index < len(source_row) else ""
                    for index in indices
                ]
            )
        if rows:
            table = ValidationCaseTable(
                name=_text(raw.get("name"), 160) or "aggregate_distribution",
                label=_text(raw.get("label"), 300) or "Aggregate case-study result",
                evidence_id=_text(raw.get("evidence_id"), 160),
                columns=selected,
                rows=rows,
            )
            if {"outcome_events", "outcome_rate_pct"}.issubset(header_set):
                return table
            fallback = fallback or table
    return fallback


def _scientific_findings(
    readiness: Mapping[str, Any],
) -> List[ValidationFinding]:
    severity_rank = {"blocker": 0, "major": 1, "minor": 2}
    raw_rows = [
        row for row in list(readiness.get("findings") or []) if isinstance(row, Mapping)
    ]
    priority = {
        "NOVELTY_POSITIONING_NOT_ESTABLISHED": 0,
        "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED": 1,
        "LITERATURE_RETRIEVAL_NOT_CONDUCTED": 2,
        "INDEPENDENT_SCIENTIFIC_REVIEW_NOT_AVAILABLE": 3,
        "MANUSCRIPT_EXACT_LITERATURE_BINDING_INCOMPLETE": 4,
        "PAPER_AUTHORITY_NOT_GRANTED": 5,
        "IDEA_PRIOR_ART_AUTHORITY_NOT_ESTABLISHED": 6,
        "PUBLICATION_FIGURE_SOURCE_DATA_NOT_VERIFIED": 7,
    }
    raw_rows.sort(
        key=lambda row: (
            severity_rank.get(_text(row.get("severity"), 40).lower(), 9),
            priority.get(_text(row.get("code"), 160), 99),
            _text(row.get("code"), 160),
        )
    )
    return [
        ValidationFinding(
            code=_text(row.get("code"), 160) or "UNCLASSIFIED_FINDING",
            severity=_text(row.get("severity"), 40) or "unknown",
            domain=_text(row.get("domain"), 80) or "scientific_readiness",
            message=_text(row.get("message"), 1_000),
            remediation=_text(row.get("remediation"), 1_000),
            evidence_refs=[
                text
                for text in (
                    _text(value, 160)
                    for value in list(row.get("evidence_refs") or [])[:12]
                )
                if text
            ],
        )
        for row in raw_rows[:12]
    ]


def build_system_validation_report(
    *,
    run_id: str,
    projections: Mapping[str, Mapping[str, Any]],
    run_status: Optional[Mapping[str, Any]] = None,
    review_checkpoint: Optional[Mapping[str, Any]] = None,
    provider_usage: Optional[Mapping[str, Any]] = None,
    projection_privacy_passed: bool,
) -> SystemValidationReport:
    """Build a report from existing projections without deriving clinical facts."""

    run_context = projections.get("run_context.json") or {}
    plan = projections.get("agent_plan.json") or {}
    gate_payload = projections.get("quality_gate.json") or {}
    gate = gate_payload.get("gate") if isinstance(gate_payload, Mapping) else {}
    gate = gate if isinstance(gate, Mapping) else {}
    readiness = projections.get("scientific_readiness.json") or {}
    source_manifest = projections.get("source_run_manifest.json") or {}
    result_tables = projections.get("result_tables.json") or {}
    figure_gallery = (
        projections.get("system_validation_figure_gallery.json")
        or projections.get("figure_gallery.json")
        or {}
    )
    status_payload = run_status if isinstance(run_status, Mapping) else {}
    status_gates = status_payload.get("gates")
    status_gates = status_gates if isinstance(status_gates, Mapping) else {}
    checkpoint = review_checkpoint if isinstance(review_checkpoint, Mapping) else {}

    steps = [row for row in list(plan.get("steps") or []) if isinstance(row, Mapping)]
    planned_steps = len(steps)
    completed_steps = int(status_gates.get("completed_step_count") or 0)
    manifest_readiness = source_manifest.get("readiness")
    manifest_readiness = (
        manifest_readiness if isinstance(manifest_readiness, Mapping) else {}
    )
    if not completed_steps and bool(manifest_readiness.get("execution_complete")):
        completed_steps = planned_steps
    execution_complete = bool(
        status_gates.get("execution_complete")
        or manifest_readiness.get("execution_complete")
    )
    checkpoint_model = None
    try:
        checkpoint_model = HumanReviewCheckpoint.model_validate(
            {key: value for key, value in checkpoint.items() if key != "_source_sha256"}
        )
    except ValueError:
        pass
    review_approved = False
    if (
        checkpoint_model is not None
        and checkpoint_model.run_id == run_id
        and checkpoint_model.state
        in {
            "approved_pending_execution",
            "executing",
            "write_in_progress",
            "finalize_in_progress",
            "completed",
        }
    ):
        requests = {
            (request.review_id, request.authority_sha256)
            for request in checkpoint_model.requests
        }
        decisions = {
            (
                _text(row.get("review_id"), 200),
                _text(row.get("authority_sha256"), 64).lower(),
            )
            for row in checkpoint_model.approved_decisions
            if _text(row.get("decision"), 40).lower() == "approved"
        }
        review_approved = (
            bool(requests)
            and len(checkpoint_model.approved_decisions) == len(requests)
            and decisions == requests
        )
    evidence_count = int(source_manifest.get("evidence_count") or 0)
    table_count = int(result_tables.get("table_count") or 0)
    figure_count = len(list(figure_gallery.get("figures") or []))
    manuscript_ready = bool(
        manifest_readiness.get("manuscript_ready")
    )
    publication_authorized = bool(
        manifest_readiness.get("paper_authorized")
    )
    if manuscript_ready or publication_authorized:
        raise ValueError(
            "system_validation_report_requires_withheld_manuscript_and_publication_authority"
        )

    usage = None
    if isinstance(provider_usage, Mapping):
        usage = ProviderUsage(
            status=_text(provider_usage.get("status"), 80) or "unknown",
            calls=max(0, int(provider_usage.get("calls") or 0)),
            accounted_tokens=max(
                0, int(provider_usage.get("accounted_tokens") or 0)
            ),
            estimated_cost_usd=max(
                0.0, float(provider_usage.get("estimated_cost_usd") or 0.0)
            ),
            ledger_sha256=(
                _text(provider_usage.get("ledger_sha256"), 64).lower() or None
            ),
        )

    figure_rows = []
    for row in list(figure_gallery.get("figures") or [])[:8]:
        if not isinstance(row, Mapping):
            continue
        figure_rows.append(
            ValidationFigure(
                name=_text(row.get("name") or row.get("relative_path"), 160),
                label=_text(row.get("label"), 240) or "Run figure",
                status=_text(row.get("status"), 80) or "available",
            )
        )

    analysis_type = _text(plan.get("analysis_type"), 120) or "not_declared"
    scientific_ceiling = "descriptive_only"
    for step in steps:
        claim = step.get("descriptive_claim")
        if isinstance(claim, Mapping) and claim.get("claim_ceiling"):
            scientific_ceiling = _text(claim.get("claim_ceiling"), 120)
            break

    metrics = [
        ValidationMetric(
            label="Execution",
            value=f"{completed_steps}/{planned_steps} steps",
            detail="Completed reviewed plan steps; this is an engineering execution measure.",
            evidence_refs=["agent_plan.json", "source_run_manifest.json"],
        ),
        ValidationMetric(
            label="Registered evidence",
            value=f"{evidence_count:,}",
            detail="Pipeline evidence records projected for this exact run.",
            evidence_refs=["source_run_manifest.json", "evidence_ledger.json"],
        ),
        ValidationMetric(
            label="Review surfaces",
            value=f"{table_count} tables / {figure_count} figures",
            detail="Bounded aggregate browser projections; no patient rows are included.",
            evidence_refs=["result_tables.json", "figure_gallery.json"],
        ),
        ValidationMetric(
            label="Formal paper gate",
            value="WITHHELD AS DESIGNED",
            detail=(
                "The safety boundary prevented technical success from becoming unsupported manuscript authority."
            ),
            evidence_refs=["quality_gate.json", "scientific_readiness.json"],
        ),
    ]
    if usage is not None:
        metrics.append(
            ValidationMetric(
                label="Provider usage",
                value=f"{usage.calls} calls / {usage.accounted_tokens:,} tokens",
                detail=f"Recorded estimated cost: ${usage.estimated_cost_usd:.5f}.",
                evidence_refs=["provider_hard_stop_ledger.json"],
            )
        )

    lifecycle = [
        ValidationStage(
            stage="plan",
            label="Typed plan",
            status="verified" if planned_steps else "blocked",
            summary=(
                f"{planned_steps} typed steps were projected from the run's current plan authority."
                if planned_steps
                else "No typed plan was available."
            ),
            evidence_refs=["agent_plan.json"],
        ),
        ValidationStage(
            stage="review",
            label="Development review",
            status="verified" if review_approved else "not_assessed",
            summary=(
                "An exact-plan development approval was recorded; it is not attributable publication review."
                if review_approved
                else "No approved development-review checkpoint was projected."
            ),
            evidence_refs=["human_review_checkpoint.json"],
        ),
        ValidationStage(
            stage="execute",
            label="Deterministic execution",
            status="verified" if execution_complete else "blocked",
            summary=f"{completed_steps} of {planned_steps} required plan steps completed.",
            evidence_refs=["source_run_manifest.json"],
        ),
        ValidationStage(
            stage="project",
            label="Bounded artifact projection",
            status="verified" if projection_privacy_passed else "blocked",
            summary=(
                "Aggregate tables and figures passed the browser projection privacy boundary."
                if projection_privacy_passed
                else "The browser projection privacy boundary did not pass."
            ),
            evidence_refs=["evidence_ledger.json", "result_tables.json"],
        ),
        ValidationStage(
            stage="write",
            label="Evidence-bound manuscript",
            status="withheld",
            summary="STRICT evidence binding withheld the formal manuscript.",
            evidence_refs=["manuscript_draft.json", "quality_gate.json"],
        ),
        ValidationStage(
            stage="publish",
            label="Publication authority",
            status="withheld",
            summary="Scientific and publication gates remain closed.",
            evidence_refs=["scientific_readiness.json", "quality_gate.json"],
        ),
    ]

    source_bindings = [
        ValidationSourceBinding(
            artifact=name,
            sha256=projection_payload_sha256(payload),
            binding_scope="browser_projection_payload",
        )
        for name, payload in sorted(projections.items())
        if name
        in {
            "agent_plan.json",
            "figure_gallery.json",
            "system_validation_figure_gallery.json",
            "manuscript_draft.json",
            "quality_gate.json",
            "result_tables.json",
            "run_context.json",
            "scientific_readiness.json",
            "source_run_manifest.json",
        }
    ]
    checkpoint_sha256 = _text(checkpoint.get("_source_sha256"), 64).lower()
    if re.fullmatch(r"[a-f0-9]{64}", checkpoint_sha256):
        source_bindings.append(
            ValidationSourceBinding(
                artifact="human_review_checkpoint.json",
                sha256=checkpoint_sha256,
                binding_scope="run_private_receipt",
            )
        )
    if usage is not None and re.fullmatch(
        r"[a-f0-9]{64}", str(usage.ledger_sha256 or "")
    ):
        source_bindings.append(
            ValidationSourceBinding(
                artifact="provider_hard_stop_ledger.json",
                sha256=str(usage.ledger_sha256),
                binding_scope="run_private_receipt",
            )
        )

    engineering_complete = bool(
        planned_steps
        and review_approved
        and execution_complete
        and projection_privacy_passed
        and table_count
        and figure_count
    )
    return SystemValidationReport(
        run_id=_text(run_id, 160),
        title="A Complete, Governed Research Workflow",
        subtitle="Reviewer demonstration dossier from one real ICU analysis run",
        status=(
            "engineering_validation_complete"
            if engineering_complete
            else "engineering_validation_incomplete"
        ),
        executive_summary=(
            "This reviewer demonstration follows one exact plan from approval through execution, aggregate "
            "results, figures, provenance, and final authority adjudication. The clinical phenotype analysis "
            "is a bounded test case; the demonstrated contribution is the governed workflow."
        ),
        thesis=(
            "The reviewer workflow completed end to end: analysis and evidence projection succeeded, and the "
            "same workflow correctly withheld a manuscript whose scientific authority was not established."
        ),
        metrics=metrics,
        lifecycle=lifecycle,
        demonstrated=[
            "A typed plan can cross development review and execute all required steps.",
            "Aggregate tables and figures can be projected without exposing identifier columns or host paths.",
            "Provider usage, run identity, evidence inventory, and projection inputs remain inspectable.",
            "A successful analysis does not automatically unlock a manuscript or publication claim.",
        ],
        not_demonstrated=[
            "Clinical novelty, causal validity, predictive performance, or transportability.",
            "A finalized publication data seal or externally validated multi-database result.",
            "Independent attributable scientific review or formal publication activation.",
            "Superiority over clinicians, conventional workflows, or competing research agents.",
        ],
        case_study=ValidationCaseStudy(
            question=_text(run_context.get("question"), 1_200),
            analysis_type=analysis_type,
            scientific_claim_ceiling=scientific_ceiling,
            primary_table=_case_table(result_tables),
            figures=figure_rows,
        ),
        provider_usage=usage,
        scientific_findings=_scientific_findings(readiness),
        source_bindings=source_bindings,
        next_validation_work=[
            "Add multi-task and multi-database benchmark cases with prespecified success criteria.",
            "Compare governed and ungoverned agents on plan drift, hallucination, evidence laundering, and recovery.",
            "Run ablations for each authority boundary and report error-detection sensitivity and false blocks.",
            "Obtain independent expert review and quantify reproducibility, time, and cost against human workflows.",
        ],
    )


def _safe_png_data_url(value: Any) -> str:
    text = str(value or "")
    if re.fullmatch(r"data:image/png;base64,[A-Za-z0-9+/=]+", text):
        return text
    return ""


def render_system_validation_html(
    report: SystemValidationReport,
    *,
    figure_gallery: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render a self-contained, script-free dossier suitable for browser/PDF QA."""

    def e(value: Any) -> str:
        return escape(str(value or ""), quote=True)
    report_payload_sha256 = projection_payload_sha256(
        report.model_dump(mode="json")
    )
    status_label = (
        "REVIEWER DEMONSTRATION COMPLETE"
        if report.status == "engineering_validation_complete"
        else "ENGINEERING VALIDATION INCOMPLETE"
    )
    metric_html = "".join(
        f'<article class="metric"><span>{e(row.label)}</span><strong>{e(row.value)}</strong><p>{e(row.detail)}</p></article>'
        for row in report.metrics
    )
    lifecycle_html = "".join(
        f'<article class="stage {e(row.status)}"><div class="stage-mark"></div><div><span>{e(row.stage)}</span><h3>{e(row.label)}</h3><p>{e(row.summary)}</p><code>{e(" · ".join(row.evidence_refs))}</code></div></article>'
        for row in report.lifecycle
    )
    finding_html = "".join(
        f'<tr><td><span class="severity {e(row.severity)}">{e(row.severity)}</span></td><td><code class="finding-code">{e(row.code.replace("_", " "))}</code><small>{e(row.domain)}</small></td><td>{e(row.message)}</td><td>{e(row.remediation)}</td></tr>'
        for row in report.scientific_findings[:6]
    )
    binding_html = "".join(
        f'<tr><td>{e(row.artifact)}<small>{e(row.binding_scope.replace("_", " "))}</small></td><td><code>{e(row.sha256)}</code></td></tr>'
        for row in report.source_bindings
    )
    case_table_html = ""
    if report.case_study.primary_table is not None:
        table = report.case_study.primary_table
        case_table_html = (
            f'<div class="table-title"><strong>{e(table.label)}</strong><code>{e(table.evidence_id)}</code></div>'
            '<div class="table-wrap case-table-wrap"><table><thead><tr>'
            + "".join(f"<th>{e(column)}</th>" for column in table.columns)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{e(cell)}</td>" for cell in row) + "</tr>"
                for row in table.rows
            )
            + "</tbody></table></div>"
        )
    figure_html = ""
    gallery = figure_gallery if isinstance(figure_gallery, Mapping) else {}
    for row in list(gallery.get("figures") or [])[:3]:
        if not isinstance(row, Mapping):
            continue
        source = _safe_png_data_url(row.get("data_url"))
        if not source:
            continue
        figure_html += (
            f'<figure><img src="{source}" alt="{e(row.get("label") or "Run figure")}">'
            f'<figcaption>{e(row.get("label") or row.get("name") or "Run figure")}'
            + (
                f'<small>{e(row.get("projection_note"))}</small>'
                if row.get("projection_note")
                else ""
            )
            + "</figcaption></figure>"
        )

    def list_html(values: List[str]) -> str:
        return "".join(f"<li>{e(value)}</li>" for value in values)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="easyicu-authority-class" content="engineering_validation_only">
  <meta name="easyicu-report-payload-sha256" content="{report_payload_sha256}">
  <title>{e(report.title)}</title>
  <style>
    :root{{--ink:#17211d;--muted:#607069;--paper:#f7f4ec;--card:#fffdf8;--line:#d8d4c8;--green:#0b725e;--teal:#0d9488;--red:#a33c32;--amber:#a66b19;--navy:#18364b}}
    *{{box-sizing:border-box}} html{{background:#dfe5e0}} body{{margin:0;color:var(--ink);background:var(--paper);font:15px/1.6 Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
    main{{max-width:1120px;margin:0 auto;background:var(--paper);box-shadow:0 0 80px #22382d24}}
    header.hero{{position:relative;overflow:hidden;padding:72px 72px 54px;color:white;background:linear-gradient(125deg,#102d2a 0%,#16483f 54%,#173548 100%)}}
    header.hero:after{{content:"";position:absolute;right:-90px;top:-190px;width:520px;height:520px;border:1px solid #a7f3d055;border-radius:50%;box-shadow:0 0 0 70px #a7f3d012,0 0 0 145px #a7f3d00a}}
    .kicker{{text-transform:uppercase;letter-spacing:.18em;font-size:12px;font-weight:800;color:#8de3cf}} h1{{max-width:760px;margin:15px 0 10px;font:700 54px/1.02 Georgia,serif;letter-spacing:-.035em}}
    .subtitle{{font-size:20px;color:#cde8e1}} .run{{margin-top:32px;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:#b8d6ce}}
    .hero-state{{position:relative;z-index:1;display:inline-flex;margin-top:28px;padding:10px 14px;border:1px solid #72d8c2aa;border-radius:999px;background:#062d287a;font-size:12px;font-weight:800;letter-spacing:.08em}}
    section{{padding:50px 72px;border-bottom:1px solid var(--line)}} h2{{margin:0 0 10px;font:700 34px/1.1 Georgia,serif;letter-spacing:-.02em}} h3{{margin:2px 0 4px;font-size:17px}} .lead{{max-width:880px;font-size:18px;color:#33433d}}
    .thesis{{margin-top:24px;padding:22px 26px;border-left:5px solid var(--teal);background:#e8f3ef;font:600 20px/1.5 Georgia,serif}}
    .metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:14px;margin-top:28px}} .metric{{padding:20px;border:1px solid var(--line);border-radius:12px;background:var(--card)}}
    .metric span{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}} .metric strong{{display:block;margin:5px 0;font-size:23px;color:var(--navy)}} .metric p{{margin:0;color:var(--muted);font-size:13px}}
    .lifecycle{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:28px}} .stage{{display:flex;gap:14px;padding:20px;border:1px solid var(--line);border-radius:12px;background:var(--card)}}
     .stage-mark{{flex:0 0 10px;width:10px;height:10px;margin-top:8px;border-radius:50%;background:#8b9792}} .stage.verified .stage-mark{{background:var(--green);box-shadow:0 0 0 5px #0b725e18}} .stage.withheld .stage-mark{{background:var(--amber);box-shadow:0 0 0 5px #a66b1918}} .stage.blocked .stage-mark{{background:var(--red);box-shadow:0 0 0 5px #a33c3218}}
    .stage span{{font:700 11px ui-monospace,monospace;text-transform:uppercase;color:var(--muted)}} .stage p{{margin:0;color:#46554f;font-size:13px}} code{{overflow-wrap:anywhere;font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;color:#4d625a}}
    .split{{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:26px}} .panel{{padding:24px;border:1px solid var(--line);border-radius:14px;background:var(--card)}} .panel.good{{border-top:5px solid var(--green)}} .panel.bounded{{border-top:5px solid var(--amber)}} ul{{margin:14px 0 0;padding-left:21px}} li{{margin:8px 0}}
    .case-banner{{display:flex;justify-content:space-between;gap:20px;margin:18px 0 24px;padding:16px 20px;border-radius:10px;background:#edf0e8}} .case-banner strong{{display:block}} .case-banner code{{text-align:right}}
    .table-title{{display:flex;justify-content:space-between;gap:16px;margin:22px 0 8px}} .table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:10px;background:white}} table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{padding:10px 12px;border-bottom:1px solid #e8e5dc;text-align:left;vertical-align:top}} th{{background:#e9eee9;color:#32443d;font-size:11px;text-transform:uppercase;letter-spacing:.04em}} td small{{display:block;color:var(--muted)}}
    .figures{{display:grid;grid-template-columns:1fr;gap:22px;margin-top:28px}} figure{{margin:0;padding:16px;border:1px solid var(--line);border-radius:12px;background:white}} figure img{{display:block;width:100%;height:auto}} figcaption{{padding:10px 4px 0;color:var(--muted);font-size:13px}} figcaption small{{display:block;margin-top:4px;color:#344940}}
    .severity{{display:inline-block;padding:3px 7px;border-radius:999px;background:#eee;font-size:10px;font-weight:800;text-transform:uppercase}} .severity.blocker{{color:#8d211d;background:#f9dedb}} .severity.major{{color:#7a4c0a;background:#fae9c8}} .finding-code{{word-break:normal;overflow-wrap:normal;white-space:normal}}
    .bindings td:last-child code{{font-size:10px}} footer{{padding:30px 72px 48px;color:#64736d;background:#edf0ea;font-size:12px}} footer strong{{color:#243a31}}
    @media(max-width:800px){{header.hero,section,footer{{padding-left:24px;padding-right:24px}} h1{{font-size:40px}} .metrics,.lifecycle,.split{{grid-template-columns:1fr}} .case-banner{{display:block}} .case-banner code{{display:block;margin-top:8px;text-align:left}}}}
    @media print{{html{{background:white}} body{{background:white;font-size:11px}} main{{max-width:none;box-shadow:none}} header.hero{{padding:38px 44px 30px;-webkit-print-color-adjust:exact;print-color-adjust:exact}} h1{{font-size:38px}} section{{padding:26px 44px}} h2{{font-size:25px;break-after:avoid}} .metrics{{grid-template-columns:repeat(5,1fr);gap:8px}} .metric{{padding:13px}} .metric strong{{font-size:17px}} .metric p{{font-size:9px}} .lifecycle{{grid-template-columns:repeat(3,1fr);gap:8px}} .stage{{padding:13px}} .stage code{{display:none}} .split{{break-inside:avoid}} .metric,.stage,.panel,figure,tr{{break-inside:avoid}} thead{{display:table-header-group}} th,td{{padding:7px 8px}} .case-table-wrap{{overflow:visible}} .case-table-wrap table{{table-layout:fixed;font-size:7px}} .case-table-wrap th,.case-table-wrap td{{padding:4px;overflow-wrap:anywhere}} .figures figure{{break-inside:avoid}} footer{{padding:22px 44px}}}}
  </style>
</head>
<body>
<main>
  <header class="hero">
    <div class="kicker">EasyICU · Reviewer Demonstration</div>
    <h1>{e(report.title)}</h1>
    <div class="subtitle">{e(report.subtitle)}</div>
    <div class="hero-state">{status_label}</div>
    <div class="run">Run {e(report.run_id)} · authority={e(report.authority_class)} · publication_authorized=false<br>report_payload_sha256={report_payload_sha256}</div>
  </header>
  <section>
    <h2>Executive finding</h2>
    <p class="lead">{e(report.executive_summary)}</p>
    <div class="thesis">{e(report.thesis)}</div>
    <div class="metrics">{metric_html}</div>
  </section>
  <section>
    <h2>Authority-aware lifecycle</h2>
    <p class="lead">Green marks a verified workflow boundary. Amber marks an authority intentionally withheld by the safety policy; it is an expected validation outcome, not a failed demonstration.</p>
    <div class="lifecycle">{lifecycle_html}</div>
  </section>
  <section>
    <h2>What the case establishes</h2>
    <div class="split">
      <div class="panel good"><h3>Demonstrated</h3><ul>{list_html(report.demonstrated)}</ul></div>
      <div class="panel bounded"><h3>Outside this demonstration</h3><ul>{list_html(report.not_demonstrated)}</ul></div>
    </div>
  </section>
  <section>
    <h2>Bounded ICU case study</h2>
    <p class="lead">The phenotype analysis is retained as an execution and evidence-binding test case. It is not presented as the system paper's novelty claim.</p>
    <div class="case-banner"><div><strong>{e(report.case_study.analysis_type)}</strong><span>scientific ceiling: {e(report.case_study.scientific_claim_ceiling)}</span></div><code>generated_numbers=false</code></div>
    {case_table_html}
    <div class="figures">{figure_html}</div>
  </section>
  <section>
    <h2>Safety outcome: manuscript authority withheld</h2>
    <p class="lead">The completed reviewer workflow exposes the strongest unresolved scientific requirements instead of laundering them into polished prose.</p>
    <div class="table-wrap"><table><thead><tr><th>Severity</th><th>Finding</th><th>Why it matters</th><th>Required remediation</th></tr></thead><tbody>{finding_html}</tbody></table></div>
  </section>
  <section>
    <h2>Validation programme needed for a systems paper</h2>
    <ol>{list_html(report.next_validation_work)}</ol>
  </section>
  <section>
    <h2>Projection bindings</h2>
    <p class="lead">Each digest binds the browser-safe projection consumed by this dossier. The report does not read raw patient rows.</p>
    <div class="table-wrap bindings"><table><thead><tr><th>Source artifact and binding scope</th><th>SHA-256</th></tr></thead><tbody>{binding_html}</tbody></table></div>
  </section>
  <footer><strong>REVIEWER DEMONSTRATION · ENGINEERING VALIDATION ONLY · NOT A CLINICAL MANUSCRIPT</strong><br>This deterministic dossier demonstrates the system workflow; it does not grant scientific, clinical, or publication authority to the bounded case study.</footer>
</main>
</body>
</html>"""


def build_system_validation_receipt(
    *,
    report_payload: Mapping[str, Any],
    html_bytes: bytes,
    pdf_bytes: Optional[bytes] = None,
    report_bytes: Optional[bytes] = None,
) -> dict[str, Any]:
    exact_report_bytes = report_bytes
    if exact_report_bytes is None:
        exact_report_bytes = json.dumps(
            report_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
    receipt: dict[str, Any] = {
        "schema_version": "easyicu.system-validation-report-receipt/1",
        "authority_class": "engineering_validation_only",
        "claim_ceiling": "engineering_validation_only",
        "reportable": False,
        "publication_authorized": False,
        "report": {
            "name": "system_validation_report.json",
            "sha256": hashlib.sha256(exact_report_bytes).hexdigest(),
            "bytes": len(exact_report_bytes),
        },
        "html": {
            "name": "system_validation_report.html",
            "sha256": hashlib.sha256(html_bytes).hexdigest(),
            "bytes": len(html_bytes),
        },
        "pdf": None,
    }
    if pdf_bytes is not None:
        receipt["pdf"] = {
            "name": "system_validation_report.pdf",
            "sha256": hashlib.sha256(pdf_bytes).hexdigest(),
            "bytes": len(pdf_bytes),
        }
    return receipt


__all__ = [
    "SystemValidationReport",
    "build_system_validation_receipt",
    "build_system_validation_report",
    "projection_payload_sha256",
    "render_system_validation_html",
]
