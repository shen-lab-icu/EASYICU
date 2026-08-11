"""Browser-safe scientific and publication-readiness projection.

The scientific owners remain Idea Mining, Literature, Data Foundation, the
Research Agent validators, and Reporting.  This module does not repeat their
science and never upgrades an authority decision.  It compiles their immutable
receipts into one bounded Web contract so a technically complete run cannot be
presented as a reliable idea or publication-ready manuscript by omission.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field


DomainName = Literal["idea", "literature", "data", "analysis", "manuscript"]
DomainState = Literal["passed", "review_required", "blocked", "not_assessed"]
FindingSeverity = Literal["blocker", "major", "minor"]


class ScientificReadinessFinding(BaseModel):
    """One source-attributable readiness defect with a stable code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    domain: DomainName
    severity: FindingSeverity
    message: str
    evidence_refs: list[str] = Field(default_factory=list, max_length=20)
    remediation: str


class ScientificDomainReadiness(BaseModel):
    """Bounded status for one owner domain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    domain: DomainName
    status: DomainState
    summary: str
    evidence_refs: list[str] = Field(default_factory=list, max_length=20)


class ScientificReadinessProjection(BaseModel):
    """Cross-owner read-only projection; never a new scientific authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.web-scientific-readiness/1"] = (
        "easyicu.web-scientific-readiness/1"
    )
    run_id: str
    status: Literal[
        "blocked", "analysis_only", "ready_for_human_review", "publication_ready"
    ]
    claim_ceiling: Literal["unsupported", "analysis_only", "reportable"]
    publication_ready: bool
    paper_authorized: bool
    human_review_required: bool = True
    source: Literal["owner_artifacts_only"] = "owner_artifacts_only"
    domains: list[ScientificDomainReadiness] = Field(max_length=5)
    findings: list[ScientificReadinessFinding] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)


def _text(value: Any, limit: int = 1_200) -> str:
    return " ".join(str(value or "").split())[:limit]


def _read_json(run_dir: Path | None, name: str) -> Mapping[str, Any]:
    if run_dir is None:
        return {}
    try:
        path = run_dir / name
        if path.stat().st_size > 2 * 1024 * 1024:
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _idea_status(
    idea_handoff: Mapping[str, Any],
) -> tuple[ScientificDomainReadiness, list[ScientificReadinessFinding], dict[str, Any]]:
    accepted = _text(idea_handoff.get("status"), 80) == "accepted"
    digest = _text(idea_handoff.get("canonical_handoff_sha256"), 80)
    prior_digest = _text(idea_handoff.get("prior_art_sha256"), 80)
    prior_status = _text(idea_handoff.get("prior_art_status"), 80)
    prior_searched_at = _text(idea_handoff.get("prior_art_searched_at"), 80)
    try:
        result_count = max(0, int(idea_handoff.get("prior_art_result_count") or 0))
    except (TypeError, ValueError):
        result_count = 0
    reviewed = bool(
        accepted
        and len(digest) == 64
        and len(prior_digest) == 64
        and prior_status not in {"", "not_checked", "search_failed", "blocked"}
        and bool(prior_searched_at)
        and result_count > 0
    )
    findings: list[ScientificReadinessFinding] = []
    if reviewed:
        domain = ScientificDomainReadiness(
            domain="idea",
            status="passed",
            summary="The selected idea carries an accepted, digest-bound prior-art receipt.",
            evidence_refs=["StudyContext.idea_handoff"],
        )
    else:
        findings.append(
            ScientificReadinessFinding(
                code="IDEA_PRIOR_ART_AUTHORITY_NOT_ESTABLISHED",
                domain="idea",
                severity="blocker",
                message=(
                    "No accepted, digest-bound prior-art review proves that this "
                    "question is a reliable or sufficiently differentiated research idea."
                ),
                evidence_refs=["StudyContext.idea_handoff"],
                remediation=(
                    "Run Idea Mining prior-art retrieval, inspect same-topic hits, and "
                    "accept the refreshed handoff before treating the idea as publishable."
                ),
            )
        )
        domain = ScientificDomainReadiness(
            domain="idea",
            status="not_assessed",
            summary="Technical executability is not evidence of novelty or publication value.",
            evidence_refs=["StudyContext.idea_handoff"],
        )
    return (
        domain,
        findings,
        {
            "handoff_accepted": accepted,
            "prior_art_status": prior_status or "not_checked",
            "prior_art_result_count": result_count,
            "prior_art_digest_bound": len(prior_digest) == 64,
            "prior_art_searched_at": prior_searched_at or None,
        },
    )


def _literature_status(
    literature: Mapping[str, Any],
) -> tuple[ScientificDomainReadiness, list[ScientificReadinessFinding], dict[str, Any]]:
    search = literature.get("search")
    search = search if isinstance(search, Mapping) else {}
    searched = bool(search.get("search_conducted"))
    sources = [
        _text(value, 80)
        for value in list(search.get("sources_returning") or [])[:12]
        if _text(value, 80)
    ]
    mapping_status = _text(
        literature.get("scientific_mapping_status") or literature.get("mapping_status"),
        40,
    )
    searched_at = _text(search.get("searched_at"), 80)
    citation_count = int(literature.get("citation_count") or 0)
    findings: list[ScientificReadinessFinding] = []
    if not searched:
        findings.append(
            ScientificReadinessFinding(
                code="LITERATURE_RETRIEVAL_NOT_CONDUCTED",
                domain="literature",
                severity="blocker",
                message=(
                    "The bundle contains curated seed references, but no retrieval "
                    "source completed a search; recency, relevance, and novelty are unverified."
                ),
                evidence_refs=["preplan_literature_bundle.json.search_provenance"],
                remediation=(
                    "Run a dated database search with inspectable queries and screening "
                    "receipt, then rebind the exact retained records to the plan."
                ),
            )
        )
    elif not searched_at:
        findings.append(
            ScientificReadinessFinding(
                code="LITERATURE_SEARCH_DATE_NOT_RECORDED",
                domain="literature",
                severity="major",
                message=(
                    "Retrieval is marked complete, but no bound search timestamp is "
                    "retained, so literature currency cannot be audited."
                ),
                evidence_refs=["preplan_literature_bundle.json.search_provenance"],
                remediation=(
                    "Refresh the search and retain its UTC searched_at value in the "
                    "digest-bound prior-art and literature receipts."
                ),
            )
        )
    if mapping_status not in {"complete", "not_applicable"}:
        findings.append(
            ScientificReadinessFinding(
                code="LITERATURE_PLAN_BINDING_INCOMPLETE",
                domain="literature",
                severity="major",
                message=(
                    "Not every governed scientific plan step has an exact retained "
                    "citation binding; a paper list alone does not justify the design."
                ),
                evidence_refs=["analysis_plan.json", "preplan_literature_bundle.json"],
                remediation=(
                    "Bind each primary, secondary, and sensitivity decision to exact "
                    "citation keys, and remove irrelevant citations from those steps."
                ),
            )
        )
    passed = (
        searched
        and bool(searched_at)
        and bool(sources)
        and mapping_status
        in {
            "complete",
            "not_applicable",
        }
    )
    domain = ScientificDomainReadiness(
        domain="literature",
        status="passed" if passed else "blocked",
        summary=(
            "A dated search returned sources and the plan mapping is complete."
            if passed
            else "The current literature package cannot establish currency or design support."
        ),
        evidence_refs=["preplan_literature_bundle.json", "analysis_plan.json"],
    )
    return (
        domain,
        findings,
        {
            "search_conducted": searched,
            "searched_at": searched_at or None,
            "sources_returning": sources,
            "citation_count": citation_count,
            "mapping_status": mapping_status or "unavailable",
        },
    )


def _data_status(
    run_dir: Path | None,
) -> tuple[ScientificDomainReadiness, list[ScientificReadinessFinding], dict[str, Any]]:
    provenance = _read_json(run_dir, "cohort_provenance.json")
    cohort_definition = provenance.get("cohort_definition")
    export_authority = provenance.get("export_authority")
    database = _text(provenance.get("database"), 80) or None
    scope_explicit = bool(cohort_definition)
    authority_present = isinstance(export_authority, Mapping) and bool(export_authority)
    findings: list[ScientificReadinessFinding] = []
    if not scope_explicit:
        findings.append(
            ScientificReadinessFinding(
                code="COHORT_SOURCE_SCOPE_NOT_EXPLICIT",
                domain="data",
                severity="major",
                message=(
                    "The materialized cohort may be technically traceable, but the source "
                    "population, selection path, and representativeness are not explicitly closed."
                ),
                evidence_refs=["cohort_provenance.json"],
                remediation=(
                    "Persist the source population, eligibility/exclusion flow, source "
                    "coverage, and final denominator as a reproducible cohort definition."
                ),
            )
        )
    passed = bool(provenance) and scope_explicit and authority_present
    domain = ScientificDomainReadiness(
        domain="data",
        status="passed" if passed else "review_required",
        summary=(
            "The cohort definition and export authority are explicit."
            if passed
            else "Provenance exists, but scientific population scope is not fully established."
        ),
        evidence_refs=["cohort_provenance.json"],
    )
    return (
        domain,
        findings,
        {
            "database": database,
            "cohort_definition_explicit": scope_explicit,
            "export_authority_present": authority_present,
        },
    )


def _analysis_status(
    run_dir: Path | None,
    axes: Mapping[str, Any],
) -> tuple[ScientificDomainReadiness, list[ScientificReadinessFinding], dict[str, Any]]:
    report = _read_json(run_dir, "reviewer_report.json")
    summary = report.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    recommendation = _text(summary.get("aggregated_recommendation"), 80)
    counts = summary.get("counts") if isinstance(summary.get("counts"), Mapping) else {}
    major_count = int(counts.get("major") or 0) + int(counts.get("reject") or 0)
    analysis_validated = bool(axes.get("analysis_validated"))
    display_complete = bool(axes.get("display_suite_complete"))
    display_errors = [
        _text(value, 500)
        for value in list(axes.get("display_suite_errors") or [])[:20]
        if _text(value, 500)
    ]
    findings: list[ScientificReadinessFinding] = []
    if recommendation in {"major_revision", "reject"} or major_count:
        findings.append(
            ScientificReadinessFinding(
                code="SCIENTIFIC_REVIEW_MAJOR_REVISION_OPEN",
                domain="analysis",
                severity="blocker",
                message=(
                    "The persisted independent reviewer package still contains an "
                    "open major-revision or reject-level scientific finding."
                ),
                evidence_refs=["reviewer_report.json"],
                remediation=(
                    "Resolve or explicitly adjudicate every major finding and regenerate "
                    "the reviewer receipt before human publication review."
                ),
            )
        )
    if not display_complete and display_errors:
        findings.append(
            ScientificReadinessFinding(
                code="PUBLICATION_DISPLAY_SUITE_INCOMPLETE",
                domain="analysis",
                severity="major",
                message="The article-level result display suite is incomplete.",
                evidence_refs=["display_suite_audit.json"],
                remediation="Repair the named display-suite defects from the owner audit.",
            )
        )
    blocked = not analysis_validated or bool(
        recommendation in {"major_revision", "reject"} or major_count
    )
    domain = ScientificDomainReadiness(
        domain="analysis",
        status="blocked" if blocked else "passed",
        summary=(
            "Automated validation passed and no major reviewer finding remains."
            if not blocked
            else "Execution completeness does not close the open scientific review defects."
        ),
        evidence_refs=["run_status.json", "reviewer_report.json"],
    )
    return (
        domain,
        findings,
        {
            "analysis_validated": analysis_validated,
            "reviewer_recommendation": recommendation or "not_available",
            "reviewer_major_or_reject_count": major_count,
            "display_suite_complete": display_complete,
            "display_suite_errors": display_errors,
        },
    )


def _manuscript_status(
    run_dir: Path | None,
    axes: Mapping[str, Any],
) -> tuple[ScientificDomainReadiness, list[ScientificReadinessFinding], dict[str, Any]]:
    checklist = _read_json(run_dir, "reporting_checklist_strobe.json")
    raw_items = checklist.get("items")
    raw_items = raw_items if isinstance(raw_items, list) else []
    open_items = [
        _text(item.get("item_id"), 40)
        for item in raw_items
        if isinstance(item, Mapping)
        and _text(item.get("status"), 40).lower()
        not in {"addressed", "complete", "completed", "not_applicable", "na"}
    ]
    manuscript_ready = bool(axes.get("manuscript_ready"))
    publication_ready = bool(axes.get("publication_ready"))
    paper_authorized = bool(axes.get("paper_authorized"))
    findings: list[ScientificReadinessFinding] = []
    if open_items:
        findings.append(
            ScientificReadinessFinding(
                code="REPORTING_CHECKLIST_ITEMS_OPEN",
                domain="manuscript",
                severity="major",
                message="The reporting checklist contains unresolved items.",
                evidence_refs=["reporting_checklist_strobe.json"],
                remediation=(
                    "Address each item or record an evidence-backed not-applicable decision "
                    "before calling the draft submission-ready."
                ),
            )
        )
    if not (publication_ready and paper_authorized):
        findings.append(
            ScientificReadinessFinding(
                code="PAPER_AUTHORITY_NOT_GRANTED",
                domain="manuscript",
                severity="blocker",
                message=(
                    "An evidence-bound draft exists at most; publication authority has "
                    "not been granted by the Research Agent gates."
                ),
                evidence_refs=["run_status.json", "manifest.json"],
                remediation=(
                    "Close scientific, display, reporting, provenance, and human-review "
                    "gates on one exact run authority before external use."
                ),
            )
        )
    passed = (
        manuscript_ready and publication_ready and paper_authorized and not open_items
    )
    domain = ScientificDomainReadiness(
        domain="manuscript",
        status="passed" if passed else "blocked",
        summary=(
            "The exact run is publication-ready and paper-authorized."
            if passed
            else "Draft generation is not equivalent to publication readiness."
        ),
        evidence_refs=["run_status.json", "reporting_checklist_strobe.json"],
    )
    return (
        domain,
        findings,
        {
            "evidence_bound_draft_generated": manuscript_ready,
            "publication_ready": publication_ready,
            "paper_authorized": paper_authorized,
            "open_reporting_items": open_items,
        },
    )


def build_scientific_readiness_projection(
    *,
    run_id: Any,
    run_dir: Path | None,
    axes: Mapping[str, Any],
    literature_evidence: Mapping[str, Any],
    study: Mapping[str, Any],
) -> ScientificReadinessProjection:
    """Compile source-owner receipts without creating a new readiness verdict."""

    idea_handoff = study.get("idea_handoff")
    idea_handoff = idea_handoff if isinstance(idea_handoff, Mapping) else {}
    pieces = [
        _idea_status(idea_handoff),
        _literature_status(literature_evidence),
        _data_status(run_dir),
        _analysis_status(run_dir, axes),
        _manuscript_status(run_dir, axes),
    ]
    domains = [piece[0] for piece in pieces]
    findings = [finding for piece in pieces for finding in piece[1]]
    facts = {
        domain.domain: piece[2] for domain, piece in zip(domains, pieces, strict=True)
    }
    publication_ready = bool(axes.get("publication_ready"))
    paper_authorized = bool(axes.get("paper_authorized"))
    analysis_validated = bool(axes.get("analysis_validated"))
    manuscript_ready = bool(axes.get("manuscript_ready"))
    if publication_ready and paper_authorized and not findings:
        status = "publication_ready"
        claim_ceiling = "reportable"
    elif (
        manuscript_ready
        and analysis_validated
        and not any(finding.severity == "blocker" for finding in findings)
    ):
        status = "ready_for_human_review"
        claim_ceiling = "analysis_only"
    elif analysis_validated:
        status = "analysis_only"
        claim_ceiling = "analysis_only"
    else:
        status = "blocked"
        claim_ceiling = "unsupported"
    return ScientificReadinessProjection(
        run_id=_text(run_id, 160) or "unknown_run",
        status=status,
        claim_ceiling=claim_ceiling,
        publication_ready=publication_ready,
        paper_authorized=paper_authorized,
        domains=domains,
        findings=findings,
        facts=facts,
    )


__all__ = [
    "ScientificDomainReadiness",
    "ScientificReadinessFinding",
    "ScientificReadinessProjection",
    "build_scientific_readiness_projection",
]
