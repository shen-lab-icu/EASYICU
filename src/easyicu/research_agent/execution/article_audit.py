"""Run-level article audit orchestration.

This module owns the read-only projection that compares one completed execute
phase with the article analysis contract and figure strategy.  It deliberately
does not mutate :class:`EvidenceStore`: the execute-phase host remains the
owner of artifact registration and manifest persistence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from ..planning.figure_strategy import (
    validate_run_against_article_figure_strategy,
)
from ..planning.study_design import study_design_family_for_analysis_type
from ..reporting.article_contract import (
    article_contract_audit_payload,
    summarize_article_contract_coverage,
    validate_run_against_article_contract,
)
from ..schema import AnalysisPlan, EvidenceRecord, ResearchContext, ValidationFinding
from .figure_plan_binding import validate_planned_figure_contract_bindings


@dataclass(frozen=True)
class ArticleAuditArtifact:
    """Immutable registration request emitted by the article audit owner."""

    evidence_id: str
    kind: str
    description: str
    source_path: Path
    producer: str
    generation_mode: str


@dataclass(frozen=True)
class RunArticleAuditResult:
    """Findings and host-owned persistence work from both article audits."""

    findings: Tuple[ValidationFinding, ...]
    artifact: Optional[ArticleAuditArtifact]
    manifest_items: Tuple[Tuple[str, str], ...]


def collect_run_article_audits(
    *,
    context: ResearchContext,
    plan: Optional[AnalysisPlan],
    evidence_records: Sequence[EvidenceRecord],
    per_step_records: Sequence[Mapping[str, Any]],
    run_dir: Path,
) -> RunArticleAuditResult:
    """Run independent article-contract and figure-strategy audits.

    The plan is the final execute-phase plan, not the initial planner draft.
    Each audit has its own failure boundary so an implementation fault in one
    projection cannot suppress the other projection's findings.
    """

    findings: list[ValidationFinding] = []
    artifact: Optional[ArticleAuditArtifact] = None
    manifest_items: Tuple[Tuple[str, str], ...] = ()

    try:
        article_contract_status = summarize_article_contract_coverage(
            context=context,
            plan=plan,
            evidence_records=evidence_records,
            per_step_records=per_step_records,
            run_dir=run_dir,
        )
        article_contract_path = run_dir / "article_contract_audit.json"
        article_contract_path.write_text(
            json.dumps(
                article_contract_audit_payload(article_contract_status),
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        artifact = ArticleAuditArtifact(
            evidence_id="article_contract_audit",
            kind="log",
            description=(
                "Run-level article analysis contract audit: compares "
                "registered artifacts against required article display roles."
            ),
            source_path=article_contract_path,
            producer="article_contract",
            generation_mode="system",
        )
        findings.extend(
            validate_run_against_article_contract(
                context=context,
                plan=plan,
                evidence_records=evidence_records,
                per_step_records=per_step_records,
                run_dir=run_dir,
            )
        )
        manifest_items = (
            (
                "article_contract_audit",
                str(article_contract_path.relative_to(run_dir)),
            ),
        )
    except Exception as exc:
        findings.append(
            ValidationFinding(
                validator="article_analysis_contract",
                severity="warning",
                message=(
                    "Run-level article analysis contract audit failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        )

    try:
        analysis_family = (
            study_design_family_for_analysis_type(plan.analysis_type)
            if plan is not None and plan.analysis_type is not None
            else None
        )
        findings.extend(
            validate_run_against_article_figure_strategy(
                context=context,
                run_dir=run_dir,
                per_step_records=per_step_records,
                analysis_family=analysis_family,
            )
        )
    except Exception as exc:
        findings.append(
            ValidationFinding(
                validator="article_figure_strategy",
                severity="warning",
                message=(
                    "Run-level article figure strategy audit failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        )

    try:
        findings.extend(
            validate_planned_figure_contract_bindings(
                plan=plan,
                run_dir=run_dir,
                per_step_records=per_step_records,
            )
        )
    except Exception as exc:
        # The binding is scientific authority, not optional display QA.  A
        # validator implementation failure must therefore stay fail-closed.
        findings.append(
            ValidationFinding(
                validator="planned_figure_contract_binding",
                severity="error",
                message=(
                    "Planned figure-contract binding audit failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                detail={"reason": "binding_audit_failed"},
            )
        )

    return RunArticleAuditResult(
        findings=tuple(findings),
        artifact=artifact,
        manifest_items=manifest_items,
    )


__all__ = [
    "ArticleAuditArtifact",
    "RunArticleAuditResult",
    "collect_run_article_audits",
]
