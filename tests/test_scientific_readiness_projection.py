"""Focused fail-closed tests for the Web scientific-readiness owner."""

from __future__ import annotations

import json
from pathlib import Path

from easyicu.webserver import agent_pipeline_runs, agent_runs
from easyicu.webserver.scientific_readiness_projection import (
    build_scientific_readiness_projection,
)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_historical_engineering_run_fails_closed_for_publication(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "cohort_provenance.json",
        {
            "database": "miiv",
            "cohort_definition": None,
            "export_authority": {"authority_sha256": "a" * 64},
        },
    )
    _write(
        tmp_path / "reviewer_report.json",
        {
            "summary": {
                "aggregated_recommendation": "major_revision",
                "counts": {"major": 1, "reject": 0},
            }
        },
    )
    _write(
        tmp_path / "reporting_checklist_strobe.json",
        {"items": [{"item_id": "12d", "status": "open"}]},
    )
    projection = build_scientific_readiness_projection(
        run_id="run-historical",
        run_dir=tmp_path,
        axes={
            "analysis_validated": True,
            "manuscript_ready": True,
            "publication_ready": False,
            "paper_authorized": False,
            "display_suite_complete": False,
            "display_suite_errors": ["absolute-risk panel missing"],
        },
        literature_evidence={
            "status": "curated_only",
            "citation_count": 9,
            "mapping_status": "not_bound",
            "search": {
                "search_conducted": False,
                "curated_seed_count": 9,
                "sources_returning": [],
            },
        },
        study={},
    )

    assert projection.status == "analysis_only"
    assert projection.claim_ceiling == "analysis_only"
    assert projection.publication_ready is False
    assert projection.paper_authorized is False
    codes = {finding.code for finding in projection.findings}
    assert codes == {
        "IDEA_PRIOR_ART_AUTHORITY_NOT_ESTABLISHED",
        "LITERATURE_RETRIEVAL_NOT_CONDUCTED",
        "LITERATURE_PLAN_BINDING_INCOMPLETE",
        "COHORT_SOURCE_SCOPE_NOT_EXPLICIT",
        "SCIENTIFIC_REVIEW_MAJOR_REVISION_OPEN",
        "PUBLICATION_DISPLAY_SUITE_INCOMPLETE",
        "REPORTING_CHECKLIST_ITEMS_OPEN",
        "PAPER_AUTHORITY_NOT_GRANTED",
    }
    by_domain = {domain.domain: domain.status for domain in projection.domains}
    assert by_domain == {
        "idea": "not_assessed",
        "literature": "blocked",
        "data": "review_required",
        "analysis": "blocked",
        "manuscript": "blocked",
    }
    assert projection.facts["literature"]["search_conducted"] is False
    assert projection.facts["analysis"]["reviewer_recommendation"] == ("major_revision")


def test_exact_owner_receipts_can_project_publication_ready(tmp_path: Path) -> None:
    _write(
        tmp_path / "cohort_provenance.json",
        {
            "database": "miiv",
            "cohort_definition": {"population": "all eligible adult ICU stays"},
            "export_authority": {"authority_sha256": "a" * 64},
        },
    )
    _write(
        tmp_path / "reviewer_report.json",
        {
            "summary": {
                "aggregated_recommendation": "accept",
                "counts": {"major": 0, "reject": 0},
            }
        },
    )
    _write(
        tmp_path / "reporting_checklist_strobe.json",
        {"items": [{"item_id": "12d", "status": "not_applicable"}]},
    )
    projection = build_scientific_readiness_projection(
        run_id="run-ready",
        run_dir=tmp_path,
        axes={
            "analysis_validated": True,
            "manuscript_ready": True,
            "publication_ready": True,
            "paper_authorized": True,
            "display_suite_complete": True,
            "display_suite_errors": [],
        },
        literature_evidence={
            "status": "searched",
            "citation_count": 12,
            "mapping_status": "complete",
            "scientific_mapping_status": "complete",
            "search": {
                "search_conducted": True,
                "searched_at": "2026-08-11T12:00:00+00:00",
                "sources_returning": ["pubmed"],
            },
        },
        study={
            "idea_handoff": {
                "status": "accepted",
                "canonical_handoff_sha256": "b" * 64,
                "prior_art_sha256": "c" * 64,
                "prior_art_status": "complete",
                "prior_art_result_count": 12,
                "prior_art_searched_at": "2026-08-11T12:00:00+00:00",
            }
        },
    )

    assert projection.status == "publication_ready"
    assert projection.claim_ceiling == "reportable"
    assert projection.findings == []
    assert all(domain.status == "passed" for domain in projection.domains)


def test_web_gate_distinguishes_draft_generation_from_publication() -> None:
    gate = agent_pipeline_runs._gate_from_axes(
        {
            "execution_complete": True,
            "analysis_validated": True,
            "evidence_complete": True,
            "numeric_verified": True,
            "manuscript_ready": True,
            "publication_ready": False,
            "paper_authorized": False,
        },
        pending=False,
    )
    by_id = {row["id"]: row for row in gate["checks"]}
    assert by_id["manuscript_ready"] == {
        "id": "manuscript_ready",
        "label": "evidence-bound draft generated",
        "passed": True,
        "reason": None,
    }
    assert by_id["publication_ready"]["passed"] is False
    assert by_id["paper_authorized"]["passed"] is False
    assert gate["reportable"] is False


def test_scientific_readiness_artifact_is_public_but_bounded() -> None:
    payload = {
        "schema_version": "easyicu.web-scientific-readiness/1",
        "status": "analysis_only",
        "findings": [{"code": "PAPER_AUTHORITY_NOT_GRANTED"}],
    }
    public = agent_runs._public_review_payloads({"scientific_readiness.json": payload})
    assert public == {"scientific_readiness.json": payload}
    assert "scientific_readiness.json" in agent_runs._RUN_ARTIFACT_NAMES


def test_maturity_authorization_request_is_projected_without_auto_upgrade(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "scientific_maturity_audit.json",
        {
            "status": "major_revision",
            "score": 72,
            "dimension_scores": {"statistical_design": 65},
            "findings": [
                {
                    "code": "UNADJUSTED_ASSOCIATION_NOT_ARTICLE_GRADE",
                    "severity": "major",
                    "dimension": "statistical_design",
                    "message": "The exact user-authorized analysis is unadjusted.",
                    "evidence_refs": ["manifest.json.current_plan_authority"],
                    "remediation": "Ask before creating a new adjusted study version.",
                    "requires_user_authorization": True,
                    "authorization_question": "Keep descriptive, or authorize a new adjusted version?",
                }
            ],
        },
    )
    projection = build_scientific_readiness_projection(
        run_id="run-auth",
        run_dir=tmp_path,
        axes={
            "analysis_validated": True,
            "manuscript_ready": True,
            "publication_ready": False,
            "paper_authorized": False,
            "display_suite_complete": True,
        },
        literature_evidence={
            "status": "searched",
            "citation_count": 1,
            "mapping_status": "complete",
            "scientific_mapping_status": "complete",
            "search": {
                "search_conducted": True,
                "sources_returning": ["pubmed"],
            },
        },
        study={
            "idea_handoff": {
                "status": "accepted",
                "canonical_handoff_sha256": "b" * 64,
                "prior_art_sha256": "c" * 64,
                "prior_art_status": "complete",
                "prior_art_result_count": 1,
                "prior_art_searched_at": "2026-08-12T00:00:00+00:00",
            }
        },
    )

    finding = next(
        row
        for row in projection.findings
        if row.code == "UNADJUSTED_ASSOCIATION_NOT_ARTICLE_GRADE"
    )
    assert projection.status == "analysis_only"
    assert projection.claim_ceiling == "analysis_only"
    assert finding.requires_user_authorization is True
    assert projection.facts["analysis"]["user_authorization_requests"] == [
        {
            "code": "UNADJUSTED_ASSOCIATION_NOT_ARTICLE_GRADE",
            "question": "Keep descriptive, or authorize a new adjusted version?",
        }
    ]
