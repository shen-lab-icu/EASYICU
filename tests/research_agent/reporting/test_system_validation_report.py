from __future__ import annotations

import hashlib

import pytest

from easyicu.research_agent.reporting.system_validation_report import (
    build_system_validation_receipt,
    build_system_validation_report,
    render_system_validation_html,
)
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.human_review_checkpoint import (
    HumanReviewCheckpoint,
)
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest


def _projections() -> dict:
    return {
        "run_context.json": {
            "run_id": "run_validation",
            "question": "Report bounded counts without causal inference.",
        },
        "agent_plan.json": {
            "analysis_type": "descriptive_epidemiology",
            "steps": [
                {"step_id": "01_cohort"},
                {
                    "step_id": "02_distribution",
                    "descriptive_claim": {"claim_ceiling": "descriptive_only"},
                },
            ],
        },
        "quality_gate.json": {
            "gate": {
                "status": "blocked",
                "reportable": False,
                "draft_unlocked": False,
            }
        },
        "scientific_readiness.json": {
            "status": "blocked",
            "paper_authorized": False,
            "findings": [
                {
                    "code": "NOVELTY_NOT_ESTABLISHED",
                    "severity": "blocker",
                    "domain": "idea",
                    "message": "No independent novelty comparison is available.",
                    "remediation": "Run a source-bound comparator review.",
                    "evidence_refs": ["novelty_positioning_audit.json"],
                }
            ],
        },
        "source_run_manifest.json": {
            "run_id": "run_validation",
            "evidence_count": 27,
            "readiness": {
                "execution_complete": True,
                "manuscript_ready": False,
                "paper_authorized": False,
            },
        },
        "result_tables.json": {
            "table_count": 1,
            "tables": [
                {
                    "name": "distribution.csv",
                    "label": "Registered aggregate distribution",
                    "evidence_id": "ev-distribution",
                    "headers": [
                        "exposure_level",
                        "n_rows",
                        "exposure_denominator",
                        "exposure_pct",
                        "outcome_events",
                        "outcome_denominator",
                        "outcome_rate_pct",
                        "interval_method",
                    ],
                    "rows": [
                        ["0", "60", "100", "60.0", "5", "60", "8.3", "none_counts_only"],
                        ["1", "40", "100", "40.0", "6", "40", "15.0", "none_counts_only"],
                    ],
                }
            ],
        },
        "figure_gallery.json": {
            "figures": [
                {
                    "name": "case.png",
                    "label": "Bounded case figure",
                    "status": "supporting",
                    "data_url": "data:image/png;base64,aGVsbG8=",
                }
            ]
        },
        "manuscript_draft.json": {
            "status": "locked_pending_human_review",
            "claims": [],
        },
    }


def _approved_checkpoint(run_id: str = "run_validation") -> dict:
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review the exact plan.",
        authority_sha256="b" * 64,
        payload={},
    )
    checkpoint = HumanReviewCheckpoint.create(
        run_id=run_id,
        pipeline_config_sha256="a" * 64,
        environment_identity={},
        llm_signature_sha256="c" * 64,
        run_input_capsule_sha256="d" * 64,
        capability_activation_sha256="e" * 64,
        runtime_capabilities=(),
        runtime_bundle=None,
        requests=(request,),
        plan_handoff={},
        execution_coordinates={},
    )
    decisions = [
        {
            "review_id": request.review_id,
            "authority_sha256": request.authority_sha256,
            "decision": "approved",
        }
    ]
    approved = checkpoint.approved(
        decisions=decisions,
        decision_records=decisions,
        decision_sha256=canonical_sha256(decisions),
    )
    return approved.model_dump(mode="json")


def test_system_validation_report_separates_execution_from_publication() -> None:
    projections = _projections()
    report = build_system_validation_report(
        run_id="run_validation",
        projections=projections,
        run_status={
            "gates": {
                "execution_complete": True,
                "completed_step_count": 2,
            }
        },
        review_checkpoint=_approved_checkpoint(),
        provider_usage={
            "status": "completed",
            "calls": 3,
            "accounted_tokens": 1234,
            "estimated_cost_usd": 0.42,
        },
        projection_privacy_passed=True,
    )

    assert report.status == "engineering_validation_complete"
    assert report.claim_ceiling == "engineering_validation_only"
    assert report.reportable is False
    assert report.publication_authorized is False
    assert [row.status for row in report.lifecycle] == [
        "verified",
        "verified",
        "verified",
        "verified",
        "withheld",
        "withheld",
    ]
    assert report.case_study.generated_numbers is False
    assert report.case_study.primary_table is not None
    assert report.case_study.primary_table.rows[1][4] == "6"
    assert len(report.source_bindings) == 8


def test_system_validation_report_prefers_semantically_corrected_gallery() -> None:
    projections = _projections()
    projections["system_validation_figure_gallery.json"] = {
        "schema_version": "easyicu.system-validation-figure-gallery/1",
        "figures": [
            {
                "name": "data_quality.png",
                "label": "figure:data quality · applicability-aware semantic correction",
                "status": "supporting_corrected_projection",
            }
        ],
    }

    report = build_system_validation_report(
        run_id="run_validation",
        projections=projections,
        run_status={
            "gates": {"execution_complete": True, "completed_step_count": 2}
        },
        review_checkpoint=_approved_checkpoint(),
        projection_privacy_passed=True,
    )

    assert report.case_study.figures[0].status == "supporting_corrected_projection"
    assert any(
        row.artifact == "system_validation_figure_gallery.json"
        for row in report.source_bindings
    )


def test_system_validation_html_is_self_contained_and_names_its_boundary() -> None:
    projections = _projections()
    report = build_system_validation_report(
        run_id="run_validation",
        projections=projections,
        run_status={"gates": {"execution_complete": True, "completed_step_count": 2}},
        review_checkpoint=_approved_checkpoint(),
        projection_privacy_passed=True,
    )

    html = render_system_validation_html(
        report,
        figure_gallery=projections["figure_gallery.json"],
    )

    assert "A Complete, Governed Research Workflow" in html
    assert "REVIEWER DEMONSTRATION COMPLETE" in html
    assert "WITHHELD AS DESIGNED" in html
    assert "ENGINEERING VALIDATION ONLY" in html
    assert "NOT A CLINICAL MANUSCRIPT" in html
    assert "NOT FOR SUBMISSION" not in html
    assert "data:image/png;base64,aGVsbG8=" in html
    assert "<script" not in html.lower()
    assert "generated_numbers=false" in html


def test_system_validation_report_requires_an_approved_review_checkpoint() -> None:
    report = build_system_validation_report(
        run_id="run_validation",
        projections=_projections(),
        run_status={"gates": {"execution_complete": True, "completed_step_count": 2}},
        review_checkpoint={},
        projection_privacy_passed=True,
    )

    assert report.status == "engineering_validation_incomplete"
    review = next(row for row in report.lifecycle if row.stage == "review")
    assert review.status == "not_assessed"

    unvalidated = build_system_validation_report(
        run_id="run_validation",
        projections=_projections(),
        run_status={"gates": {"execution_complete": True, "completed_step_count": 2}},
        review_checkpoint={"approved_decisions": [{"decision": "approved"}]},
        projection_privacy_passed=True,
    )
    assert unvalidated.status == "engineering_validation_incomplete"

    mismatched = build_system_validation_report(
        run_id="run_validation",
        projections=_projections(),
        run_status={"gates": {"execution_complete": True, "completed_step_count": 2}},
        review_checkpoint=_approved_checkpoint("run_other"),
        projection_privacy_passed=True,
    )
    assert mismatched.status == "engineering_validation_incomplete"


@pytest.mark.parametrize("authority_field", ["manuscript_ready", "paper_authorized"])
def test_system_validation_report_rejects_positive_paper_authority(
    authority_field: str,
) -> None:
    projections = _projections()
    projections["source_run_manifest.json"]["readiness"][authority_field] = True

    with pytest.raises(
        ValueError,
        match="requires_withheld_manuscript_and_publication_authority",
    ):
        build_system_validation_report(
            run_id="run_validation",
            projections=projections,
            review_checkpoint=_approved_checkpoint(),
            projection_privacy_passed=True,
        )


def test_system_validation_case_table_preserves_numeric_zeroes() -> None:
    projections = _projections()
    projections["result_tables.json"]["tables"][0]["rows"][0] = [
        0,
        60,
        100,
        60.0,
        0,
        60,
        0.0,
        "none_counts_only",
    ]

    report = build_system_validation_report(
        run_id="run_validation",
        projections=projections,
        review_checkpoint=_approved_checkpoint(),
        projection_privacy_passed=True,
    )

    assert report.case_study.primary_table is not None
    assert report.case_study.primary_table.rows[0][0] == "0"
    assert report.case_study.primary_table.rows[0][4] == "0"
    assert report.case_study.primary_table.rows[0][6] == "0.0"


def test_system_validation_receipt_binds_exact_json_and_html_bytes() -> None:
    projections = _projections()
    report = build_system_validation_report(
        run_id="run_validation",
        projections=projections,
        projection_privacy_passed=True,
    )
    payload = report.model_dump(mode="json")
    html_bytes = render_system_validation_html(report).encode("utf-8")

    receipt = build_system_validation_receipt(
        report_payload=payload,
        html_bytes=html_bytes,
    )

    assert receipt["authority_class"] == "engineering_validation_only"
    assert receipt["publication_authorized"] is False
    assert receipt["html"]["sha256"] == hashlib.sha256(html_bytes).hexdigest()
    assert receipt["pdf"] is None
