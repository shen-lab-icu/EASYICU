"""Paper maturity cannot override the executed analysis owner's verdict."""

import json

import pytest

from easyicu.webserver.scientific_readiness_projection import build_scientific_readiness_projection


@pytest.mark.parametrize("code,dimension", [
    ("NOVELTY_POSITIONING_NOT_ESTABLISHED", "novelty"),
    ("FORMAL_PREREGISTRATION_NOT_ESTABLISHED", "clinical_review"),
    ("MANUSCRIPT_CORE_SECTIONS_TOO_THIN", "manuscript"),
    ("ROBUSTNESS_AXES_TOO_NARROW", "robustness"),
])
def test_paper_maturity_findings_leave_a_valid_analysis_reportable(tmp_path, code, dimension):
    (tmp_path / "scientific_maturity_audit.json").write_text(json.dumps({
        "findings": [{"code": code, "dimension": dimension, "severity": "blocker",
                      "message": "Paper review remains open.", "remediation": "Review the manuscript."}],
    }))
    projection = _project(tmp_path)
    assert next(domain for domain in projection.domains if domain.domain == "analysis").status == "passed"
    assert next(finding for finding in projection.findings if finding.code == code).domain == "manuscript"
    assert projection.status == "analysis_only"
    assert not projection.paper_authorized
    assert not projection.publication_ready


def _project(tmp_path, **axes):
    return build_scientific_readiness_projection(
        run_id="synthetic", run_dir=tmp_path, study={}, literature_evidence={},
        axes={"analysis_validated": True, "display_suite_complete": True, **axes},
    )


def test_manuscript_reviewer_recommendation_does_not_replace_analysis_owner(tmp_path):
    (tmp_path / "reviewer_report.json").write_text(json.dumps({
        "summary": {"aggregated_recommendation": "major_revision", "counts": {"major": 1}},
    }))
    projection = _project(tmp_path)
    assert next(domain for domain in projection.domains if domain.domain == "analysis").status == "passed"
    assert next(finding for finding in projection.findings if finding.code == "SCIENTIFIC_REVIEW_MAJOR_REVISION_OPEN").domain == "manuscript"
    assert not projection.paper_authorized


def test_paper_findings_remain_visible_against_a_ready_manuscript_axis(tmp_path):
    (tmp_path / "reviewer_report.json").write_text(json.dumps({
        "summary": {"aggregated_recommendation": "major_revision", "counts": {"major": 1}},
    }))
    projection = _project(tmp_path, manuscript_ready=True, publication_ready=True, paper_authorized=True)
    assert next(domain for domain in projection.domains if domain.domain == "manuscript").status == "blocked"
    assert next(domain for domain in projection.domains if domain.domain == "analysis").status == "passed"


@pytest.mark.parametrize("defect", ["analysis_error", "absolute_risk_missing", "scientific_defect"])
def test_genuine_analysis_defects_still_block(tmp_path, defect):
    axes = {}
    if defect == "analysis_error":
        axes["analysis_validated"] = False
    elif defect == "absolute_risk_missing":
        axes.update(analysis_validated=False, display_suite_complete=False,
                    display_suite_errors=["Primary association figure lacks an absolute-risk panel."])
    else:
        (tmp_path / "scientific_maturity_audit.json").write_text(json.dumps({
            "findings": [{"code": "ENDPOINT_SEMANTICS_UNRESOLVED", "dimension": "icu_clinical_design",
                          "severity": "blocker", "message": "Endpoint unresolved.", "remediation": "Resolve the endpoint."}],
        }))
    projection = _project(tmp_path, **axes)
    assert next(domain for domain in projection.domains if domain.domain == "analysis").status == "blocked"


def test_paper_robustness_authorization_is_projected_in_manuscript(tmp_path, monkeypatch):
    (tmp_path / "scientific_maturity_audit.json").write_text(json.dumps({
        "findings": [{"code": "ROBUSTNESS_AXES_TOO_NARROW", "dimension": "robustness", "severity": "major",
                      "message": "More axes are needed for publication.", "remediation": "Review a new study version.",
                      "requires_user_authorization": True, "authorization_question": "Authorize a new study version?"}],
    }))
    projection = _project(tmp_path)
    assert projection.facts["analysis"]["user_authorization_requests"] == []
    assert projection.facts["manuscript"]["user_authorization_requests"] == [
        {"code": "ROBUSTNESS_AXES_TOO_NARROW", "question": "Authorize a new study version?"},
    ]
    from easyicu.webserver.pi_copilot import tools
    from easyicu.webserver.pi_copilot.contracts import PiSessionRecord, ToolExecutionContext

    review = {"artifact_payloads": {"scientific_readiness.json": projection.model_dump(mode="json")}}
    monkeypatch.setattr(tools, "_select_run", lambda *_args: {"run_id": "synthetic"})
    monkeypatch.setattr(tools, "_run_review", lambda _row: review)
    result = tools.execute_tool("easyicu_inspect_validation", {"run_id": "synthetic"}, ToolExecutionContext(
        data_handling="zero_retention_declared", session=PiSessionRecord(session_id="synthetic"),
    ))
    assert result["details"]["scientific_readiness"]["user_authorization_requests"] == projection.facts["manuscript"]["user_authorization_requests"]
    assert result["details"]["analysis_execution"]["analysis_validated"] is True
