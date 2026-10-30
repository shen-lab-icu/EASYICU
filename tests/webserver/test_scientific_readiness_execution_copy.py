"""Scientific readiness must not invent completion for unexecuted runs."""

from easyicu.webserver.scientific_readiness_projection import _analysis_status


def test_missing_or_false_execution_receipt_never_claims_completion() -> None:
    for axes in ({}, {"execution_complete": False}, {"execution_complete": None}):
        domain, _, _ = _analysis_status(None, axes)
        assert domain.status == "blocked"
        assert domain.summary == (
            "Analysis completion has not been confirmed; the scientific assessment remains open."
        )


def test_completed_execution_does_not_imply_validation() -> None:
    domain, _, _ = _analysis_status(None, {"execution_complete": True})
    assert domain.status == "blocked"
    assert domain.summary == (
        "Analysis completion does not establish validation or close open scientific design issues."
    )


def test_validated_analysis_can_pass_without_open_analysis_findings() -> None:
    domain, _, _ = _analysis_status(None, {"execution_complete": True, "analysis_validated": True})
    assert domain.status == "passed"
    assert domain.summary == "Automated analysis validation passed with no open analysis findings."
