"""Scientific review of owner-bound outcome missingness policy."""
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.schema import AnalysisPlan

from .scientific_review_fixtures import _absolute_risk_distribution_step, _context


def test_scientific_review_blocks_unproven_structural_outcome_absence() -> None:
    step = _absolute_risk_distribution_step()
    spec = step.exposure_outcome_distribution_spec.model_copy(update={
        "missing_outcome_policy": "structural_absence_is_non_event",
    })
    plan = AnalysisPlan(
        research_question=_context().research_question, analysis_type="descriptive_study",
        steps=[step.model_copy(update={"exposure_outcome_distribution_spec": spec})],
    )
    review = build_plan_scientific_review(context=_context(), plan=plan)
    finding = next(f for f in review.findings if f.code == "DISTRIBUTION_MISSINGNESS_AUTHORITY_INVALID")
    assert finding.severity == "blocker"
    assert finding.remediation_route == "agent_plan_revision"
    assert not finding.requires_user_authorization
    assert not review.approval_allowed
    assert "death" in finding.message
