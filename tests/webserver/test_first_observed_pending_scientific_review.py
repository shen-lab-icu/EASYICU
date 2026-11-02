"""Real scientific reviewer checks on an invented first-ICU proposal."""
from tests.support.first_observed_synthetic_review import positive_case
from tests.support.first_observed_synthetic_literature import KEY


def test_real_review_95_retains_pending_and_unverified_findings(tmp_path):
    _, _, population, plan, _, literature, review = positive_case(tmp_path)
    assert population.criteria[0].disposition == 'pending_materialization'
    assert not population.criteria[0].applied and population.blocking
    assert plan.steps[-1].exposure_outcome_distribution_spec.confidence_level == 0.95
    assert plan.steps[-1].exposure_outcome_distribution_spec.interval_method == 'wilson'
    assert KEY in plan.steps[-1].literature_citation_keys
    assert literature.search_provenance.note.startswith('Synthetic test')
    assert review.approval_allowed
    assert any(f.code == 'LITERATURE_SOURCE_EVIDENCE_UNVERIFIED' for f in review.findings)
    assert any(f.code == 'POPULATION_CRITERION_NOT_APPLIED' for f in review.findings)


def test_real_review_90_is_not_forced_to_95(tmp_path):
    _, _, population, plan, _, _, review = positive_case(tmp_path, confidence=0.90)
    assert population.blocking and plan.steps[-1].exposure_outcome_distribution_spec.confidence_level == 0.90
    assert review.approval_allowed
    assert not any(f.code == 'QUESTION_DESCRIPTIVE_UNCERTAINTY_NOT_MET' for f in review.findings)
