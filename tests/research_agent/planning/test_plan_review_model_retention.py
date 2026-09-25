"""A plan is judged by the rows its primary model would fit, not by its labels.

The review used to score a plan whose primary model silently fitted 28% of its
cohort as highly as one that fitted all of it: it read ``complete_case`` labels
and never counted rows.  These tests pin how measured retention becomes a
finding and where each finding routes.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.contracts.model_retention import (
    CovariateRetention,
    PrimaryModelRetentionEvidence,
    RequirementRetention,
)
from easyicu.research_agent.planning.robustness_contract import RobustnessSpec
from easyicu.research_agent.planning.scientific_review import (
    CURRENT_SCIENTIFIC_REVIEW_SCHEMA_VERSION,
    PlanScientificReview,
    build_plan_scientific_review,
    plan_revision_blocker_codes,
)

from .scientific_review_fixtures import _context, _plan

RETENTION_CODES = {
    "PRIMARY_MODEL_RETENTION_INSUFFICIENT",
    "PRIMARY_MODEL_RETENTION_REDUCED",
    "PRIMARY_MODEL_RETENTION_UNVERIFIED",
    "PRIMARY_MODEL_NOT_EVALUABLE_ON_SEALED_COHORT",
    "PRIMARY_MODEL_RETENTION_PROBE_FAILED",
}


def _measured(model_n: int, evaluable_n: int = 43_518) -> PrimaryModelRetentionEvidence:
    return PrimaryModelRetentionEvidence(
        status="measured",
        population_source="analysis_cohort",
        cohort_source_sha256="f" * 64,
        requirements=(
            RequirementRetention(
                step_id="primary_model",
                requirement_id="primary",
                policy="drop_missing_baseline",
                population_n=62_231,
                evaluable_n=evaluable_n,
                model_n=model_n,
                complete_case_n=model_n,
                retention=round(model_n / evaluable_n, 4),
                complete_case_retention=round(model_n / evaluable_n, 4),
                outcome_rate_retained=0.123,
                outcome_rate_dropped=0.033,
                covariates=(
                    CovariateRetention(
                        name="age",
                        handling="drop_row",
                        n_missing=evaluable_n - model_n,
                        missing_share=round(1 - model_n / evaluable_n, 4),
                    ),
                ),
            ),
        ),
    )


def _review(evidence, plan=None):
    return build_plan_scientific_review(
        context=_context(), plan=plan or _plan(), model_retention=evidence
    )


def _retention_findings(review):
    return [item for item in review.findings if item.code in RETENTION_CODES]


def test_a_primary_model_that_fits_a_minority_of_its_cohort_cannot_be_approved() -> None:
    review = _review(_measured(17_580))

    (finding,) = _retention_findings(review)
    assert finding.code == "PRIMARY_MODEL_RETENTION_INSUFFICIENT"
    assert finding.severity == "blocker"
    assert "17580 of 43518" in finding.message and "age" in finding.message
    assert "12.3% in fitted rows vs 3.3% in dropped rows" in finding.message
    assert review.status == "changes_required" and not review.approval_allowed
    # A runtime gap stops futile automatic plan retries; no Planner fix is offered.
    assert finding.remediation_route == "runtime_capability"
    assert finding.code in plan_revision_blocker_codes(review.findings)
    assert review.facts["primary_model_retention"]["requirements"][0]["model_n"] == 17_580


@pytest.mark.parametrize(
    "retention,code",
    [
        (0.4999, "PRIMARY_MODEL_RETENTION_INSUFFICIENT"),
        (0.50, "PRIMARY_MODEL_RETENTION_REDUCED"),
        (0.8999, "PRIMARY_MODEL_RETENTION_REDUCED"),
        (0.90, None),
    ],
)
def test_the_thresholds_are_inclusive_of_the_better_band(retention, code) -> None:
    review = _review(_measured(round(retention * 10_000), 10_000))

    assert [item.code for item in _retention_findings(review)] == ([code] if code else [])


@pytest.mark.parametrize(
    "status,reason,expected",
    [
        ("rows_unavailable", "metadata_only_planning", None),
        ("rows_unavailable", "cohort_not_materialized", ("PRIMARY_MODEL_RETENTION_UNVERIFIED", "minor")),
        ("not_evaluable", "model_term_model_term_source_missing", ("PRIMARY_MODEL_NOT_EVALUABLE_ON_SEALED_COHORT", "blocker")),
        ("probe_failed", "unexpected_OSError", ("PRIMARY_MODEL_RETENTION_PROBE_FAILED", "major")),
        ("not_applicable", "no_primary_adjusted_model", None),
    ],
)
def test_an_unmeasured_retention_is_reported_for_what_it_is(status, reason, expected) -> None:
    review = _review(PrimaryModelRetentionEvidence(status=status, reason_code=reason))

    found = [(item.code, item.severity) for item in _retention_findings(review)]
    assert found == ([expected] if expected else [])
    assert review.facts["primary_model_retention"]["status"] == status


def test_without_a_measurement_the_review_is_unchanged() -> None:
    baseline = _review(None)
    measured_fine = _review(_measured(43_000))

    assert baseline.facts["primary_model_retention"] is None
    assert [item.code for item in measured_fine.findings] == [
        item.code for item in baseline.findings
    ]


def test_a_kept_unmeasured_state_needs_a_complete_case_refit_beside_it() -> None:
    plan = _plan()
    step = plan.steps[0]
    requirement = type(step.model_requirements[0]).model_validate(
        {
            **step.model_requirements[0].model_dump(mode="json"),
            "baseline_missing_handling": {"covariates": ["age"]},
        }
    )
    declared = plan.model_copy(
        update={
            "steps": [
                step.model_copy(update={"model_requirements": [requirement]}),
                *plan.steps[1:],
            ]
        }
    )
    covered = declared.model_copy(
        update={
            "robustness_specs": [
                RobustnessSpec(
                    spec_id="complete_case",
                    axis="missing",
                    description="Refit on rows complete for every model input.",
                    missing_override={
                        "strategy": "complete_case",
                        "variables": ["exposure", "age", "death"],
                    },
                )
            ]
        }
    )

    codes = [item.code for item in _review(None, declared).findings]
    assert "MISSING_CATEGORY_WITHOUT_COMPLETE_CASE_SENSITIVITY" in codes
    assert "MISSING_CATEGORY_WITHOUT_COMPLETE_CASE_SENSITIVITY" not in [
        item.code for item in _review(None, covered).findings
    ]


def test_the_review_version_moves_and_archived_reviews_stay_readable() -> None:
    review = _review(None)
    assert review.schema_version == CURRENT_SCIENTIFIC_REVIEW_SCHEMA_VERSION
    assert CURRENT_SCIENTIFIC_REVIEW_SCHEMA_VERSION.endswith("/14")

    archived = review.model_dump(mode="json")
    archived["schema_version"] = "easyicu.plan_scientific_review/13"
    assert PlanScientificReview.model_validate(archived).schema_version.endswith("/13")
