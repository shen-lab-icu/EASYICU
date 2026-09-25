"""A complete-case refit over the primary's own rows is the primary again.

When every locked complete-case variable is a column the primary model already
requires, and none of them is kept as an unmeasured state, the refit drops no
row the primary fits.  It reproduces the primary estimate and tests nothing, so
the plan review says so and credits it no robustness axis.  The host decides
which covariates keep unmeasured rows, so the finding is the host's record,
not a Planner revision.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.cohort.schema import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
)
from easyicu.research_agent.planning.robustness_contract import RobustnessSpec
from easyicu.research_agent.planning.scientific_review import (
    _complete_case_repeats_primary_findings,
    build_plan_scientific_review,
)

from .scientific_review_fixtures import _context, _plan

CODE = "COMPLETE_CASE_SENSITIVITY_REPEATS_PRIMARY"
KEPT_AGE = {"baseline_missing_handling": {"covariates": ["age"]}}


def _with(*specs: RobustnessSpec, **requirement_update):
    plan = _plan()
    step = plan.steps[0]
    requirement = type(step.model_requirements[0]).model_validate(
        {**step.model_requirements[0].model_dump(mode="json"), **requirement_update}
    )
    return plan.model_copy(
        update={
            "steps": [
                step.model_copy(update={"model_requirements": [requirement]}),
                *plan.steps[1:],
            ],
            "robustness_specs": list(specs),
        }
    )


def _complete_case(variables: list[str], **overrides) -> RobustnessSpec:
    return RobustnessSpec(
        spec_id="complete_case",
        axis="missing",
        description="Refit on rows complete for the locked variables.",
        missing_override={"strategy": "complete_case", "variables": variables},
        **overrides,
    )


def _codes(plan) -> list[str]:
    return [item.code for item in _complete_case_repeats_primary_findings(plan)]


def test_a_refit_over_the_primary_columns_is_reported_before_approval() -> None:
    plan = _with(_complete_case(["exposure", "death", "age"]))

    review = build_plan_scientific_review(context=_context(), plan=plan)

    (finding,) = [item for item in review.findings if item.code == CODE]
    assert finding.severity == "minor"
    assert finding.remediation_route == "runtime_capability"
    assert finding.message.endswith(": complete_case")
    assert CODE not in review.facts["remediation_buckets"]["agent_plan_revision"]


@pytest.mark.parametrize(
    "variables,update,expected",
    [
        # A subset of the primary's columns still drops no fitted row.
        (["exposure", "death"], {}, [CODE]),
        # A column outside the model can drop fitted rows.
        (["exposure", "death", "age", "lactate"], {}, []),
        # Rows the primary keeps as unmeasured are dropped here.
        (["exposure", "death", "age"], KEPT_AGE, []),
        # The kept covariate is not among the locked variables.
        (["exposure", "death"], KEPT_AGE, [CODE]),
    ],
)
def test_only_a_refit_that_drops_no_fitted_row_repeats_the_primary(
    variables, update, expected
) -> None:
    assert _codes(_with(_complete_case(variables), **update)) == expected


def test_the_patient_group_column_is_one_the_primary_requires() -> None:
    variables = ["exposure", "death", "age", "subject_id"]
    grouped = {
        "dependence": {"group_source": "subject_id", "group_derivation": "identity"}
    }

    assert _codes(_with(_complete_case(variables), **grouped)) == [CODE]
    assert _codes(_with(_complete_case(variables))) == []


def test_a_refit_that_also_changes_the_cohort_or_outcome_is_a_different_analysis() -> None:
    adults = CohortDefinition(
        name="adults",
        inclusion=[
            ConceptPredicate(
                concept_id="age",
                time_window=TimeWindow(
                    anchor="icu_admit", start_offset_hours=0.0, end_offset_hours=24.0
                ),
                aggregation="first",
                op=">=",
                value=18,
            )
        ],
    )
    variables = ["exposure", "death", "age"]

    assert _codes(_with(_complete_case(variables, cohort_override=adults))) == []
    assert _codes(
        _with(_complete_case(variables, outcome_override={"outcome": "death_90d"}))
    ) == []


def test_without_a_primary_model_nothing_is_judged() -> None:
    assert _codes(_with(_complete_case(["exposure"]), analysis_role="secondary")) == []


def test_a_refit_is_judged_against_every_primary_model() -> None:
    plan = _with(_complete_case(["exposure", "death", "age"]))
    step = plan.steps[0]
    death = step.model_requirements[0]
    stay = type(death).model_validate(
        {
            **death.model_dump(mode="json"),
            "requirement_id": "primary_stay",
            "outcome": "los",
            "outcome_type": "continuous",
            "method_family": "statsmodels_ols",
        }
    )
    both = plan.model_copy(
        update={
            "steps": [
                step.model_copy(update={"model_requirements": [death, stay]}),
                *plan.steps[1:],
            ]
        }
    )

    # The plan does not say which primary a replay binds; for the stay model
    # the locked death column can drop fitted rows.
    assert _codes(plan) == [CODE]
    assert _codes(both) == []


def test_a_refit_that_restates_the_primary_earns_no_robustness_axis() -> None:
    from easyicu.research_agent.planning.scientific_review import _sensitivity_facts
    from easyicu.research_agent.schema import AnalysisStep

    replay = AnalysisStep.model_validate(
        {
            "step_id": "complete_case_replay",
            "planned_analysis_role": "sensitivity",
            "intent": "Replay the locked specification.",
            "method": "robustness_sensitivity",
            "inputs": ["table:adjusted_association_estimates"],
            "expected_outputs": ["table:robustness_matrix"],
            "sensitivity_spec_ids": ["complete_case"],
            "robustness_replay_spec": {
                "products": [{"product_id": "robustness_matrix", "output": "robustness_matrix"}]
            },
        }
    )

    def axes(plan) -> list[str]:
        plan = plan.model_copy(update={"steps": [*plan.steps, replay]})
        return _sensitivity_facts(_context(), plan)["typed_executable"]

    assert axes(_with(_complete_case(["exposure", "death", "age"]))) == []
    assert axes(_with(_complete_case(["exposure", "death", "age"]), **KEPT_AGE)) == [
        "missing"
    ]
