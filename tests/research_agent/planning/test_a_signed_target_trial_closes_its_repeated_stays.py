"""A signed target trial owns its primary result and its repeated stays.

The suite's bootstrap draws whole patients when the study verified a patient
group, so a cohort that keeps readmissions has its repeated stays handled by
the estimator itself.  A signed suite that resamples ICU stays leaves them
open, as any estimator without a dependence contract does.  Its strategies,
endpoint and confounders are signed in its authority, so a causal plan needs
no Planner-written family result contract beside it.

Synthetic authority only (``tests/support/target_trial.py``).
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.planning.primary_result_contract import (
    validate_required_primary_result,
)
from easyicu.research_agent.planning.scientific_review import (
    repeated_unit_design_closed,
    repeated_unit_estimator_present,
)
from easyicu.research_agent.schema import AnalysisPlan
from tests.support.target_trial import target_trial_authority_body

from .scientific_review_fixtures import _context


def _patient_context():
    context = _context()
    return context.model_copy(
        update={
            "cohort": context.cohort.model_copy(
                update={
                    "id_columns": ["stay_id", "subject_id"],
                    "provenance": {
                        "analysis_unit": "icu_stay",
                        "patient_id_columns": ["subject_id"],
                    },
                }
            )
        }
    )


def _bound(**overrides) -> AnalysisPlan:
    authority = build_current_case_scientific_runtime_authority(
        target_trial_authority_body(**overrides)
    )
    draft = AnalysisPlan.model_validate(
        {
            "research_question": "Does an early start change 28-day mortality?",
            "analysis_type": "causal_inference",
            "steps": [
                {
                    "step_id": "01_primary",
                    "planned_analysis_role": "primary",
                    "intent": "Draft primary step.",
                    "inputs": ["table:analysis_cohort"],
                    "expected_outputs": ["table:draft"],
                    "method": "draft_method",
                }
            ],
        }
    )
    return authority.bind_plan(draft)


def test_a_patient_bootstrap_closes_the_repeated_stays() -> None:
    context = _patient_context()
    plan = _bound()

    assert repeated_unit_estimator_present(context, plan)
    assert repeated_unit_design_closed(context, plan)


def test_a_stay_bootstrap_leaves_the_repeated_stays_open() -> None:
    context = _patient_context()
    plan = _bound(
        resampling_unit="icu_stay",
        patient_group_column=None,
        patient_group_derivation=None,
    )

    assert repeated_unit_estimator_present(context, plan)
    assert not repeated_unit_design_closed(context, plan)


def test_an_unsigned_suite_does_not_close_them() -> None:
    context = _patient_context()
    plan = _bound()
    suite = plan.steps[1].model_copy(update={"icu_rule_refs": []})
    unsigned = plan.model_copy(update={"steps": [plan.steps[0], suite, plan.steps[2]]})

    assert not repeated_unit_design_closed(context, unsigned)


def test_the_signed_suite_is_the_causal_plan_s_primary_result_owner() -> None:
    context = _patient_context()
    plan = _bound()

    validate_required_primary_result(plan=plan, context=context)
    drafted = plan.model_copy(
        update={
            "steps": [
                plan.steps[0],
                plan.steps[1].model_copy(update={"method": "draft_method"}),
                plan.steps[2],
            ]
        }
    )
    with pytest.raises(ValueError, match="family_primary_result_requirement"):
        validate_required_primary_result(plan=drafted, context=context)
