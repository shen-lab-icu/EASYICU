"""A survival plan times its exposure by the first record, not a verified onset.

The signed suite classes an exposed stay as prevalent or incident from the
exposure source's first recorded time in the extract, which the cohort owner
documents as observation coverage, not a clinical onset.  The reviewed plan
still assumed "exposure onset times are recorded" and promised that prevalent
exposure at time zero is excluded, although exposure begun before its first
record (before ICU admission, say) is not observed.  The plan now states the
timing it runs and that limit.

Synthetic study (renal replacement therapy, 90-day mortality); zero patient rows.
"""

from __future__ import annotations

from tests.support.survival_proposal import (
    AGE,
    SEX,
    proposed_survival_plan,
    survival_context,
    survival_request,
    survival_spec,
)


def test_the_plan_states_first_record_timing_and_its_limit():
    context = survival_context()
    plan, _llm = proposed_survival_plan(context, survival_spec(survival_request(context), [AGE, SEX]))
    selected = plan.design_selection.selected

    stated = " ".join([selected.observation_window, *selected.assumptions, *(selected.reviewable_plan or [])])

    assert "onset" not in stated.lower()
    assert "first recorded time of the exposure source" in stated
    assert "before its first record, such as before ICU admission, is not observed" in stated
    assert "exposure first recorded at or before time zero is excluded as prevalent" in stated
    # Only the adjusted models are complete-case.
    assert "Kaplan-Meier and the restricted mean use the whole risk set" in stated
