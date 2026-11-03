"""A question that states a first ICU stay drafts its first plan with that scope.

Otherwise the Planner compiles the stated criterion against an unrestricted
study, the plan stops on it, the population proposal adds the scope, and a
second plan follows.  The first plan uses the scope the bound source supports;
any scope already chosen, or wording the source cannot honour as written,
keeps the study as it is.

Synthetic studies only.
"""

from __future__ import annotations

from typing import Any

import pytest

from easyicu.webserver.pi_copilot.plan_decisions import (
    repeated_stays_planning_default,
    repeated_stays_system_default_route,
)
from easyicu.webserver.pi_copilot.population_preparation import (
    stated_first_stay_planning_default,
)

_QUESTION = (
    "在 eICU 演示数据中，成人患者首次入 ICU 后 24 小时内的最高乳酸，"
    "是否与院内死亡相关？"
)


def _study(database: str = "eicu", question: str = _QUESTION, **overrides: Any) -> dict[str, Any]:
    study = {
        "id": "study_a",
        "question": question,
        "data_source": {"database": database},
        "cohort": {},
        "analysis_design": {"analysis_family": "association_study"},
        "confirmations": {},
    }
    study.update(overrides)
    return study


def test_eicu_drafts_the_first_stay_within_each_hospitalization() -> None:
    patch = stated_first_stay_planning_default(_study())

    assert patch == {"cohort": {"first_icu_stay_per_hospitalization": True}}
    # One patient's hospitalizations still repeat: the clustered default
    # follows, and the card names this population, not every stay.
    study = {**_study(), **patch}
    clustered = repeated_stays_planning_default(study, source_groups_stays=True)
    assert clustered is not None
    assert repeated_stays_system_default_route({**study, **clustered}) == (
        "patient_clustered_per_hospitalization"
    )


def test_a_source_that_orders_a_patients_stays_drafts_each_patients_first_stay() -> None:
    patch = stated_first_stay_planning_default(
        _study("mimic_iv", "Among adults at their first ICU stay, is peak lactate associated with death?")
    )

    assert patch == {"cohort": {"exclude_readmissions": True}}


@pytest.mark.parametrize(
    "study",
    [
        # No first stay in the question.
        _study(question="成人患者入 ICU 后 24 小时内的最高乳酸是否与院内死亡相关？"),
        # Every stay, stated alongside.
        _study(question="纳入所有符合条件的成人 ICU 入住，不限于首次入 ICU。"),
        # The hospitalization named: the Planner and the proposal keep it.
        _study(question="每次住院的首次入 ICU 后 24 小时内最高乳酸是否与死亡相关？"),
        # Each patient named on a source that cannot order a patient's stays.
        _study(question="每位患者首次入 ICU 后 24 小时内最高乳酸是否与死亡相关？"),
        # A scope already chosen.
        _study(cohort={"exclude_readmissions": False}),
        _study(cohort={"preset": "adult_all"}),
        _study(confirmations={"plan_repeated_stays_clustered": True}),
        # A source with neither a first-stay recipe nor this substitute.
        _study("hirid"),
        _study("unknown_source"),
    ],
)
def test_otherwise_the_study_is_unchanged(study: dict[str, Any]) -> None:
    assert stated_first_stay_planning_default(study) is None


def test_the_submission_persists_the_scope_and_survives_a_refused_write(monkeypatch) -> None:
    from easyicu.webserver import research_run_submission as submission
    from easyicu.webserver import study_contexts

    writes = []

    def upsert(patch, **kwargs):
        writes.append((patch, kwargs))
        return {**_study(revision=4), **patch, "revision": 5}

    monkeypatch.setattr(study_contexts, "upsert_context", upsert)
    study = _study(revision=4)
    updated = submission._with_stated_first_stay(study)
    assert updated["revision"] == 5
    [(patch, kwargs)] = writes
    assert patch == {"id": "study_a", "cohort": {"first_icu_stay_per_hospitalization": True}}
    assert kwargs == {"expected_revision": 4, "require_revision": True, "lifecycle_write": False}

    def refuse(*_args, **_kwargs):
        raise study_contexts.StudyContextError({"error": "study_context_revision_conflict"})

    monkeypatch.setattr(study_contexts, "upsert_context", refuse)
    assert submission._with_stated_first_stay(study) is study
