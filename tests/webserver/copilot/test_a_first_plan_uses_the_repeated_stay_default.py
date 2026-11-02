"""A first plan on a source that groups stays is drafted with patient clustering.

Otherwise the first plan is reviewed as lacking patient identity, the host
compiles this same route from it, and a second plan starts back to back.
The route stays the system's default; any recorded choice, a one-stay
population, or another analysis unit keeps the study as it is.

Synthetic studies only.
"""

from __future__ import annotations

from typing import Any

import pytest

from easyicu.webserver.pi_copilot.plan_decisions import (
    REPEATED_STAYS_SYSTEM_DEFAULT,
    repeated_stays_planning_default,
    repeated_stays_system_default_route,
)


def _study(**overrides: Any) -> dict[str, Any]:
    study = {
        "id": "study_a",
        "question": "Is peak lactate associated with hospital death?",
        "cohort": {"age_min": 18, "exclude_readmissions": False},
        "analysis_design": {"analysis_family": "association_study"},
        "confirmations": {"outcome_confirmed": True},
    }
    study.update(overrides)
    return study


def test_a_grouping_source_drafts_the_first_plan_clustered_by_patient() -> None:
    patch = repeated_stays_planning_default(_study(), source_groups_stays=True)
    assert patch is not None
    assert patch["analysis_design"] == {
        "analysis_family": "association_study", "analysis_unit": "icu_stay",
        "variance_estimator": "cluster_robust", "cluster_unit": "patient",
    }
    assert patch["confirmations"]["outcome_confirmed"] is True
    assert patch["confirmations"]["plan_repeated_stays_clustered"] is True
    assert patch["confirmations"]["plan_repeated_stays_first"] is False
    assert "agent_plan_configuration_compiled" not in patch["confirmations"]
    # The plan card names it as the system's default.
    assert repeated_stays_system_default_route({**_study(), **patch}) == "patient_clustered"
    # A first stay per hospitalization still lets one patient contribute
    # several stays, so it is clustered too.
    per_hospitalization = _study(cohort={"first_icu_stay_per_hospitalization": True})
    assert repeated_stays_planning_default(per_hospitalization, source_groups_stays=True) is not None


@pytest.mark.parametrize(
    "study, groups",
    [
        (_study(), False),
        (_study(confirmations={REPEATED_STAYS_SYSTEM_DEFAULT: False}), True),
        (_study(confirmations={"plan_repeated_stays_first": True}), True),
        (_study(cohort={"exclude_readmissions": True}), True),
        (_study(analysis_design={"analysis_unit": "hospitalization"}), True),
        (_study(analysis_design={"variance_estimator": "cluster_robust", "cluster_unit": "patient"}), True),
    ],
)
def test_a_decided_or_ungrouped_study_is_left_as_it_is(study, groups) -> None:
    assert repeated_stays_planning_default(study, source_groups_stays=groups) is None


def test_the_submission_persists_the_default_and_survives_a_refused_write(monkeypatch) -> None:
    from easyicu.webserver import research_run_submission as submission
    from easyicu.webserver import source_identity_authority
    from easyicu.webserver import study_contexts

    monkeypatch.setattr(source_identity_authority, "resolve_study_patient_grouping", lambda **_: object())
    writes = []

    def upsert(patch, **kwargs):
        writes.append((patch, kwargs))
        return {**_study(revision=4), **patch, "revision": 5}

    monkeypatch.setattr(study_contexts, "upsert_context", upsert)
    study = _study(revision=4, data_source={"database": "eicu", "path": "/export"})
    updated = submission._with_repeated_stays_default(study, path="/export")
    assert updated["revision"] == 5
    [(patch, kwargs)] = writes
    assert patch["id"] == "study_a" and patch["analysis_design"]["cluster_unit"] == "patient"
    assert kwargs == {"expected_revision": 4, "require_revision": True, "lifecycle_write": False}

    def refuse(*_args, **_kwargs):
        raise study_contexts.StudyContextError({"error": "study_context_revision_conflict"})

    monkeypatch.setattr(study_contexts, "upsert_context", refuse)
    assert submission._with_repeated_stays_default(study, path="/export") is study
