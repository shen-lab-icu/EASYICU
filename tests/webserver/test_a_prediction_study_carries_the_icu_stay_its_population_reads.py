"""A study declared a prediction study carries the ICU length of stay its population reads.

Its model predicts at the end of the feature window for the stays still in
the ICU after it, a bound the plan writes on ``los_icu``.  The host puts that
concept in the planning catalog and the materialized universe, as it does for
a typed minimum ICU stay; without it the Research Agent refuses to plan.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
from easyicu.webserver import agent_pipeline_runs, primary_cohort, research_launch_scientific

_PREDICTION = {"analysis_family": "prediction_model", "analysis_unit": "icu_stay", "variance_estimator": "model_based"}


@pytest.mark.parametrize(
    ("cohort", "design", "expected"),
    [
        (None, _PREDICTION, ("los_icu",)),
        ({"preset": "adult_all", "age_min": 18}, _PREDICTION, ("los_icu",)),
        ({"min_icu_los_hours": 24}, _PREDICTION, ("los_icu",)),
        (None, {"analysis_family": "association_study"}, ()),
        (None, {"analysis_family": "descriptive_epidemiology"}, ()),
        (None, {}, ()),
        (None, None, ()),
        ({"min_icu_los_hours": 24}, {"analysis_family": "association_study"}, ("los_icu",)),
    ],
)
def test_a_declared_prediction_study_requires_the_icu_length_of_stay(
    cohort: object, design: object, expected: tuple[str, ...]
) -> None:
    assert primary_cohort.cohort_required_concepts(cohort, design) == expected


def _catalog(monkeypatch: pytest.MonkeyPatch, *, duration_available: bool) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module

    concepts = [
        ("age", "demographics.parquet", "value"),
        ("death", "outcome.parquet", "event_status"),
        *([("los_icu", "outcome.parquet", "value")] if duration_available else []),
    ]
    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="typed-demo",
            concepts=[
                CatalogConcept(
                    concept_id=concept, file_name=file_name, typed_metadata=True,
                    column_role=role,
                )
                for concept, file_name, role in concepts
            ],
        ),
    )


def test_the_universe_of_a_declared_prediction_study_carries_the_icu_length_of_stay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _catalog(monkeypatch, duration_available=True)
    study = {"modules": ["demographics", "outcome"], "cohort": {}, "analysis_design": _PREDICTION}

    profile = research_launch_scientific._data_foundation_profile(
        export_path="/typed/demo", study=study, target="death",
    )

    assert profile["static_concepts"] == ("age", "los_icu")
    assert profile["outcome_concepts"] == ("death",)


def test_another_family_universe_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _catalog(monkeypatch, duration_available=True)
    study = {
        "modules": ["demographics", "outcome"],
        "cohort": {},
        "analysis_design": {**_PREDICTION, "analysis_family": "association_study"},
    }

    profile = research_launch_scientific._data_foundation_profile(
        export_path="/typed/demo", study=study, target="death",
    )

    assert profile["static_concepts"] == ("age",)


def test_a_prediction_study_without_the_icu_length_of_stay_names_its_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _catalog(monkeypatch, duration_available=False)
    study = {"modules": ["demographics", "outcome"], "cohort": {}, "analysis_design": _PREDICTION}

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as raised:
        research_launch_scientific._data_foundation_profile(
            export_path="/typed/demo", study=study, target="death",
        )

    assert raised.value.code == "research_pipeline_cohort_concept_unavailable"
    assert raised.value.details == {
        "field": "analysis_design.analysis_family", "concept_id": "los_icu",
    }
