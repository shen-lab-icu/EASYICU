"""A declared cross-concept exposure reaches data preparation through its sources.

The strict KDIGO stage has no single exported source concept: the acquisition
owner materializes it from the concepts its host-derivation declaration reads.
The launch profile must accept it exactly when every one of those concepts is
in the selected modules, and fail closed naming the ones it lacks otherwise.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.acquisition import catalog as catalog_owner
from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
from easyicu.research_agent.contracts.host_derivations import (
    STRICT_KDIGO_DERIVATION_ID,
    host_derivation,
)
from easyicu.webserver import research_launch_scientific
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from easyicu.webserver.scientific_runtime_projection import required_host_derivations

_STRICT_SOURCES = host_derivation(STRICT_KDIGO_DERIVATION_ID).source_concepts


def _profile(
    monkeypatch: pytest.MonkeyPatch,
    *,
    primary_exposure: str,
    renal: tuple[str, ...] = _STRICT_SOURCES,
    require_primary_exposure: bool = True,
):
    rows = [
        ("age", "demographics", "value"),
        ("sex", "demographics", "value"),
        ("adm", "demographics", "value"),
        ("death", "outcome", "event_status"),
        *((concept, "renal", "value") for concept in renal),
    ]
    monkeypatch.setattr(
        catalog_owner,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="synthetic-metadata-only",
            concepts=[
                CatalogConcept(
                    concept_id=concept,
                    file_name=f"{module}.parquet",
                    typed_metadata=True,
                    column_role=role,
                )
                for concept, module, role in rows
            ],
        ),
    )
    return research_launch_scientific._data_foundation_profile(
        export_path="/synthetic/not-read",
        study={"modules": ["demographics", "outcome", "renal"]},
        target="death",
        primary_exposure=primary_exposure,
        require_primary_exposure=require_primary_exposure,
        covariates=("age", "sex", "adm"),
    )


def test_a_strict_kdigo_exposure_is_prepared_from_its_declared_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(monkeypatch, primary_exposure="aki_stage_strict")

    assert profile["primary_exposure_source_concept"] is None
    assert profile["static_concepts"] == ("age", "sex", "adm")
    assert profile["outcome_concepts"] == ("death",)
    # The derivation, not the concept loader, produces the exposure.
    assert "aki_stage_strict" not in profile["required_feature_concepts"]
    assert required_host_derivations(
        "aki_stage_strict", profile["primary_exposure_source_concept"], "death"
    ) == (STRICT_KDIGO_DERIVATION_ID,)


def test_a_strict_kdigo_exposure_missing_a_source_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renal = tuple(
        concept for concept in _STRICT_SOURCES if concept != "rrt_evidence_status"
    )

    with pytest.raises(ResearchPipelineRunError) as raised:
        _profile(monkeypatch, primary_exposure="aki_stage_strict", renal=renal)

    assert raised.value.code == (
        "research_pipeline_primary_exposure_outside_configured_modules"
    )
    assert raised.value.details["missing_source_concepts"] == [
        "rrt_evidence_status"
    ]


def test_an_unrequired_derived_exposure_without_sources_is_left_to_the_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(
        monkeypatch,
        primary_exposure="aki_stage_strict",
        renal=(),
        require_primary_exposure=False,
    )

    assert profile["primary_exposure_source_concept"] is None


def test_a_derivation_receipt_is_not_a_design_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ResearchPipelineRunError) as raised:
        _profile(monkeypatch, primary_exposure="kidney_window_row_count")

    assert raised.value.code == (
        "research_pipeline_primary_exposure_outside_configured_modules"
    )
    assert "missing_source_concepts" not in raised.value.details


def _baseline_grouped_by(group: str):
    from easyicu.research_agent.planning.baseline_requirements import (
        AcceptedBaselineRequirements,
        BaselineCoordinate,
        BaselineTableRequirement,
    )

    return AcceptedBaselineRequirements(
        source_plan_sha256="b" * 64,
        tables=(
            BaselineTableRequirement(
                source_step_id="table_one",
                group_by=BaselineCoordinate(name=group, source_concept=group),
                variables=tuple(
                    BaselineCoordinate(name=name, source_concept=name)
                    for name in ("age", "sex", "adm")
                ),
            ),
        ),
    )


def test_a_table_one_grouped_by_the_strict_stage_reaches_the_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver import agent_pipeline_runs

    profile = _profile(monkeypatch, primary_exposure="aki_stage_strict")

    roster = agent_pipeline_runs._materialization_concept_roster(
        foundation_profile=profile,
        development_resume_acquisition=None,
        baseline_requirements=_baseline_grouped_by("aki_stage_strict"),
    )

    assert roster == {
        "outcome_concepts": ("death",),
        # The acquisition owner routes this declared name to its derivation.
        "required_feature_concepts": ("aki_stage_strict",),
        "static_concepts": ("age", "sex", "adm"),
    }


def test_a_derived_table_one_group_without_its_sources_stays_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver import agent_pipeline_runs

    profile = _profile(
        monkeypatch,
        primary_exposure="aki_stage_strict",
        renal=(),
        require_primary_exposure=False,
    )

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as raised:
        agent_pipeline_runs._materialization_concept_roster(
            foundation_profile=profile,
            development_resume_acquisition=None,
            baseline_requirements=_baseline_grouped_by("aki_stage_strict"),
        )

    assert raised.value.code == "accepted_baseline_source_unresolved"
    assert raised.value.details == {"concepts": ["aki_stage_strict"]}
