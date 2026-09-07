"""Accepted clinical names bind to prepared outputs before patient access."""

from __future__ import annotations

import pytest

from easyicu.research_agent.acquisition import catalog as catalog_owner
from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
from easyicu.research_agent.planning.baseline_requirements import (
    AcceptedBaselineRequirements,
    BaselineCoordinate,
    BaselineTableRequirement,
)
from easyicu.webserver import agent_pipeline_runs, research_launch_scientific
from easyicu.webserver.research_launch_resume import _DevelopmentResumeAcquisition


def _requirements(group: str = "sep3") -> AcceptedBaselineRequirements:
    return AcceptedBaselineRequirements(
        source_plan_sha256="a" * 64,
        tables=(BaselineTableRequirement(
            source_step_id="accepted_baseline",
            group_by=BaselineCoordinate(name=group, source_concept=group),
            variables=tuple(BaselineCoordinate(name=name, source_concept=name)
                            for name in ("age", "sex", "adm", "charlson")),
        ),),
    )


def _profile(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(catalog_owner, "build_available_catalog", lambda _path: AvailableCatalog(
        source="synthetic-metadata-only",
        concepts=[CatalogConcept(concept_id=concept, file_name=f"{module}.parquet",
                                 typed_metadata=True, column_role=role)
                  for concept, module, role in (
                      ("age", "demographics", "value"),
                      ("sex", "demographics", "value"),
                      ("adm", "demographics", "value"),
                      ("death", "outcome", "event_status"),
                      ("charlson", "comorbidity", "value"),
                      ("sep3_sofa1", "sepsis3_sofa1", "event_status"),
                      ("outside_module", "not_selected", "value"),
                  )],
    ))
    return research_launch_scientific._data_foundation_profile(
        export_path="/synthetic/not-read",
        study={"modules": ["demographics", "outcome", "comorbidity", "sepsis3_sofa1"]},
        target="death", primary_exposure="sep3",
    )


@pytest.mark.parametrize("checkpoint_kind", [None, "metadata_only_planning_catalog"])
def test_reviewed_baseline_resolves_same_owner_output_as_primary_exposure(
    monkeypatch: pytest.MonkeyPatch, checkpoint_kind: str | None,
) -> None:
    profile = _profile(monkeypatch)
    assert "outside_module" not in profile["available_concepts"]
    requirements = _requirements()
    original = requirements.model_dump(mode="json")
    roster = agent_pipeline_runs._materialization_concept_roster(
        foundation_profile=profile,
        development_resume_acquisition=(
            _DevelopmentResumeAcquisition(kind=checkpoint_kind)
            if checkpoint_kind else None
        ),
        baseline_requirements=requirements,
    )
    assert profile["primary_exposure_source_concept"] == "sep3_sofa1"
    assert roster == {
        "outcome_concepts": ("death",),
        "static_concepts": ("age", "sex"),
        "required_feature_concepts": ("sep3_sofa1", "adm", "charlson"),
    }
    assert requirements.model_dump(mode="json") == original


@pytest.mark.parametrize("group", ["sep3_sofa2", "outside_module", "comorbidity_loader"])
def test_unavailable_or_ambiguous_baseline_is_not_silently_replaced(
    monkeypatch: pytest.MonkeyPatch, group: str,
) -> None:
    profile = _profile(monkeypatch)
    # A family with two different scores must not select one on the user's behalf.
    profile["available_concepts"] += ("elixhauser",)
    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as raised:
        agent_pipeline_runs._materialization_concept_roster(
            foundation_profile=profile, development_resume_acquisition=None,
            baseline_requirements=_requirements(group),
        )
    assert raised.value.code == "accepted_baseline_source_unresolved"


@pytest.mark.parametrize("missing_score", [False, True])
def test_frozen_patient_roster_cannot_be_widened_to_repair_a_baseline(
    monkeypatch: pytest.MonkeyPatch, missing_score: bool,
) -> None:
    profile = _profile(monkeypatch)
    frozen = _DevelopmentResumeAcquisition(
        kind="materialized_patient_universe",
        feature_concepts=("sep3_sofa1", "adm") if missing_score else ("sep3_sofa1", "adm", "charlson"),
        outcome_concepts=("death",), static_concepts=("age", "sex"),
    )
    if missing_score:
        with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as raised:
            agent_pipeline_runs._materialization_concept_roster(
                foundation_profile=profile, development_resume_acquisition=frozen,
                baseline_requirements=_requirements(),
            )
        assert raised.value.code == "accepted_baseline_resume_materialization_mismatch"
    else:
        roster = agent_pipeline_runs._materialization_concept_roster(
            foundation_profile=profile, development_resume_acquisition=frozen,
            baseline_requirements=_requirements(),
        )
        assert roster["required_feature_concepts"] == frozen.feature_concepts
