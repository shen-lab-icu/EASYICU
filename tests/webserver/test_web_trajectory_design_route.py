"""A declared trajectory design reaches the panel it is checked against.

The signed fixed-window trajectory route had three gaps between a declared
``trajectory_design`` and its execution:

* the projection read the long panel's window under a key the cohort
  materializer never writes (``window``, not ``trajectory_window_hours``), so a
  design reaching past the materialized window passed on every real panel;
* the launch materialized the panel over the outer cohort window and over the
  association roster only, never over the design's own concepts and window;
* a planner-only candidate, which plans on a zero-row catalog and has no panel
  at all, was always refused (``web_trajectory_longitudinal_panel_missing``),
  so no trajectory study could reach a reviewable plan on the sealed suite.

The fixtures are generic synthetic data, not any benchmark study.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from easyicu.research_agent.acquisition import catalog as catalog_owner
from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.webserver import agent_pipeline_runs, provider_adapter
from easyicu.webserver import research_launch_scientific
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from easyicu.webserver.scientific_runtime_projection import (
    WebScientificRuntimeProjectionError,
)
from easyicu.webserver.trajectory_runtime_projection import (
    compile_web_trajectory_runtime_projection,
    trajectory_provenance_path,
)
from tests.support.typed_trajectory import typed_trajectory_export

_COORDINATES = ("sofa2_resp", "sofa2_cardio", "lact")


def _study(**design_overrides: Any) -> dict[str, Any]:
    design = {
        "coordinate_concepts": list(_COORDINATES),
        "descriptive_only_concepts": ["sofa2"],
        "window_start_hours": 0,
        "window_end_hours": 48,
        "grid_width_hours": 12,
        "candidate_cluster_min": 2,
        "candidate_cluster_max": 4,
    }
    design.update(design_overrides)
    return {
        "analysis_design": {
            "analysis_family": "trajectory_clustering",
            "analysis_unit": "icu_stay",
            "variance_estimator": "model_based",
        },
        "trajectory_design": study_context_owner.normalize_trajectory_design(design),
    }


def _compile(universe: Path, study: dict[str, Any], **kwargs: Any):
    return compile_web_trajectory_runtime_projection(
        study=study,
        universe_path=universe,
        scientific_configuration_sha256="c" * 64,
        **kwargs,
    )


def _planning_catalog(path: Path, columns: tuple[str, ...]) -> Path:
    frame = pd.DataFrame({name: pd.Series(dtype="float64") for name in columns})
    frame.to_parquet(path, index=False)
    return path


# -- the window the materializer records --------------------------------------


def test_the_window_check_reads_the_window_the_materializer_records(tmp_path):
    source = typed_trajectory_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        stem="universe",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
        emit_trajectory=True,
        trajectory_concepts=("lact",),
        trajectory_window=(0.0, 24.0),
    )
    universe = Path(paths["parquet"])
    design = {"coordinate_concepts": ["sofa2_resp", "lact"], "descriptive_only_concepts": []}

    with pytest.raises(WebScientificRuntimeProjectionError) as wider:
        _compile(universe, _study(**design, window_end_hours=48))
    assert wider.value.code == "web_trajectory_window_outside_materialization"
    assert wider.value.details["materialized_window_hours"] == [0.0, 24.0]

    # Inside the recorded window the next check runs: the panel carries only
    # the concept it was cut for.
    with pytest.raises(WebScientificRuntimeProjectionError) as inside:
        _compile(universe, _study(**design, window_end_hours=24))
    assert inside.value.code == "web_trajectory_concepts_unavailable"
    assert inside.value.details["missing_concepts"] == ["sofa2_resp"]


def test_a_panel_that_does_not_record_its_window_is_refused(tmp_path):
    universe = tmp_path / "web_research_universe.parquet"
    pd.DataFrame({"stay_id": [1, 2]}).to_parquet(universe, index=False)
    trajectory_provenance_path(universe).write_text(
        json.dumps(
            {
                "trajectory_concepts_materialized": [*_COORDINATES, "sofa2"],
                "available_unobserved_concepts": [],
                "unavailable_concepts": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(WebScientificRuntimeProjectionError) as excinfo:
        _compile(universe, _study())
    assert excinfo.value.code == "web_trajectory_materialized_window_unproven"
    assert excinfo.value.details["expected_field"] == "trajectory_window_hours"


# -- a planner-only candidate --------------------------------------------------


def _formal_panel(tmp_path: Path) -> Path:
    universe = tmp_path / "formal" / "web_research_universe.parquet"
    universe.parent.mkdir()
    pd.DataFrame({"stay_id": [1, 2]}).to_parquet(universe, index=False)
    trajectory_provenance_path(universe).write_text(
        json.dumps(
            {
                "trajectory_window_hours": [0.0, 48.0],
                "trajectory_concepts_materialized": [*_COORDINATES, "sofa2"],
                "available_unobserved_concepts": [],
                "unavailable_concepts": [],
            }
        ),
        encoding="utf-8",
    )
    return universe


def test_a_candidate_signs_the_contract_the_formal_run_signs(tmp_path):
    catalog = _planning_catalog(
        tmp_path / "planner_catalog.parquet",
        ("stay_id", "age", *_COORDINATES, "sofa2", "death"),
    )

    candidate = _compile(catalog, _study(), planning_catalog=True)
    formal = _compile(_formal_panel(tmp_path), _study())

    assert candidate is not None and formal is not None
    assert candidate.authority == formal.authority
    assert candidate.projection_sha256 == formal.projection_sha256


def test_a_candidate_catalog_without_a_design_concept_is_refused(tmp_path):
    catalog = _planning_catalog(
        tmp_path / "planner_catalog.parquet",
        ("stay_id", "sofa2_resp", "sofa2_cardio", "sofa2", "death"),
    )

    with pytest.raises(WebScientificRuntimeProjectionError) as excinfo:
        _compile(catalog, _study(), planning_catalog=True)
    assert excinfo.value.code == "web_trajectory_concepts_unavailable"
    assert excinfo.value.details["missing_concepts"] == ["lact"]
    assert excinfo.value.details["admission"] == "planning_catalog"


def test_catalog_admission_never_applies_to_a_universe_with_rows(tmp_path):
    universe = tmp_path / "web_research_universe.parquet"
    pd.DataFrame(
        {name: [1.0] for name in ("stay_id", *_COORDINATES, "sofa2")}
    ).to_parquet(universe, index=False)

    with pytest.raises(WebScientificRuntimeProjectionError) as excinfo:
        _compile(universe, _study(), planning_catalog=True)
    assert excinfo.value.code == "web_trajectory_planning_catalog_not_empty"


def test_a_formal_run_without_its_panel_still_fails_closed(tmp_path):
    catalog = _planning_catalog(
        tmp_path / "planner_catalog.parquet", ("stay_id", *_COORDINATES, "sofa2")
    )

    with pytest.raises(WebScientificRuntimeProjectionError) as excinfo:
        _compile(catalog, _study())
    assert excinfo.value.code == "web_trajectory_longitudinal_panel_missing"


# -- the launch materializes the design ----------------------------------------


def _profile(monkeypatch: pytest.MonkeyPatch, *, lab_concepts: tuple[str, ...]):
    rows = [
        ("age", "demographics", "value"),
        ("sex", "demographics", "value"),
        ("death", "outcome", "event_status"),
        *((concept, "labs", "value") for concept in lab_concepts),
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
        study={"modules": ["demographics", "outcome", "labs"]},
        target="death",
        require_primary_exposure=False,
        trajectory_concepts=(*_COORDINATES, "sofa2"),
    )


def test_the_design_concepts_are_materialized_as_features(monkeypatch):
    profile = _profile(monkeypatch, lab_concepts=(*_COORDINATES, "sofa2"))

    assert profile["required_feature_concepts"] == (*_COORDINATES, "sofa2")
    assert profile["outcome_concepts"] == ("death",)


def test_a_design_concept_outside_the_modules_is_refused_before_materialization(
    monkeypatch,
):
    with pytest.raises(ResearchPipelineRunError) as excinfo:
        _profile(monkeypatch, lab_concepts=("sofa2_resp", "sofa2_cardio", "sofa2"))
    assert excinfo.value.code == (
        "research_pipeline_trajectory_concept_outside_configured_modules"
    )
    assert excinfo.value.details == {"field": "trajectory_design", "concept_id": "lact"}


def _export(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"stay_id": [1], "age": [65]}).to_parquet(
        root / "demographics.parquet", index=False
    )
    (root / "_manifest.json").write_text(
        json.dumps(
            {
                "database": "miiv",
                "format": "parquet",
                "concept_selection": {
                    "mode": "explicit",
                    "modules": {"demographics": ["age"]},
                },
                "feature_definitions": {"included": False},
                "files": [
                    {
                        "file": "demographics.parquet",
                        "module": "demographics",
                        "concepts": 1,
                        "concept_ids": ["age"],
                        "rows": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


class _Job:
    id = "job-trajectory-route"
    cancel_requested = False

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))


def _stop(code: str, captured: dict[str, Any]):
    def capture(**kwargs: Any) -> Any:
        captured.update(kwargs)
        raise ResearchPipelineRunError(code, "stop once the acquisition request is bound")

    return capture


def _mock_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    monkeypatch.setattr(
        provider_adapter,
        "build_research_agent_provider_client",
        lambda *_args, **_kwargs: (
            ScriptedMockLLMClient([]),
            {"provider": "mock", "model": "trajectory-route-test"},
        ),
    )


def _declared_study(export: Path) -> dict[str, Any]:
    return {
        "id": "study-trajectory-route",
        "revision": 1,
        "question": "Do organ-dysfunction trajectories over the first two ICU days form stable groups?",
        "data_source": {"path": str(export), "database": "miiv"},
        "outcome": "death",
        **_study(),
    }


def test_a_candidate_run_puts_the_design_concepts_on_its_planning_catalog(
    tmp_path, monkeypatch
):
    _mock_provider(monkeypatch)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        agent_pipeline_runs,
        "_metadata_only_planning_acquisition",
        _stop("test_planning_roster_captured", captured),
    )
    export = _export(tmp_path / "export")
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(export),
        study_context=_declared_study(export),
        project_root=str(tmp_path / "projects"),
        provider={"provider": "openai", "external": True},
        provider_environment={"OPENAI_API_KEY": "test-key"},
        credential_source="pi_verified",
        budget_mode="planner_canary",
    )

    with pytest.raises(ResearchPipelineRunError) as raised:
        runner(_Job())

    assert raised.value.code == "test_planning_roster_captured"
    assert set(_COORDINATES) | {"sofa2"} <= set(captured["required_concepts"])


def test_the_formal_run_cuts_the_panel_over_the_design_window(tmp_path, monkeypatch):
    from easyicu.research_agent.acquisition import foundation
    from easyicu.research_agent.execution import runner as runner_module
    from easyicu.webserver import research_pipeline_run_preparation

    _mock_provider(monkeypatch)
    monkeypatch.setattr(
        runner_module,
        "probe_runner_availability",
        lambda kind, **_kwargs: runner_module.RunnerAvailability(
            kind=kind, available=True, image="easyicu-research-agent:test"
        ),
    )
    profiles: list[dict[str, Any]] = []

    def profile(**kwargs: Any) -> dict[str, Any]:
        profiles.append(kwargs)
        return {
            "allowed_modules": ("demographics", "outcome", "labs"),
            "static_concepts": ("age",),
            "outcome_concepts": ("death",),
            "required_feature_concepts": tuple(kwargs["trajectory_concepts"]),
            "require_outcome": True,
            "primary_exposure_source_concept": None,
        }

    monkeypatch.setattr(research_pipeline_run_preparation, "_data_foundation_profile", profile)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        foundation,
        "acquire_universe_for_question",
        _stop("test_materialization_captured", captured),
    )
    export = _export(tmp_path / "export")
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(export),
        study_context=_declared_study(export),
        project_root=str(tmp_path / "projects"),
        provider={"provider": "openai", "external": True},
        provider_environment={"OPENAI_API_KEY": "test-key"},
        credential_source="pi_verified",
        budget_mode="full_reviewed",
    )

    with pytest.raises(ResearchPipelineRunError) as raised:
        runner(_Job())

    assert raised.value.code == "test_materialization_captured"
    assert profiles[0]["trajectory_concepts"] == (*_COORDINATES, "sofa2")
    assert captured["trajectory_window"] == (0.0, 48.0)
    # The cohort's own outer window is unchanged by the design.
    assert captured["cohort_window"] != captured["trajectory_window"]
    assert captured["emit_trajectory"] is True


def test_a_candidate_run_asks_for_catalog_admission(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from easyicu.webserver import trajectory_runtime_projection

    _mock_provider(monkeypatch)
    catalog = _planning_catalog(
        tmp_path / "planner_catalog.parquet",
        ("stay_id", "age", *_COORDINATES, "sofa2", "death"),
    )
    monkeypatch.setattr(
        agent_pipeline_runs,
        "_metadata_only_planning_acquisition",
        lambda **_kwargs: SimpleNamespace(
            blocked=False,
            universe_path=catalog,
            provenance_path=None,
            materialized_concepts=["age", *_COORDINATES, "sofa2", "death"],
        ),
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        trajectory_runtime_projection,
        "compile_web_trajectory_runtime_projection",
        _stop("test_trajectory_projection_captured", captured),
    )
    export = _export(tmp_path / "export")
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(export),
        study_context=_declared_study(export),
        project_root=str(tmp_path / "projects"),
        provider={"provider": "openai", "external": True},
        provider_environment={"OPENAI_API_KEY": "test-key"},
        credential_source="pi_verified",
        budget_mode="planner_canary",
    )

    with pytest.raises(ResearchPipelineRunError) as raised:
        runner(_Job())

    assert raised.value.code == "test_trajectory_projection_captured"
    assert captured["planning_catalog"] is True
    assert captured["universe_path"] == catalog
