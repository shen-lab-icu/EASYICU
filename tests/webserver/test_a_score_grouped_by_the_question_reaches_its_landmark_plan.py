"""A total score the question groups by cut points reaches its landmark plan.

Metadata-only planning compiles the Web runtime projection before the run's
grouping phase. A total score's integer range is ordinal without a closed
level set, so the categorical landmark contract cannot be signed on the source
there (``web_landmark_categorical_levels_unavailable`` stopped the run before
any grouping was asked). The projection now waits for the grouping: the run
forms the stated groups, and the run that binds them signs the ordered
categorical contract. A source authority that a grouping replaces yields to
it. Prepared data without a closed level set still stop. Synthetic zero-row
catalogs, groupings and universes only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    LandmarkCategoricalAssociationRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.orchestration.exposure_grouping_phase import (
    planning_run_groupings,
    run_exposure_grouping_phase,
)
from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
from easyicu.research_agent.planning.family_spec import family_template_id_for_context
from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.webserver.research_launch_scientific import _runtime_projection_sensitivity_specs
from easyicu.webserver.scientific_runtime_projection import (
    WebScientificRuntimeProjectionError,
    compile_web_scientific_runtime_projection,
)

LANDMARK = PrespecifiedSensitivitySpec(
    spec_id="landmark_24h", axis="timing", strategy="landmark", landmark_hours=24,
    require_alive_at_landmark=True, exclude_negative_event_times=True,
    event_time_variable="death_time_hours",
    observation_duration_variable="hospital_followup_time_hours",
    observation_duration_unit="hours",
)
QUESTION = "Do three bands of the first-day score carry a rising risk of hospital death?"
# (concept, summary, lower cut, upper cut): a levelless total score and a
# closed-level ordinal score, each grouped into three ordered bands.
SCORES = {"sofa": ("max", 6, 10), "gcs": ("min", 9, 13)}


def _catalog(path: Path, concept: str) -> Path:
    """The metadata-only planning catalog: zero rows, every concept float64."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "stay_id": pd.Series(dtype="int64"),
        **{name: pd.Series(dtype="float64") for name in (
            "death_time_hours", "hospital_followup_time_hours", concept, "age", "death",
        )},
    })
    frame.attrs["easyicu_planning_authority"] = {
        "kind": "metadata_only_planning_catalog", "patient_rows_read": False,
    }
    frame.to_parquet(path, index=False)
    return path


def _context(cohort, *, primary_exposure=None):
    return build_research_context(
        research_question=QUESTION, cohort=cohort, cohort_name="synthetic", database="eicu",
        primary_exposure=primary_exposure, target_outcome="death",
        id_columns=("stay_id",), outcome_columns=("death",),
        user_preferences={
            "landmark_hours": 24, "sensitivity_specs": [LANDMARK.model_dump(mode="json")],
            "data_constraints": json.dumps({
                "materialization_window": {
                    "role": "outer_observation_window", "anchor": "icu_admission", "hours": 24,
                },
                "analysis_design": {
                    "analysis_family": "association_study", "analysis_unit": "icu_stay",
                    "variance_estimator": "model_based",
                },
            }),
        },
    )


def _grouped(tmp_path: Path, concept: str):
    """Run the real grouping phase on the catalog with the question's three bands."""
    summary, low, high = SCORES[concept]
    answer = {"groupings": [{
        "id": "x1", "concept": concept, "scale": "ordinal",
        "window": {"start_hours": 0, "end_hours": 24},
        "groups": [
            {"id": "g1", "label": "Low band", "rule": {"summary": summary, "op": "<", "value": low}},
            {"id": "g3", "label": "High band", "rule": {"summary": summary, "op": ">=", "value": high}},
            {"id": "g2", "label": "Middle band", "rule": "otherwise"},
        ],
        "unmeasured": {"handling": "exclude"},
        "reference": "g1", "contrast": "g3",
        "quote": "three bands of the first-day score", "source": "question",
    }]}
    catalog = _catalog(tmp_path / "planning/cohort.parquet", concept)
    run_exposure_grouping_phase(
        context=_context(catalog, primary_exposure=concept), cohort_path=catalog,
        run_dir=catalog.parent, planner=ScriptedMockLLMClient([json.dumps(answer)]),
        rebuild_context=_context, capability_review_pending=False, trajectory_staged=False,
        emit_progress=lambda *_args, **_kwargs: None,
    )
    groupings = planning_run_groupings(
        catalog.parent, cohort_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
    )
    return catalog, groupings


def _project(universe, *, exposure, source, groupings=None):
    specs = _runtime_projection_sensitivity_specs(
        (LANDMARK,), primary_exposure=exposure, primary_exposure_source=source,
        universe_path=universe, exposure_groupings=groupings,
    )
    return compile_web_scientific_runtime_projection(
        study={"covariate_selection": "exact"}, sensitivity_specs=specs,
        primary_exposure=exposure, primary_exposure_source=source,
        target_outcome="death", target_is_event_status=True,
        declared_covariates=("age",), covariate_operationalizations={},
        universe_path=universe, scientific_configuration_sha256="a" * 64,
        exposure_groupings=groupings,
    )


def _universe(path: Path, column: str) -> Path:
    """Prepared rows for the candidate run (no patient data: synthetic values)."""
    pd.DataFrame({
        "stay_id": [1, 2, 3], column: [2.0, 7.0, 12.0], "age": [40.0, 50.0, 60.0],
        "death": [0, 1, 0], "death_time_hours": [np.nan, 48.0, np.nan],
        "hospital_followup_time_hours": [72.0, 48.0, 96.0],
    }).to_parquet(path, index=False)
    return path


@pytest.mark.parametrize("exposure", ["sofa", "sofa_max"])
def test_a_levelless_score_waits_for_the_runs_grouping_on_the_planning_catalog(tmp_path, exposure):
    catalog = _catalog(tmp_path / "planner_catalog.parquet", exposure)
    assert _project(catalog, exposure=exposure, source="sofa") is None


def test_prepared_data_without_a_closed_level_set_still_stop(tmp_path):
    universe = _universe(tmp_path / "universe.parquet", "sofa_max")
    with pytest.raises(WebScientificRuntimeProjectionError) as stopped:
        _project(universe, exposure="sofa_max", source="sofa")
    assert stopped.value.code == "web_landmark_categorical_levels_unavailable"


@pytest.mark.parametrize("concept", sorted(SCORES))
def test_the_run_forms_the_stated_bands_and_plans_them_as_an_ordered_landmark_exposure(tmp_path, concept):
    source = _project(_catalog(tmp_path / "source/planner_catalog.parquet", concept),
                      exposure=concept, source=concept)
    catalog, groupings = _grouped(tmp_path, concept)
    column, = groupings.columns
    assert column.variable == f"{concept}_group_x1" and column.levels == 3
    context = _context(catalog, primary_exposure=column.variable)
    assert family_template_id_for_context(
        context, analysis_types=("association_study",),
    ) == "landmark_categorical_association"
    # Whatever the source projection signed (nothing for a levelless score, a
    # categorical contract on a closed-level one) cannot govern the grouped
    # primary exposure the plan is made on.
    authorities = ScientificRuntimeAuthorities.load(
        trajectory=None, current_case=source.authority if source is not None else None,
    )
    assert authorities.for_planning_context(
        context=context, run_dir=catalog.parent, cohort_path=catalog,
    ).current_case is None


@pytest.mark.parametrize("concept", sorted(SCORES))
def test_the_run_that_binds_the_bands_signs_an_ordered_categorical_contract(tmp_path, concept):
    summary = SCORES[concept][0]
    _catalog_path, groupings = _grouped(tmp_path, concept)
    universe = _universe(tmp_path / "universe.parquet", f"{concept}_{summary}")
    projection = _project(universe, exposure=f"{concept}_group_x1", source=concept, groupings=groupings)
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert isinstance(authority, LandmarkCategoricalAssociationRuntimeAuthority)
    assert authority.exposure_column == f"{concept}_group_x1"
    assert authority.exposure_kind == "ordinal"
    assert authority.exposure_levels == ("1", "2", "3")
    assert (authority.exposure_reference_level, authority.primary_contrast_level) == ("1", "3")
