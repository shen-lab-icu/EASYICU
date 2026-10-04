"""A window summary of an event status is binary on the planning catalog.

Metadata-only planning names an aggregated exposure ``<concept>_<aggregation>``
before any row exists.  The materializer later writes the window maximum,
minimum or first value of an event-status concept as an event status with
values 0 and 1 and publishes that domain in the universe's column metadata,
but the zero-row planning catalog has no column metadata.  The exposure
therefore had no levels there, and the categorical landmark runtime refused
to sign it (``web_landmark_categorical_levels_unavailable``) before any
analysis.  Planning now reads the owners the materializer applies: the concept
owner's event-status declaration and the summary the column names.  A mean,
a count, a summary of a factor or a measurement, and a column that is not its
declared source's own summary keep no levels.  Synthetic schemas and exports
only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    LandmarkCategoricalAssociationRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.icu_rules import VariableKind
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)
from easyicu.webserver.research_launch_scientific import (
    _runtime_projection_sensitivity_specs,
)
from easyicu.webserver.scientific_runtime_projection import (
    compile_web_scientific_runtime_projection,
    primary_exposure_kind,
)

from tests.support.typed_export import typed_export

_LANDMARK = PrespecifiedSensitivitySpec.model_validate(
    {
        "spec_id": "landmark_24h",
        "axis": "timing",
        "strategy": "landmark",
        "landmark_hours": 24,
        "require_alive_at_landmark": True,
        "exclude_negative_event_times": True,
        "event_time_variable": "death_time_hours",
        "observation_duration_variable": "hospital_followup_time_hours",
        "observation_duration_unit": "hours",
    }
)


def _planning_catalog(path: Path, *columns: str) -> Path:
    """The metadata-only planning catalog: identity int64, every column float64."""

    pd.DataFrame(
        {
            "stay_id": pd.Series(dtype="int64"),
            **{column: pd.Series(dtype="float64") for column in columns},
        }
    ).to_parquet(path, index=False)
    return path


@pytest.mark.parametrize("concept", ["abx", "rrt", "vaso_ind"])
@pytest.mark.parametrize("aggregation", ["max", "min", "first"])
def test_a_window_summary_of_an_event_status_has_the_status_levels(
    tmp_path: Path, concept: str, aggregation: str
) -> None:
    column = f"{concept}_{aggregation}"
    catalog = _planning_catalog(tmp_path / "planner_catalog.parquet", column)

    assert primary_exposure_kind(
        universe_path=catalog,
        primary_exposure=column,
        primary_exposure_source=concept,
    ) == (VariableKind.BINARY, ("0", "1"))


@pytest.mark.parametrize(
    "column, source",
    [
        ("abx_mean", "abx"),  # an event fraction
        ("abx_n", "abx"),  # a count
        ("mech_vent_max", "mech_vent"),  # a summary of a two-level factor
        ("adm_max", "adm"),  # a summary of a three-level factor
        ("lact_max", "lact"),  # a summary of a measurement
        ("abx_max", "rrt"),  # not the declared source's own summary
    ],
)
def test_other_summaries_keep_no_levels(tmp_path: Path, column: str, source: str) -> None:
    catalog = _planning_catalog(tmp_path / "planner_catalog.parquet", column)

    _kind, levels = primary_exposure_kind(
        universe_path=catalog,
        primary_exposure=column,
        primary_exposure_source=source,
    )
    assert levels == ()


def test_planning_agrees_with_the_domain_the_materializer_publishes(
    tmp_path: Path,
) -> None:
    source = typed_export(tmp_path / "export", event_concepts=("abx",))
    universe = Path(
        cohort_materializer.materialize_to_parquet(
            tmp_path / "materialized",
            data_path=source,
            database="miiv",
            static_concepts=("age",),
            feature_concepts=("abx", "lact"),
            outcome_concepts=("death",),
        )["parquet"]
    )
    import pyarrow.parquet as pq

    summaries = [
        name
        for name in pq.read_schema(universe).names
        if name in {"abx_max", "abx_min", "abx_mean", "abx_first"}
    ]
    assert {"abx_max", "abx_mean"} <= set(summaries)
    catalog = _planning_catalog(tmp_path / "planner_catalog.parquet", *summaries)

    for column in summaries:
        published = primary_exposure_kind(
            universe_path=universe, primary_exposure=column, primary_exposure_source="abx"
        )
        planned = primary_exposure_kind(
            universe_path=catalog, primary_exposure=column, primary_exposure_source="abx"
        )
        assert planned == published, column
    assert primary_exposure_kind(
        universe_path=universe, primary_exposure="abx_max", primary_exposure_source="abx"
    ) == (VariableKind.BINARY, ("0", "1"))


def test_the_categorical_landmark_runtime_is_signed_on_the_planning_catalog(
    tmp_path: Path,
) -> None:
    catalog = _planning_catalog(
        tmp_path / "planner_catalog.parquet",
        "abx_max",
        "death",
        "death_time_hours",
        "hospital_followup_time_hours",
        "age",
    )
    specs = _runtime_projection_sensitivity_specs(
        (_LANDMARK,),
        primary_exposure_source="abx",
        primary_exposure="abx_max",
        universe_path=catalog,
    )
    # A binary exposure never receives the continuous spline safeguard.
    assert specs == (_LANDMARK,)

    projection = compile_web_scientific_runtime_projection(
        study={"covariate_selection": "exact"},
        sensitivity_specs=specs,
        primary_exposure="abx_max",
        primary_exposure_source="abx",
        target_outcome="death",
        declared_covariates=("age",),
        covariate_operationalizations={},
        target_is_event_status=True,
        universe_path=catalog,
        scientific_configuration_sha256="c" * 64,
    )

    assert projection is not None
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert isinstance(authority, LandmarkCategoricalAssociationRuntimeAuthority)
    assert authority.exposure_kind == "binary"
    assert authority.exposure_levels == ("0", "1")
    assert authority.exposure_reference_level == "0"
    assert authority.primary_contrast_level == "1"
