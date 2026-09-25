"""A landmark exposure whose concept owner declares a closed two-level domain.

Planning reads the concept owner's declared domain before any row exists, so a
source-published event status (an antibiotic exposure, a positive culture) is
a binary exposure there even though the zero-row catalog types every column
float64 and the materializer later writes it as int64.  The signed projection
and the launch's automatic functional-form safeguard read the same owner, so
the reviewed categorical plan and its signed runtime agree in both universes
instead of the exposure being signed to the continuous spline runtime.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    LandmarkCategoricalAssociationRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.authority.declared_levels import closed_planning_levels_for
from easyicu.research_agent.contracts.model_terms import ModelTermSpec, level_spelling
from easyicu.research_agent.execution.model_matrix import compile_model_terms
from easyicu.research_agent.icu_rules import VariableKind
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)
from easyicu.research_agent.schema import ConceptDescriptor
from easyicu.webserver.research_launch_scientific import (
    _materialized_column_dtype,
    _runtime_projection_sensitivity_specs,
)
from easyicu.webserver.scientific_runtime_projection import (
    compile_web_scientific_runtime_projection,
    exposure_kind_for_dtype,
)

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


def _zero_row_universe(path):
    """The metadata-only planning catalog: identity int64, every column float64."""

    pd.DataFrame(
        {
            "stay_id": pd.Series(dtype="int64"),
            "abx": pd.Series(dtype="float64"),
            "death": pd.Series(dtype="float64"),
            "death_time_hours": pd.Series(dtype="float64"),
            "hospital_followup_time_hours": pd.Series(dtype="float64"),
            "age": pd.Series(dtype="float64"),
        }
    ).to_parquet(path, index=False)
    return path


def _materialized_universe(path):
    """Synthetic rows typed the way the materializer writes event status."""

    pd.DataFrame(
        {
            "stay_id": pd.Series([1, 2, 3, 4], dtype="int64"),
            "abx": pd.Series([0, 1, 0, 1], dtype="int64"),
            "death": pd.Series([0, 1, 1, 0], dtype="int64"),
            "death_time_hours": [float("nan"), 60.0, 90.0, float("nan")],
            "hospital_followup_time_hours": [120.0, 60.0, 90.0, 150.0],
            "age": [55.0, 71.0, 64.0, 48.0],
        }
    ).to_parquet(path, index=False)
    return path


def _signed(universe):
    specs = _runtime_projection_sensitivity_specs(
        (_LANDMARK,),
        primary_exposure_source="abx",
        primary_exposure_dtype=_materialized_column_dtype(universe, "abx"),
        primary_exposure="abx",
    )
    # A binary exposure never receives the continuous spline safeguard.
    assert specs == (_LANDMARK,)
    projection = compile_web_scientific_runtime_projection(
        study={"covariate_selection": "exact"},
        sensitivity_specs=specs,
        primary_exposure="abx",
        primary_exposure_source="abx",
        target_outcome="death",
        declared_covariates=("age",),
        covariate_operationalizations={},
        target_is_event_status=True,
        universe_path=universe,
        scientific_configuration_sha256="c" * 64,
    )
    assert projection is not None
    return projection


def test_the_zero_row_and_materialized_universes_sign_one_binary_contract(tmp_path):
    planned = _signed(_zero_row_universe(tmp_path / "planner_catalog.parquet"))
    executed = _signed(_materialized_universe(tmp_path / "universe.parquet"))

    authority = load_current_case_scientific_runtime_authority(planned.authority)
    assert isinstance(authority, LandmarkCategoricalAssociationRuntimeAuthority)
    assert authority.exposure_kind == "binary"
    assert authority.exposure_levels == ("0", "1")
    assert authority.exposure_reference_level == "0"
    assert authority.primary_contrast_level == "1"
    assert executed.projection_sha256 == planned.projection_sha256


def test_the_projection_reads_the_levels_planning_reads(tmp_path):
    descriptor = ConceptDescriptor(name="abx", dtype="float64", source_concept="abx")
    planning = tuple(
        level_spelling(value)
        for value in closed_planning_levels_for(name="abx", variables={"abx": descriptor})
    )

    assert exposure_kind_for_dtype(
        primary_exposure="abx", primary_exposure_source="abx", dtype="double"
    ) == (VariableKind.BINARY, planning)


@pytest.mark.parametrize("concept", ["abx", "heparin", "culture_positive"])
@pytest.mark.parametrize("dtype", ["double", "float", "int64"])
def test_every_declared_event_status_is_binary_whatever_its_numeric_type(concept, dtype):
    assert exposure_kind_for_dtype(
        primary_exposure=concept, primary_exposure_source=concept, dtype=dtype
    ) == (VariableKind.BINARY, ("0", "1"))


def test_a_declared_two_level_factor_keeps_its_declared_order():
    assert exposure_kind_for_dtype(
        primary_exposure="sex", primary_exposure_source="sex", dtype="large_string"
    ) == (VariableKind.BINARY, ("Female", "Male"))


def test_a_boolean_column_binds_the_spelling_of_its_own_type():
    assert exposure_kind_for_dtype(
        primary_exposure="abx", primary_exposure_source="abx", dtype="bool"
    ) == (VariableKind.BINARY, ("false", "true"))


def test_an_operationalized_column_does_not_inherit_its_concepts_domain():
    # The factor's levels describe the concept's own values, not a maximum of
    # them; a continuous summary keeps the continuous rule and its safeguard.
    _kind, levels = exposure_kind_for_dtype(
        primary_exposure="mech_vent_max", primary_exposure_source="mech_vent", dtype="double"
    )
    assert levels == ()
    assert exposure_kind_for_dtype(
        primary_exposure="lact_max", primary_exposure_source="lact", dtype="double"
    ) == (VariableKind.CONTINUOUS, ())
    specs = _runtime_projection_sensitivity_specs(
        (_LANDMARK,),
        primary_exposure_source="lact",
        primary_exposure_dtype="double",
        primary_exposure="lact_max",
    )
    assert [spec.spec_id for spec in specs] == [
        "landmark_24h",
        "easyicu_auto_primary_exposure_rcs",
    ]


def test_a_nominal_domain_needs_a_reviewed_reference_level():
    # Three unordered levels have no row-free reference level, so the
    # declared domain is not turned into signed contrasts here.
    kind, levels = exposure_kind_for_dtype(
        primary_exposure="adm", primary_exposure_source="adm", dtype="large_string"
    )
    assert levels == ()
    assert kind is not VariableKind.BINARY


def test_ordinal_rule_levels_keep_precedence():
    assert exposure_kind_for_dtype(
        primary_exposure="aki_stage", primary_exposure_source="aki_stage", dtype="int64"
    ) == (VariableKind.ORDINAL, ("0", "1", "2", "3"))


@pytest.mark.parametrize("values", [[0.0, 1.0, float("nan"), 1.0], [0, 1, 0, 1]])
def test_the_signed_levels_bind_the_materialized_encoding(values):
    kind, levels = exposure_kind_for_dtype(
        primary_exposure="abx", primary_exposure_source="abx", dtype="double"
    )
    term = ModelTermSpec(
        name="abx",
        role="exposure",
        coding=kind.value,
        levels=list(levels),
        reference_level=levels[0],
        transform="treatment_contrast",
    )

    compiled = compile_model_terms(pd.DataFrame({"abx": values}), terms=(term,), exposure="abx")

    assert compiled.exposure_columns == ("abx__is_1",)
