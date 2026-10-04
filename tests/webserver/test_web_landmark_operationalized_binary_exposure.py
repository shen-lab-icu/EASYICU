"""A landmark exposure that the materializer writes as a window summary.

A reviewed candidate plan signs the categorical landmark runtime on the
zero-row catalog, where the exposure is the concept itself.  After data
preparation the physical exposure is the concept's window maximum
(``<concept>_max``): an int64 column whose name no longer matches the concept,
so the name and dtype rules read it as a count and the study lost its signed
runtime at the formal replan.  The materializer publishes that column's own
two-value domain in the universe's verified column metadata, and routing now
reads it.  A fraction, a count or a numeric summary publishes no domain, so
its routing is exactly what the name and dtype rules gave before.  The
universe here is built with the host's own materializer from a synthetic
export.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.icu_rules import VariableKind
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)
from easyicu.webserver import scientific_runtime_projection as projection_owner
from easyicu.webserver.research_launch_scientific import (
    _runtime_projection_sensitivity_specs,
)
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from easyicu.webserver.scientific_runtime_projection import (
    WebScientificRuntimeProjectionError,
    compile_web_scientific_runtime_projection,
    exposure_kind_for_dtype,
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


@pytest.fixture
def universe(tmp_path: Path) -> Path:
    source = typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("mech_vent", "lact"),
        outcome_concepts=("death",),
    )
    return Path(paths["parquet"])


def test_a_window_maximum_of_an_event_status_is_binary(universe: Path) -> None:
    assert primary_exposure_kind(
        universe_path=universe,
        primary_exposure="mech_vent_max",
        primary_exposure_source="mech_vent",
    ) == (VariableKind.BINARY, ("0", "1"))


def _name_rule(path: Path, column: str, source: str):
    return exposure_kind_for_dtype(
        primary_exposure=column,
        primary_exposure_source=source,
        dtype=str(pq.read_schema(path).field(column).type),
    )


@pytest.mark.parametrize("column", ["mech_vent_mean", "mech_vent_n", "lact_max"])
def test_a_fraction_count_or_numeric_summary_keeps_its_rule(
    universe: Path, column: str
) -> None:
    source = column.rsplit("_", 1)[0]
    routed = primary_exposure_kind(
        universe_path=universe, primary_exposure=column, primary_exposure_source=source
    )
    assert routed[1] == ()
    assert routed == _name_rule(universe, column, source)


def test_the_launch_safeguard_reads_the_same_universe(universe: Path) -> None:
    # The binary summary gets no continuous safeguard; a numeric one does.
    assert _runtime_projection_sensitivity_specs(
        (_LANDMARK,),
        primary_exposure_source="mech_vent",
        primary_exposure="mech_vent_max",
        universe_path=universe,
    ) == (_LANDMARK,)
    numeric = _runtime_projection_sensitivity_specs(
        (_LANDMARK,),
        primary_exposure_source="lact",
        primary_exposure="lact_max",
        universe_path=universe,
    )
    assert [spec.spec_id for spec in numeric] == [
        "landmark_24h",
        "easyicu_auto_primary_exposure_rcs",
    ]


def test_the_reviewed_categorical_design_is_signed_after_preparation(
    universe: Path, monkeypatch
) -> None:
    routed: list[str] = []
    monkeypatch.setattr(
        projection_owner,
        "compile_landmark_categorical_runtime_projection",
        lambda **coordinates: routed.append("categorical") or "signed",
    )
    monkeypatch.setattr(
        projection_owner,
        "compile_landmark_spline_runtime_projection",
        lambda **coordinates: routed.append("spline"),
    )
    specs = _runtime_projection_sensitivity_specs(
        (_LANDMARK,),
        primary_exposure_source="mech_vent",
        primary_exposure="mech_vent_max",
        universe_path=universe,
    )

    result = compile_web_scientific_runtime_projection(
        study={"covariate_selection": "exact"},
        sensitivity_specs=specs,
        primary_exposure="mech_vent_max",
        primary_exposure_source="mech_vent",
        target_outcome="death",
        declared_covariates=("age",),
        covariate_operationalizations={},
        target_is_event_status=True,
        universe_path=universe,
        scientific_configuration_sha256="c" * 64,
    )

    assert routed == ["categorical"]
    assert result == "signed"


def test_metadata_that_does_not_verify_is_refused(universe: Path) -> None:
    provenance = universe.with_name(f"{universe.stem}_provenance.json")
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    payload["column_metadata"]["authority"]["sha256"] = "0" * 64
    provenance.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(WebScientificRuntimeProjectionError) as raised:
        primary_exposure_kind(
            universe_path=universe,
            primary_exposure="mech_vent_max",
            primary_exposure_source="mech_vent",
        )
    assert raised.value.code == "web_scientific_runtime_metadata_unverified"


def test_a_universe_without_materialized_metadata_keeps_the_name_rule(
    tmp_path: Path,
) -> None:
    import pandas as pd

    catalog = tmp_path / "planner_catalog.parquet"
    pd.DataFrame({"mech_vent_max": pd.Series(dtype="float64")}).to_parquet(
        catalog, index=False
    )

    routed = primary_exposure_kind(
        universe_path=catalog,
        primary_exposure="mech_vent_max",
        primary_exposure_source="mech_vent",
    )
    assert routed[1] == ()
    assert routed == _name_rule(catalog, "mech_vent_max", "mech_vent")


def test_a_published_domain_overrides_the_count_reading_of_an_int64_flag() -> None:
    # The prepared Sepsis-3 shape: an int64 window maximum of a flag whose
    # name and dtype read as a count when its metadata publishes no domain.
    assert exposure_kind_for_dtype(
        primary_exposure="abx_max",
        primary_exposure_source="abx",
        dtype="int64",
        published_levels=(),
    ) == (VariableKind.COUNT, ())
    assert exposure_kind_for_dtype(
        primary_exposure="abx_max",
        primary_exposure_source="abx",
        dtype="int64",
        published_levels=(0, 1),
    ) == (VariableKind.BINARY, ("0", "1"))


@pytest.mark.parametrize("published", [(0,), (0, 1, 2), (1, 1)])
def test_only_a_two_value_domain_is_binary(published) -> None:
    assert exposure_kind_for_dtype(
        primary_exposure="abx_max",
        primary_exposure_source="abx",
        dtype="int64",
        published_levels=published,
    ) == (VariableKind.COUNT, ())


def test_the_safeguard_follows_the_published_domain_not_the_dtype(
    tmp_path: Path, monkeypatch
) -> None:
    import pandas as pd

    universe = tmp_path / "universe.parquet"
    pd.DataFrame({"abx_max": pd.Series(dtype="float64")}).to_parquet(universe, index=False)
    monkeypatch.setattr(
        projection_owner, "_published_column_domain", lambda path, column: (0, 1)
    )

    # Read on the universe it signs, a binary summary gets no spline safeguard,
    # so the categorical runtime is not handed a continuous request for it.
    assert _runtime_projection_sensitivity_specs(
        (_LANDMARK,),
        primary_exposure_source="abx",
        primary_exposure="abx_max",
        universe_path=universe,
    ) == (_LANDMARK,)


def test_the_safeguard_reads_the_schema_of_the_universe_it_signs(tmp_path: Path) -> None:
    import pandas as pd

    universe = tmp_path / "universe.parquet"
    pd.DataFrame({"x_max": pd.Series(dtype="float64")}).to_parquet(universe, index=False)

    # A numeric summary with no published domain is continuous on its schema,
    # so it keeps the automatic spline safeguard.
    specs = _runtime_projection_sensitivity_specs(
        (_LANDMARK,),
        primary_exposure_source="x",
        primary_exposure="x_max",
        universe_path=universe,
    )
    assert [spec.spec_id for spec in specs] == [
        "landmark_24h",
        "easyicu_auto_primary_exposure_rcs",
    ]


def test_a_design_without_a_landmark_never_reads_the_universe(tmp_path: Path) -> None:
    placeholder = tmp_path / "universe.parquet"
    placeholder.write_text("not parquet", encoding="utf-8")
    other = PrespecifiedSensitivitySpec(
        spec_id="complete_case",
        axis="missing_data",
        strategy="complete_case",
        execution_variables=("lact",),
    )

    assert _runtime_projection_sensitivity_specs(
        (other,),
        primary_exposure_source="lact",
        primary_exposure="lact_max",
        universe_path=placeholder,
    ) == (other,)


def test_an_unreadable_universe_keeps_the_projection_reason_code(tmp_path: Path) -> None:
    placeholder = tmp_path / "universe.parquet"
    placeholder.write_text("not parquet", encoding="utf-8")

    with pytest.raises(ResearchPipelineRunError) as raised:
        _runtime_projection_sensitivity_specs(
            (_LANDMARK,),
            primary_exposure_source="lact",
            primary_exposure="lact_max",
            universe_path=placeholder,
        )
    assert raised.value.code == "web_scientific_runtime_schema_unavailable"
