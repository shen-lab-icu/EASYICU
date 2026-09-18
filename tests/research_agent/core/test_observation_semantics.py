from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.research_context.observation_semantics import (
    compile_observation_semantics,
)
from easyicu.research_agent.schema import ConceptDescriptor, MissingnessProfile


@pytest.mark.parametrize("time_column,event_column", [
    ("event_elapsed", "event_flag"), ("failure_day", "failed"),
    ("outcome_recorded_at", "outcome_status"),
])
def test_declared_event_time_uses_event_denominator_without_name_inference(time_column, event_column):
    frame = pd.DataFrame({event_column: [0, 1, 1, 1], time_column: [np.nan, 30.0, np.nan, -2.0]})
    original = frame.copy(deep=True)
    descriptors = [_descriptor(event_column, is_binary=True), _descriptor(time_column, n_missing=2)]
    result = compile_observation_semantics(
        frame=frame, descriptors=descriptors, event_time_bindings={time_column: event_column},
    )[-1]
    assert result.role.value == "time"
    assert result.observation_semantics.event_status_column == event_column
    assert result.missingness.eligible_n == 3
    assert result.missingness.not_applicable_n == 1
    assert result.missingness.raw_n_missing == 2
    assert result.missingness.n_missing == 1
    assert result.missingness.fraction_missing == pytest.approx(1 / 3)
    assert any("precede the declared time origin" in note for note in result.clinical_caveats)
    pd.testing.assert_frame_equal(frame, original)
    assert compile_observation_semantics(frame=frame, descriptors=descriptors)[-1].observation_semantics is None


@pytest.mark.parametrize("events,times", [
    ([0, 1], [2.0, 3.0]), ([None, 1], [None, 3.0]),
    ([2, 1], [None, 3.0]), ([0, 1], [None, "invalid"]),
])
def test_invalid_declared_event_time_fails_instead_of_reverting_to_raw_missingness(events, times):
    with pytest.raises(ValueError):
        compile_observation_semantics(
            frame=pd.DataFrame({"flag": events, "elapsed": times}),
            descriptors=[_descriptor("flag"), _descriptor("elapsed")],
            event_time_bindings={"elapsed": "flag"},
        )


def test_landmark_declared_event_time_is_excluded_from_ordinary_mcar_screen(monkeypatch):
    from easyicu.research_agent.research_context import builder

    seen = []
    def screen(frame):
        seen.append(list(frame.columns))
        return {"name": "not_run", "columns": []}
    monkeypatch.setattr(builder, "_compute_missingness_test_metadata", screen)
    context = builder.build_research_context(
        research_question="Assess exposure and the binary event after a landmark.",
        cohort=pd.DataFrame({"stay_id": [1, 2, 3, 4], "event_flag": [0, 1, 0, 1],
            "event_elapsed": [np.nan, 30.0, np.nan, np.nan], "observed_hours": [48.0] * 4}),
        cohort_name="test", database="miiv", target_outcome="event_flag",
        user_preferences={"sensitivity_specs": [{
            "spec_id": "landmark", "axis": "timing", "strategy": "landmark",
            "execution_variables": ["event_elapsed", "observed_hours"], "landmark_hours": 24,
            "event_time_variable": "event_elapsed", "observation_duration_variable": "observed_hours",
            "observation_duration_unit": "hours", "require_alive_at_landmark": True,
        }]},
    )
    assert "event_elapsed" not in seen[0]
    time = context.variable("event_elapsed")
    assert time.role.value == "time"
    assert time.missingness.n_missing == 1
    assert time.missingness.not_applicable_n == 2


def test_declared_event_time_cannot_rebind_an_existing_verified_event():
    frame = pd.DataFrame({"event_a": [0, 1], "event_b": [0, 1], "event_a_time": [np.nan, 3.0]})
    with pytest.raises(ValueError, match="verified representation"):
        compile_observation_semantics(
            frame=frame, descriptors=[_descriptor("event_a", is_binary=True),
                _descriptor("event_b", is_binary=True), _descriptor("event_a_time", n_missing=1)],
            event_time_bindings={"event_a_time": "event_b"},
        )


def _descriptor(
    name: str,
    *,
    source_concept: str | None = None,
    unit_normalization: str | None = None,
    temporal_resolution: str | None = None,
    is_binary: bool = False,
    n_missing: int = 0,
    n_total: int = 4,
) -> ConceptDescriptor:
    return ConceptDescriptor(
        name=name,
        dtype="float64",
        source_concept=source_concept,
        unit_normalization=unit_normalization,
        temporal_resolution=temporal_resolution,
        observed_domain={"is_binary": is_binary},
        missingness=MissingnessProfile(
            fraction_missing=n_missing / n_total if n_total else 0.0,
            n_missing=n_missing,
            n_total=n_total,
            missingness_severity="high" if n_missing else "low",
        ),
    )


def test_positive_only_event_triad_is_complete_status_not_missingness() -> None:
    frame = pd.DataFrame(
        {
            "susp_inf_n": [0, 1, 2, 0],
            "susp_inf_measured": [0, 1, 1, 0],
            "susp_inf_first": [np.nan, 1.0, 1.0, np.nan],
        }
    )
    descriptors = [
        _descriptor(
            "susp_inf_first",
            source_concept="susp_inf",
            n_missing=2,
        )
    ]

    compiled = compile_observation_semantics(
        frame=frame,
        descriptors=descriptors,
    )

    descriptor = compiled[0]
    assert descriptor.observation_semantics is not None
    assert descriptor.observation_semantics.kind == "positive_only_event"
    assert descriptor.observation_semantics.event_count_column == "susp_inf_n"
    assert descriptor.missingness is not None
    assert descriptor.missingness.raw_n_missing == 2
    assert descriptor.missingness.n_missing == 0
    assert descriptor.missingness.fraction_missing == 0.0
    assert descriptor.missingness.eligible_n == 4


def test_normalized_binary_event_triad_retains_positive_only_semantics() -> None:
    frame = pd.DataFrame(
        {
            "susp_inf_n": [0, 1, 2, 0],
            "susp_inf_measured": [0, 1, 1, 0],
            "susp_inf_max": [0, 1, 1, 0],
        }
    )
    descriptors = [
        _descriptor(
            "susp_inf_max",
            source_concept="susp_inf",
            is_binary=True,
        )
    ]

    compiled = compile_observation_semantics(
        frame=frame,
        descriptors=descriptors,
    )

    descriptor = compiled[0]
    assert descriptor.observation_semantics is not None
    assert descriptor.observation_semantics.kind == "positive_only_event"
    assert descriptor.observation_semantics.event_count_column == "susp_inf_n"
    assert descriptor.observation_semantics.measured_column == "susp_inf_measured"
    assert descriptor.missingness is not None
    assert descriptor.missingness.raw_n_missing == 0
    assert descriptor.missingness.n_missing == 0
    assert descriptor.missingness.fraction_missing == 0.0


def test_conditional_event_time_uses_event_positive_denominator() -> None:
    frame = pd.DataFrame(
        {
            "death": [0, 1, 1, 0],
            "death_time": [np.nan, 48.0, -2.0, np.nan],
        }
    )
    descriptors = [
        _descriptor(
            "death",
            source_concept="death",
            is_binary=True,
        ),
        _descriptor(
            "death_time",
            source_concept="death",
            unit_normalization="first_truthy_event_time",
            temporal_resolution="relative to icu_admission in h",
            n_missing=2,
        ),
    ]

    compiled = compile_observation_semantics(
        frame=frame,
        descriptors=descriptors,
    )

    death_time = next(item for item in compiled if item.name == "death_time")
    assert death_time.observation_semantics is not None
    assert death_time.observation_semantics.kind == "conditional_event_time"
    assert death_time.observation_semantics.event_status_column == "death"
    assert death_time.observation_semantics.time_origin == "icu_admission"
    assert death_time.observation_semantics.time_unit == "h"
    assert death_time.missingness is not None
    assert death_time.missingness.raw_n_missing == 2
    assert death_time.missingness.not_applicable_n == 2
    assert death_time.missingness.eligible_n == 2
    assert death_time.missingness.n_missing == 0
    assert any("precede the declared time origin" in item for item in death_time.clinical_caveats)


def test_legacy_export_event_time_is_reconciled_without_treating_absence_as_missing() -> None:
    frame = pd.DataFrame(
        {
            "death": [0, 1, 1, 0],
            "death_time": [np.nan, 48.0, np.nan, np.nan],
        }
    )
    descriptors = [
        _descriptor("death", is_binary=True),
        _descriptor("death_time", n_missing=3),
    ]

    compiled = compile_observation_semantics(
        frame=frame,
        descriptors=descriptors,
    )

    death_time = next(item for item in compiled if item.name == "death_time")
    assert death_time.observation_semantics is not None
    assert death_time.observation_semantics.kind == "conditional_event_time"
    assert death_time.observation_semantics.event_status_column == "death"
    assert death_time.missingness is not None
    assert death_time.missingness.raw_n_missing == 3
    assert death_time.missingness.not_applicable_n == 2
    assert death_time.missingness.eligible_n == 2
    assert death_time.missingness.n_missing == 1
    assert death_time.missingness.fraction_missing == 0.5


def test_builder_wires_positive_only_event_semantics_into_context() -> None:
    frame = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "death": [0, 1, 0, 0],
            "susp_inf_n": [0, 1, 2, 0],
            "susp_inf_measured": [0, 1, 1, 0],
            "susp_inf_first": [np.nan, 1.0, 1.0, np.nan],
        }
    )

    context = build_research_context(
        research_question="Describe suspected infection and mortality.",
        cohort=frame,
        cohort_name="test",
        database="miiv",
        target_outcome="death",
    )

    descriptor = context.variable("susp_inf_first")
    assert descriptor is not None
    assert descriptor.observation_semantics is not None
    assert descriptor.observation_semantics.kind == "positive_only_event"
    assert descriptor.missingness is not None
    assert descriptor.missingness.n_missing == 0


@pytest.mark.parametrize("transform", ["window_first_time", "window_last_time"])
def test_typed_observation_time_uses_verified_measurement_opportunity(transform) -> None:
    # Deliberately arbitrary physical names: the materialized transform and
    # source, not a suffix or a coincidentally binary clinical value, own this.
    frame = pd.DataFrame({
        "count": [0, 1, 2, 0], "available": [0, 1, 1, 0],
        "assay_value": [0, 0, 1, 0], "observed_at": [np.nan, 2.0, np.nan, np.nan],
    })
    descriptors = [
        _descriptor("count", source_concept="assay", unit_normalization="window_nonnull_count"),
        _descriptor("available", source_concept="assay", unit_normalization="window_measurement_status", is_binary=True),
        _descriptor("assay_value", source_concept="assay", is_binary=True),
        _descriptor("observed_at", source_concept="assay", unit_normalization=transform, n_missing=3),
    ]

    result = compile_observation_semantics(frame=frame, descriptors=descriptors)[-1]

    assert result.observation_semantics.event_status_column == "available"
    assert result.missingness.raw_n_missing == 3
    assert result.missingness.not_applicable_n == 2
    assert result.missingness.eligible_n == 2
    assert result.missingness.n_missing == 1
    assert result.missingness.fraction_missing == 0.5
    assert result.missingness.missingness_test_p_value is None
    assert "observation" in result.missingness.notes


@pytest.mark.parametrize("mutation", ["discordant_count", "missing_flag", "wrong_source", "ambiguous_flag"])
def test_observation_time_does_not_guess_applicability_without_one_valid_pair(mutation) -> None:
    frame = pd.DataFrame({
        "count": [0, 1, 2, 0], "available": [0, 1, 1, 0],
        "observed_at": [np.nan, 2.0, 3.0, np.nan],
    })
    descriptors = [
        _descriptor("count", source_concept="assay", unit_normalization="window_nonnull_count"),
        _descriptor("available", source_concept="assay", unit_normalization="window_measurement_status", is_binary=True),
        _descriptor("observed_at", source_concept="assay", unit_normalization="window_last_time", n_missing=2),
    ]
    if mutation == "discordant_count":
        frame.loc[0, "count"] = 1
    elif mutation == "missing_flag":
        frame.loc[0, "available"] = np.nan
    elif mutation == "wrong_source":
        descriptors[1] = descriptors[1].model_copy(update={"source_concept": "other"})
    else:
        frame["other_flag"] = frame["available"]
        descriptors.insert(1, descriptors[1].model_copy(update={"name": "other_flag"}))

    result = compile_observation_semantics(frame=frame, descriptors=descriptors)[-1]

    assert result.observation_semantics is None
    assert result.missingness.n_missing == 2


def test_builder_excludes_structural_absence_before_global_mcar_screen(monkeypatch) -> None:
    from easyicu.research_agent.research_context import builder

    seen = []
    def screen(frame):
        seen.append(set(frame.columns))
        return {"name": "little_mcar_em", "p_value": 0.31, "note": "test panel", "columns": ["lab_x", "lab_y"]}
    monkeypatch.setattr(builder, "_compute_missingness_test_metadata", screen)
    frame = pd.DataFrame({
        "stay_id": [1, 2, 3, 4], "death": [0, 1, 0, 0],
        "susp_inf_n": [0, 1, 2, 0], "susp_inf_measured": [0, 1, 1, 0],
        "susp_inf_first": [0, 1, 1, 0],
        "susp_inf_first_time": [np.nan, 2, 3, np.nan],
        "lab_x": [1.0, np.nan, 3.0, 4.0], "lab_y": [1.0, 2.0, np.nan, 4.0],
        "lab_not_in_panel": [np.nan, 2.0, 3.0, 4.0],
    })

    context = builder.build_research_context(
        research_question="Describe suspected infection and mortality.", cohort=frame,
        cohort_name="test", database="miiv", target_outcome="death",
    )

    assert len(seen) == 1
    assert "susp_inf_first" not in seen[0]
    assert "susp_inf_first_time" not in seen[0]
    assert "susp_inf_n" not in seen[0]
    assert context.variable("lab_x").missingness.missingness_test_p_value == 0.31
    assert context.variable("lab_not_in_panel").missingness.missingness_test_p_value is None


@pytest.mark.parametrize("flag", [0, 1])
def test_verified_constant_measurement_flag_does_not_need_two_observed_levels(flag):
    frame = pd.DataFrame({"n": [flag] * 4, "flag": [flag] * 4, "time": [np.nan] * 4})
    descriptors = [
        _descriptor("n", source_concept="assay", unit_normalization="window_nonnull_count"),
        _descriptor("flag", source_concept="assay", unit_normalization="window_measurement_status"),
        _descriptor("time", source_concept="assay", unit_normalization="window_last_time", n_missing=4),
    ]
    result = compile_observation_semantics(frame=frame, descriptors=descriptors)[-1]
    assert result.observation_semantics.event_status_column == "flag"
    assert result.missingness.eligible_n == 4 * flag
    assert result.missingness.not_applicable_n == 4 * (1 - flag)
    assert result.missingness.n_missing == 4 * flag
