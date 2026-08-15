from __future__ import annotations

import numpy as np
import pandas as pd

from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.research_context.observation_semantics import (
    compile_observation_semantics,
)
from easyicu.research_agent.schema import ConceptDescriptor, MissingnessProfile


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
