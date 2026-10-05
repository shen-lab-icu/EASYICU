"""An event status is timed by its first record as present.

``<c>_first_time`` is the first record of any value, an observation time.  The
landmark survival suite classified exposure by it, so a stay whose source first
recorded the exposure absent at ICU admission and present hours later counted
as prevalent and was excluded.  A typed event status now also publishes
``<c>_onset_time``, the first record inside the window whose status is present,
declared as an event time.  The research context reads that time as applicable
where the event is present in its window, never where the source was merely
recorded.

Synthetic export rows only (invasive ventilation and kidney replacement
therapy); no patient data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.metadata_projection import ConceptColumnRole
from easyicu.concept.metadata_sidecar import SidecarRef, read_content_addressed_sidecar
from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.literature_concepts import concept_id
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.research_context.observation_semantics import (
    compile_observation_semantics,
)
from easyicu.research_agent.research_context.outbound import outbound_safe_context_payload
from easyicu.research_agent.research_context.prompt_scope import _variable_family
from easyicu.research_agent.schema import ConceptDescriptor, MissingnessProfile
from tests.support.typed_export import typed_export

# Stay 1 is recorded absent at admission, then present at hour 3; stay 2 is
# present at its first record; stay 3 is only ever recorded absent; stay 4 has
# a record only when every stay is recorded.
_RECORDS = ((1, 0.0, False), (1, 3.0, True), (1, 5.0, True), (2, 2.0, True), (3, 1.0, False), (3, 4.0, False))


def _labs(concept: str, *, every_stay_recorded: bool) -> pd.DataFrame:
    rows = [*_RECORDS, (4, 2.0, False if every_stay_recorded else None)]
    status = pd.array([row[2] for row in rows], dtype="boolean")
    return pd.DataFrame(
        {
            "stay_id": [row[0] for row in rows],
            "charttime": [row[1] for row in rows],
            "age": [60] * len(rows),
            "lact": [1.5] * len(rows),
            "mech_vent": status if concept == "mech_vent" else pd.array([False] * len(rows), dtype="boolean"),
            **({concept: status} if concept != "mech_vent" else {}),
        }
    )


def _materialize(tmp_path: Path, concept: str, *, every_stay_recorded: bool) -> dict[str, Path]:
    source = typed_export(
        tmp_path / "export",
        labs=_labs(concept, every_stay_recorded=every_stay_recorded),
        outcomes=pd.DataFrame({"stay_id": [1, 2, 3, 4], "death": [False, True, False, False]}),
        event_concepts=() if concept == "mech_vent" else (concept,),
    )
    return cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        stem="universe",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=tuple(dict.fromkeys(("lact", "mech_vent", concept))),
        outcome_concepts=("death",),
    )


@pytest.mark.parametrize("concept", ["mech_vent", "rrt"])
def test_the_onset_is_the_first_record_as_present(tmp_path, concept):
    paths = _materialize(tmp_path, concept, every_stay_recorded=False)
    cohort = pd.read_parquet(paths["parquet"]).set_index("stay_id")
    onset = cohort[f"{concept}_onset_time"]

    # Classified by its first record (hour 0, at the prevalence cutoff) the
    # first stay was prevalent; by its first present record it is incident.
    assert cohort.loc[1, f"{concept}_first_time"] == 0.0
    assert onset.loc[1] == 3.0
    assert onset.loc[2] == 2.0
    assert math.isnan(onset.loc[3]) and math.isnan(onset.loc[4])
    assert onset.notna().tolist() == cohort[f"{concept}_max"].eq(1).tolist()
    # A value has no presence to time.
    assert "lact_onset_time" not in cohort.columns

    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    reference = SidecarRef.from_dict(provenance["column_metadata"]["sidecar"])
    sidecar = read_content_addressed_sidecar(
        paths["column_metadata"], expected_sha256=reference.sha256, expected_size=reference.size
    )
    columns = sidecar.files[0].columns
    declared = columns[f"{concept}_onset_time"]
    assert declared.metadata.role is ConceptColumnRole.EVENT_TIME
    assert declared.representation_transform == "first_truthy_event_time"
    assert declared.derivation_window == columns[f"{concept}_max"].derivation_window
    assert columns[f"{concept}_first_time"].metadata.role is ConceptColumnRole.FIRST_OBSERVATION_TIME


def test_a_source_that_records_only_presence_keeps_its_first_record_time(tmp_path):
    # The shape of the current MIIV export: an event status stored as True or
    # null.  Its first record is its first present record, so the onset and
    # the first-observation time agree there.
    labs = pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 3],
            "charttime": [0.0, 4.0, 2.0, 1.0],
            "age": [60] * 4,
            "lact": [1.5] * 4,
            "mech_vent": pd.array([False] * 4, dtype="boolean"),
            "vent_ind": pd.array([True, True, True, None], dtype="boolean"),
        }
    )
    source = typed_export(
        tmp_path / "export",
        labs=labs,
        outcomes=pd.DataFrame({"stay_id": [1, 2, 3], "death": [False, True, False]}),
        event_concepts=("vent_ind",),
    )
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized", stem="universe", data_path=source, database="miiv",
        static_concepts=("age",), feature_concepts=("lact", "mech_vent", "vent_ind"),
        outcome_concepts=("death",),
    )
    cohort = pd.read_parquet(paths["parquet"]).set_index("stay_id")

    present = cohort["vent_ind_max"].eq(1)
    assert present.tolist() == [True, True, False]
    assert cohort.loc[present, "vent_ind_onset_time"].tolist() == [0.0, 2.0]
    assert cohort.loc[present, "vent_ind_first_time"].tolist() == [0.0, 2.0]
    assert cohort.loc[~present, "vent_ind_onset_time"].isna().all()


def test_the_context_reads_the_onset_where_the_window_recorded_the_event(tmp_path):
    paths = _materialize(tmp_path, "mech_vent", every_stay_recorded=True)

    context = build_research_context(
        research_question="Describe when ventilation began after ICU admission.",
        cohort=paths["parquet"],
        cohort_name="onset",
        database="miiv",
        target_outcome="death",
        primary_exposure="mech_vent_max",
        id_columns=("stay_id",),
        outcome_columns=("death",),
    )

    onset = context.variable("mech_vent_onset_time")
    # A companion of its concept, not a concept of its own.
    assert onset.concept_enrichment_degraded is False
    assert onset.observation_semantics is not None
    assert onset.observation_semantics.kind == "conditional_event_time"
    # Not the first recorded status, which is absent for the first stay.
    assert onset.observation_semantics.event_status_column == "mech_vent_max"
    assert onset.missingness.not_applicable_n == 2
    assert onset.missingness.n_missing == 0
    (sent,) = [
        item for item in outbound_safe_context_payload(context)["variables"]
        if item["name"] == "mech_vent_onset_time"
    ]
    assert sent["materialized_representation"] == "first_truthy_event_time"


@pytest.mark.parametrize("family", [_variable_family, concept_id], ids=["prompt_scope", "literature"])
def test_the_onset_belongs_to_its_concepts_family(family):
    assert family("rrt_onset_time") == "rrt"


def _descriptor(name: str, transform: str | None, *, is_binary: bool, n_missing: int = 0) -> ConceptDescriptor:
    return ConceptDescriptor(
        name=name,
        dtype="float64",
        source_concept="rrt",
        unit_normalization=transform,
        temporal_resolution="relative to icu_admission in h" if transform == "first_truthy_event_time" else None,
        observed_domain={"is_binary": is_binary},
        missingness=MissingnessProfile(
            fraction_missing=n_missing / 4,
            n_missing=n_missing,
            n_total=4,
            missingness_severity="high" if n_missing else "low",
        ),
    )


def _window(*, maximum, measured, count, onset, first=None) -> tuple[pd.DataFrame, list[ConceptDescriptor]]:
    frame = pd.DataFrame(
        {
            **({"rrt_first": first} if first is not None else {}),
            "rrt_max": maximum, "rrt_measured": measured, "rrt_n": count, "rrt_onset_time": onset,
        }
    )
    descriptors = [
        *(
            [_descriptor("rrt_first", "window_presence_first", is_binary=True)]
            if first is not None
            else []
        ),
        _descriptor("rrt_max", "window_presence_max", is_binary=True, n_missing=int(frame["rrt_max"].isna().sum())),
        _descriptor("rrt_measured", "window_measurement_status", is_binary=True),
        _descriptor("rrt_n", "window_nonnull_count", is_binary=False),
        _descriptor("rrt_onset_time", "first_truthy_event_time", is_binary=False, n_missing=int(frame["rrt_onset_time"].isna().sum())),
    ]
    return frame, descriptors


def _onset(frame, descriptors) -> ConceptDescriptor:
    compiled = compile_observation_semantics(frame=frame, descriptors=descriptors)
    return next(item for item in compiled if item.name == "rrt_onset_time")


def test_a_recorded_absence_is_not_a_missing_onset():
    # The source records absence and two stays have no record, so only the
    # coverage flag is complete -- and it is not the event.
    frame, descriptors = _window(
        maximum=[1.0, 1.0, 0.0, np.nan], measured=[1, 1, 1, 0], count=[3, 1, 2, 0],
        onset=[3.0, 2.0, np.nan, np.nan],
    )

    assert _onset(frame, descriptors).observation_semantics is None


def test_a_source_that_records_only_presence_is_timed_where_it_was_recorded():
    frame, descriptors = _window(
        maximum=[1.0, 1.0, np.nan, np.nan], measured=[1, 1, 0, 0], count=[2, 1, 0, 0],
        onset=[3.0, 2.0, np.nan, np.nan],
    )

    onset = _onset(frame, descriptors)
    assert onset.observation_semantics is not None
    assert onset.observation_semantics.event_status_column == "rrt_measured"
    assert onset.missingness.not_applicable_n == 2
    assert onset.missingness.n_missing == 0


def test_the_onset_is_described_by_presence_not_by_a_coincident_first_record():
    # Every stay recorded and none first recorded absent then present: the
    # first recorded status matches presence on this frame, but only presence
    # is what the onset is conditional on.
    frame, descriptors = _window(
        first=[1.0, 0.0, 0.0, 1.0], maximum=[1.0, 0.0, 0.0, 1.0], measured=[1, 1, 1, 1],
        count=[2, 1, 1, 3], onset=[3.0, np.nan, np.nan, 0.5],
    )

    assert _onset(frame, descriptors).observation_semantics.event_status_column == "rrt_max"
