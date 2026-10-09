"""An event's onset can be read over a window of its own.

A target trial summarizes its covariates over the hours before time zero and
times the start of its treatment through the grace period after it.  The
materializer reads a typed event status's ``<c>_onset_time`` over the window a
design names and keeps every other summary over the cohort window: a stay that
starts after the cohort window but inside the onset window keeps its onset,
the column's metadata states the window it was read over, the receipt records
it, and the research context neither reads the onset over the cohort window
nor pairs it with a status read there.  A trial whose onset is read only over
the cohort window cannot see its grace period; it names the extraction it
needs, and compiles on it.

Synthetic export rows only (two vasoactive indicators and a lactate); no
patient data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.acquisition.foundation import acquire_universe_for_question
from easyicu.research_agent.cohort.materializer import materialize_cohort
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedMetadataError,
    load_verified_materialized_cohort_authority,
)
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.planning.target_trial_compile import compile_target_trial
from easyicu.research_agent.planning.target_trial_spec import TargetTrialSpec
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.research_context.materialization_window import (
    column_materialized_window,
)
from easyicu.research_agent.research_context.observation_semantics import (
    compile_observation_semantics,
)
from easyicu.research_agent.schema import ConceptDescriptor, MissingnessProfile
from tests.support.native_outcome_export import native_outcome, typed_native_export

_TIME_ZERO = 6.0
_GRACE_END = 30.0
_TREATMENTS = ("vaso_ind", "other_vaso")
_TRIAL_WINDOWS = {concept: (0.0, _GRACE_END) for concept in _TREATMENTS}

# Stay 1 starts at hour 3, inside the cohort window; stay 2 at hour 10, inside
# the grace period; stay 3 at hour 40, after both; stay 4 is recorded only at
# hour 6, the first hour after the cohort window ends.
_ROWS = (
    (1, 0.0, False),
    (1, 3.0, True),
    (2, 1.0, False),
    (2, 10.0, True),
    (3, 2.0, False),
    (3, 40.0, True),
    (4, 6.0, True),
)


def _export(tmp_path: Path) -> Path:
    root = tmp_path / "export"
    if root.exists():
        return root
    medications = pd.DataFrame(
        {
            "stay_id": [row[0] for row in _ROWS],
            "charttime": [row[1] for row in _ROWS],
            "vaso_ind": pd.array([row[2] for row in _ROWS], dtype="boolean"),
            # The other class members are never started.
            "other_vaso": pd.array([False] * len(_ROWS), dtype="boolean"),
            "lact": [1.5, 2.5, 1.0, 4.0, 2.0, 3.0, 5.0],
        }
    )
    return typed_native_export(
        root,
        outcome=native_outcome(
            death=[False, True, False, True],
            death_time=[None, 200.0, None, 90.0],
            los_icu=[3.0, 9.0, 4.0, 2.0],
            followup_days_28d=[28.0, 200.0 / 24, 28.0, 90.0 / 24],
            mort_28d=[False, True, False, True],
        ),
        outcome_concepts=["death", "los_icu", "followup_days_28d", "mort_28d"],
        longitudinal=medications,
        longitudinal_concepts=[*_TREATMENTS, "lact"],
    )


def _acquire(
    tmp_path: Path,
    *,
    window: tuple[float, float] = (0.0, _TIME_ZERO),
    onset_windows=None,
    name: str = "universe",
):
    result = acquire_universe_for_question(
        export_dir=_export(tmp_path),
        question="Does an early start of a vasoactive drug change death by day 28?",
        llm=ScriptedMockLLMClient([]),
        output_dir=tmp_path / name,
        target_outcome="mort_28d",
        outcome_concepts=["mort_28d"],
        required_feature_concepts=[*_TREATMENTS, "lact"],
        static_concepts=["age"],
        concept_selection_authority="host_exact",
        cohort_window=window,
        emit_trajectory=False,
        event_onset_windows=onset_windows,
    )
    assert result.blocked is False and result.universe_path is not None
    return result


def _column(frame: pd.DataFrame, name: str) -> dict[int, float | None]:
    return {
        int(stay): (None if pd.isna(value) else float(value))
        for stay, value in frame.set_index("stay_id")[name].items()
    }


def _sidecar_columns(verified) -> dict:
    return {
        name: binding
        for file in verified.sidecar.files
        for name, binding in file.columns.items()
    }


def _context(result):
    return build_research_context(
        research_question="Does an early start of a vasoactive drug change death?",
        cohort=result.universe_path,
        cohort_name="trial",
        database="miiv",
        target_outcome="mort_28d",
        id_columns=("stay_id",),
        outcome_columns=("mort_28d",),
    )


def test_a_start_after_the_cohort_window_keeps_its_onset(tmp_path: Path) -> None:
    cohort = pd.read_parquet(
        _acquire(tmp_path, onset_windows=_TRIAL_WINDOWS).universe_path
    )

    # Stay 2 starts in the grace period and stay 4 at its first hour; stay 3
    # starts after it.
    assert _column(cohort, "vaso_ind_onset_time") == {1: 3.0, 2: 10.0, 3: None, 4: 6.0}
    # Every other summary is still read over the cohort window [0, 6) h.
    assert _column(cohort, "vaso_ind_max") == {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0}
    assert _column(cohort, "vaso_ind_n") == {1: 2.0, 2: 1.0, 3: 1.0, 4: 0.0}
    assert _column(cohort, "lact_max") == {1: 2.5, 2: 1.0, 3: 2.0, 4: None}
    assert _column(cohort, "other_vaso_onset_time") == dict.fromkeys(range(1, 5))


def test_over_the_cohort_window_alone_the_grace_period_is_unseen(
    tmp_path: Path,
) -> None:
    cohort = pd.read_parquet(_acquire(tmp_path).universe_path)

    assert _column(cohort, "vaso_ind_onset_time") == {1: 3.0, 2: None, 3: None, 4: None}


def test_the_onset_states_the_window_it_was_read_over(tmp_path: Path) -> None:
    result = _acquire(tmp_path, onset_windows=_TRIAL_WINDOWS)

    verified = load_verified_materialized_cohort_authority(Path(result.universe_path))
    columns = _sidecar_columns(verified)
    onset = columns["vaso_ind_onset_time"]
    assert onset.representation_transform == "first_truthy_event_time"
    assert (onset.derivation_window.start_hours, onset.derivation_window.end_hours) == (
        0.0,
        _GRACE_END,
    )
    status = columns["vaso_ind_max"].derivation_window
    assert (status.start_hours, status.end_hours) == (0.0, _TIME_ZERO)
    recorded = {concept: [0.0, _GRACE_END] for concept in _TREATMENTS}
    sealed = verified.authority.producer_parameters["event_onset_windows"]
    assert {concept: list(window) for concept, window in sealed.items()} == recorded
    provenance = json.loads(Path(result.provenance_path).read_text(encoding="utf-8"))
    assert provenance["event_onset_windows"] == recorded

    context = _context(result)
    window = column_materialized_window(context, "vaso_ind_onset_time")
    assert (window.anchor, window.start_hours, window.end_hours) == (
        "icu_admission",
        0.0,
        _GRACE_END,
    )
    covariate = column_materialized_window(context, "lact_max")
    assert (covariate.start_hours, covariate.end_hours) == (0.0, _TIME_ZERO)
    # The cohort-window status is not what the onset is applicable on.
    assert context.variable("vaso_ind_onset_time").observation_semantics is None


def test_without_an_onset_window_the_receipt_names_none(tmp_path: Path) -> None:
    result = _acquire(tmp_path)

    verified = load_verified_materialized_cohort_authority(Path(result.universe_path))
    assert "event_onset_windows" not in verified.authority.producer_parameters
    provenance = json.loads(Path(result.provenance_path).read_text(encoding="utf-8"))
    assert "event_onset_windows" not in provenance
    columns = _sidecar_columns(verified)
    assert (
        columns["vaso_ind_onset_time"].derivation_window
        == columns["vaso_ind_max"].derivation_window
    )


def _spec() -> TargetTrialSpec:
    return TargetTrialSpec.model_validate(
        {
            "treatment": {
                "quote": "a vasoactive drug",
                "source": "question",
                "concepts": list(_TREATMENTS),
                "treatment_class": "vasoactive",
            },
            "strategies": {
                "quote": "start it early or not",
                "source": "question",
                "initiate_label": "Early start",
                "defer_label": "No early start",
            },
            "time_zero": {
                "quote": "six hours after ICU admission",
                "source": "question",
                "hours_after_icu_admission": int(_TIME_ZERO),
            },
            "grace_period": {
                "quote": "within a day",
                "source": "question",
                "hours": int(_GRACE_END - _TIME_ZERO),
            },
            "outcome": {
                "quote": "death by day 28",
                "source": "question",
                "endpoint": "mort_28d",
            },
            "confounders": [
                {
                    "name": "age",
                    "source": "question",
                    "clinical_rationale": "Older patients are started later and die more often.",
                },
                {
                    "name": "lact_max",
                    "source": "conversation",
                    "clinical_rationale": "A higher lactate prompts the start and predicts death.",
                },
            ],
        }
    )


def _compiled(result):
    context = _context(result)
    population = compile_population(
        PopulationSpec.model_validate(
            {
                "criteria": [
                    {
                        "id": "c1",
                        "source": "question",
                        "role": "include",
                        "quote": "adults",
                        "kind": "age_years",
                        "min_years": 18,
                    }
                ]
            }
        ),
        context,
        time_zero_hours=int(_TIME_ZERO),
    )
    return compile_target_trial(_spec(), context, population=population)


def test_a_trial_compiles_over_the_two_windows(tmp_path: Path) -> None:
    trial = _compiled(_acquire(tmp_path, onset_windows=_TRIAL_WINDOWS))

    dispositions = {item.element: item.disposition for item in trial.elements}
    assert dispositions["treatment"] == "applied"
    assert dispositions["grace_period"] == "applied"
    assert dispositions["time_zero"] == "applied"
    assert [(item.name, item.disposition) for item in trial.confounders] == [
        ("age", "applied"),
        ("lact_max", "applied"),
    ]
    assert trial.record()["materialization"]["treatment_onset_window"] == {
        "start_hours": 0,
        "end_hours": int(_GRACE_END),
    }


def test_a_trial_names_the_extraction_it_compiles_on(tmp_path: Path) -> None:
    # One window for everything, as an extraction for another design reads it.
    first = _compiled(_acquire(tmp_path, window=(0.0, 24.0), name="one_window"))

    grace = first.element("grace_period")
    assert (grace.disposition, grace.reason) == (
        "requires_extraction",
        "tte_grace_beyond_capture",
    )
    assert [(item.name, item.reason) for item in first.confounders_waiting] == [
        ("lact_max", "tte_confounder_window_after_time_zero")
    ]
    assert first.acquisition_windows() == ((0.0, _TIME_ZERO), _TRIAL_WINDOWS)

    cohort_window, onset_windows = first.acquisition_windows()
    second = _compiled(
        _acquire(
            tmp_path, window=cohort_window, onset_windows=onset_windows, name="trial"
        )
    )
    assert second.element("grace_period").disposition == "applied"
    assert second.confounders_waiting == ()


@pytest.mark.parametrize(
    ("windows", "message"),
    [
        ({"lact": (0.0, 30.0)}, "applies only to a typed event status"),
        ({"mech_vent": (0.0, 30.0)}, "names a concept that is not a feature"),
        ({"vaso_ind": (30.0, 0.0)}, "not a finite, increasing span"),
        ({"vaso_ind": (0.0, math.inf)}, "not a finite, increasing span"),
        ({"vaso_ind": (0.0,)}, "not a pair of hours"),
        ({"vaso_ind": ("start", "end")}, "not a pair of hours"),
    ],
)
def test_an_onset_window_it_cannot_honour_is_refused(
    tmp_path: Path, windows, message
) -> None:
    with pytest.raises(MaterializedMetadataError, match=message):
        materialize_cohort(
            feature_concepts=[*_TREATMENTS, "lact"],
            database="miiv",
            data_path=str(_export(tmp_path)),
            cohort_window=(0.0, _TIME_ZERO),
            outcome_concepts=["mort_28d"],
            static_concepts=["age"],
            event_onset_windows=windows,
        )


def _descriptor(name: str, transform: str, window: str, **domain) -> ConceptDescriptor:
    return ConceptDescriptor(
        name=name,
        dtype="float64",
        source_concept="rrt",
        unit_normalization=transform,
        temporal_resolution=(
            "relative to icu_admission in h"
            if transform == "first_truthy_event_time"
            else None
        ),
        analysis_window=window,
        observed_domain=domain,
        missingness=MissingnessProfile(
            fraction_missing=0.0, n_missing=0, n_total=3, missingness_severity="low"
        ),
    )


def _semantics(onset_window: str, bindings=None):
    # On this frame every onset lies inside the status's window, so the two
    # agree whatever window either was read over.
    frame = pd.DataFrame({"rrt_max": [1, 0, 1], "rrt_onset_time": [2.0, np.nan, 4.0]})
    descriptors = [
        _descriptor(
            "rrt_max", "window_presence_max", "icu_admission[0,6]h", is_binary=True
        ),
        _descriptor("rrt_onset_time", "first_truthy_event_time", onset_window),
    ]
    compiled = compile_observation_semantics(
        frame=frame, descriptors=descriptors, event_time_bindings=bindings
    )
    return next(item for item in compiled if item.name == "rrt_onset_time")


def test_a_status_read_over_another_window_does_not_time_the_onset() -> None:
    same = _semantics("icu_admission[0,6]h").observation_semantics
    assert same is not None and same.event_status_column == "rrt_max"

    assert _semantics("icu_admission[0,30]h").observation_semantics is None
    with pytest.raises(ValueError, match="read over another window than its status"):
        _semantics("icu_admission[0,30]h", bindings={"rrt_onset_time": "rrt_max"})
