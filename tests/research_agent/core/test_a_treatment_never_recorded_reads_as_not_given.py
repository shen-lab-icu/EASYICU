"""A treatment recorded only when given reads as not given where it has no record.

Characterization of the host owners a question about vasopressor use rests
on.  A typed export declares the vasopressor status ``vaso_ind`` an event
status recorded only as present.  The wide materializer summarizes it over the
analysis window as a status of every stay -- present where a record says so,
absent where the window holds none -- and keeps whether the window held a
record in its own column.  The research context declares the status a
positive-only event, so a share of treated stays is over every stay, and the
onset of treatment applies only where the treatment was given.

Synthetic export rows only; no patient data.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.research_context.builder import build_research_context
from tests.support.typed_export import typed_export

# Stay 1 is given the treatment twice in the window and stay 2 once; stays 3
# and 4 have rows (other concepts recorded) that carry no treatment status;
# stay 5 is given it only after the window ends.
_ROWS = (
    (1, 1.0, True),
    (1, 2.0, True),
    (2, 3.0, True),
    (3, 1.0, None),
    (4, 2.0, None),
    (5, 30.0, True),
)


def _materialize(tmp_path: Path) -> dict[str, Path]:
    labs = pd.DataFrame(
        {
            "stay_id": [row[0] for row in _ROWS],
            "charttime": [row[1] for row in _ROWS],
            "age": [60] * len(_ROWS),
            "lact": [1.5] * len(_ROWS),
            "mech_vent": pd.array([False] * len(_ROWS), dtype="boolean"),
            "vaso_ind": pd.array([row[2] for row in _ROWS], dtype="boolean"),
        }
    )
    source = typed_export(
        tmp_path / "export",
        labs=labs,
        outcomes=pd.DataFrame(
            {"stay_id": [1, 2, 3, 4, 5], "death": [False, True, False, False, True]}
        ),
        event_concepts=("vaso_ind",),
    )
    return cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        stem="universe",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact", "vaso_ind"),
        outcome_concepts=("death",),
        cohort_window=(0.0, 24.0),
    )


def test_a_stay_without_a_record_in_the_window_is_not_given(tmp_path):
    paths = _materialize(tmp_path)
    cohort = pd.read_parquet(paths["parquet"]).set_index("stay_id").sort_index()

    assert cohort["vaso_ind_max"].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert cohort["vaso_ind_n"].tolist() == [2, 1, 0, 0, 0]
    assert cohort["vaso_ind_measured"].tolist() == [1, 1, 0, 0, 0]
    # The share of treated stays is over every stay, not over recorded stays.
    assert cohort["vaso_ind_max"].mean() == 2 / 5
    onset = cohort["vaso_ind_onset_time"].tolist()
    assert onset[:2] == [1.0, 3.0]
    assert all(math.isnan(value) for value in onset[2:])


def test_the_context_reads_the_status_as_a_positive_only_event(tmp_path):
    paths = _materialize(tmp_path)

    context = build_research_context(
        research_question="What share of ICU stays received a vasopressor in the first 24 hours?",
        cohort=paths["parquet"],
        cohort_name="treated_share",
        database="miiv",
        target_outcome="death",
        primary_exposure="vaso_ind_max",
        id_columns=("stay_id",),
        outcome_columns=("death",),
    )

    status = context.variable("vaso_ind_max")
    assert status.observation_semantics is not None
    assert status.observation_semantics.kind == "positive_only_event"
    assert status.missingness.n_missing == 0

    onset = context.variable("vaso_ind_onset_time")
    assert onset.observation_semantics is not None
    assert onset.observation_semantics.kind == "conditional_event_time"
    assert onset.observation_semantics.event_status_column == "vaso_ind_max"
    assert onset.missingness.not_applicable_n == 3
    assert onset.missingness.n_missing == 0
