"""An outcome-module concept is read at stay level wherever it is requested.

A native export publishes its outcome module as stay-level rows at one
coordinate, 0 h from ICU admission: no event happened and nothing was measured
there.  An outcome was already read at stay level, timed by the event time the
export issues beside it.  Requested as a feature or a cohort predicate, the same
concept was summarized over a window by that coordinate:

- a window that held 0 h made a death at any time a death within the window,
  so an early-death exclusion removed every death;
- a window that left 0 h out found no row: a feature lost its columns, and a
  predicate's cohort could not be built;
- the summaries claimed an observation at 0 h (``*_first_time``), and the
  stay's own value (``followup_days_28d``) never appeared under its name.

The concept is now read as the stay-level fact it is wherever it is requested:
an event status as its whole-stay status with its issued time, a value as the
stay's value.  The cohort builder applies a predicate's window to the event by
its issued time.  Synthetic exports only.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import easyicu.research_agent.acquisition.foundation as foundation
from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
from easyicu.research_agent.acquisition.foundation import acquire_universe_for_question
from easyicu.research_agent.cohort import materializer as M
from easyicu.research_agent.cohort.materializer import (
    materialize_cohort,
    materialize_to_parquet,
)
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedMetadataError,
)
from easyicu.research_agent.planning.adjustment_authority import host_window_bound_roles
from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.research_context.builder import build_research_context
from tests.support.native_outcome_export import (
    native_outcome,
    typed_native_export,
    untyped_native_export,
)

#: Deaths at 87 h (stay 1) and 12 h (stay 3); stays 2 and 4 survive.
_OUTCOME = dict(
    death=[True, False, True, False],
    death_time=[87.0, None, 12.0, None],
    persistent_critical_illness=[False, True, False, False],
    los_icu=[4.0, 1.5, 0.6, 2.0],
    followup_days_28d=[87.0 / 24, 28.0, 0.5, 28.0],
    mort_28d=[True, False, True, False],
)
_STAY_LEVEL = ["death", "persistent_critical_illness", "los_icu", "followup_days_28d"]


def _export(tmp_path: Path) -> Path:
    medications = pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 3],
            "charttime": [9.0, 30.0, 4.0, 30.0],
            "abx": [True, True, False, True],
        }
    )
    return typed_native_export(
        tmp_path / "export",
        outcome=native_outcome(**_OUTCOME),
        outcome_concepts=[*_STAY_LEVEL, "mort_28d"],
        longitudinal=medications,
        longitudinal_concepts=["abx"],
    )


def _materialize(root: Path, *, window=(0.0, 24.0), features=(), definition=None):
    return materialize_cohort(
        feature_concepts=list(features),
        database="miiv",
        data_path=str(root),
        cohort_window=window,
        outcome_concepts=["mort_28d"],
        static_concepts=["age"],
        cohort_definition=definition,
    )


def _column(cohort: pd.DataFrame, name: str) -> dict[int, float | None]:
    return {
        int(stay): (None if pd.isna(value) else float(value))
        for stay, value in cohort.set_index("stay_id")[name].items()
    }


def _predicate(
    concept: str, op: str, value, start: float, end: float, *, anchor="icu_admit"
) -> ConceptPredicate:
    return ConceptPredicate(
        concept_id=concept,
        time_window=TimeWindow(
            anchor=anchor, start_offset_hours=start, end_offset_hours=end
        ),
        aggregation="max",
        op=op,
        value=value,
    )


def _kept(root: Path, *, inclusion=(), exclusion=()) -> list[int]:
    cohort, _ = _materialize(
        root,
        definition=CohortDefinition(
            name="primary", inclusion=tuple(inclusion), exclusion=tuple(exclusion)
        ),
    )
    return sorted(int(stay) for stay in cohort["stay_id"])


@pytest.mark.parametrize("window", [(0.0, 24.0), (24.0, 48.0)])
def test_a_value_feature_is_the_stays_value_whatever_the_window(
    tmp_path: Path, window
) -> None:
    cohort, provenance = _materialize(
        _export(tmp_path), window=window, features=["los_icu", "followup_days_28d"]
    )

    # Before: summaries over the 0 h row, and none at all when 0 h fell outside.
    assert _column(cohort, "los_icu") == {1: 4.0, 2: 1.5, 3: 0.6, 4: 2.0}
    assert _column(cohort, "followup_days_28d") == {
        1: 87.0 / 24,
        2: 28.0,
        3: 0.5,
        4: 28.0,
    }
    assert provenance["stay_level_concepts"] == {
        "los_icu": {"source": "stay_level_value"},
        "followup_days_28d": {"source": "stay_level_value"},
    }


def test_no_stay_level_column_claims_an_observation_in_the_window(
    tmp_path: Path,
) -> None:
    cohort, _ = _materialize(_export(tmp_path), features=_STAY_LEVEL)

    summaries = [
        column
        for column in cohort.columns
        for concept in _STAY_LEVEL
        if column.startswith(f"{concept}_") and column != "death_time"
    ]
    assert summaries == []


def test_an_event_feature_is_its_whole_stay_status_with_its_issued_time(
    tmp_path: Path,
) -> None:
    cohort, provenance = _materialize(
        _export(tmp_path), window=(24.0, 48.0), features=["death"]
    )

    # Not "a death within 24-48 h": the status says the stay died, the time when.
    assert _column(cohort, "death") == {1: 1, 2: 0, 3: 1, 4: 0}
    assert _column(cohort, "death_time") == {1: 87.0, 2: None, 3: 12.0, 4: None}
    assert provenance["stay_level_concepts"] == {
        "death": {"source": "issued_event_time", "column": "death_time"}
    }


def test_an_event_without_an_issued_time_is_untimed(tmp_path: Path) -> None:
    cohort, provenance = _materialize(
        _export(tmp_path), features=["persistent_critical_illness"]
    )

    assert _column(cohort, "persistent_critical_illness") == {1: 0, 2: 1, 3: 0, 4: 0}
    assert "persistent_critical_illness_time" not in cohort.columns
    assert provenance["stay_level_concepts"] == {
        "persistent_critical_illness": {"source": "untimed_stay_level_event"}
    }


def test_an_outcome_requested_as_a_feature_too_is_read_once(tmp_path: Path) -> None:
    cohort, _ = materialize_cohort(
        feature_concepts=["death"],
        database="miiv",
        data_path=str(_export(tmp_path)),
        cohort_window=(0.0, 24.0),
        outcome_concepts=["death"],
        static_concepts=["age"],
    )

    assert [column for column in cohort.columns if column.startswith("death")] == [
        "death",
        "death_time",
    ]


def test_an_untyped_export_reads_its_declared_event_at_stay_level(
    tmp_path: Path,
) -> None:
    root = untyped_native_export(
        tmp_path / "export",
        outcome=native_outcome(
            death=[True, False, True, False],
            los_icu=[4.0, 1.5, 0.6, 2.0],
            mort_28d=[True, False, True, False],
        ),
        outcome_concepts=["death", "los_icu", "mort_28d"],
        unrecorded_stays=(5,),
    )

    # Before: no summary column of the declared event fell in the window.
    cohort, provenance = materialize_cohort(
        feature_concepts=["death", "los_icu"],
        database="miiv",
        data_path=str(root),
        cohort_window=(24.0, 48.0),
        outcome_concepts=["mort_28d"],
        static_concepts=["age"],
        positive_only_event_concepts=["death"],
    )

    # A positive-only event the module does not record is absent, as an
    # outcome's is.
    assert cohort.set_index("stay_id")["death"].to_dict() == {
        1: 1,
        2: 0,
        3: 1,
        4: 0,
        5: 0,
    }
    assert str(cohort["death"].dtype) == "int64"
    assert _column(cohort, "los_icu") == {1: 4.0, 2: 1.5, 3: 0.6, 4: 2.0, 5: None}
    assert provenance["stay_level_concepts"] == {
        "death": {"source": "untimed_stay_level_event"},
        "los_icu": {"source": "stay_level_value"},
    }


def test_a_longitudinal_feature_is_still_summarized_over_its_window(
    tmp_path: Path,
) -> None:
    cohort, provenance = _materialize(
        _export(tmp_path), window=(24.0, 48.0), features=["abx"]
    )

    # A control: rows that record their own time keep their window.
    assert _column(cohort, "abx_max") == {1: 1, 2: 0, 3: 1, 4: 0}
    assert provenance["stay_level_concepts"] == {}


def test_an_early_death_exclusion_removes_only_the_deaths_within_it(
    tmp_path: Path,
) -> None:
    kept = _kept(_export(tmp_path), exclusion=[_predicate("death", "==", 1, 0, 24)])

    # Before: the 0 h row made the death at 87 h a death within 24 h.
    assert kept == [1, 2, 4]


def test_a_later_window_reads_the_deaths_within_it(tmp_path: Path) -> None:
    # Before: no row fell in the window, and the cohort could not be built.
    assert _kept(
        _export(tmp_path), exclusion=[_predicate("death", "==", 1, 72, 96)]
    ) == [2, 3, 4]


def test_two_windows_on_one_stay_level_event_are_each_applied(
    tmp_path: Path,
) -> None:
    # Before: a typed export refused two derivations of one predicate column.
    assert _kept(
        _export(tmp_path),
        exclusion=[
            _predicate("death", "==", 1, 0, 24),
            _predicate("death", "==", 1, 72, 96),
        ],
    ) == [2, 4]


def test_a_value_predicate_reads_the_stays_value_whatever_its_window(
    tmp_path: Path,
) -> None:
    # Before: a window leaving 0 h out read no value, and the cohort could not
    # be built.
    assert _kept(
        _export(tmp_path), inclusion=[_predicate("los_icu", ">=", 1, 24, 48)]
    ) == [1, 2, 4]


def test_a_stay_level_feature_its_predicate_also_reads_is_read_once(
    tmp_path: Path,
) -> None:
    cohort, _ = _materialize(
        _export(tmp_path),
        features=["death"],
        definition=CohortDefinition(
            name="primary", exclusion=(_predicate("death", "==", 1, 0, 24),)
        ),
    )

    assert sorted(int(stay) for stay in cohort["stay_id"]) == [1, 2, 4]
    assert [column for column in cohort.columns if column.startswith("death")] == [
        "death",
        "death_time",
    ]


def test_a_value_predicate_has_no_time_axis_to_anchor(tmp_path: Path) -> None:
    # A stay's value is no reading at a time, so no anchor can misplace it.
    kept = _kept(
        _export(tmp_path),
        inclusion=[_predicate("los_icu", ">=", 1, 0, 24, anchor="hospital_admit")],
    )

    assert kept == [1, 2, 4]


def test_a_stay_level_event_predicate_from_another_anchor_is_refused(
    tmp_path: Path,
) -> None:
    with pytest.raises(MaterializedMetadataError, match="anchored at"):
        _kept(
            _export(tmp_path),
            exclusion=[_predicate("death", "==", 1, 0, 24, anchor="hospital_admit")],
        )


def _package(kind: str, **modules: str) -> SimpleNamespace:
    return SimpleNamespace(
        manifest_kind=kind,
        concept_index={
            concept: {"module": module, "file": f"{module}.parquet"}
            for concept, module in modules.items()
        },
    )


def test_only_a_native_outcome_module_is_stay_level() -> None:
    native = _package("native", death="outcome", abx="medications")
    legacy = _package("legacy", death="outcome")

    assert M._export_concept_is_stay_level(native, "death") is True
    assert M._export_concept_is_stay_level(native, "abx") is False
    assert M._export_concept_is_stay_level(native, "lact") is False
    # A legacy export keeps the row times of its outcome rows (a control).
    assert M._export_concept_is_stay_level(legacy, "death") is False


def test_the_acquisition_binds_a_stay_level_event_feature_to_its_column(
    tmp_path: Path,
) -> None:
    result = acquire_universe_for_question(
        export_dir=_export(tmp_path),
        question="Is an early death associated with 28-day mortality?",
        llm=ScriptedMockLLMClient([]),
        output_dir=tmp_path / "universe",
        target_outcome="mort_28d",
        outcome_concepts=["mort_28d"],
        required_feature_concepts=["death"],
        static_concepts=["age"],
        concept_selection_authority="host_exact",
        emit_trajectory=False,
    )

    assert result.blocked is False
    # Before: only ``death_max`` was a public coordinate; now the bare status is.
    assert result.analysis_columns["death"] == "death"
    assert {"death", "death_time"} <= set(result.materialized_columns)


@pytest.mark.parametrize("read_at_stay_level", [True, False])
def test_only_a_stay_level_reading_binds_a_bare_event_column(
    monkeypatch, tmp_path: Path, read_at_stay_level: bool
) -> None:
    catalog = AvailableCatalog(
        source="typed",
        concepts=[
            CatalogConcept(
                concept_id=concept,
                file_name="outcome.parquet",
                column_role="event_status",
                typed_metadata=True,
            )
            for concept in ("persistent_critical_illness", "mort_28d")
        ]
        + [CatalogConcept(concept_id="age", typed_metadata=True)],
    )
    monkeypatch.setattr(foundation, "build_available_catalog", lambda _root: catalog)

    def materialize(**_kwargs):
        provenance = tmp_path / "universe_provenance.json"
        provenance.write_text(
            json.dumps(
                {
                    "columns": ["stay_id", "age", "persistent_critical_illness"],
                    "stay_level_concepts": (
                        {"persistent_critical_illness": {"source": "x"}}
                        if read_at_stay_level
                        else {}
                    ),
                }
            ),
            encoding="utf-8",
        )
        parquet = tmp_path / "universe.parquet"
        parquet.write_bytes(b"placeholder")
        return {"parquet": str(parquet), "provenance": str(provenance)}

    monkeypatch.setattr(M, "materialize_to_parquet", materialize)

    result = acquire_universe_for_question(
        export_dir=tmp_path,
        question="q",
        llm=ScriptedMockLLMClient([]),
        output_dir=tmp_path,
        target_outcome="mort_28d",
        outcome_concepts=["mort_28d"],
        required_feature_concepts=["persistent_critical_illness"],
        static_concepts=["age"],
        concept_selection_authority="host_exact",
        emit_trajectory=False,
    )

    # A bare column the materializer did not read at stay level stays unbound,
    # as before (a control).
    assert ("persistent_critical_illness" in result.analysis_columns) is (
        read_at_stay_level
    )


def test_a_stay_level_column_is_not_proven_known_at_baseline(tmp_path: Path) -> None:
    paths = materialize_to_parquet(
        tmp_path / "universe",
        feature_concepts=["los_icu", "followup_days_28d", "death"],
        database="miiv",
        data_path=str(_export(tmp_path)),
        cohort_window=(0.0, 24.0),
        outcome_concepts=["mort_28d"],
        static_concepts=["age"],
    )
    context = build_research_context(
        research_question="Is the length of stay associated with 28-day mortality?",
        cohort=paths["parquet"],
        cohort_name="stay_level",
        database="miiv",
        target_outcome="mort_28d",
    )

    timing = host_window_bound_roles(context, reference_hours=24.0)

    # A stay-level reading is known only when the stay is over: never a baseline.
    assert not {"los_icu", "followup_days_28d", "death", "death_time"} & set(timing)
    assert timing["age"] == "baseline_static"
