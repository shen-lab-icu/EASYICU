"""An outcome read from an export is timed by the time the export issues for it.

A native export publishes its outcome module as stay-level rows at one
coordinate, 0 h from ICU admission, and issues an event's own time beside it as
a typed event-time companion (``death_time`` beside ``death``).  The cohort
materializer read only the event status and the row coordinate, so every death
in such an export was timed at ICU admission: an "alive at 24 hours" guard
removed every death, and a time-to-death analysis placed every event at 0 h.

These tests build small synthetic typed exports and fix the rule from both
sides: the issued time is the event's time, the stay-level coordinate never is,
and a longitudinal concept keeps the time its first recorded row carries.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept.export_metadata import build_export_file_metadata_binding
from easyicu.concept.metadata_projection import ConceptColumnRole
from easyicu.concept.metadata_sidecar import (
    EXPORT_PHYSICAL_SCOPE,
    ColumnMetadataFileBinding,
    ColumnMetadataSidecar,
    write_content_addressed_sidecar,
)
from easyicu.resources import load_dictionary
from easyicu.research_agent.cohort import materializer as M
from easyicu.research_agent.cohort.materializer import (
    materialize_cohort,
    materialize_to_parquet,
)
from easyicu.research_agent.intake import export_package as intake
from easyicu.research_agent.intake.export_package import open_export_package
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedMetadataError,
    load_verified_materialized_cohort_authority,
)


def _binding(
    *, relative_path: str, module: str, frame: pd.DataFrame, concepts: list[str]
) -> ColumnMetadataFileBinding:
    """Type one physical file with the module exporter's own binder."""

    return build_export_file_metadata_binding(
        relative_path=relative_path,
        module=module,
        frame=frame,
        concept_ids=concepts,
        database="miiv",
        database_class_prefixes=(),
        dictionary=load_dictionary(include_sofa2=True),
    )


def _typed_export(
    root: Path,
    *,
    outcome: pd.DataFrame,
    outcome_concepts: list[str],
    longitudinal: pd.DataFrame | None = None,
    longitudinal_concepts: list[str] | None = None,
) -> Path:
    root.mkdir()
    stays = sorted(int(value) for value in outcome["stay_id"].unique())
    statics = pd.DataFrame(
        {"stay_id": stays, "age": [50 + 5 * index for index in range(len(stays))]}
    )
    members: list[tuple[str, str, pd.DataFrame, list[str]]] = [
        ("demographics.parquet", "demographics", statics, ["age"]),
        ("outcome.parquet", "outcome", outcome, list(outcome_concepts)),
    ]
    if longitudinal is not None:
        members.append(
            (
                "medications.parquet",
                "medications",
                longitudinal,
                list(longitudinal_concepts or ()),
            )
        )
    bindings = []
    files = []
    for relative_path, module, frame, concepts in members:
        frame.to_parquet(root / relative_path, index=False)
        binding = _binding(
            relative_path=relative_path, module=module, frame=frame, concepts=concepts
        )
        bindings.append(binding)
        files.append(
            {
                "file": relative_path,
                "module": module,
                "concepts": len(concepts),
                "concept_ids": concepts,
                "rows": len(frame),
                "column_metadata_columns": list(binding.columns),
            }
        )
    sidecar = ColumnMetadataSidecar(
        source_database="miiv",
        source_database_class_prefixes=(),
        scope=EXPORT_PHYSICAL_SCOPE,
        files=tuple(bindings),
    )
    reference = write_content_addressed_sidecar(root, sidecar)
    (root / intake.NATIVE_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": intake.NATIVE_MANIFEST_SCHEMA_V2,
                "database": "miiv",
                "format": "parquet",
                "concept_selection": {
                    "mode": "explicit",
                    "modules": {item["module"]: item["concept_ids"] for item in files},
                },
                "files": files,
                "feature_definitions": {"included": False},
                "column_metadata": reference.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    return root


def _native_outcome(**columns: list) -> pd.DataFrame:
    """Stay-level outcome rows exactly as the native publisher writes them."""

    stays = [1, 2, 3, 4]
    frame = pd.DataFrame({"stay_id": stays, "charttime": [0.0] * len(stays)})
    for name, values in columns.items():
        frame[name] = values
    return frame


def _materialize(root: Path, outcomes: list[str]):
    return materialize_cohort(
        feature_concepts=[],
        database="miiv",
        data_path=str(root),
        cohort_window=(0.0, 24.0),
        outcome_concepts=outcomes,
        static_concepts=["age"],
    )


def _times(cohort: pd.DataFrame, column: str) -> dict[int, float | None]:
    return {
        int(stay): (None if pd.isna(value) else float(value))
        for stay, value in cohort.set_index("stay_id")[column].items()
    }


def test_a_death_is_timed_by_the_time_the_export_issues(tmp_path: Path) -> None:
    root = _typed_export(
        tmp_path / "export",
        outcome=_native_outcome(
            death=[True, False, True, False],
            death_time=[87.0, None, 12.0, None],
        ),
        outcome_concepts=["death"],
    )

    cohort, _ = _materialize(root, ["death"])

    assert cohort.set_index("stay_id")["death"].to_dict() == {1: 1, 2: 0, 3: 1, 4: 0}
    # Not 0 h: the stay-level coordinate is when the stay began, not when it died.
    assert _times(cohort, "death_time") == {1: 87.0, 2: None, 3: 12.0, 4: None}


def test_an_alive_at_24_hours_guard_keeps_the_death_after_it(tmp_path: Path) -> None:
    """The consequence the defect hid: every death looked like an early death."""

    root = _typed_export(
        tmp_path / "export",
        outcome=_native_outcome(
            death=[True, False, True, False],
            death_time=[87.0, None, 12.0, None],
        ),
        outcome_concepts=["death"],
    )

    cohort, _ = _materialize(root, ["death"])

    alive_at_24h = cohort[~(cohort["death_time"] <= 24.0)]
    assert sorted(alive_at_24h["stay_id"]) == [1, 2, 4]
    assert int(alive_at_24h["death"].sum()) == 1


def test_a_death_the_export_left_untimed_stays_untimed(tmp_path: Path) -> None:
    # A source that records no time of death (the export issues the companion
    # empty) leaves the death untimed: neither 0 h nor any other coordinate.
    root = _typed_export(
        tmp_path / "export",
        outcome=_native_outcome(
            death=[True, False, True, False],
            death_time=[None, None, 12.0, None],
        ),
        outcome_concepts=["death"],
    )

    cohort, _ = _materialize(root, ["death"])

    assert _times(cohort, "death_time") == {1: None, 2: None, 3: 12.0, 4: None}


def test_a_time_beside_a_survivor_times_nothing(tmp_path: Path) -> None:
    root = _typed_export(
        tmp_path / "export",
        outcome=_native_outcome(
            death=[True, False, False, False],
            death_time=[30.0, 5.0, None, None],
        ),
        outcome_concepts=["death"],
    )

    cohort, _ = _materialize(root, ["death"])

    assert _times(cohort, "death_time") == {1: 30.0, 2: None, 3: None, 4: None}


def test_a_stay_level_outcome_without_an_issued_time_is_untimed(
    tmp_path: Path,
) -> None:
    # The row coordinate here is whatever the stay-level row carries -- an ICU
    # discharge offset, for one source -- and it is not when the death happened.
    outcome = _native_outcome(death=[True, False, True, False])
    outcome["charttime"] = [40.0, 40.0, 70.0, 10.0]
    root = _typed_export(
        tmp_path / "export", outcome=outcome, outcome_concepts=["death"]
    )

    cohort, provenance = _materialize(root, ["death"])

    assert cohort.set_index("stay_id")["death"].to_dict() == {1: 1, 2: 0, 3: 1, 4: 0}
    assert "death_time" not in cohort.columns
    assert provenance["outcome_event_time_sources"] == {
        "death": {"source": "untimed_stay_level_outcome"}
    }


def test_another_outcome_module_event_is_untimed_without_its_own_time(
    tmp_path: Path,
) -> None:
    root = _typed_export(
        tmp_path / "export",
        outcome=_native_outcome(
            death=[True, False, False, False],
            death_time=[30.0, None, None, None],
            persistent_critical_illness=[False, True, True, False],
        ),
        outcome_concepts=["death", "persistent_critical_illness"],
    )

    cohort, provenance = _materialize(root, ["death", "persistent_critical_illness"])

    assert cohort.set_index("stay_id")["persistent_critical_illness"].to_dict() == {
        1: 0,
        2: 1,
        3: 1,
        4: 0,
    }
    assert "persistent_critical_illness_time" not in cohort.columns
    assert _times(cohort, "death_time") == {1: 30.0, 2: None, 3: None, 4: None}
    assert provenance["outcome_event_time_sources"]["persistent_critical_illness"] == {
        "source": "untimed_stay_level_outcome"
    }


def test_an_export_with_no_row_time_still_times_its_death(tmp_path: Path) -> None:
    # A stay-level outcome file may carry no time coordinate at all; the issued
    # time is still the event's time.
    outcome = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "death": [True, False, True, False],
            "death_time": [87.0, None, 12.0, None],
        }
    )
    root = _typed_export(
        tmp_path / "export", outcome=outcome, outcome_concepts=["death"]
    )

    cohort, provenance = _materialize(root, ["death"])

    assert _times(cohort, "death_time") == {1: 87.0, 2: None, 3: 12.0, 4: None}
    assert provenance["outcome_event_time_sources"] == {
        "death": {"source": "issued_event_time", "column": "death_time"}
    }


def test_a_longitudinal_event_keeps_the_time_its_first_row_recorded(
    tmp_path: Path,
) -> None:
    medications = pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 3],
            "charttime": [9.0, 3.0, 4.0, 30.0],
            "abx": [True, True, False, True],
        }
    )
    root = _typed_export(
        tmp_path / "export",
        outcome=_native_outcome(
            death=[False, False, False, False],
            death_time=[None, None, None, None],
        ),
        outcome_concepts=["death"],
        longitudinal=medications,
        longitudinal_concepts=["abx"],
    )

    cohort, provenance = _materialize(root, ["abx"])

    assert _times(cohort, "abx_time") == {1: 3.0, 2: None, 3: 30.0, 4: None}
    assert provenance["outcome_event_time_sources"] == {
        "abx": {"source": "first_recorded_event_row"}
    }


def test_the_sealed_cohort_types_the_issued_time_as_the_event_time(
    tmp_path: Path,
) -> None:
    root = _typed_export(
        tmp_path / "export",
        outcome=_native_outcome(
            death=[True, False, True, False],
            death_time=[87.0, None, 12.0, None],
        ),
        outcome_concepts=["death"],
    )
    out = tmp_path / "materialized"
    out.mkdir()

    with open_export_package(root) as package:
        paths = materialize_to_parquet(
            out,
            stem="cohort",
            source_package=package,
            feature_concepts=[],
            database="miiv",
            data_path=str(root),
            cohort_window=(0.0, 24.0),
            outcome_concepts=["death"],
            static_concepts=["age"],
        )

    verified = load_verified_materialized_cohort_authority(Path(paths["parquet"]))
    assert verified is not None
    binding = verified.sidecar.files[0].columns["death_time"]
    assert binding.metadata.role is ConceptColumnRole.EVENT_TIME
    assert binding.metadata.source_concept == "death"
    assert (binding.metadata.time_origin, binding.metadata.time_unit) == (
        "icu_admission",
        "h",
    )
    sealed = pd.read_parquet(paths["parquet"])
    assert _times(sealed, "death_time") == {1: 87.0, 2: None, 3: 12.0, 4: None}


def test_an_issued_time_is_read_only_beside_the_rows_of_its_event() -> None:
    frame = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 3],
            "charttime": [0.0, 0.0, 0.0, 0.0],
            "death": [1, 0, 1, 1],
            "death_time": [87.0, 5.0, None, 12.0],
        }
    )

    out = M._event_time_column(
        frame,
        "death",
        source_role=ConceptColumnRole.EVENT_STATUS,
        row_time_is_event_time=False,
    )

    times = out.set_index("stay_id")["death_time"]
    assert times.loc[1] == 87.0
    assert math.isnan(times.loc[2])
    assert times.loc[3] == 12.0


def test_an_untimed_row_coordinate_times_nothing() -> None:
    frame = pd.DataFrame({"stay_id": [1, 2], "charttime": [40.0, 0.0], "death": [1, 0]})

    assert M._event_time_column(frame, "death", row_time_is_event_time=False).empty
    # The converted-database default is unchanged: an indexed event row is
    # timed by its index.
    timed = M._event_time_column(frame, "death")
    assert timed.set_index("stay_id")["death_time"].to_dict() == {1: 40.0}


def test_an_issued_time_that_cannot_be_read_as_a_number_is_refused() -> None:
    frame = pd.DataFrame({"stay_id": [1], "death": [1], "death_time": ["late"]})

    with pytest.raises(MaterializedMetadataError, match="lossy numeric coercion"):
        M._event_time_column(frame, "death")


def _declared(kind: str, **entries: dict) -> SimpleNamespace:
    """An export's declarations only: its manifest kind and concept index."""

    return SimpleNamespace(manifest_kind=kind, concept_index=entries)


def _typed_entry(role: str, *, file: str = "outcome.parquet", **extra) -> dict:
    return {
        "column_metadata_v2": True,
        "source_concept": "death",
        "column_metadata_role": role,
        "file": file,
        "module": "outcome",
        **extra,
    }


def _issued(*, origin: str = "icu_admission", unit: str = "h", **extra) -> dict:
    metadata = SimpleNamespace(time_origin=origin, time_unit=unit)
    return _typed_entry(
        "event_time",
        column_metadata_binding=SimpleNamespace(metadata=metadata),
        **extra,
    )


def test_an_issued_time_must_be_in_hours_from_icu_admission() -> None:
    for origin, unit in (("icu_admission", "min"), ("hospital_admission", "h")):
        package = _declared(
            "native",
            death=_typed_entry("event_status"),
            death_time=_issued(origin=origin, unit=unit),
        )
        with pytest.raises(MaterializedMetadataError, match="hours from ICU admission"):
            M._export_event_time_source(package, "death")


def test_two_issued_times_for_one_event_are_refused() -> None:
    package = _declared(
        "native",
        death=_typed_entry("event_status"),
        death_time=_issued(),
        death_time_recorded=_issued(),
    )

    with pytest.raises(MaterializedMetadataError, match="more than one event time"):
        M._export_event_time_source(package, "death")


def test_a_time_issued_in_another_file_does_not_time_the_event() -> None:
    package = _declared(
        "native",
        death=_typed_entry("event_status"),
        death_time=_issued(file="followup.parquet"),
    )

    assert M._export_event_time_source(package, "death") == (None, False)


def test_a_legacy_export_keeps_the_row_times_of_its_outcome_rows() -> None:
    # A legacy export predates the native stay-level contract: its outcome rows
    # sit at the time the loader indexed them.
    package = _declared(
        "legacy", death={"file": "outcome.parquet", "module": "outcome"}
    )

    assert M._export_event_time_source(package, "death") == (None, True)
