"""A MIMIC export keeps each patient's first ICU stay within the database's record.

B20: a first-stay cohort (``exclude_readmissions`` or the ``adult_first``
preset) asked PatientFilter for a generic first-stay column, which it never
publishes for MIMIC (no absolute patient ICU ordinal), so every MIMIC
extraction with a first-stay cohort failed with
``patient_filter_criterion_unavailable``. Planning and the run already apply
the host's declared recipe (``raw_source_authority``: icustays ordered by
intime within subject_id). Data Extraction now applies that same recipe over
the source's whole stay table and records the rule in its cohort report; a
missing table or a tied time still fails closed with a typed reason, and a
database without the recipe keeps PatientFilter's own answer. Synthetic
tables only.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.patient_filter import PatientFilter, PatientFilterCriterionError
from easyicu.webserver import dataio, raw_source_authority

_FIRST = {"preset": "all_icu", "age_min": 18, "exclude_readmissions": True}


def _api() -> SimpleNamespace:
    return SimpleNamespace(
        get_id_col_for_database=lambda database: "stay_id",
        get_all_patient_ids=lambda *args, **kwargs: ([], "stay_id"),
    )


def _source(tmp_path: Path, rows, *, stay_column: str = "stay_id") -> Path:
    root = tmp_path / "raw"
    root.mkdir()
    pd.DataFrame(rows, columns=[stay_column, "subject_id", "intime"]).assign(
        intime=lambda frame: pd.to_datetime(frame["intime"])
    ).to_parquet(root / "icustays.parquet", index=False)
    return root


_STAYS = [
    # Patient 1: two stays, 101 first. Patient 2: one stay. Patient 3: 302 first.
    (101, 1, "2150-01-01 08:00"),
    (102, 1, "2150-03-01 08:00"),
    (201, 2, "2151-05-01 10:00"),
    (301, 3, "2152-07-09 12:00"),
    (302, 3, "2152-02-01 12:00"),
]


@pytest.fixture
def demographics(monkeypatch: pytest.MonkeyPatch):
    """Every stay is an adult; the frame publishes no first-stay column (MIMIC)."""
    calls: list[dict] = []
    frame = pd.DataFrame({"patient_id": [101, 102, 201, 301, 302], "age": [60.0] * 5})
    original = PatientFilter.filter

    def spy(self, **kwargs):
        calls.append(dict(kwargs))
        return original(self, **kwargs)

    monkeypatch.setattr(PatientFilter, "_load_demographics", lambda self: frame)
    monkeypatch.setattr(PatientFilter, "filter", spy)
    return calls


@pytest.mark.parametrize(
    ("database", "stay_column"), [("miiv", "stay_id"), ("mimic_demo", "icustay_id")]
)
def test_a_mimic_first_stay_cohort_extracts_by_the_host_recipe(
    tmp_path, demographics, database, stay_column
) -> None:
    root = _source(tmp_path, _STAYS, stay_column=stay_column)

    resolved = dataio._resolve_export_cohort(str(root), database, _FIRST, None, _api())

    assert sorted(resolved["patient_ids"]["stay_id"]) == [101, 201, 302]
    assert resolved["cohort_contract"]["exclude_readmissions"] is True
    # PatientFilter is not asked for a first-stay column it cannot publish.
    assert demographics[0]["first_icu_stay"] is None
    report = resolved["cohort_report"]
    assert "first_icu_stay" in report["applied_filters"]
    first = report["first_icu_stay"]
    assert first["rule"] == "earliest_order_time_per_patient"
    assert (
        first["label"]["zh"] == "数据库记录中的首次 ICU 入住（MIMIC 无院外/更早入住史）"
    )
    assert first["scope"] == "source_global"
    assert first["identity_table"] == "icustays"
    assert (first["selected_before"], first["selected"], first["excluded"]) == (5, 3, 2)
    assert first["identifier_values_returned"] is False
    assert str(root) not in repr(report)


def test_the_adult_first_preset_uses_the_same_recipe(tmp_path, demographics) -> None:
    root = _source(tmp_path, _STAYS)

    resolved = dataio._resolve_export_cohort(
        str(root), "miiv", {"preset": "adult_first", "age_min": 18}, None, _api()
    )

    assert sorted(resolved["patient_ids"]["stay_id"]) == [101, 201, 302]


def test_a_tied_first_admission_time_fails_closed(tmp_path, demographics) -> None:
    root = _source(
        tmp_path, [*_STAYS, (401, 4, "2153-01-01 00:00"), (402, 4, "2153-01-01 00:00")]
    )

    with pytest.raises(dataio.ExportCohortError) as caught:
        dataio._resolve_export_cohort(str(root), "miiv", _FIRST, None, _api())

    assert caught.value.error == "export_first_icu_stay_unavailable"
    assert (
        caught.value.detail["cause_code"]
        == "raw_source_authority_first_icu_stay_invalid"
    )


def test_a_source_without_its_stay_table_fails_closed(tmp_path, demographics) -> None:
    root = tmp_path / "raw"
    root.mkdir()

    with pytest.raises(dataio.ExportCohortError) as caught:
        dataio._resolve_export_cohort(str(root), "miiv", _FIRST, None, _api())

    assert caught.value.error == "export_first_icu_stay_unavailable"


@pytest.mark.parametrize("database", ["eicu", "hirid"])
def test_a_database_without_the_recipe_keeps_patient_filters_answer(
    tmp_path, demographics, database
) -> None:
    assert not raw_source_authority.extraction_applies_first_stay_recipe(database)

    with pytest.raises(PatientFilterCriterionError) as caught:
        dataio._resolve_export_cohort(str(tmp_path), database, _FIRST, None, _api())

    assert caught.value.code == "patient_filter_criterion_unavailable"
    assert demographics[0]["first_icu_stay"] is True


def test_only_mimic_takes_the_recipe_and_its_own_ordinal_stays_elsewhere() -> None:
    applies = raw_source_authority.extraction_applies_first_stay_recipe
    assert [
        db
        for db in (
            "miiv",
            "mimic",
            "mimic_demo",
            "aumc",
            "sic",
            "eicu",
            "eicu_demo",
            "hirid",
        )
        if applies(db)
    ] == [
        "miiv",
        "mimic",
        "mimic_demo",
    ]
