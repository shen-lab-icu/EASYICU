"""Official patient identity for MIMIC-III, SICdb and AmsterdamUMCdb; none for HiRID.

These databases carry an official patient identifier in one raw table the
export's sealed ``data_path`` names (MIMIC-III ``icustays.subject_id``, SICdb
``cases.PatientID``, AmsterdamUMCdb ``admissions.patientid``), but have no
hospital follow-up profile.  The host binds that table by digest through the
same owner as eICU and MIMIC-IV, derives the private stay-to-patient bridge
and, where an official column orders a patient's ICU stays, the first-stay
coordinate.  HiRID gives every ICU admission a new ``patientid``, so its stays
stay ungrouped.  All tables here are synthetic.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.first_icu_stay import (
    COORDINATE_FLAG_COLUMN,
    COORDINATE_STAY_COLUMN,
    load_verified_first_icu_stay,
)
from easyicu.research_agent.acquisition.patient_grouping import (
    load_verified_patient_grouping,
)
from easyicu.webserver import raw_source_authority, source_identity_authority
from easyicu.webserver.raw_source_authority import (
    RawSourceAuthorityError,
    resolve_raw_hospital_source_binding,
    resolve_raw_source_identity_binding,
)


@pytest.fixture(autouse=True)
def _private_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        source_identity_authority.state_paths, "state_root", lambda: tmp_path / "state"
    )
    monkeypatch.setattr(raw_source_authority, "_DEFAULT_CONFIG_PATH", tmp_path / "absent.env")
    monkeypatch.setattr(
        source_identity_authority, "_DEFAULT_CONFIG_PATH", tmp_path / "absent-grouping.env"
    )


def _export(tmp_path: Path, database: str, tables: dict[str, pd.DataFrame]) -> Path:
    """A modern export whose manifest names its sealed raw root."""

    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_parquet(raw / f"{name}.parquet", index=False)
    export = tmp_path / "export"
    export.mkdir()
    (export / "_manifest.json").write_text(
        json.dumps({"database": database, "data_path": str(raw), "files": []}),
        encoding="utf-8",
    )
    return export


def _mimic_iii_icustays(*, upper: bool = True, intimes: list[str] | None = None) -> pd.DataFrame:
    """Stays 201 and 202 share subject 9; 202 was admitted first."""

    frame = pd.DataFrame(
        {
            "row_id": [1, 2, 3],
            "subject_id": [9, 9, 4],
            "hadm_id": [91, 92, 41],
            "icustay_id": [201, 202, 203],
            "intime": intimes or ["2150-03-01 10:00:00", "2150-01-01 10:00:00", "2149-06-01 10:00:00"],
            "outtime": ["2150-03-03 10:00:00", "2150-01-02 10:00:00", "2149-06-04 10:00:00"],
        }
    )
    # The PhysioNet CSV release, as converted, keeps upper-case headers.
    return frame.rename(columns=str.upper) if upper else frame


def _sicdb_cases(offsets: list[int] | None = None) -> pd.DataFrame:
    """Patient 70 has two cases; patient 72's first admission is not in the table."""

    return pd.DataFrame(
        {
            "CaseID": [301, 302, 303, 304],
            "PatientID": [70, 70, 71, 72],
            "AdmissionYear": [2015, 2016, 2017, 2018],
            "ICUOffset": [0, 600, 0, 0],
            "OffsetAfterFirstAdmission": offsets or [0, 86_400, 0, 3_600],
        }
    )


def _aumc_admissions(counts: list[int] | None = None) -> pd.DataFrame:
    """Admissions 401 and 402 share patient 1; 402 is the first (count 1)."""

    return pd.DataFrame(
        {
            "patientid": [1, 1, 2],
            "admissionid": [401, 402, 403],
            "admissioncount": counts or [2, 1, 1],
            "admittedat": [900_000, 0, 0],
            "dischargedat": [1_900_000, 500_000, 400_000],
            "location": ["IC", "IC", "MC"],
        }
    )


def _flags(binding) -> dict[int, bool]:
    frame = load_verified_first_icu_stay(
        binding.coordinate_path, expected_sha256=binding.coordinate_sha256
    )
    return frame.set_index(COORDINATE_STAY_COLUMN)[COORDINATE_FLAG_COLUMN].to_dict()


CASES = {
    "mimic": (
        lambda: {"icustays": _mimic_iii_icustays()},
        "icustays",
        "SUBJECT_ID",
        "INTIME",
        {201: False, 202: True, 203: True},
    ),
    "sic": (
        lambda: {"cases": _sicdb_cases()},
        "cases",
        "PatientID",
        "OffsetAfterFirstAdmission",
        {301: True, 302: False, 303: True, 304: False},
    ),
    "aumc": (
        lambda: {"admissions": _aumc_admissions()},
        "admissions",
        "patientid",
        "admissioncount",
        {401: False, 402: True, 403: True},
    ),
}


@pytest.mark.parametrize("database", sorted(CASES))
def test_each_database_groups_stays_by_its_official_patient_identifier(
    tmp_path: Path, database: str
) -> None:
    tables, table, patient_column, _order, flags = CASES[database]
    export = _export(tmp_path, database, tables())

    binding = source_identity_authority.resolve_study_patient_grouping(
        export_path=export, database=database
    )

    assert binding is not None
    assert stat.S_IMODE(os.stat(binding.mapping_path).st_mode) == 0o600
    coordinates = dict(binding.authority_coordinates)
    family = {"mimic": "mimic_iii", "sic": "sicdb", "aumc": "aumc"}[database]
    assert coordinates["authority_ref"] == f"export_manifest_data_path/{family}/1/patient_grouping"
    assert coordinates["identity_table"] == table
    assert coordinates["patient_identifier_column"] == patient_column
    assert coordinates["stays"] == len(flags)
    assert coordinates["patients_with_repeated_stays"] == 1
    assert coordinates["provider_visible_values"] is False
    grouped = load_verified_patient_grouping(binding).frame.set_axis(
        ["stay", "patient"], axis=1
    ).set_index("stay")["patient"]
    repeated = [stay for stay, first in flags.items() if not first][:1]
    partner = {201: 202, 302: 301, 401: 402}[repeated[0]]
    assert grouped.loc[repeated[0]] == grouped.loc[partner]
    assert grouped.nunique() == coordinates["patients"]


@pytest.mark.parametrize("database", sorted(set(CASES) - {"sic"}))
def test_each_database_proves_the_first_icu_stay_from_its_official_order(
    tmp_path: Path, database: str
) -> None:
    tables, table, _patient, order_column, flags = CASES[database]
    export = _export(tmp_path, database, tables())

    binding = source_identity_authority.resolve_study_first_icu_stay(
        export_path=export, database=database
    )

    assert binding is not None
    assert _flags(binding) == flags
    coordinates = dict(binding.authority_coordinates)
    assert coordinates["identity_table"] == table
    assert coordinates["order_column"] == order_column
    assert coordinates["non_first_icu_stays"] == sum(not first for first in flags.values())
    # It agrees with the grouping derived from the same verified table.
    grouping = source_identity_authority.resolve_study_patient_grouping(
        export_path=export, database=database
    )
    assert coordinates["patient_grouping_mapping_sha256"] == grouping.mapping_sha256
    if database == "mimic":
        assert coordinates["order_rule"] == "earliest_order_time_per_patient"
        assert "first_order_value" not in coordinates
    else:
        assert coordinates["order_rule"] == "declared_first_admission_value_per_patient"
        assert coordinates["first_order_value"] == 1
        assert coordinates["patients_without_first_icu_stay_in_source"] == 0
    # Receipts carry counts and digests, never identifier values or paths.
    serialized = json.dumps(coordinates)
    assert str(tmp_path / "raw") not in serialized


def test_mimic_iii_lower_case_columns_bind_the_same_identity(tmp_path: Path) -> None:
    export = _export(tmp_path, "mimic_demo", {"icustays": _mimic_iii_icustays(upper=False)})

    first = source_identity_authority.resolve_study_first_icu_stay(
        export_path=export, database="mimic_demo"
    )

    assert first is not None
    assert first.authority_coordinates["order_column"] == "intime"
    assert _flags(first) == {201: False, 202: True, 203: True}


@pytest.mark.parametrize("database", sorted(CASES))
def test_an_identity_only_binding_never_claims_hospital_follow_up(
    tmp_path: Path, database: str
) -> None:
    tables, table, _patient, _order, _flags_ = CASES[database]
    export = _export(tmp_path, database, tables())

    # The hospital follow-up owner still refuses these databases.
    with pytest.raises(RawSourceAuthorityError) as excinfo:
        resolve_raw_hospital_source_binding(export_path=export, database=database)
    assert excinfo.value.code == "raw_source_authority_database_unsupported"

    binding = resolve_raw_source_identity_binding(export_path=export, database=database)
    assert binding is not None
    receipt = binding.public_receipt()
    assert "hospital_mortality_followup" not in receipt
    assert set(receipt["raw_table_sha256"]) == {table}
    assert receipt["patient_grouping"]["identity_table"] == table
    assert str(tmp_path) not in json.dumps(receipt)
    with pytest.raises(RawSourceAuthorityError) as refused:
        binding.materialize_hospital_mortality_followup()
    assert refused.value.code == "raw_source_authority_database_unsupported"
    assert binding.first_icu_stay_per_hospitalization_available is False


def test_hirid_stays_are_never_grouped(tmp_path: Path) -> None:
    """HiRID's ``patientid`` names one ICU admission; it is not a patient identity."""

    export = _export(
        tmp_path,
        "hirid",
        {
            "general": pd.DataFrame(
                {
                    "patientid": [1, 2],
                    "admissiontime": ["2150-01-01 08:00:00", "2150-02-01 08:00:00"],
                    "sex": ["M", "F"],
                    "age": [60, 70],
                    "discharge_status": ["alive", "dead"],
                }
            )
        },
    )

    assert resolve_raw_source_identity_binding(export_path=export, database="hirid") is None
    assert (
        source_identity_authority.resolve_study_patient_grouping(
            export_path=export, database="hirid"
        )
        is None
    )
    assert (
        source_identity_authority.resolve_study_first_icu_stay(
            export_path=export, database="hirid"
        )
        is None
    )
    assert (
        source_identity_authority.resolve_study_first_icu_stay_per_hospitalization(
            export_path=export, database="hirid"
        )
        is None
    )


@pytest.mark.parametrize(
    ("database", "tables"),
    [
        ("mimic", {"icustays": _mimic_iii_icustays().assign(ICUSTAY_ID=[201, 201, 203])}),
        ("sic", {"cases": _sicdb_cases().assign(CaseID=[301, 302, 302, 304])}),
        ("aumc", {"admissions": _aumc_admissions().assign(admissionid=[401, 401, 403])}),
    ],
)
def test_a_repeated_stay_identifier_refuses_the_grouping(
    tmp_path: Path, database: str, tables: dict[str, pd.DataFrame]
) -> None:
    export = _export(tmp_path, database, tables)

    with pytest.raises(source_identity_authority.PatientGroupingAuthorityError) as excinfo:
        source_identity_authority.resolve_study_patient_grouping(
            export_path=export, database=database
        )

    assert excinfo.value.code == "raw_source_authority_patient_grouping_invalid"
    assert not (tmp_path / "state" / "private" / "patient-grouping").exists() or not any(
        (tmp_path / "state" / "private" / "patient-grouping").iterdir()
    )


@pytest.mark.parametrize(
    ("database", "tables"),
    [
        (
            "mimic",
            {
                "icustays": _mimic_iii_icustays(
                    intimes=["2150-01-01 10:00:00", "2150-01-01 10:00:00", "2149-06-01 10:00:00"]
                )
            },
        ),
        ("aumc", {"admissions": _aumc_admissions(counts=[1, 1, 1])}),
    ],
)
def test_a_tied_first_admission_refuses_the_first_stay(
    tmp_path: Path, database: str, tables: dict[str, pd.DataFrame]
) -> None:
    export = _export(tmp_path, database, tables)

    with pytest.raises(source_identity_authority.PatientGroupingAuthorityError) as excinfo:
        source_identity_authority.resolve_study_first_icu_stay(
            export_path=export, database=database
        )

    assert excinfo.value.code == "raw_source_authority_first_icu_stay_invalid"
    assert excinfo.value.details["cause_code"] == "first_icu_stay_order_tied"
    assert not (tmp_path / "state" / "private" / "first-icu-stay").exists() or not any(
        (tmp_path / "state" / "private" / "first-icu-stay").iterdir()
    )


def test_an_identity_table_without_its_patient_column_is_not_inferred(tmp_path: Path) -> None:
    export = _export(tmp_path, "sic", {"cases": _sicdb_cases().drop(columns=["PatientID"])})

    assert (
        source_identity_authority.resolve_study_patient_grouping(
            export_path=export, database="sic"
        )
        is None
    )
    assert (
        source_identity_authority.resolve_study_first_icu_stay(
            export_path=export, database="sic"
        )
        is None
    )


def test_a_broken_identity_table_fails_closed(tmp_path: Path) -> None:
    missing_stay = _export(
        tmp_path / "a", "aumc", {"admissions": _aumc_admissions().drop(columns=["admissionid"])}
    )
    with pytest.raises(source_identity_authority.PatientGroupingAuthorityError) as excinfo:
        source_identity_authority.resolve_study_patient_grouping(
            export_path=missing_stay, database="aumc"
        )
    assert excinfo.value.code == "raw_source_authority_table_schema_mismatch"
    assert excinfo.value.details["missing_columns"] == ["admissionid"]

    # Two columns that differ only in case could each be the identity.
    ambiguous = _export(
        tmp_path / "b",
        "mimic",
        {"icustays": _mimic_iii_icustays().assign(Subject_Id=[1, 2, 3])},
    )
    with pytest.raises(source_identity_authority.PatientGroupingAuthorityError) as excinfo:
        source_identity_authority.resolve_study_patient_grouping(
            export_path=ambiguous, database="mimic"
        )
    assert excinfo.value.code == "raw_source_authority_identity_column_ambiguous"

    absent = tmp_path / "c"
    absent.mkdir()
    export = _export(absent, "sic", {"other": pd.DataFrame({"x": [1]})})
    with pytest.raises(source_identity_authority.PatientGroupingAuthorityError) as excinfo:
        source_identity_authority.resolve_study_patient_grouping(
            export_path=export, database="sic"
        )
    assert excinfo.value.code == "raw_source_authority_table_unavailable"


@pytest.mark.parametrize(
    ("database", "expected"),
    [("mimic", ("bound", None)), ("sic", ("bound", None)), ("aumc", ("bound", None)),
     ("hirid", ("source_has_none", None))],
)
def test_planning_reads_the_same_grouping_status(
    tmp_path: Path, database: str, expected: tuple
) -> None:
    from easyicu.webserver import run_patient_grouping

    tables = (
        CASES[database][0]()
        if database in CASES
        else {"general": pd.DataFrame({"patientid": [1], "admissiontime": ["2150-01-01"]})}
    )
    export = _export(tmp_path, database, tables)
    study = {
        "data_source": {"path": str(export), "database": database},
        "analysis_design": {"analysis_family": "association_study", "analysis_unit": "icu_stay"},
    }

    grouping, status = run_patient_grouping.planning_patient_grouping(study, bound=None)

    assert status == expected
    assert (grouping is not None) is (expected[0] == "bound")


@pytest.mark.parametrize("database", ["mimic", "sic", "aumc", "hirid"])
def test_a_quick_answer_offers_first_stays_only_where_they_can_be_prepared(
    tmp_path: Path, database: str
) -> None:
    from easyicu.webserver.quick_answer.preparation import _first_stays_orderable

    tables = (
        CASES[database][0]()
        if database in CASES
        else {"general": pd.DataFrame({"patientid": [1], "admissiontime": ["2150-01-01"]})}
    )
    export = _export(tmp_path, database, tables)

    assert _first_stays_orderable(str(export), database) is (database not in {"hirid", "sic"})


def test_sicdb_groups_stays_by_patient_without_claiming_a_first_stay(tmp_path: Path) -> None:
    """A shared zero offset marks a first admission, not a first ICU stay."""

    export = _export(tmp_path, "sic", {"cases": _sicdb_cases(offsets=[0, 0, 0, 3_600])})

    assert source_identity_authority.resolve_study_patient_grouping(
        export_path=export, database="sic"
    ) is not None
    assert source_identity_authority.resolve_study_first_icu_stay(
        export_path=export, database="sic"
    ) is None
