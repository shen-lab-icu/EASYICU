"""Cohort-domain contracts for the public EasyICU API."""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu import api, datasource
from easyicu.patient_filter import PatientFilter
from easyicu.storage_layout import ORDINAL, TableScanError


def test_unknown_database_does_not_inherit_miiv_id_contract() -> None:
    with pytest.raises(ValueError, match="Unsupported database"):
        api.get_id_col_for_database("unknown-icu")
    with pytest.raises(ValueError, match="Unsupported database"):
        api.get_patient_table_for_database("unknown-icu")


def test_patient_id_read_failure_is_not_reported_as_an_empty_cohort(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No stay-table file: discovery falls to the loader, whose failure must
    # surface without its message. (A present but unreadable stay table is a
    # typed refusal before this point: test_d17c_strict_csv_and_stay_discovery.)
    class FailingLoader:
        def __init__(self, **_kwargs):
            raise PermissionError("patient 123")

    monkeypatch.setattr(api, "BaseICULoader", FailingLoader)

    with pytest.raises(api.PatientIdDiscoveryError) as caught:
        api.get_all_patient_ids(tmp_path, database="miiv")

    message = str(caught.value)
    assert "PermissionError" in message
    assert "123" not in message
    assert str(tmp_path) not in message


def test_patient_id_error_is_exported_at_package_root() -> None:
    import easyicu

    assert easyicu.PatientIdDiscoveryError is api.PatientIdDiscoveryError


@pytest.fixture
def stay_fixture(tmp_path_factory, monkeypatch):
    """Use a configured file alias so the on-disk name is synthetic."""
    root = tmp_path_factory.mktemp("fixture")
    config = datasource.registered_source_config("mimic").model_copy(deep=True)
    monkeypatch.setattr(datasource, "registered_source_config", lambda _: config)

    def configure(filename, *, table="icustays"):
        config.tables[table].files = [{"path": filename}]
        return root / filename

    return configure


@pytest.mark.parametrize("column", ["ICUSTAY_ID", "icustay_id"])
@pytest.mark.parametrize(
    ("filename", "source_format"),
    [("rows.parquet", "parquet"), ("rows.csv.gz", "csv_fallback")],
)
def test_mimic_stay_projection(stay_fixture, column, filename, source_format):
    path = stay_fixture(filename)
    rows = pd.DataFrame({column: [23, 23, 7, 11]})
    if source_format == "parquet":
        rows.to_parquet(path, index=False)
    else:
        rows.to_csv(path, index=False)

    listing = {}
    ids, id_col = api.get_all_patient_ids(
        path.parent, database="mimic", max_patients=2, listing=listing,
    )

    assert ids == [23, 7]
    assert id_col == "icustay_id"
    assert listing == {"order": "source_file_order", "source_format": source_format}


@pytest.mark.parametrize("filename", ["rows.parquet", "rows.csv"])
def test_mimic_stay_projection_refuses_two_physical_id_columns(stay_fixture, filename):
    path = stay_fixture(filename)
    rows = pd.DataFrame({"icustay_id": [23], "ICUSTAY_ID": [7]})
    if path.suffix == ".parquet":
        rows.to_parquet(path, index=False)
    else:
        rows.to_csv(path, index=False)

    listing = {}
    with pytest.raises(TableScanError) as error:
        api.get_all_patient_ids(path.parent, database="mimic", listing=listing)

    assert error.value.reason == "identity_column_ambiguous"
    assert error.value.table == "icustays"
    assert listing == {}


def test_mimic_shards_bind_spelling_per_schema_and_restore_source_order(stay_fixture):
    root = stay_fixture("parts")
    root.mkdir()
    pd.DataFrame({"ICUSTAY_ID": [23, 11], ORDINAL: [0, 3]}).to_parquet(root / "a.pq", index=False)
    pd.DataFrame({"icustay_id": [7, 23], ORDINAL: [1, 2]}).to_parquet(root / "b.pq", index=False)

    assert api.get_all_patient_ids(root.parent, database="mimic") == ([23, 7, 11], "icustay_id")


def test_mimic_demographic_cohort_binds_uppercase_keys(stay_fixture):
    stays = stay_fixture("rows.parquet")
    people = stay_fixture("people.parquet", table="patients")
    visits = stay_fixture("visits.parquet", table="admissions")
    pd.DataFrame({
        "ICUSTAY_ID": [23, 7], "SUBJECT_ID": [101, 102], "HADM_ID": [1, 2], "LOS": [2, 1],
    }).to_parquet(stays, index=False)
    pd.DataFrame({"SUBJECT_ID": [101, 102]}).to_parquet(people, index=False)
    pd.DataFrame({"SUBJECT_ID": [101, 102], "HADM_ID": [1, 2]}).to_parquet(visits, index=False)

    selection = PatientFilter(database="mimic", data_path=stays.parent).filter(los_min=30)

    assert selection == [23]
