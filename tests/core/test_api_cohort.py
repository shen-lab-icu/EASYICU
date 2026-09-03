"""Cohort-domain contracts for the public EasyICU API."""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu import api


def test_unknown_database_does_not_inherit_miiv_id_contract() -> None:
    with pytest.raises(ValueError, match="Unsupported database"):
        api.get_id_col_for_database("unknown-icu")
    with pytest.raises(ValueError, match="Unsupported database"):
        api.get_patient_table_for_database("unknown-icu")


def test_patient_id_read_failure_is_not_reported_as_an_empty_cohort(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "icustays.parquet").touch()
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("private path")),
    )

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
