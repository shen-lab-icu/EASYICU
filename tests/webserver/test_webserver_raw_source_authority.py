"""Contracts for the registered-export/raw-source binding owner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.webserver import dataio, raw_source_authority
from easyicu.webserver.raw_source_authority import (
    RawHospitalSourceBinding,
    RawSourceAuthorityError,
    resolve_manifest_raw_source_binding,
    resolve_raw_hospital_source_binding,
    resolve_raw_mimic_iv_source_binding,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authority_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    export_root = tmp_path / "export"
    export_root.mkdir()
    manifest = export_root / "easyicu_export_manifest.json"
    manifest.write_text('{"database":"miiv"}', encoding="utf-8")
    raw_root = tmp_path / "raw"
    (raw_root / "icu").mkdir(parents=True)
    (raw_root / "hosp").mkdir()
    icustays = raw_root / "icu" / "icustays.parquet"
    admissions = raw_root / "hosp" / "admissions.parquet"
    pd.DataFrame(
        {
            "stay_id": [11, 12],
            "hadm_id": [101, 102],
            "intime": ["2026-01-01", "2026-01-02"],
        }
    ).to_parquet(icustays, index=False)
    pd.DataFrame(
        {
            "hadm_id": [101, 102],
            "dischtime": ["2026-01-03", "2026-01-04"],
            "deathtime": [None, "2026-01-02"],
            "hospital_expire_flag": [0, 1],
        }
    ).to_parquet(admissions, index=False)
    env = {
        "EASYICU_RAW_SOURCE_EXPORT_ROOT": str(export_root),
        "EASYICU_RAW_SOURCE_EXPORT_MANIFEST": manifest.name,
        "EASYICU_RAW_SOURCE_EXPORT_MANIFEST_SHA256": _sha256(manifest),
        "EASYICU_RAW_SOURCE_DATABASE": "miiv",
        "EASYICU_RAW_SOURCE_SOURCE_ROOT": str(raw_root),
        "EASYICU_RAW_SOURCE_ICUSTAYS_FILE": "icu/icustays.parquet",
        "EASYICU_RAW_SOURCE_ICUSTAYS_SHA256": _sha256(icustays),
        "EASYICU_RAW_SOURCE_ADMISSIONS_FILE": "hosp/admissions.parquet",
        "EASYICU_RAW_SOURCE_ADMISSIONS_SHA256": _sha256(admissions),
        "EASYICU_RAW_SOURCE_AUTHORITY_REF": "owner/raw-mimiciv/v1",
    }
    return env, export_root


def test_raw_source_authority_absent_is_not_inferred(tmp_path: Path) -> None:
    assert (
        resolve_raw_mimic_iv_source_binding(
            export_path=tmp_path,
            database="miiv",
            environ={},
        )
        is None
    )


def test_raw_source_authority_binds_exact_export_and_tables(tmp_path: Path) -> None:
    env, export_root = _authority_environment(tmp_path)
    binding = resolve_raw_mimic_iv_source_binding(
        export_path=export_root,
        database="miiv",
        environ=env,
    )

    assert binding is not None
    assert binding.database == "miiv"
    receipt = binding.public_receipt()
    assert receipt["source_paths_returned"] is False
    assert str(tmp_path) not in str(receipt)
    followup = binding.materialize_hospital_mortality_followup()
    status = binding.materialize_hospital_mortality_status()
    assert status.frame.hospital_death.tolist() == [False, True]
    assert status.receipt["clock_required"] is False
    assert followup.frame["stay_id"].tolist() == [11, 12]
    assert followup.receipt["event_stays"] == 1


def test_raw_source_authority_never_cross_binds_another_export(tmp_path: Path) -> None:
    env, _ = _authority_environment(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    assert (
        resolve_raw_mimic_iv_source_binding(
            export_path=other,
            database="miiv",
            environ=env,
        )
        is None
    )


def test_raw_source_authority_digest_drift_fails_closed(tmp_path: Path) -> None:
    env, export_root = _authority_environment(tmp_path)
    env["EASYICU_RAW_SOURCE_ADMISSIONS_SHA256"] = "0" * 64
    with pytest.raises(RawSourceAuthorityError) as raised:
        resolve_raw_mimic_iv_source_binding(
            export_path=export_root,
            database="miiv",
            environ=env,
        )
    assert raised.value.code == "raw_source_authority_table_digest_mismatch"


def test_materialization_rechecks_the_bytes_after_binding(tmp_path: Path) -> None:
    env, export_root = _authority_environment(tmp_path)
    binding = resolve_raw_mimic_iv_source_binding(
        export_path=export_root, database="miiv", environ=env
    )
    assert binding is not None
    changed = pd.read_parquet(binding.admissions_path)
    changed["hospital_expire_flag"] = 0
    changed["deathtime"] = None
    changed.to_parquet(binding.admissions_path, index=False)

    with pytest.raises(RawSourceAuthorityError) as raised:
        binding.materialize_hospital_mortality_followup()

    assert raised.value.code == "raw_source_authority_table_digest_mismatch"


def test_materialization_does_not_follow_a_replaced_table_link(tmp_path: Path) -> None:
    env, export_root = _authority_environment(tmp_path)
    binding = resolve_raw_mimic_iv_source_binding(
        export_path=export_root, database="miiv", environ=env
    )
    assert binding is not None
    original = binding.admissions_path.with_name("original.parquet")
    binding.admissions_path.rename(original)
    binding.admissions_path.symlink_to(original)

    with pytest.raises(RawSourceAuthorityError) as raised:
        binding.materialize_hospital_mortality_followup()

    assert raised.value.code == "raw_source_authority_tables_unreadable"


def test_raw_source_authority_rejects_incomplete_table_schema(tmp_path: Path) -> None:
    env, export_root = _authority_environment(tmp_path)
    admissions = tmp_path / "raw" / "hosp" / "admissions.parquet"
    pd.DataFrame({"hadm_id": [101]}).to_parquet(admissions, index=False)
    env["EASYICU_RAW_SOURCE_ADMISSIONS_SHA256"] = _sha256(admissions)

    with pytest.raises(RawSourceAuthorityError) as raised:
        resolve_raw_mimic_iv_source_binding(
            export_path=export_root,
            database="miiv",
            environ=env,
        )
    assert raised.value.code == "raw_source_authority_table_schema_mismatch"


def test_raw_source_authority_partial_configuration_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RawSourceAuthorityError) as raised:
        resolve_raw_mimic_iv_source_binding(
            export_path=tmp_path,
            database="miiv",
            environ={"EASYICU_RAW_SOURCE_DATABASE": "miiv"},
        )
    assert raised.value.code == "raw_source_authority_incomplete"


def test_legacy_registered_export_uses_only_exact_raw_source_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, export_root = _authority_environment(tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    binding = dataio.resolve_registered_export_binding(str(export_root), "miiv")

    assert binding["source_data_path"] == str((tmp_path / "raw").resolve())
    receipt = binding["source_authority_receipt"]
    assert receipt is not None
    assert receipt["source_paths_returned"] is False
    assert str(tmp_path) not in str(receipt)


# --- modern exports: the manifest's sealed ``data_path`` binds the raw tables ---


def _modern_export(tmp_path: Path, *, database: str, manifest_database: str | None = None) -> Path:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    if database == "miiv":
        pd.DataFrame(
            {"stay_id": [11, 12], "hadm_id": [101, 102], "intime": ["2026-01-01", "2026-01-02"]}
        ).to_parquet(raw_root / "icustays.parquet", index=False)
        pd.DataFrame(
            {
                "hadm_id": [101, 102],
                "dischtime": ["2026-01-03", "2026-01-04"],
                "deathtime": [None, "2026-01-02 12:00:00"],
                "hospital_expire_flag": [0, 1],
            }
        ).to_parquet(raw_root / "admissions.parquet", index=False)
    else:
        pd.DataFrame(
            {
                "patientunitstayid": [201, 202, 203],
                "hospitaldischargestatus": ["Alive", "Expired", None],
                "hospitaldischargeoffset": [2880, 600, 100],
                "unitdischargeoffset": [2000, 660, 90],
                "gender": ["Female", "Male", "Male"],
            }
        ).to_parquet(raw_root / "patient.parquet", index=False)
    export_root = tmp_path / "export"
    export_root.mkdir()
    (export_root / "_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu_native_export_v2",
                "database": manifest_database or database,
                "data_path": str(raw_root),
            }
        ),
        encoding="utf-8",
    )
    return export_root


def test_modern_eicu_export_binds_its_patient_table_without_private_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_root = _modern_export(tmp_path, database="eicu_demo")
    monkeypatch.setattr(
        raw_source_authority,
        "resolve_raw_mimic_iv_source_binding",
        lambda **_kwargs: pytest.fail("legacy authority must not be consulted"),
    )

    binding = resolve_raw_hospital_source_binding(
        export_path=export_root, database="eicu_demo"
    )

    assert isinstance(binding, RawHospitalSourceBinding)
    assert binding.family == "eicu"
    assert binding.database == "eicu_demo"
    receipt = binding.public_receipt()
    assert receipt["authority_kind"] == "export_manifest_data_path"
    assert receipt["export_manifest_file"] == "_manifest.json"
    assert set(receipt["raw_table_sha256"]) == {"patient"}
    assert (
        receipt["hospital_mortality_followup"]["materializer"]
        == "eicu_hospital_death_or_discharge_censor"
    )
    assert str(tmp_path) not in json.dumps(receipt)

    followup = binding.materialize_hospital_mortality_followup()
    assert followup.frame[["stay_id", "hospital_death", "hospital_followup_time_hours"]].to_dict(
        orient="records"
    ) == [
        {"stay_id": 201, "hospital_death": 0, "hospital_followup_time_hours": 48.0},
        {"stay_id": 202, "hospital_death": 1, "hospital_followup_time_hours": 10.0},
    ]
    assert followup.exclusions.to_dict(orient="records") == [
        {"stay_id": 203, "reason_code": "hospital_mortality_status_missing"}
    ]
    assert followup.receipt["database"] == "eicu_demo"
    assert followup.receipt["chronology_notes"] == {
        "hospital_discharge_before_icu_discharge_stays": 1
    }
    with pytest.raises(RawSourceAuthorityError) as excinfo:
        binding.materialize_hospital_mortality_status()
    assert excinfo.value.code == "raw_source_authority_status_unsupported"


def test_modern_mimic_iv_export_binds_both_raw_tables(tmp_path: Path) -> None:
    export_root = _modern_export(tmp_path, database="miiv")

    binding = resolve_raw_hospital_source_binding(export_path=export_root, database="miiv")

    assert isinstance(binding, RawHospitalSourceBinding)
    assert binding.family == "mimic_iv"
    receipt = binding.public_receipt()
    assert set(receipt["raw_table_sha256"]) == {"icustays", "admissions"}
    assert (
        receipt["hospital_mortality_followup"]["materializer"]
        == "mimic_iv_hospital_death_or_discharge_censor"
    )
    followup = binding.materialize_hospital_mortality_followup()
    assert followup.frame["hospital_death"].tolist() == [0, 1]
    assert binding.materialize_hospital_mortality_status().frame["hospital_death"].tolist() == [
        False,
        True,
    ]


def test_modern_export_binding_rechecks_bytes_and_fails_closed(tmp_path: Path) -> None:
    export_root = _modern_export(tmp_path, database="eicu_demo")
    binding = resolve_raw_hospital_source_binding(
        export_path=export_root, database="eicu_demo"
    )
    assert binding is not None
    patient = tmp_path / "raw" / "patient.parquet"
    pd.DataFrame(
        {
            "patientunitstayid": [201],
            "hospitaldischargestatus": ["Alive"],
            "hospitaldischargeoffset": [10],
        }
    ).to_parquet(patient, index=False)

    with pytest.raises(RawSourceAuthorityError) as excinfo:
        binding.materialize_hospital_mortality_followup()
    assert excinfo.value.code == "raw_source_authority_table_digest_mismatch"


@pytest.mark.parametrize(
    ("database", "manifest_database", "break_table", "code"),
    [
        ("eicu_demo", "miiv", False, "raw_source_authority_database_mismatch"),
        ("eicu_demo", None, True, "raw_source_authority_table_unavailable"),
        ("hirid", "hirid", False, "raw_source_authority_database_unsupported"),
    ],
)
def test_modern_export_with_broken_coordinate_never_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    database: str,
    manifest_database: str | None,
    break_table: bool,
    code: str,
) -> None:
    export_root = _modern_export(tmp_path, database="eicu_demo", manifest_database=manifest_database)
    if break_table:
        (tmp_path / "raw" / "patient.parquet").unlink()
    monkeypatch.setattr(
        raw_source_authority,
        "resolve_raw_mimic_iv_source_binding",
        lambda **_kwargs: pytest.fail("a broken sealed coordinate must fail closed"),
    )

    with pytest.raises(RawSourceAuthorityError) as excinfo:
        resolve_raw_hospital_source_binding(export_path=export_root, database=database)
    assert excinfo.value.code == code


def test_modern_export_requires_the_minimal_raw_schema(tmp_path: Path) -> None:
    export_root = _modern_export(tmp_path, database="eicu_demo")
    patient = tmp_path / "raw" / "patient.parquet"
    pd.read_parquet(patient).drop(columns=["hospitaldischargestatus"]).to_parquet(
        patient, index=False
    )

    with pytest.raises(RawSourceAuthorityError) as excinfo:
        resolve_raw_hospital_source_binding(export_path=export_root, database="eicu_demo")
    assert excinfo.value.code == "raw_source_authority_table_schema_mismatch"
    assert excinfo.value.details == {
        "object": "patient",
        "missing_columns": ["hospitaldischargestatus"],
    }


def test_legacy_export_without_data_path_uses_only_the_private_authority(
    tmp_path: Path,
) -> None:
    env, export_root = _authority_environment(tmp_path)
    assert (
        resolve_manifest_raw_source_binding(export_path=export_root, database="miiv")
        is None
    )
    legacy = resolve_raw_mimic_iv_source_binding(
        export_path=export_root, database="miiv", environ=env
    )
    assert legacy is not None
    assert (
        legacy.public_receipt()["schema_version"]
        == "easyicu.registered_export_raw_source_authority/1"
    )
    # Without the private authority nothing is inferred from the sibling ``raw``
    # directory that the legacy fixture happens to have.
    assert (
        resolve_raw_hospital_source_binding(export_path=tmp_path / "missing", database="miiv")
        is None
    )
