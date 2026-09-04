"""Contracts for the private legacy-export/raw-MIMIC-IV binding owner."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from easyicu.webserver import dataio
from easyicu.webserver.raw_source_authority import (
    RawSourceAuthorityError,
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
