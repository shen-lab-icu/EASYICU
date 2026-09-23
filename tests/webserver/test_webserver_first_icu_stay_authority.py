"""The host resolves a verified first-ICU-stay coordinate from a bound source.

A first-stay study keeps each patient's first ICU stay.  The host may apply
that restriction only when the raw stay table it already binds by digest can
prove which stay came first: MIMIC-IV's ``icustays`` carries ``subject_id``
and ``intime``.  eICU cannot order stays across hospitalizations, so it is
never inferred there.
"""

from __future__ import annotations

import hashlib
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
from easyicu.webserver import source_identity_authority


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_mimic_iv(raw: Path, *, intimes: list, subjects: list | None = None) -> tuple[Path, Path]:
    raw.mkdir(parents=True, exist_ok=True)
    stays = [11, 12, 13][: len(intimes)]
    columns: dict[str, list] = {
        "stay_id": stays,
        "hadm_id": [100 + stay for stay in stays],
        "intime": intimes,
    }
    if subjects is not None:
        columns["subject_id"] = subjects
    icustays = raw / "icustays.parquet"
    admissions = raw / "admissions.parquet"
    pd.DataFrame(columns).to_parquet(icustays, index=False)
    pd.DataFrame(
        {
            "hadm_id": [100 + stay for stay in stays],
            "dischtime": ["2026-06-01"] * len(stays),
            "deathtime": [None] * len(stays),
            "hospital_expire_flag": [0] * len(stays),
        }
    ).to_parquet(admissions, index=False)
    return icustays, admissions


def _modern_export(tmp_path: Path, **raw_kwargs) -> Path:
    _raw_mimic_iv(tmp_path / "raw", **raw_kwargs)
    export = tmp_path / "export"
    export.mkdir()
    (export / "_manifest.json").write_text(
        json.dumps({"database": "miiv", "data_path": str(tmp_path / "raw"), "files": []}),
        encoding="utf-8",
    )
    return export


@pytest.fixture(autouse=True)
def _private_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from easyicu.webserver import raw_source_authority

    monkeypatch.setattr(
        source_identity_authority.state_paths, "state_root", lambda: tmp_path / "state"
    )
    monkeypatch.setattr(raw_source_authority, "_DEFAULT_CONFIG_PATH", tmp_path / "absent.env")
    monkeypatch.setattr(
        source_identity_authority, "_DEFAULT_CONFIG_PATH", tmp_path / "absent-grouping.env"
    )


def test_a_modern_mimic_iv_export_proves_each_patients_first_stay(tmp_path: Path) -> None:
    # Stays 11 and 12 share patient 5, and 12 was admitted first.
    export = _modern_export(
        tmp_path,
        intimes=["2026-03-01 08:00", "2026-01-01 08:00", "2026-02-01 08:00"],
        subjects=[5, 5, 7],
    )

    binding = source_identity_authority.resolve_study_first_icu_stay(
        export_path=export, database="miiv"
    )

    assert binding is not None
    # Private host state, bound by digest and named after the verified table.
    assert stat.S_IMODE(os.stat(binding.coordinate_path).st_mode) == 0o600
    assert binding.coordinate_path.parent == tmp_path / "state" / "private" / "first-icu-stay"
    assert _sha256(binding.coordinate_path) == binding.coordinate_sha256
    coordinates = dict(binding.authority_coordinates)
    assert coordinates["identity_table_sha256"] in binding.coordinate_path.name
    assert coordinates["authority_ref"] == "export_manifest_data_path/mimic_iv/1/first_icu_stay"
    assert coordinates["order_column"] == "intime"
    assert (coordinates["stays"], coordinates["patients"], coordinates["non_first_icu_stays"]) == (3, 2, 1)
    assert coordinates["provider_visible_values"] is False
    # It agrees with the grouping derived from the same table.
    grouping = source_identity_authority.resolve_study_patient_grouping(
        export_path=export, database="miiv"
    )
    assert coordinates["patient_grouping_mapping_sha256"] == grouping.mapping_sha256
    flags = load_verified_first_icu_stay(
        binding.coordinate_path, expected_sha256=binding.coordinate_sha256
    ).set_index(COORDINATE_STAY_COLUMN)[COORDINATE_FLAG_COLUMN]
    assert flags.to_dict() == {11: False, 12: True, 13: True}


def test_a_legacy_export_uses_its_private_raw_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    icustays, admissions = _raw_mimic_iv(
        tmp_path / "raw", intimes=["2026-01-01", "2026-02-01"], subjects=[5, 5]
    )
    export = tmp_path / "export"
    export.mkdir()
    manifest = export / "easyicu_export_manifest.json"
    manifest.write_text('{"database":"miiv"}', encoding="utf-8")
    for key, value in {
        "EXPORT_ROOT": str(export),
        "EXPORT_MANIFEST": manifest.name,
        "EXPORT_MANIFEST_SHA256": _sha256(manifest),
        "DATABASE": "miiv",
        "SOURCE_ROOT": str(tmp_path / "raw"),
        "ICUSTAYS_FILE": icustays.name,
        "ICUSTAYS_SHA256": _sha256(icustays),
        "ADMISSIONS_FILE": admissions.name,
        "ADMISSIONS_SHA256": _sha256(admissions),
        "AUTHORITY_REF": "owner/raw-mimiciv/v1",
    }.items():
        monkeypatch.setenv(f"EASYICU_RAW_SOURCE_{key}", value)

    binding = source_identity_authority.resolve_study_first_icu_stay(
        export_path=export, database="miiv"
    )

    assert binding is not None
    assert binding.authority_coordinates["authority_ref"] == "owner/raw-mimiciv/v1/first_icu_stay"
    assert binding.authority_coordinates["non_first_icu_stays"] == 1


def test_a_source_that_cannot_order_stays_is_not_inferred(tmp_path: Path) -> None:
    # MIMIC-IV without subject_id: the stays cannot be attributed to patients.
    export = _modern_export(tmp_path, intimes=["2026-01-01", "2026-02-01"])
    assert (
        source_identity_authority.resolve_study_first_icu_stay(
            export_path=export, database="miiv"
        )
        is None
    )


def test_eicu_never_derives_a_first_stay(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {
            "patientunitstayid": [11, 12],
            "hospitaldischargeoffset": [100, 200],
            "hospitaldischargestatus": ["Alive", "Alive"],
            "uniquepid": ["002-9", "002-9"],
        }
    ).to_parquet(raw / "patient.parquet", index=False)
    export = tmp_path / "export"
    export.mkdir()
    (export / "_manifest.json").write_text(
        json.dumps({"database": "eicu", "data_path": str(raw), "files": []}),
        encoding="utf-8",
    )

    assert (
        source_identity_authority.resolve_study_first_icu_stay(
            export_path=export, database="eicu"
        )
        is None
    )


def test_a_tied_first_admission_fails_closed_with_its_cause(tmp_path: Path) -> None:
    export = _modern_export(
        tmp_path, intimes=["2026-01-01 08:00", "2026-01-01 08:00"], subjects=[5, 5]
    )

    with pytest.raises(source_identity_authority.PatientGroupingAuthorityError) as exc:
        source_identity_authority.resolve_study_first_icu_stay(
            export_path=export, database="miiv"
        )

    assert exc.value.code == "raw_source_authority_first_icu_stay_invalid"
    assert exc.value.details["cause_code"] == "first_icu_stay_order_tied"
    assert not (tmp_path / "state" / "private" / "first-icu-stay").exists() or not any(
        (tmp_path / "state" / "private" / "first-icu-stay").iterdir()
    )


def test_the_public_raw_receipt_keeps_its_shape(tmp_path: Path) -> None:
    """Runs already sealed against the receipt keep their identity."""

    from easyicu.webserver.raw_source_authority import resolve_manifest_raw_source_binding

    export = _modern_export(tmp_path, intimes=["2026-01-01"], subjects=[5])
    receipt = resolve_manifest_raw_source_binding(export_path=export, database="miiv").public_receipt()

    assert "first_icu_stay" not in json.dumps(receipt)
