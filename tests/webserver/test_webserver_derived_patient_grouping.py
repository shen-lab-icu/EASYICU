"""Patient grouping derived from a source's own official identity table.

A cluster-robust estimate needs to know which ICU stays belong to the same
patient.  Prepared exports deliberately carry no patient identifier, so the
host has always needed an owner-approved private bridge.  For a source whose
raw tables are already bound -- eICU's ``patient`` and MIMIC-IV's ``icustays``
are digest-verified for hospital-mortality follow-up -- the same verified bytes
carry ``uniquepid`` or ``subject_id``, and the host derives the bridge instead
of asking for a second approval of a column it already reads.

These tests fix that boundary: derived only from a verified binding, private on
disk, never inferred, and never preferred over an owner's own bridge.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.patient_grouping import (
    PatientGroupingError,
    derive_patient_grouping,
    load_verified_patient_grouping,
)
from easyicu.webserver import source_identity_authority


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _eicu_source(
    tmp_path: Path,
    *,
    stays: list[int] | None = None,
    patients: list[str] | None = None,
    with_identifier: bool = True,
) -> Path:
    """An export that names its sealed raw root, as a modern export does."""

    stays = stays or [11, 12, 13, 14]
    patients = patients or ["002-9", "002-9", "001-3", "003-7"]
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    columns: dict[str, list] = {
        "patientunitstayid": stays,
        "hospitaldischargeoffset": [100 * (index + 1) for index in range(len(stays))],
        "hospitaldischargestatus": ["Alive"] * len(stays),
    }
    if with_identifier:
        columns["uniquepid"] = patients
    pd.DataFrame(columns).to_parquet(raw / "patient.parquet", index=False)
    export = tmp_path / "export"
    export.mkdir()
    (export / "_manifest.json").write_text(
        json.dumps({"database": "eicu", "data_path": str(raw), "files": []}),
        encoding="utf-8",
    )
    return export


def _mimic_iv_raw_tables(raw: Path, *, with_identifier: bool = True) -> tuple[Path, Path]:
    """MIMIC-IV ``icustays``/``admissions``; stays 11 and 12 share a patient."""

    raw.mkdir(parents=True, exist_ok=True)
    icustays_columns: dict[str, list] = {
        "stay_id": [11, 12, 13],
        "hadm_id": [101, 102, 103],
        "intime": ["2026-01-01", "2026-02-01", "2026-03-01"],
    }
    if with_identifier:
        icustays_columns["subject_id"] = [5, 5, 7]
    icustays = raw / "icustays.parquet"
    admissions = raw / "admissions.parquet"
    pd.DataFrame(icustays_columns).to_parquet(icustays, index=False)
    pd.DataFrame(
        {
            "hadm_id": [101, 102, 103],
            "dischtime": ["2026-01-03", "2026-02-04", "2026-03-05"],
            "deathtime": [None, None, "2026-03-02"],
            "hospital_expire_flag": [0, 0, 1],
        }
    ).to_parquet(admissions, index=False)
    return icustays, admissions


def _legacy_mimic_iv_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, with_identifier: bool = True
) -> Path:
    """A legacy export (no ``data_path``) bound by the private raw-source authority."""

    from easyicu.webserver import raw_source_authority

    monkeypatch.setattr(
        raw_source_authority, "_DEFAULT_CONFIG_PATH", tmp_path / "absent.env"
    )
    icustays, admissions = _mimic_iv_raw_tables(
        tmp_path / "raw", with_identifier=with_identifier
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
    return export


def test_repeated_stays_are_grouped_by_the_official_patient_identifier() -> None:
    table = pd.DataFrame(
        {
            "patientunitstayid": [11, 12, 13, 14],
            # Two stays for one patient; the identifier is not an integer.
            "uniquepid": ["002-9", "002-9", "001-3", "003-7"],
        }
    )

    derived = derive_patient_grouping(
        table,
        stay_column="patientunitstayid",
        patient_column="uniquepid",
        identity_table_name="patient",
    )

    grouped = derived.frame.set_index("stay_id")["patient_key"]
    assert grouped.loc[11] == grouped.loc[12]
    assert len({grouped.loc[11], grouped.loc[13], grouped.loc[14]}) == 3
    # The surrogate is a dense rank over the sorted distinct identifiers, so it
    # is reproducible from the source and carries nothing but the grouping.
    assert sorted(derived.frame["patient_key"].tolist()) == [1, 2, 2, 3]
    assert derived.receipt["stays"] == 4
    assert derived.receipt["patients"] == 3
    assert derived.receipt["patients_with_repeated_stays"] == 1
    assert derived.receipt["stays_in_repeated_patients"] == 2
    assert derived.receipt["max_stays_per_patient"] == 2
    # A receipt is read by people; it must carry no identifier values.
    serialized = json.dumps(dict(derived.receipt))
    assert "002-9" not in serialized and "001-3" not in serialized


def test_an_unusable_identity_table_fails_closed() -> None:
    with pytest.raises(PatientGroupingError, match="declared columns"):
        derive_patient_grouping(
            pd.DataFrame({"patientunitstayid": [1]}),
            stay_column="patientunitstayid",
            patient_column="uniquepid",
            identity_table_name="patient",
        )
    with pytest.raises(PatientGroupingError, match="repeats a stay identifier"):
        derive_patient_grouping(
            pd.DataFrame(
                {"patientunitstayid": [1, 1], "uniquepid": ["a", "b"]}
            ),
            stay_column="patientunitstayid",
            patient_column="uniquepid",
            identity_table_name="patient",
        )
    for broken in (None, "   "):
        with pytest.raises(PatientGroupingError, match="present and non-empty"):
            derive_patient_grouping(
                pd.DataFrame(
                    {"patientunitstayid": [1, 2], "uniquepid": ["a", broken]}
                ),
                stay_column="patientunitstayid",
                patient_column="uniquepid",
                identity_table_name="patient",
            )


def test_a_derived_bridge_is_private_and_bound_to_its_source_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        source_identity_authority.state_paths, "state_root", lambda: tmp_path / "state"
    )
    export = _eicu_source(tmp_path)

    binding = source_identity_authority.resolve_derived_patient_grouping(
        export_path=export, database="eicu"
    )

    assert binding is not None
    # Private host state: readable only by the owner, outside the export.
    assert stat.S_IMODE(os.stat(binding.mapping_path).st_mode) == 0o600
    assert binding.mapping_path.parent == tmp_path / "state" / "private" / (
        "patient-grouping"
    )
    assert _sha256(binding.mapping_path) == binding.mapping_sha256
    coordinates = dict(binding.authority_coordinates)
    assert coordinates["provider_visible_values"] is False
    assert coordinates["patient_identifier_column"] == "uniquepid"
    assert coordinates["patients"] == 3
    # The file name is keyed on the verified source table, so a changed raw
    # source can never be served from a stale bridge.
    assert coordinates["identity_table_sha256"] in binding.mapping_path.name
    # The binding round-trips through the runtime's own verification.
    assert len(load_verified_patient_grouping(binding).frame) == 4


def test_a_changed_source_table_produces_a_different_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        source_identity_authority.state_paths, "state_root", lambda: tmp_path / "state"
    )
    first = source_identity_authority.resolve_derived_patient_grouping(
        export_path=_eicu_source(tmp_path / "a"), database="eicu"
    )
    second = source_identity_authority.resolve_derived_patient_grouping(
        export_path=_eicu_source(
            tmp_path / "b",
            stays=[11, 12, 13, 14],
            patients=["002-9", "001-3", "001-3", "003-7"],
        ),
        database="eicu",
    )

    assert first is not None and second is not None
    assert first.mapping_path != second.mapping_path
    assert first.mapping_sha256 != second.mapping_sha256


def test_a_source_without_an_official_identifier_is_not_inferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent identity is reported as absent, never guessed from stay ids."""

    monkeypatch.setattr(
        source_identity_authority.state_paths, "state_root", lambda: tmp_path / "state"
    )
    export = _eicu_source(tmp_path, with_identifier=False)

    assert (
        source_identity_authority.resolve_derived_patient_grouping(
            export_path=export, database="eicu"
        )
        is None
    )
    # An export with no sealed raw root at all is equally not inferred.
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "_manifest.json").write_text(
        json.dumps({"database": "eicu", "files": []}), encoding="utf-8"
    )
    assert (
        source_identity_authority.resolve_derived_patient_grouping(
            export_path=bare, database="eicu"
        )
        is None
    )


def test_a_modern_mimic_iv_export_groups_stays_by_subject_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from easyicu.webserver.raw_source_authority import (
        resolve_manifest_raw_source_binding,
    )

    monkeypatch.setattr(
        source_identity_authority.state_paths, "state_root", lambda: tmp_path / "state"
    )
    _mimic_iv_raw_tables(tmp_path / "raw")
    export = tmp_path / "export"
    export.mkdir()
    (export / "_manifest.json").write_text(
        json.dumps({"database": "miiv", "data_path": str(tmp_path / "raw"), "files": []}),
        encoding="utf-8",
    )

    binding = source_identity_authority.resolve_derived_patient_grouping(
        export_path=export, database="miiv"
    )

    assert binding is not None
    coordinates = dict(binding.authority_coordinates)
    assert coordinates["identity_table"] == "icustays"
    assert coordinates["patient_identifier_column"] == "subject_id"
    assert coordinates["patients"] == 2
    assert coordinates["patients_with_repeated_stays"] == 1
    grouped = load_verified_patient_grouping(binding).frame
    assert len(grouped) == 3
    # The export's own sealed raw root advertises the grouping it can derive.
    raw = resolve_manifest_raw_source_binding(export_path=export, database="miiv")
    assert raw is not None
    assert raw.public_receipt()["patient_grouping"]["identity_table"] == "icustays"


def test_a_legacy_mimic_iv_export_derives_grouping_from_its_private_raw_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The private authority already binds ``icustays``; its ``subject_id`` suffices."""

    from easyicu.webserver.raw_source_authority import (
        resolve_raw_hospital_source_binding,
    )

    monkeypatch.setattr(
        source_identity_authority.state_paths, "state_root", lambda: tmp_path / "state"
    )
    export = _legacy_mimic_iv_export(tmp_path, monkeypatch)

    binding = source_identity_authority.resolve_derived_patient_grouping(
        export_path=export, database="miiv"
    )

    assert binding is not None
    coordinates = dict(binding.authority_coordinates)
    assert coordinates["authority_ref"] == "owner/raw-mimiciv/v1/patient_grouping"
    assert coordinates["patient_identifier_column"] == "subject_id"
    assert coordinates["patients"] == 2
    assert stat.S_IMODE(os.stat(binding.mapping_path).st_mode) == 0o600
    assert len(load_verified_patient_grouping(binding).frame) == 3
    # The legacy public receipt keeps its version-1 shape, so runs already
    # bound to it keep their identity.
    legacy = resolve_raw_hospital_source_binding(export_path=export, database="miiv")
    receipt = legacy.public_receipt()
    assert receipt["schema_version"] == "easyicu.registered_export_raw_source_authority/1"
    assert "patient_grouping" not in receipt


def test_a_mimic_iv_source_without_subject_id_is_not_inferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        source_identity_authority.state_paths, "state_root", lambda: tmp_path / "state"
    )
    export = _legacy_mimic_iv_export(tmp_path, monkeypatch, with_identifier=False)

    assert (
        source_identity_authority.resolve_derived_patient_grouping(
            export_path=export, database="miiv"
        )
        is None
    )


def test_an_owner_approved_bridge_is_never_replaced_by_a_derived_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        source_identity_authority.state_paths, "state_root", lambda: tmp_path / "state"
    )
    export = _eicu_source(tmp_path)
    mapping = tmp_path / "owner-bridge.parquet"
    pd.DataFrame(
        {"stay_id": [11, 12, 13, 14], "patient_key": [7, 7, 7, 8]}
    ).to_parquet(mapping, index=False)
    manifest = export / "_manifest.json"
    monkeypatch.setenv("EASYICU_PATIENT_GROUPING_EXPORT_ROOT", str(export))
    monkeypatch.setenv("EASYICU_PATIENT_GROUPING_EXPORT_MANIFEST", manifest.name)
    monkeypatch.setenv(
        "EASYICU_PATIENT_GROUPING_EXPORT_MANIFEST_SHA256", _sha256(manifest)
    )
    monkeypatch.setenv("EASYICU_PATIENT_GROUPING_DATABASE", "eicu")
    monkeypatch.setenv("EASYICU_PATIENT_GROUPING_MAPPING_PATH", str(mapping))
    monkeypatch.setenv("EASYICU_PATIENT_GROUPING_MAPPING_SHA256", _sha256(mapping))
    monkeypatch.setenv("EASYICU_PATIENT_GROUPING_STAY_COLUMN", "stay_id")
    monkeypatch.setenv("EASYICU_PATIENT_GROUPING_PATIENT_COLUMN", "patient_key")
    monkeypatch.setenv("EASYICU_PATIENT_GROUPING_AUTHORITY_REF", "owner/bridge/v1")

    binding = source_identity_authority.resolve_study_patient_grouping(
        export_path=export, database="eicu"
    )

    assert binding is not None
    assert binding.mapping_path == mapping.resolve()
    assert binding.authority_coordinates["authority_ref"] == "owner/bridge/v1"
