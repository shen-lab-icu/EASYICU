from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from easyicu.webserver.source_identity_authority import (
    PatientGroupingAuthorityError,
    resolve_patient_grouping_authority,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authority_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    export_root = tmp_path / "export"
    export_root.mkdir()
    manifest = export_root / "easyicu_export_manifest.json"
    manifest.write_text('{"database":"miiv"}', encoding="utf-8")
    mapping = tmp_path / "stay_patient.parquet"
    pd.DataFrame(
        {"stay_id": [11, 12, 13], "patient_key": [1, 1, 2]}
    ).to_parquet(mapping, index=False)
    env = {
        "EASYICU_PATIENT_GROUPING_EXPORT_ROOT": str(export_root),
        "EASYICU_PATIENT_GROUPING_EXPORT_MANIFEST": manifest.name,
        "EASYICU_PATIENT_GROUPING_EXPORT_MANIFEST_SHA256": _sha256(manifest),
        "EASYICU_PATIENT_GROUPING_DATABASE": "mimic_iv",
        "EASYICU_PATIENT_GROUPING_MAPPING_PATH": str(mapping),
        "EASYICU_PATIENT_GROUPING_MAPPING_SHA256": _sha256(mapping),
        "EASYICU_PATIENT_GROUPING_STAY_COLUMN": "stay_id",
        "EASYICU_PATIENT_GROUPING_PATIENT_COLUMN": "patient_key",
        "EASYICU_PATIENT_GROUPING_AUTHORITY_REF": "owner/bridge/v1",
    }
    return env, export_root


def test_patient_grouping_authority_absent_is_not_inferred(tmp_path: Path) -> None:
    assert (
        resolve_patient_grouping_authority(
            export_path=tmp_path,
            database="miiv",
            environ={},
        )
        is None
    )


def test_patient_grouping_authority_binds_exact_export_and_mapping(
    tmp_path: Path,
) -> None:
    env, export_root = _authority_environment(tmp_path)
    binding = resolve_patient_grouping_authority(
        export_path=export_root,
        database="miiv",
        environ=env,
    )

    assert binding is not None
    assert binding.mapping_stay_column == "stay_id"
    assert binding.mapping_patient_column == "patient_key"
    assert binding.output_identity_column == "patient_stay_id"
    assert binding.authority_coordinates["provider_visible_values"] is False


def test_patient_grouping_authority_wrong_source_does_not_cross_bind(
    tmp_path: Path,
) -> None:
    env, _ = _authority_environment(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    assert (
        resolve_patient_grouping_authority(
            export_path=other,
            database="miiv",
            environ=env,
        )
        is None
    )


def test_patient_grouping_authority_digest_drift_fails_closed(
    tmp_path: Path,
) -> None:
    env, export_root = _authority_environment(tmp_path)
    env["EASYICU_PATIENT_GROUPING_MAPPING_SHA256"] = "0" * 64
    with pytest.raises(
        PatientGroupingAuthorityError,
        match="does not match its authority digest",
    ) as exc:
        resolve_patient_grouping_authority(
            export_path=export_root,
            database="miiv",
            environ=env,
        )
    assert exc.value.code == "patient_grouping_mapping_digest_mismatch"


def test_patient_grouping_authority_partial_configuration_fails_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(PatientGroupingAuthorityError) as exc:
        resolve_patient_grouping_authority(
            export_path=tmp_path,
            database="miiv",
            environ={"EASYICU_PATIENT_GROUPING_DATABASE": "miiv"},
        )
    assert exc.value.code == "patient_grouping_authority_incomplete"
