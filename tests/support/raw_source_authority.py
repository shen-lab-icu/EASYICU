"""Synthetic raw-source authority and digest fixtures."""
from __future__ import annotations

import hashlib
from pathlib import Path
import pandas as pd


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
