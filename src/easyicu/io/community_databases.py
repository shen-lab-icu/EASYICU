"""Preparation helpers for community ICU databases.

The four sources in this module are locally supplied public datasets rather
than redistributable package fixtures.  EasyICU keeps their raw layouts intact,
converts CSV tables through :class:`~easyicu.io.data_converter.DataConverter`,
and materialises only the small stay-index table required by the common API.

No raw rows are copied into the source repository.  The generated manifest is
deliberately explicit about whether ``stay_id`` is a native ICU identifier or
an admission-level proxy.
"""

from __future__ import annotations

import json
import shutil
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


COMMUNITY_DATABASES = frozenset({"nwicu", "zhejiang_eicu", "jinhua", "zigong"})
COMMUNITY_MANIFEST = "community_preparation_manifest.json"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_extract_csv_zip(archive: Path, destination: Path) -> int:
    """Extract CSV members without allowing links or path traversal."""

    root = destination.resolve()
    extracted = 0
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            name = member.filename
            if member.is_dir() or name.startswith("__MACOSX/"):
                continue
            if not name.casefold().endswith(".csv"):
                continue
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Unsafe ZIP symlink member: {name}")
            target = (destination / name).resolve()
            if not _is_relative_to(target, root):
                raise ValueError(f"Unsafe ZIP member outside extraction root: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                raise ValueError(f"Refusing to overwrite ZIP target symlink: {target}")
            if target.is_file() and target.stat().st_size == member.file_size:
                continue
            temporary = target.with_name(f".{target.name}.easyicu-part")
            with bundle.open(member) as source, temporary.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
            temporary.replace(target)
            extracted += 1
    return extracted


def _has_csv(root: Path, basename: str) -> bool:
    target = basename.casefold()
    return any(
        path.is_file() and path.name.casefold() == target
        for path in root.rglob("*.csv")
    )


def prepare_community_archives(data_path: Path, database: str) -> list[str]:
    """Expand the selected release archive for a supported community source.

    Jinhua has two local archive snapshots.  ``-02`` is the corrected/latest
    snapshot and is selected deterministically so duplicate tables are never
    combined.
    """

    if database not in COMMUNITY_DATABASES:
        return []
    selections: list[tuple[str, Path | None]] = []
    if database == "zhejiang_eicu" and not _has_csv(data_path, "PtAdmiTable.csv"):
        selections.append(
            (
                "PtAdmiTable.csv",
                next(iter(sorted(data_path.glob("OMIX005817-*.zip"))), None),
            )
        )
    elif database == "jinhua" and not _has_csv(data_path, "transfer.csv"):
        archives = sorted(data_path.glob("OMIX007493-*.zip"))
        selections.append(("transfer.csv", archives[-1] if archives else None))
    elif database == "zigong" and not _has_csv(data_path, "dtBaseline.csv"):
        archives = sorted(data_path.glob("DataTables.zip"))
        selections.append(("dtBaseline.csv", archives[-1] if archives else None))

    extracted: list[str] = []
    for marker, archive in selections:
        if archive is None:
            raise FileNotFoundError(
                f"{database} raw layout is missing both {marker} and its release ZIP"
            )
        _safe_extract_csv_zip(archive, data_path)
        if not _has_csv(data_path, marker):
            raise ValueError(f"{archive.name} did not contain required marker {marker}")
        extracted.append(archive.name)
    return extracted


def _read_table(data_path: Path, name: str, columns: list[str]) -> pd.DataFrame:
    path = data_path / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Required prepared table is missing: {path}")
    return pd.read_parquet(path, columns=columns)


def _write_parquet_atomic(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.easyicu-part")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(destination)


def _ensure_nwicu_outcomes(data_path: Path, *, force: bool) -> None:
    """Materialise stay-keyed hospital mortality without duplicating raw data."""

    destination = data_path / "nwicu_outcomes.parquet"
    if destination.is_file() and not force:
        return
    admissions_path = data_path / "admissions.parquet"
    icustays_path = data_path / "icustays.parquet"
    if not admissions_path.is_file() or not icustays_path.is_file():
        return
    admissions = pd.read_parquet(
        admissions_path,
        columns=["hadm_id", "dischtime", "hospital_expire_flag"],
    )
    stays = pd.read_parquet(icustays_path, columns=["stay_id", "hadm_id"])
    outcomes = stays.merge(
        admissions,
        on="hadm_id",
        how="left",
        validate="many_to_one",
    )[["stay_id", "hadm_id", "dischtime", "hospital_expire_flag"]]
    _write_parquet_atomic(outcomes, destination)


def _zhejiang_stays(data_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = _read_table(
        data_path,
        "ptadmitable",
        ["patient_SN", "Hospital_ID", "Discharge_DateTime", "DaysHospitalStay"],
    )
    discharge_days = pd.to_numeric(source["Discharge_DateTime"], errors="coerce")
    days = pd.to_numeric(source["DaysHospitalStay"], errors="coerce")
    source["outtime"] = discharge_days * 24.0
    source["intime"] = (discharge_days - days) * 24.0
    source = source.dropna(subset=["patient_SN"]).copy()
    source["stay_id"] = source["patient_SN"]
    source["patient_SN"] = source["stay_id"]
    source["hadm_id"] = source["patient_SN"]
    source["subject_id"] = source["Hospital_ID"]
    stays = source.groupby("stay_id", as_index=False, dropna=False).agg(
        patient_SN=("patient_SN", "first"),
        subject_id=("subject_id", "first"),
        hadm_id=("hadm_id", "first"),
        intime=("intime", "min"),
        outtime=("outtime", "max"),
    )[["subject_id", "hadm_id", "stay_id", "patient_SN", "intime", "outtime"]]
    return stays, {
        "stay_semantics": "patient_SN encounter on a relative-hour axis; admission start reconstructed as discharge-day offset minus reported hospital days",
        "limitations": [
            "No independent native ICU entry timestamp is present in PtAdmiTable.",
            "The generated window is an encounter-level proxy and must not be described as a validated ICU-only window.",
        ],
    }


def _icu_text_mask(series: pd.Series) -> pd.Series:
    return series.astype("string").str.contains(
        r"ICU|intensive\s+care|重症监护|急诊重症", case=False, na=False, regex=True
    )


def _jinhua_stays(data_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    transfers = _read_table(
        data_path,
        "transfer",
        [
            "subject_id",
            "hadm_id",
            "outtime_base",
            "out_department",
            "out_department_en",
            "intime_base",
            "transfer_department",
            "transfer_department_en",
        ],
    )
    entered = _icu_text_mask(transfers["transfer_department"]) | _icu_text_mask(
        transfers["transfer_department_en"]
    )
    exited = _icu_text_mask(transfers["out_department"]) | _icu_text_mask(
        transfers["out_department_en"]
    )
    entry = transfers.loc[
        entered, ["subject_id", "hadm_id", "intime_base", "outtime_base"]
    ].copy()
    entry["intime"] = pd.to_numeric(entry.pop("intime_base"), errors="coerce") / 60.0
    # Some transfer rows omit the arrival field but retain the immediately
    # preceding departure offset. It is the only available boundary for that
    # ICU entry and is preferable to silently losing the admission.
    entry["intime"] = entry["intime"].fillna(
        pd.to_numeric(entry.pop("outtime_base"), errors="coerce") / 60.0
    )
    entry = entry.groupby(["subject_id", "hadm_id"], as_index=False).agg(
        intime=("intime", "min")
    )
    leave = transfers.loc[exited, ["hadm_id", "outtime_base"]].copy()
    leave["outtime"] = pd.to_numeric(leave.pop("outtime_base"), errors="coerce") / 60.0
    leave = leave.groupby("hadm_id", as_index=False).agg(outtime=("outtime", "max"))
    stays = entry.merge(leave, on="hadm_id", how="left")

    front = _read_table(
        data_path,
        "medical_record_front_page",
        ["hadm_id", "dischtime_base"],
    )
    front["hospital_outtime"] = (
        pd.to_numeric(front["dischtime_base"], errors="coerce") / 60.0
    )
    front = front.groupby("hadm_id", as_index=False).agg(
        hospital_outtime=("hospital_outtime", "max")
    )
    stays = stays.merge(front, on="hadm_id", how="left")
    stays["outtime"] = stays["outtime"].fillna(stays["hospital_outtime"])
    stays["stay_id"] = stays["hadm_id"]
    stays = stays[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]]
    return stays, {
        "stay_semantics": "one ICU-containing hospital admission per hadm_id on a relative-hour axis, spanning the first recorded ICU entry to the last recorded ICU exit",
        "limitations": [
            "Multiple ICU episodes in one admission are collapsed into one analysis window.",
            "Hospital discharge is used only when a recorded ICU exit is absent.",
        ],
    }


def _zigong_stays(data_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    transfers = _read_table(
        data_path,
        "dttransfer",
        ["PATIENT_ID", "INP_NO", "StartTime", "StopTime"],
    )
    transfers["intime"] = pd.to_numeric(transfers["StartTime"], errors="coerce")
    transfers["outtime"] = pd.to_numeric(transfers["StopTime"], errors="coerce")
    transfers = transfers.dropna(subset=["INP_NO"])
    stays = transfers.groupby("INP_NO", as_index=False, dropna=False).agg(
        PATIENT_ID=("PATIENT_ID", "first"),
        intime=("intime", "min"),
        outtime=("outtime", "max"),
    )
    stays["subject_id"] = stays["PATIENT_ID"]
    stays["hadm_id"] = stays["INP_NO"]
    stays["stay_id"] = stays["INP_NO"]
    stays = stays[
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "PATIENT_ID",
            "INP_NO",
            "intime",
            "outtime",
        ]
    ]
    return stays, {
        "stay_semantics": "one infection-cohort ICU encounter per INP_NO on the source relative-hour axis, bounded by transfer records",
        "limitations": [
            "This is an infection-enriched cohort, not an unselected general ICU population."
        ],
    }


def ensure_community_stays(
    data_path: Path,
    database: str,
    *,
    extracted_archives: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any] | None:
    """Materialise a common stay index and an explicit provenance receipt."""

    if database not in COMMUNITY_DATABASES:
        return None
    destination = data_path / "stays.parquet"
    details: dict[str, Any]
    if database == "nwicu":
        icustays = data_path / "icustays.parquet"
        if not icustays.is_file():
            raise FileNotFoundError(f"Required prepared table is missing: {icustays}")
        _ensure_nwicu_outcomes(data_path, force=force)
        row_count = int(pd.read_parquet(icustays, columns=["stay_id"]).shape[0])
        details = {
            "stay_semantics": "native NWICU ICU stay_id and intime/outtime",
            "limitations": [],
        }
        stay_table = "icustays"
    else:
        builders = {
            "zhejiang_eicu": _zhejiang_stays,
            "jinhua": _jinhua_stays,
            "zigong": _zigong_stays,
        }
        if force or not destination.is_file():
            stays, details = builders[database](data_path)
            if stays.empty:
                raise ValueError(f"{database} produced an empty ICU stay index")
            if stays["stay_id"].duplicated().any():
                raise ValueError(
                    f"{database} stay index contains duplicate stay_id values"
                )
            _write_parquet_atomic(stays, destination)
        else:
            stays = pd.read_parquet(destination, columns=["stay_id"])
            _, details = builders[database](data_path)
        row_count = int(len(stays))
        stay_table = "stays"

    retained_archives = list(extracted_archives or [])
    existing_manifest = data_path / COMMUNITY_MANIFEST
    if not retained_archives and existing_manifest.is_file():
        try:
            previous = json.loads(existing_manifest.read_text(encoding="utf-8"))
            retained_archives = list(previous.get("extracted_archives", []))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            retained_archives = []

    manifest = {
        "schema_version": 1,
        "database": database,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "stay_table": stay_table,
        "stay_id_column": {
            "nwicu": "stay_id",
            "zhejiang_eicu": "patient_SN",
            "jinhua": "hadm_id",
            "zigong": "INP_NO",
        }[database],
        "stay_count": row_count,
        "extracted_archives": sorted(retained_archives),
        **details,
    }
    path = data_path / COMMUNITY_MANIFEST
    temporary = path.with_name(f".{path.name}.easyicu-part")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return manifest


__all__ = [
    "COMMUNITY_DATABASES",
    "COMMUNITY_MANIFEST",
    "ensure_community_stays",
    "prepare_community_archives",
]
