"""Build a host-only Canonical9 stay-to-patient identity bridge.

The historical ``full6_20260717`` export is the sole clinical payload.  It
contains stable ICU-stay keys but deliberately omits the patient relation used
to prevent a repeated admission leaking across a patient-level split.  This
module reads only those keys from the export and from frozen source relations,
then writes a protected mapping plus the small descriptor consumed by
``identity_bridge_contract``.

It is deliberately *not* connected to the real-run launcher.  A successful
build merely makes the output reviewable by native typed materialization; it
never becomes production input authority or a provider permit.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .identity_bridge_contract import (
    IDENTITY_BRIDGE_REF,
    IDENTITY_BRIDGE_SCHEMA,
    IdentityBridgeContractError,
    assess_identity_bridge_contract,
    load_identity_bridge_contract,
)

_EXPORT_LABEL = "full6_20260717"
_READ_BLOCK = 1024 * 1024


class IdentityBridgeBuildError(ValueError):
    """The controlled identity bridge cannot be built safely."""


@dataclasses.dataclass(frozen=True)
class IdentitySourceSpec:
    """One source relation, its full0717 key, and its documented semantics."""

    source_id: str
    export_directory: str
    export_stay_key: str
    raw_relative_path: str
    raw_stay_key: str
    raw_patient_key: str
    contract_patient_key: str
    mapping_semantics: str
    documentation_url: str
    source_note: str


_SOURCES: tuple[IdentitySourceSpec, ...] = (
    IdentitySourceSpec(
        "mimic_iv",
        "miiv",
        "stay_id",
        "mimiciv/icustays.parquet",
        "stay_id",
        "subject_id",
        "subject_id",
        "attested_icu_stay_to_patient",
        "https://physionet.org/content/mimiciv/",
        "The ICU-stay relation is read from the frozen MIMIC-IV icustays table.",
    ),
    IdentitySourceSpec(
        "mimic_iii",
        "mimic",
        "icustay_id",
        "mimiciii/icustays.parquet",
        "ICUSTAY_ID",
        "SUBJECT_ID",
        "subject_id",
        "attested_icu_stay_to_patient",
        "https://physionet.org/content/mimiciii/1.4/",
        "The ICU-stay relation is read from the frozen MIMIC-III ICUSTAYS table.",
    ),
    IdentitySourceSpec(
        "eicu",
        "eicu",
        "patientunitstayid",
        "eicu/patient.parquet",
        "patientunitstayid",
        "uniquepid",
        "uniquepid",
        "attested_icu_stay_to_patient",
        "https://eicu.mit.edu/eicutables/patient/",
        "patientUnitStayID identifies an ICU stay and uniquepid identifies a patient.",
    ),
    IdentitySourceSpec(
        "amsterdamumcdb",
        "aumc",
        "admissionid",
        "aumc/admissions.parquet",
        "admissionid",
        "patientid",
        "patientid",
        "attested_icu_stay_to_patient",
        "https://github.com/AmsterdamUMC/AmsterdamUMCdb/wiki",
        "The admissions relation binds admissionid to the de-identified patientid.",
    ),
    IdentitySourceSpec(
        "hirid",
        "hirid",
        "patientid",
        "hirid/general_table.csv",
        "patientid",
        "patientid",
        "patientid",
        "attested_source_key_semantics",
        "https://www.physionet.org/content/hirid/1.0/",
        "HiRID defines Patient ID per ICU admission, so cross-admission linkage is unavailable.",
    ),
    IdentitySourceSpec(
        "sicdb",
        "sic",
        "CaseID",
        "sic/cases.parquet",
        "CaseID",
        "PatientID",
        "PatientID",
        "attested_source_key_semantics",
        "https://www.sicdb.com/Documentation/SICdb_Documentation",
        "SICdb CaseID identifies an admission and PatientID identifies readmissions.",
    ),
)


@dataclasses.dataclass(frozen=True)
class IdentityBridgeBuildResult:
    """Paths and non-sensitive identities produced by a successful build."""

    output_root: Path
    contract_path: Path
    contract_sha256: str
    mapping_paths: Mapping[str, Path]


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_existing_directory(path: Path | str, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_dir():
        raise IdentityBridgeBuildError(f"{label} must be an absolute real directory")
    return candidate.resolve(strict=True)


def _safe_regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise IdentityBridgeBuildError(f"{label} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _ensure_new_output_root(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or candidate.is_symlink():
        raise IdentityBridgeBuildError(
            "output root must be an absolute non-symlink path"
        )
    if candidate.exists():
        raise IdentityBridgeBuildError("output root must not already exist")
    parent = candidate.parent
    if not parent.is_dir() or parent.is_symlink():
        raise IdentityBridgeBuildError("output root parent must be a real directory")
    return candidate


def _write_private_bytes(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while publishing identity bridge")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_export_stays(export_root: Path, spec: IdentitySourceSpec) -> pd.Series:
    demographics = _safe_regular_file(
        export_root / spec.export_directory / "demographics.parquet",
        label=f"{spec.source_id} full0717 demographics",
    )
    try:
        values = pd.read_parquet(demographics, columns=[spec.export_stay_key])[
            spec.export_stay_key
        ]
    except Exception as exc:  # noqa: BLE001 - preserves source context
        raise IdentityBridgeBuildError(
            f"cannot read {spec.source_id} stay key from full0717 demographics"
        ) from exc
    if values.isna().any():
        raise IdentityBridgeBuildError(f"{spec.source_id} full0717 has null stay keys")
    return values.drop_duplicates().reset_index(drop=True)


def _read_raw_mapping(raw_root: Path, spec: IdentitySourceSpec) -> pd.DataFrame:
    source = _safe_regular_file(
        raw_root / spec.raw_relative_path,
        label=f"{spec.source_id} identity source",
    )
    raw_columns = list(dict.fromkeys((spec.raw_stay_key, spec.raw_patient_key)))
    try:
        if source.suffix.lower() == ".parquet":
            frame = pd.read_parquet(source, columns=raw_columns)
        elif source.suffix.lower() == ".csv":
            frame = pd.read_csv(source, usecols=raw_columns, low_memory=False)
        else:
            raise IdentityBridgeBuildError("identity source must be parquet or csv")
    except IdentityBridgeBuildError:
        raise
    except Exception as exc:  # noqa: BLE001 - preserves source context
        raise IdentityBridgeBuildError(
            f"cannot read {spec.source_id} identity relation"
        ) from exc
    if spec.raw_stay_key == spec.raw_patient_key:
        frame = pd.DataFrame(
            {
                spec.export_stay_key: frame[spec.raw_stay_key],
                "patient_key": frame[spec.raw_patient_key],
            }
        )
    else:
        frame = frame.rename(
            columns={
                spec.raw_stay_key: spec.export_stay_key,
                spec.raw_patient_key: "patient_key",
            }
        )
    if frame[spec.export_stay_key].isna().any() or frame["patient_key"].isna().any():
        raise IdentityBridgeBuildError(
            f"{spec.source_id} identity relation has null keys"
        )
    if frame[spec.export_stay_key].duplicated().any():
        raise IdentityBridgeBuildError(
            f"{spec.source_id} identity relation has duplicate stays"
        )
    return frame[[spec.export_stay_key, "patient_key"]]


def _normalised_join_key(values: pd.Series) -> pd.Series:
    """Canonicalise only representation, never the patient key itself."""

    if pd.api.types.is_integer_dtype(values.dtype):
        return values.astype("int64").astype(str)
    return values.astype(str)


def _export_content_sha256(export_root: Path) -> str:
    """Hash every immediate full0717 file through a path-and-content manifest."""

    members: list[dict[str, object]] = []
    for file in sorted(export_root.rglob("*")):
        if not file.is_file():
            continue
        if file.is_symlink():
            raise IdentityBridgeBuildError("full0717 export contains a symlink")
        members.append(
            {
                "relative_path": file.relative_to(export_root).as_posix(),
                "size_bytes": file.stat().st_size,
                "sha256": _sha256_file(file),
            }
        )
    if not members:
        raise IdentityBridgeBuildError("full0717 export contains no regular files")
    return hashlib.sha256(_canonical_json({"members": members})).hexdigest()


def _attestation_payload(spec: IdentitySourceSpec) -> dict[str, object]:
    return {
        "schema_version": "easyicu.figure2_identity_source_attestation/1",
        "source_id": spec.source_id,
        "stay_key": spec.export_stay_key,
        "patient_key": spec.contract_patient_key,
        "mapping_semantics": spec.mapping_semantics,
        "documentation_url": spec.documentation_url,
        "source_note": spec.source_note,
    }


def _build_source_mapping(
    *,
    export_root: Path,
    raw_root: Path,
    output_root: Path,
    spec: IdentitySourceSpec,
) -> tuple[dict[str, object], Path]:
    export_stays = _read_export_stays(export_root, spec)
    source_mapping = _read_raw_mapping(raw_root, spec)
    export_frame = pd.DataFrame({spec.export_stay_key: export_stays})
    export_frame["_join_key"] = _normalised_join_key(export_frame[spec.export_stay_key])
    source_mapping["_join_key"] = _normalised_join_key(
        source_mapping[spec.export_stay_key]
    )
    if source_mapping["_join_key"].duplicated().any():
        raise IdentityBridgeBuildError(
            f"{spec.source_id} identity relation aliases duplicate stay keys"
        )
    joined = export_frame.merge(
        source_mapping[["_join_key", "patient_key"]],
        on="_join_key",
        how="left",
        validate="one_to_one",
    )
    unmapped = int(joined["patient_key"].isna().sum())
    if unmapped:
        raise IdentityBridgeBuildError(
            f"{spec.source_id} leaves {unmapped} full0717 stays without identity mapping"
        )
    mapping = joined[[spec.export_stay_key, "patient_key"]].copy()
    mapping_path = output_root / f"{spec.source_id}_stay_patient.parquet"
    mapping.to_parquet(mapping_path, index=False)
    os.chmod(mapping_path, 0o600)
    relation_schema_sha256 = hashlib.sha256(
        _canonical_json(
            {
                "columns": [spec.export_stay_key, "patient_key"],
                "stay_key": spec.export_stay_key,
                "patient_key": spec.contract_patient_key,
            }
        )
    ).hexdigest()
    raw_path = _safe_regular_file(
        raw_root / spec.raw_relative_path, label="identity source"
    )
    attestation = _attestation_payload(spec)
    attestation_raw = _canonical_json(attestation)
    attestation_path = output_root / f"{spec.source_id}_semantics_attestation.json"
    _write_private_bytes(attestation_path, attestation_raw)
    unique_patients = int(mapping["patient_key"].nunique(dropna=True))
    return (
        {
            "source_id": spec.source_id,
            "stay_key": spec.export_stay_key,
            "patient_key": spec.contract_patient_key,
            "mapping_semantics": spec.mapping_semantics,
            "source_semantics_attestation_sha256": hashlib.sha256(
                attestation_raw
            ).hexdigest(),
            "projection": {
                "artifact_sha256": _sha256_file(mapping_path),
                "artifact_size_bytes": mapping_path.stat().st_size,
                "source_snapshot_sha256": _sha256_file(raw_path),
                "relation_schema_sha256": relation_schema_sha256,
            },
            "mapped_stay_count": int(len(mapping)),
            "unmapped_stay_count": 0,
            "duplicate_stay_count": 0,
            "max_stays_per_patient": (
                int(mapping.groupby("patient_key", dropna=False).size().max())
                if unique_patients
                else 0
            ),
        },
        mapping_path,
    )


def build_identity_bridge(
    *,
    full_export_root: Path | str,
    raw_source_root: Path | str,
    output_root: Path | str,
    owner_authorization_reference: str,
) -> IdentityBridgeBuildResult:
    """Build one private bridge descriptor from exact full0717 and raw relations.

    The caller must supply a non-empty owner authorization reference.  The
    reference is metadata only: neither it nor this function creates clinical
    review, typed authority, or real-run authorization.
    """

    if (
        not isinstance(owner_authorization_reference, str)
        or not owner_authorization_reference.strip()
    ):
        raise IdentityBridgeBuildError("owner authorization reference is required")
    export_root = _safe_existing_directory(
        full_export_root, label="full0717 export root"
    )
    raw_root = _safe_existing_directory(raw_source_root, label="raw source root")
    requested_output = _ensure_new_output_root(output_root)
    if export_root.name != _EXPORT_LABEL:
        raise IdentityBridgeBuildError(
            "identity bridge requires the full6_20260717 export"
        )
    manifest = _safe_regular_file(
        export_root / "run_manifest.json", label="full0717 manifest"
    )
    temporary = requested_output.with_name(
        f".{requested_output.name}.building-{uuid.uuid4().hex}"
    )
    temporary.mkdir(mode=0o700)
    try:
        mappings: list[dict[str, object]] = []
        mapping_paths: dict[str, Path] = {}
        for spec in _SOURCES:
            mapping, mapping_path = _build_source_mapping(
                export_root=export_root,
                raw_root=raw_root,
                output_root=temporary,
                spec=spec,
            )
            mappings.append(mapping)
            mapping_paths[spec.source_id] = mapping_path
        payload = {
            "schema_version": IDENTITY_BRIDGE_SCHEMA,
            "bridge_ref": IDENTITY_BRIDGE_REF,
            "historical_export": {
                "export_label": _EXPORT_LABEL,
                "export_manifest_sha256": _sha256_file(manifest),
                "export_content_sha256": _export_content_sha256(export_root),
            },
            "data_lane": {
                "status": "authorized",
                "authorization_reference": owner_authorization_reference.strip(),
            },
            "source_mappings": mappings,
        }
        contract_path = temporary / "identity_bridge_contract.json"
        _write_private_bytes(contract_path, _canonical_json(payload))
        try:
            contract, contract_sha256 = load_identity_bridge_contract(contract_path)
            report = assess_identity_bridge_contract(
                contract, contract_sha256=contract_sha256
            )
        except IdentityBridgeContractError as exc:
            raise IdentityBridgeBuildError(
                "built identity bridge failed self-verification"
            ) from exc
        if (
            not report.eligible_for_native_materialization_review
            or report.real_run_authorized
        ):
            raise IdentityBridgeBuildError(
                "built identity bridge has an invalid readiness state"
            )
        descriptor = {
            "schema_version": "easyicu.figure2_identity_bridge_build_receipt/1",
            "contract_file": contract_path.name,
            "contract_sha256": contract_sha256,
            "real_run_authorized": False,
            "next_required_step": "native_typed_materialization_review",
        }
        _write_private_bytes(
            temporary / "build_receipt.json", _canonical_json(descriptor)
        )
        os.replace(temporary, requested_output)
        final_contract = requested_output / contract_path.name
        return IdentityBridgeBuildResult(
            output_root=requested_output,
            contract_path=final_contract,
            contract_sha256=contract_sha256,
            mapping_paths={
                key: requested_output / value.name
                for key, value in mapping_paths.items()
            },
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-export-root", required=True, type=Path)
    parser.add_argument("--raw-source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--owner-authorization-reference", required=True)
    args = parser.parse_args(argv)
    result = build_identity_bridge(
        full_export_root=args.full_export_root,
        raw_source_root=args.raw_source_root,
        output_root=args.output_root,
        owner_authorization_reference=args.owner_authorization_reference,
    )
    print(
        json.dumps(
            {
                "output_root": str(result.output_root),
                "contract_path": str(result.contract_path),
                "contract_sha256": result.contract_sha256,
                "real_run_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(_main())
