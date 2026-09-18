#!/usr/bin/env python3
"""Reproduce the developer-prepared MIMIC-IV E3 cohort without changing a foundation export.

The materialization reads the current renal AKI implementation from raw
MIMIC-IV in bounded stay batches.  It emits a strict 0--24 hour KDIGO stage in
which an observed positive component establishes stage 1--3 and stage 0
requires complete negative creatinine, urine-output, and RRT evidence.

This tools-only recipe is not a Research Agent acquisition capability and its
output does not count as autonomous extraction in Dev9 acceptance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable

import pandas as pd

from easyicu.research_agent.acquisition.hospital_mortality_followup import (
    derive_mimic_iv_hospital_mortality_followup,
)
from tools.e3_strict_kdigo_window import (
    STRICT_KDIGO_WINDOW_SCHEMA_VERSION,
    derive_strict_kdigo_window,
    strict_kdigo_summary,
)
from easyicu.scores.aki_profiles import load_renal_aki_bundle
from easyicu.scores.kdigo_aki import load_kdigo_aki


SCHEMA_VERSION = "easyicu.e3_kdigo_gradient_materialization/1"
_SOURCE_DIRS = (
    "labevents_bucket",
    "chartevents_bucket",
)
_SOURCE_FILES = (
    "conversion_manifest.json",
    "admissions.parquet",
    "icustays.parquet",
    "outputevents.parquet",
    "procedureevents.parquet",
    "hosp/admissions.parquet",
    "icu/icustays.parquet",
    "icu/outputevents.parquet",
    "icu/procedureevents.parquet",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _regular_file(path: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be an absolute regular non-symlink file")
    return candidate.resolve(strict=True)


def _regular_dir(path: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_dir():
        raise ValueError(f"{label} must be an absolute non-symlink directory")
    return candidate.resolve(strict=True)


def _source_snapshot(raw_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for relative in _SOURCE_FILES:
        path = _regular_file(raw_root / relative, label=f"raw source {relative}")
        stat = path.stat()
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": _sha256(path),
            }
        )
    for relative in _SOURCE_DIRS:
        directory = _regular_dir(raw_root / relative, label=f"raw source {relative}")
        for path in sorted(directory.rglob("*.parquet")):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"raw source shard is not a regular file: {relative}")
            stat = path.stat()
            rows.append(
                {
                    "relative_path": path.relative_to(raw_root).as_posix(),
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
    return {
        "schema_version": "easyicu.e3_raw_source_snapshot/1",
        "database": "miiv",
        "mimic_release": "3.1",
        "files": rows,
        "snapshot_sha256": _canonical_sha256(rows),
        "large_shard_content_hashing": "conversion_manifest_and_stable_file_metadata",
    }


def _batches(values: list[int], size: int) -> Iterable[tuple[int, list[int]]]:
    for start in range(0, len(values), size):
        yield start // size, values[start : start + size]


def _batch_receipt_path(parts: Path, index: int) -> Path:
    return parts / f"part-{index:04d}.receipt.json"


def _batch_path(parts: Path, index: int) -> Path:
    return parts / f"part-{index:04d}.parquet"


def _materialize_renal_parts(
    *,
    raw_root: Path,
    stay_ids: list[int],
    parts: Path,
    batch_size: int,
    resume: bool,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    total_batches = (len(stay_ids) + batch_size - 1) // batch_size
    for index, selected in _batches(stay_ids, batch_size):
        output = _batch_path(parts, index)
        receipt_path = _batch_receipt_path(parts, index)
        selection_sha256 = _canonical_sha256(selected)
        if resume and output.is_file() and receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                receipt.get("selection_sha256") != selection_sha256
                or receipt.get("output_sha256") != _sha256(output)
                or int(receipt.get("selected_stays") or -1) != len(selected)
            ):
                raise ValueError(f"batch {index} resume receipt is invalid")
            receipts.append(receipt)
            print(
                f"[{index + 1}/{total_batches}] verified existing batch",
                flush=True,
            )
            continue
        if output.exists() or receipt_path.exists():
            raise ValueError(f"batch {index} has an incomplete prior output")
        started = time.monotonic()
        strict_profile = load_kdigo_aki(
            "miiv",
            data_path=str(raw_root),
            patient_ids=selected,
            verbose=False,
        )
        reference_profile = load_renal_aki_bundle(
            "miiv",
            data_path=str(raw_root),
            patient_ids=selected,
            verbose=False,
        )
        strict = derive_strict_kdigo_window(
            strict_profile,
            reference_profile=reference_profile,
        )
        unknown_stays = sorted(set(strict["stay_id"].astype(int)) - set(selected))
        if unknown_stays:
            raise ValueError(f"batch {index} returned out-of-scope stays")
        temporary = output.with_suffix(".parquet.partial")
        strict.to_parquet(temporary, index=False)
        temporary.replace(output)
        receipt = {
            "schema_version": "easyicu.e3_kdigo_gradient_batch/1",
            "batch_index": index,
            "selected_stays": len(selected),
            "observed_stays": int(len(strict)),
            "selection_sha256": selection_sha256,
            "output_file": output.name,
            "output_sha256": _sha256(output),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "summary": strict_kdigo_summary(strict),
        }
        _write_json(receipt_path, receipt)
        receipts.append(receipt)
        print(
            f"[{index + 1}/{total_batches}] selected={len(selected)} "
            f"observed={len(strict)} elapsed={receipt['elapsed_seconds']}s",
            flush=True,
        )
    return receipts


def _one_row_per_stay(
    path: Path,
    *,
    columns: list[str],
    label: str,
) -> pd.DataFrame:
    frame = pd.read_parquet(_regular_file(path, label=label), columns=columns)
    if frame["stay_id"].isna().any() or frame["stay_id"].duplicated().any():
        raise ValueError(f"{label} must have one non-null row per stay")
    return frame


def _charlson_24h(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        _regular_file(path, label="full6 other_scores"),
        columns=["stay_id", "charttime", "charlson"],
    )
    time_values = pd.to_numeric(frame["charttime"], errors="coerce")
    if bool((frame["charttime"].notna() & time_values.isna()).any()):
        raise ValueError("full6 other_scores charttime is non-numeric")
    frame = frame.loc[time_values.between(0, 24, inclusive="both")].copy()
    value = pd.to_numeric(frame["charlson"], errors="coerce")
    if bool((frame["charlson"].notna() & value.isna()).any()):
        raise ValueError("full6 charlson is non-numeric")
    frame["charlson"] = value
    return frame.groupby("stay_id", sort=False, observed=True)["charlson"].max().reset_index()


def _identity_frame(mapping_path: Path, mapping_sha256: str) -> pd.DataFrame:
    if _sha256(mapping_path) != mapping_sha256:
        raise ValueError("identity bridge digest mismatch")
    mapping = pd.read_parquet(mapping_path, columns=["stay_id", "patient_key"])
    if (
        mapping["stay_id"].isna().any()
        or mapping["patient_key"].isna().any()
        or mapping["stay_id"].duplicated().any()
    ):
        raise ValueError("identity bridge is incomplete or ambiguous")
    stay = mapping["stay_id"].astype("int64")
    patient = mapping["patient_key"].astype("int64")
    result = pd.DataFrame(
        {"stay_id": stay, "patient_stay_id": "p" + patient.astype(str) + ":s" + stay.astype(str)}
    )
    if result["patient_stay_id"].duplicated().any():
        raise ValueError("identity bridge does not produce unique stay identities")
    return result


def _assemble_cohort(
    *,
    raw_root: Path,
    export_root: Path,
    parts: Path,
    identity_path: Path,
    identity_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    icustays = pd.read_parquet(
        _regular_file(raw_root / "icu/icustays.parquet", label="raw icustays"),
        columns=["subject_id", "hadm_id", "stay_id", "intime"],
    )
    admissions = pd.read_parquet(
        _regular_file(raw_root / "hosp/admissions.parquet", label="raw admissions"),
        columns=["hadm_id", "dischtime", "deathtime", "hospital_expire_flag"],
    )
    followup = derive_mimic_iv_hospital_mortality_followup(icustays, admissions)
    demographics = _one_row_per_stay(
        export_root / "demographics.parquet",
        columns=["stay_id", "age", "bmi", "height", "sex", "weight", "adm"],
        label="full6 demographics",
    )
    outcomes = _one_row_per_stay(
        export_root / "outcome.parquet",
        columns=["stay_id", "los_icu", "los_hosp", "icu_readmission"],
        label="full6 outcome",
    )
    charlson = _charlson_24h(export_root / "other_scores.parquet")
    renal = pd.concat(
        [pd.read_parquet(path) for path in sorted(parts.glob("part-*.parquet"))],
        ignore_index=True,
    )
    if renal["stay_id"].duplicated().any():
        raise ValueError("strict renal batches contain duplicate stays")
    identity = _identity_frame(identity_path, identity_sha256)
    ordered = icustays.sort_values(
        ["subject_id", "intime", "stay_id"], kind="stable"
    ).copy()
    ordered["first_icu_stay"] = ~ordered["subject_id"].duplicated(keep="first")
    first_stay = ordered[["stay_id", "first_icu_stay"]]

    cohort = (
        followup.frame.rename(columns={"hospital_death": "death"})
        .merge(demographics, on="stay_id", how="left", validate="one_to_one")
        .merge(outcomes, on="stay_id", how="left", validate="one_to_one")
        .merge(charlson, on="stay_id", how="left", validate="one_to_one")
        .merge(renal, on="stay_id", how="left", validate="one_to_one")
        .merge(first_stay, on="stay_id", how="left", validate="one_to_one")
        .merge(identity, on="stay_id", how="left", validate="one_to_one")
    )
    if cohort["patient_stay_id"].isna().any():
        raise ValueError("patient identity bridge does not cover the analysis cohort")
    if cohort[["age", "sex"]].isna().any().any():
        raise ValueError("required complete demographic values are missing")
    if not cohort["death"].isin((0, 1)).all():
        raise ValueError("death is not a complete binary outcome")
    if not cohort["age"].ge(18).all():
        raise ValueError("MIMIC-IV E3 source unexpectedly contains a minor")
    cohort["aki_stage_strict"] = cohort["aki_stage_strict"].astype("Int64")
    cohort["aki_stage_reference"] = cohort["aki_stage_reference"].astype("Int64")
    cohort["first_icu_stay"] = cohort["first_icu_stay"].astype("boolean")
    cohort["aki_ascertainment"] = cohort["aki_ascertainment"].astype("string").fillna(
        "indeterminate"
    )
    cohort = cohort.sort_values("stay_id", kind="stable").reset_index(drop=True)
    # Keep the source stay key only inside the registered package so the host's
    # digest-bound private bridge can replace it with patient_stay_id during
    # materialization.  The precomputed bridge value is a coverage check, not a
    # public source column, because the replacement owner refuses overwrites.
    cohort = cohort.drop(columns=["patient_stay_id"])
    cohort.insert(0, "stay_id", cohort.pop("stay_id"))
    receipt = {
        "hospital_followup": dict(followup.receipt),
        "hospital_followup_exclusions": {
            str(key): int(value)
            for key, value in followup.exclusions["reason_code"]
            .value_counts()
            .sort_index()
            .items()
        },
        "all_rows_adult": True,
        "analysis_missingness": {
            column: int(cohort[column].isna().sum())
            for column in (
                "adm",
                "charlson",
                "los_icu",
                "los_hosp",
                "aki_stage_strict",
                "aki_stage_reference",
            )
        },
        "missingness_policy": (
            "retain_source_missingness; model-specific complete-case denominators "
            "must remain explicit"
        ),
        "patient_grouping": {
            "output_identity_column": "patient_stay_id",
            "group_derivation": "prefix_before_:s",
            "provider_visible_values": False,
            "mapping_sha256": identity_sha256,
        },
    }
    return cohort, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--full6-export-root", type=Path, required=True)
    parser.add_argument("--identity-bridge", type=Path, required=True)
    parser.add_argument("--identity-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-stays", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_stays is not None and args.max_stays < 1:
        parser.error("--max-stays must be positive")
    if len(args.identity_sha256) != 64:
        parser.error("--identity-sha256 must be a lowercase SHA-256")

    raw_root = _regular_dir(args.raw_root, label="raw MIMIC-IV root")
    export_root = _regular_dir(args.full6_export_root, label="full6 MIIV export")
    identity_path = _regular_file(args.identity_bridge, label="identity bridge")
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute() or output_dir.is_symlink():
        parser.error("--output-dir must be an absolute non-symlink path")
    if output_dir.exists() and not args.resume:
        parser.error("--output-dir already exists; use --resume to verify and continue")
    output_dir.mkdir(parents=True, exist_ok=True)
    parts = output_dir / ".parts"
    parts.mkdir(exist_ok=True)

    started_at = _utc_now()
    started = time.monotonic()
    initial_snapshot = _source_snapshot(raw_root)
    icustays = pd.read_parquet(
        raw_root / "icu/icustays.parquet", columns=["stay_id"]
    )
    if icustays["stay_id"].isna().any() or icustays["stay_id"].duplicated().any():
        raise ValueError("raw icustays must contain unique non-null stay_id values")
    stay_ids = sorted(icustays["stay_id"].astype("int64").tolist())
    if args.max_stays is not None:
        stay_ids = stay_ids[: args.max_stays]
    receipts = _materialize_renal_parts(
        raw_root=raw_root,
        stay_ids=stay_ids,
        parts=parts,
        batch_size=args.batch_size,
        resume=args.resume,
    )
    final_snapshot = _source_snapshot(raw_root)
    if final_snapshot != initial_snapshot:
        raise RuntimeError("raw MIMIC-IV source changed during materialization")

    if args.max_stays is not None:
        renal = pd.concat(
            [pd.read_parquet(path) for path in sorted(parts.glob("part-*.parquet"))],
            ignore_index=True,
        )
        cohort = renal.copy()
        assembly_receipt: dict[str, Any] = {"scope": "renal_smoke_only"}
        cohort_name = "renal_smoke.parquet"
    else:
        cohort, assembly_receipt = _assemble_cohort(
            raw_root=raw_root,
            export_root=export_root,
            parts=parts,
            identity_path=identity_path,
            identity_sha256=args.identity_sha256,
        )
        cohort_name = "cohort.parquet"

    cohort_path = output_dir / cohort_name
    temporary = cohort_path.with_suffix(".parquet.partial")
    cohort.to_parquet(temporary, index=False)
    temporary.replace(cohort_path)
    export_manifest = {
        "database": "miiv",
        "entry_mode": "study_local_prepared_cohort",
        "export_format": "parquet",
        "patient_count": int(len(cohort)),
        "selected_concepts": sorted(
            column
            for column in cohort.columns
            if column not in {"stay_id", "patient_stay_id"}
        ),
        "exported_files": [cohort_name],
        "files": [
            {
                "file": cohort_name,
                "module": "e3_kdigo_gradient",
                "rows": int(len(cohort)),
            }
        ],
        "note": (
            "Study-local strict KDIGO analysis package; not a data-foundation release."
        ),
    }
    _write_json(output_dir / "easyicu_export_manifest.json", export_manifest)
    strict_summary = strict_kdigo_summary(cohort)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "started_at": started_at,
        "status": "completed",
        "database": "miiv",
        "analysis_unit": "icu_stay",
        "scope": "study_local_e3_not_foundation_release",
        "window": {"anchor": "icu_admission", "start_hours": 0, "end_hours": 24},
        "primary_exposure": {
            "column": "aki_stage_strict",
            "schema_version": STRICT_KDIGO_WINDOW_SCHEMA_VERSION,
            "missing_policy": "retain_unknown_never_recode_to_stage_0",
            "positive_policy": "maximum_observed_positive_component_stage",
            "zero_policy": "all_components_negative_and_window_coverage_complete",
        },
        "sensitivity_exposure": {"column": "aki_stage_reference"},
        "cohort_file": cohort_name,
        "cohort_rows": int(len(cohort)),
        "cohort_columns": list(cohort.columns),
        "cohort_sha256": _sha256(cohort_path),
        "strict_kdigo": strict_summary,
        "batch_size": args.batch_size,
        "batch_count": len(receipts),
        "batch_receipts_sha256": _canonical_sha256(receipts),
        "raw_source_snapshot": initial_snapshot,
        "input_bindings": {
            "full6_export_manifest_sha256": _sha256(
                _regular_file(
                    export_root / "easyicu_export_manifest.json",
                    label="full6 export manifest",
                )
            ),
            "full6_demographics_sha256": _sha256(export_root / "demographics.parquet"),
            "full6_outcome_sha256": _sha256(export_root / "outcome.parquet"),
            "full6_other_scores_sha256": _sha256(export_root / "other_scores.parquet"),
            "identity_bridge_sha256": args.identity_sha256,
        },
        "assembly": assembly_receipt,
        "privacy": {
            "raw_patient_key_in_cohort": False,
            "raw_stay_id_in_registered_package": True,
            "provider_visible_identifier_values": False,
            "analysis_identity": "host_replaces_stay_id_with_patient_stay_id",
        },
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }
    _write_json(output_dir / "manifest.json", manifest)
    if args.max_stays is None:
        # E-P2-10: rmtree guard — parts dir must stay under output_dir and
        # never be a symlink.
        assert parts.resolve().is_relative_to(output_dir.resolve()), parts
        assert not parts.is_symlink(), parts
        shutil.rmtree(parts)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
