#!/usr/bin/env python3
"""Materialize the exact Canonical9 MIMIC-IV inputs to an external volume.

The tool opens the verified native-v2 export once, reuses its immutable file
snapshots across all nine sequential cases, and writes every case directly to
its final external directory.  It makes no Provider, Docker, or network calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
import uuid

from benchmarks.figure2_canonical9.evaluator.suite import (
    easyicu_evaluation_protocol_suite,
)
from benchmarks.figure2_canonical9.identity_bridge_contract import (
    assess_identity_bridge_contract,
    load_identity_bridge_contract,
)
from benchmarks.figure2_canonical9.materialization_plan import (
    CANONICAL9_MIMIC_IV_PLAN,
    validate_canonical9_mimic_iv_plan,
)
from easyicu.research_agent.cohort.materializer import materialize_to_parquet
from easyicu.research_agent.intake.export_package import (
    open_export_package,
    require_column_metadata,
    resolve_exported_concept,
    verify_export_package,
)
from easyicu.research_agent.intake.materialized_metadata import (
    load_verified_materialized_cohort_authority,
    prepare_real_directory,
)
from easyicu.research_agent.intake.materialized_trajectory import (
    load_verified_materialized_trajectory_authority,
)

_RECEIPT_SCHEMA = "easyicu.canonical9_miiv_materialization/1"
_JSONL_SCHEMA = "easyicu.canonical9_ehrflowbench_jsonl/1"


def _canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _atomic_write(path: Path, raw: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_external_output(path: Path, *, resume_existing: bool) -> Path:
    absolute = path.expanduser()
    if not absolute.is_absolute():
        raise ValueError("--output-root must be absolute")
    home = Path.home().resolve()
    resolved_parent = absolute.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(home)
    except ValueError:
        pass
    else:
        raise ValueError(
            "--output-root must not be inside the local home directory; use the "
            "mounted external volume"
        )
    if absolute.exists():
        if not resume_existing:
            raise FileExistsError("--output-root must be fresh and absent")
        if absolute.is_symlink() or not absolute.is_dir():
            raise ValueError("resumed --output-root must be a real directory")
        return absolute.resolve(strict=True)
    if resume_existing:
        raise FileNotFoundError("--resume-existing requires an existing output root")
    return prepare_real_directory(
        absolute, label="Canonical9 external materialization root"
    )


def _require_external_temp() -> None:
    temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
    try:
        temp_root.relative_to(Path.home().resolve())
    except ValueError:
        return
    raise ValueError(
        "TMPDIR resolves inside the local home directory; point TMPDIR at the "
        "mounted external volume before materialization"
    )


def _load_bridge(
    *,
    contract_path: Path,
    mapping_path: Path,
) -> tuple[Mapping[str, Any], str]:
    contract, contract_sha256 = load_identity_bridge_contract(contract_path)
    readiness = assess_identity_bridge_contract(
        contract, contract_sha256=contract_sha256
    )
    if not readiness.eligible_for_native_materialization_review:
        raise PermissionError("identity bridge is not authorized for materialization")
    mapping = next(
        item for item in contract.source_mappings if item.source_id == "mimic_iv"
    )
    if mapping.unmapped_stay_count != 0 or mapping.duplicate_stay_count != 0:
        raise ValueError("MIMIC-IV identity bridge is incomplete")
    if not mapping_path.is_absolute() or mapping_path.is_symlink():
        raise ValueError("identity mapping must be an absolute non-symlink file")
    info = mapping_path.stat()
    if (
        int(info.st_size) != mapping.projection.artifact_size_bytes
        or _sha256_file(mapping_path) != mapping.projection.artifact_sha256
    ):
        raise ValueError("identity mapping does not match its contract projection")
    coordinates = {
        "bridge_ref": contract.bridge_ref,
        "bridge_contract_sha256": contract_sha256,
        "data_lane_authorization_reference": (
            contract.data_lane.authorization_reference
        ),
        "mapping_semantics": mapping.mapping_semantics,
        "source_semantics_attestation_sha256": (
            mapping.source_semantics_attestation_sha256
        ),
        "source_snapshot_sha256": mapping.projection.source_snapshot_sha256,
        "relation_schema_sha256": mapping.projection.relation_schema_sha256,
    }
    return {
        "mapping_path": mapping_path,
        "mapping_sha256": mapping.projection.artifact_sha256,
        "stay_column": mapping.stay_key,
        "patient_column": "patient_key",
        "coordinates": coordinates,
    }, contract_sha256


def _build_jsonl_row(
    *,
    task: object,
    spec: object,
    case_dir: Path,
    cohort_path: Path,
    cohort_verified: object,
    trajectory_path: Path | None,
    trajectory_verified: object | None,
) -> dict[str, object]:
    authority_ref = cohort_verified.reference
    row: dict[str, object] = {
        "schema_version": _JSONL_SCHEMA,
        "key": task.task_id,
        "name": task.title,
        "question": task.objective,
        "database": "miiv",
        "target_outcome": "death",
        "primary_predictor": spec.operational_exposure,
        "operational_exposure": spec.operational_exposure,
        "kind": task.kind,
        "difficulty": task.difficulty,
        "category": task.category,
        "expected_outputs": list(task.expected_outputs),
        "semantic_guardrails": list(task.semantic_guardrails),
        "evaluation_notes": list(task.evaluation_notes),
        "target_databases": list(task.target_databases),
        "gold_answer_status": task.gold_answer_status,
        "benchmark_family": "easyicu_figure2_canonical9",
        "evidence_basis": "native_typed_mimic_iv_materialization",
        "claim_scope": "owner_authorized_development_until_final_freeze",
        "protocol_version": "easyicu_evaluation_protocol_suite/v2",
        "rubric_version": "easyicu.figure2_paper_rubric/20260719-v3",
        "cohort_path": str(cohort_path),
        "cohort_authority_required": True,
        "cohort_authority_path": str(case_dir / authority_ref.file),
        "cohort_authority_ref": authority_ref.to_dict(),
        "id_columns": [
            (
                "patient_stay_id"
                if spec.identity_mode == "patient_grouped_stay"
                else "stay_id"
            )
        ],
        "candidate_variables": list(spec.feature_concepts),
        "notes": spec.notes,
    }
    if trajectory_path is not None and trajectory_verified is not None:
        trajectory_ref = trajectory_verified.reference
        row.update(
            {
                "trajectory_path": str(trajectory_path),
                "trajectory_authority_required": True,
                "trajectory_authority_path": str(case_dir / trajectory_ref.file),
                "trajectory_authority_ref": trajectory_ref.to_dict(),
            }
        )
    return row


def materialize(args: argparse.Namespace) -> Path:
    validate_canonical9_mimic_iv_plan()
    _require_external_temp()
    export_root = args.export_root.expanduser().resolve(strict=True)
    output_root = _require_external_output(
        args.output_root,
        resume_existing=bool(args.resume_existing),
    )
    bridge, bridge_contract_sha256 = _load_bridge(
        contract_path=args.identity_bridge_contract.expanduser().resolve(strict=True),
        mapping_path=args.identity_mapping.expanduser().resolve(strict=True),
    )
    tasks = easyicu_evaluation_protocol_suite().tasks
    task_by_id = {task.task_id: task for task in tasks}
    rows: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []

    with open_export_package(export_root) as package:
        require_column_metadata(package)
        if package.database != "miiv" or package.manifest_kind != "native":
            raise ValueError(
                "Canonical9 materializer requires a native MIMIC-IV export"
            )
        if package.source_seal_kind is not None:
            raise ValueError("development-sealed exports cannot become paper inputs")
        requested = {
            concept
            for spec in CANONICAL9_MIMIC_IV_PLAN
            for concept in (
                *spec.feature_concepts,
                *spec.static_concepts,
                *spec.outcome_concepts,
                *spec.trajectory_concepts,
            )
        }
        missing = sorted(
            concept
            for concept in requested
            if resolve_exported_concept(package.concept_index, concept) is None
        )
        if missing:
            raise ValueError(f"typed export is missing concepts: {missing}")

        for index, spec in enumerate(CANONICAL9_MIMIC_IV_PLAN, start=1):
            task = task_by_id[spec.task_id]
            case_dir = prepare_real_directory(
                output_root / spec.task_id,
                label=f"{spec.task_id} materialization directory",
            )
            cohort_candidate = case_dir / "cohort.parquet"
            if args.resume_existing and cohort_candidate.is_file():
                print(f"[{index}/9] verifying existing {spec.task_id}", flush=True)
                cohort_path = cohort_candidate.resolve(strict=True)
            else:
                if any(case_dir.iterdir()):
                    raise RuntimeError(
                        f"{spec.task_id}: incomplete case directory is not empty"
                    )
                print(f"[{index}/9] materializing {spec.task_id}", flush=True)
                identity_options: dict[str, object] = {}
                if spec.identity_mode == "patient_grouped_stay":
                    identity_options = {
                        "replacement_identity_path": bridge["mapping_path"],
                        "replacement_identity_sha256": bridge["mapping_sha256"],
                        "replacement_identity_stay_column": bridge["stay_column"],
                        "replacement_identity_patient_column": bridge["patient_column"],
                        "output_identity_column": "patient_stay_id",
                        "identity_authority_coordinates": bridge["coordinates"],
                    }
                paths = materialize_to_parquet(
                    case_dir,
                    stem="cohort",
                    emit_trajectory=spec.emit_trajectory,
                    trajectory_concepts=spec.trajectory_concepts or None,
                    trajectory_window=spec.trajectory_window,
                    source_package=package,
                    feature_concepts=spec.feature_concepts,
                    database="miiv",
                    data_path=export_root,
                    cohort_definition=None,
                    cohort_window=(0.0, 24.0),
                    outcome_concepts=spec.outcome_concepts,
                    static_concepts=spec.static_concepts,
                    patient_ids=None,
                    prefer_existing=True,
                    bounds_violation_policy="exclude_with_receipt",
                    positive_only_event_concepts=(spec.positive_only_event_concepts),
                    **identity_options,
                )
                cohort_path = paths["parquet"].resolve(strict=True)
            cohort_verified = load_verified_materialized_cohort_authority(cohort_path)
            if cohort_verified is None:
                raise RuntimeError(f"{spec.task_id}: typed cohort authority missing")
            trajectory_path: Path | None = None
            trajectory_verified = None
            if spec.emit_trajectory:
                trajectory_path = (case_dir / "cohort_trajectory.parquet").resolve(
                    strict=True
                )
                trajectory_verified = load_verified_materialized_trajectory_authority(
                    trajectory_path,
                    expected_universe_authority=cohort_verified.reference,
                )
                if trajectory_verified is None:
                    raise RuntimeError(
                        f"{spec.task_id}: typed trajectory authority missing"
                    )
            row = _build_jsonl_row(
                task=task,
                spec=spec,
                case_dir=case_dir,
                cohort_path=cohort_path,
                cohort_verified=cohort_verified,
                trajectory_path=trajectory_path,
                trajectory_verified=trajectory_verified,
            )
            rows.append(row)
            cases.append(
                {
                    "task_id": spec.task_id,
                    "cohort_rows": cohort_verified.authority.cohort_rows,
                    "cohort_columns": list(cohort_verified.authority.cohort_columns),
                    "cohort_authority_ref": cohort_verified.reference.to_dict(),
                    "trajectory_authority_ref": (
                        trajectory_verified.reference.to_dict()
                        if trajectory_verified is not None
                        else None
                    ),
                }
            )
        verify_export_package(package)
        export_authority_sha256 = package.authority_sha256
        export_manifest_sha256 = package.manifest_sha256

    jsonl_path = output_root / "canonical9_miiv.jsonl"
    jsonl_raw = b"".join(_canonical_json_bytes(row, newline=True) for row in rows)
    _atomic_write(jsonl_path, jsonl_raw)
    receipt = {
        "schema_version": _RECEIPT_SCHEMA,
        "paper_authority": False,
        "status": "materialized_awaiting_scientific_identity_and_operator_freeze",
        "export_root": str(export_root),
        "export_manifest_sha256": export_manifest_sha256,
        "export_authority_sha256": export_authority_sha256,
        "identity_bridge_contract_sha256": bridge_contract_sha256,
        "identity_mapping_sha256": bridge["mapping_sha256"],
        "ehrflowbench_jsonl_path": str(jsonl_path),
        "ehrflowbench_jsonl_sha256": hashlib.sha256(jsonl_raw).hexdigest(),
        "cases": cases,
    }
    _atomic_write(
        output_root / "materialization_receipt.json",
        _canonical_json_bytes(receipt),
    )
    return output_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--identity-bridge-contract", type=Path, required=True)
    parser.add_argument("--identity-mapping", type=Path, required=True)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Verify completed cases in an existing root and continue only empty cases.",
    )
    return parser


def main() -> int:
    root = materialize(_parser().parse_args())
    print(f"Canonical9 materialization complete: {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
