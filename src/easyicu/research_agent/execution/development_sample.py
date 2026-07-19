"""Deterministic post-cohort sampling for non-paper development runs.

The Planner first owns and locks the scientific cohort definition.  This
module may then take an exact row subset of that already-materialized analysis
cohort to reduce development latency.  It never chooses an exposure, outcome,
method, estimand, stratum, or inclusion rule.

Every sample is explicitly non-paper authority.  The parent and child bytes,
row positions, seed, and any filtered trajectory are sealed in a manifest so
resume cannot silently select a different development population.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..intake.materialized_metadata import (
    MaterializedCohortAuthorityRef,
    VerifiedMaterializedCohortAuthority,
    implementation_bundle_sha256,
    load_verified_materialized_cohort_authority,
    publish_ordered_subset_materialized_cohort,
)
from ..intake.materialized_trajectory import (
    StagedTrajectoryBinding,
    load_verified_materialized_trajectory_authority,
    publish_materialized_trajectory_authority,
)
from ..schema import ValidationFinding

DEVELOPMENT_SAMPLE_SCHEMA = "easyicu.development_execution_sample/1"
DEVELOPMENT_SAMPLE_FILENAME = "development_execution_sample.json"
DEVELOPMENT_COHORT_FILENAME = "cohort_analysis_development_sample.parquet"
DEVELOPMENT_TRAJECTORY_FILENAME = "cohort_trajectory_development_sample.parquet"


class DevelopmentSampleError(RuntimeError):
    """A post-cohort development sample is missing or cannot be verified."""


@dataclass(frozen=True, slots=True)
class DevelopmentSampleBinding:
    cohort_path: Path
    manifest_path: Path
    parent_cohort_path: Path
    target_rows: int
    selected_rows: int
    seed: int
    sample_sha256: str
    sample_size: int
    parent_sha256: str
    parent_size: int
    identity_column: str
    selected_positions_sha256: str
    cohort_authority_ref: Optional[MaterializedCohortAuthorityRef]
    trajectory_binding: Optional[StagedTrajectoryBinding]
    trajectory_bound_cohort_authority_ref: Optional[MaterializedCohortAuthorityRef] = (
        None
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _stable_identity_bytes(value: object) -> bytes:
    try:
        return _canonical_bytes(value)
    except (TypeError, ValueError) as exc:
        raise DevelopmentSampleError(
            "development-sample identities must be canonical JSON values"
        ) from exc


def _selected_positions(
    identities: Sequence[object], *, target_rows: int, seed: int
) -> tuple[int, ...]:
    if target_rows <= 0:
        raise DevelopmentSampleError("development sample size must be positive")
    if any(value is None for value in identities):
        raise DevelopmentSampleError(
            "development sampling requires non-null row identities"
        )
    canonical = tuple(_stable_identity_bytes(value) for value in identities)
    if len(canonical) != len(set(canonical)):
        raise DevelopmentSampleError(
            "development sampling requires unique row identities"
        )
    seed_bytes = str(int(seed)).encode("ascii")
    ranked = sorted(
        range(len(canonical)),
        key=lambda position: (
            hashlib.sha256(seed_bytes + b"\0" + canonical[position]).digest(),
            canonical[position],
        ),
    )
    return tuple(sorted(ranked[: min(target_rows, len(ranked))]))


def _positions_sha256(positions: Sequence[int]) -> str:
    return _canonical_sha256(list(positions))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists():
        raise DevelopmentSampleError(
            f"development sample target already exists: {path.name}"
        )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _atomic_write_table(path: Path, table: pa.Table) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists():
        raise DevelopmentSampleError(
            f"development sample target already exists: {path.name}"
        )
    try:
        pq.write_table(table, temporary, compression="zstd")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _identity_column(
    table: pa.Table,
    *,
    typed_parent: Optional[VerifiedMaterializedCohortAuthority],
    declared_id_columns: Sequence[str],
) -> str:
    candidates: list[str] = []
    if typed_parent is not None:
        candidates.append(typed_parent.authority.identity_column)
    candidates.extend(str(value) for value in declared_id_columns)
    candidates.extend(("stay_id", "patientunitstayid", "icustay_id"))
    for name in candidates:
        if name in table.column_names:
            return name
    raise DevelopmentSampleError(
        "post-cohort development sampling requires one explicit identity column"
    )


def _filter_trajectory(
    source: Path,
    target: Path,
    *,
    selected_identities: Sequence[object],
    identity_column: str,
) -> int:
    parquet = pq.ParquetFile(source)
    if identity_column not in parquet.schema_arrow.names:
        raise DevelopmentSampleError(
            f"trajectory lacks sampled identity column {identity_column!r}"
        )
    identity_type = parquet.schema_arrow.field(identity_column).type
    values = pa.array(list(selected_identities), type=identity_type)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if target.exists() or target.is_symlink() or temporary.exists():
        raise DevelopmentSampleError(
            f"development trajectory target already exists: {target.name}"
        )
    rows = 0
    try:
        with pq.ParquetWriter(
            temporary,
            parquet.schema_arrow,
            compression="zstd",
        ) as writer:
            for batch in parquet.iter_batches(batch_size=250_000):
                mask = pc.is_in(batch.column(identity_column), value_set=values)
                selected = batch.filter(mask)
                if selected.num_rows:
                    writer.write_batch(selected)
                    rows += selected.num_rows
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return rows


def _manifest_binding(
    *, run_dir: Path, payload: Mapping[str, Any]
) -> DevelopmentSampleBinding:
    manifest_path = run_dir / DEVELOPMENT_SAMPLE_FILENAME
    cohort_path = run_dir / DEVELOPMENT_COHORT_FILENAME
    expected_keys = {
        "schema",
        "paper_authority",
        "algorithm",
        "seed",
        "target_rows",
        "parent",
        "sample",
        "trajectory",
    }
    if set(payload) != expected_keys:
        raise DevelopmentSampleError("development sample manifest schema mismatch")
    if (
        payload.get("schema") != DEVELOPMENT_SAMPLE_SCHEMA
        or payload.get("paper_authority") is not False
        or payload.get("algorithm") != "sha256_identity_rank_v1"
    ):
        raise DevelopmentSampleError("development sample manifest is not canonical")
    parent = payload.get("parent")
    sample = payload.get("sample")
    trajectory = payload.get("trajectory")
    if not isinstance(parent, Mapping) or not isinstance(sample, Mapping):
        raise DevelopmentSampleError("development sample manifest is incomplete")
    if (
        parent.get("file") != "cohort_analysis.parquet"
        or sample.get("file") != DEVELOPMENT_COHORT_FILENAME
        or cohort_path.is_symlink()
        or not cohort_path.is_file()
        or _sha256_file(cohort_path) != sample.get("sha256")
        or int(cohort_path.stat().st_size) != sample.get("size")
    ):
        raise DevelopmentSampleError("development sample bytes do not match manifest")
    typed = load_verified_materialized_cohort_authority(cohort_path)
    raw_authority = sample.get("authority_ref")
    if raw_authority is None:
        if typed is not None:
            raise DevelopmentSampleError("untyped sample unexpectedly gained authority")
        authority_ref = None
    else:
        if not isinstance(raw_authority, Mapping):
            raise DevelopmentSampleError("sample authority reference is invalid")
        authority_ref = MaterializedCohortAuthorityRef.from_dict(raw_authority)
        if typed is None or typed.reference != authority_ref:
            raise DevelopmentSampleError("sample authority reference changed")
    trajectory_binding = None
    trajectory_bound_ref = None
    if trajectory is not None:
        if not isinstance(trajectory, Mapping):
            raise DevelopmentSampleError("development trajectory manifest is invalid")
        trajectory_path = run_dir / DEVELOPMENT_TRAJECTORY_FILENAME
        if (
            trajectory.get("file") != DEVELOPMENT_TRAJECTORY_FILENAME
            or trajectory_path.is_symlink()
            or not trajectory_path.is_file()
            or _sha256_file(trajectory_path) != trajectory.get("sha256")
            or int(trajectory_path.stat().st_size) != trajectory.get("size")
        ):
            raise DevelopmentSampleError(
                "development trajectory bytes do not match manifest"
            )
        raw_trajectory_authority = trajectory.get("authority_ref")
        trajectory_authority_ref = None
        if raw_trajectory_authority is not None:
            from ..intake.materialized_trajectory import (
                MaterializedTrajectoryAuthorityRef,
            )

            if not isinstance(raw_trajectory_authority, Mapping):
                raise DevelopmentSampleError(
                    "development trajectory authority is invalid"
                )
            trajectory_authority_ref = MaterializedTrajectoryAuthorityRef.from_dict(
                raw_trajectory_authority
            )
            raw_bound = trajectory.get("bound_cohort_authority_ref")
            if not isinstance(raw_bound, Mapping):
                raise DevelopmentSampleError(
                    "development trajectory lost its cohort authority"
                )
            trajectory_bound_ref = MaterializedCohortAuthorityRef.from_dict(raw_bound)
            verified = load_verified_materialized_trajectory_authority(
                trajectory_path,
                expected_authority=trajectory_authority_ref,
                expected_universe_authority=trajectory_bound_ref,
            )
            if verified is None:
                raise DevelopmentSampleError(
                    "development trajectory lost its typed authority"
                )
        trajectory_binding = StagedTrajectoryBinding(
            path=trajectory_path,
            sha256=str(trajectory["sha256"]),
            size=int(trajectory["size"]),
            authority_ref=trajectory_authority_ref,
        )
    return DevelopmentSampleBinding(
        cohort_path=cohort_path,
        manifest_path=manifest_path,
        parent_cohort_path=run_dir / "cohort_analysis.parquet",
        target_rows=int(payload["target_rows"]),
        selected_rows=int(sample["rows"]),
        seed=int(payload["seed"]),
        sample_sha256=str(sample["sha256"]),
        sample_size=int(sample["size"]),
        parent_sha256=str(parent["sha256"]),
        parent_size=int(parent["size"]),
        identity_column=str(sample["identity_column"]),
        selected_positions_sha256=str(sample["selected_positions_sha256"]),
        cohort_authority_ref=authority_ref,
        trajectory_binding=trajectory_binding,
        trajectory_bound_cohort_authority_ref=trajectory_bound_ref,
    )


def materialize_development_execution_sample(
    *,
    run_dir: Path,
    target_rows: int,
    seed: int,
    declared_id_columns: Sequence[str],
    trajectory_binding: Optional[StagedTrajectoryBinding],
) -> DevelopmentSampleBinding:
    """Sample the locked analysis cohort and optionally its trajectory.

    The function is idempotent for resume: an existing manifest is fully
    verified and returned.  Partial or mismatched state fails closed.
    """

    run_dir = Path(run_dir)
    parent_path = run_dir / "cohort_analysis.parquet"
    manifest_path = run_dir / DEVELOPMENT_SAMPLE_FILENAME
    if manifest_path.exists() or manifest_path.is_symlink():
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise DevelopmentSampleError("development sample manifest is unsafe")
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DevelopmentSampleError(
                "development sample manifest is unreadable"
            ) from exc
        binding = _manifest_binding(run_dir=run_dir, payload=payload)
        if binding.target_rows != int(target_rows) or binding.seed != int(seed):
            raise DevelopmentSampleError(
                "development sample configuration changed across resume"
            )
        if (
            not parent_path.is_file()
            or parent_path.is_symlink()
            or _sha256_file(parent_path) != binding.parent_sha256
            or int(parent_path.stat().st_size) != binding.parent_size
        ):
            raise DevelopmentSampleError(
                "locked analysis cohort changed across development resume"
            )
        parent_table = pq.read_table(parent_path)
        if binding.identity_column not in parent_table.column_names:
            raise DevelopmentSampleError(
                "development sample identity disappeared from its locked parent"
            )
        expected_positions = _selected_positions(
            parent_table.column(binding.identity_column)
            .combine_chunks()
            .to_pylist(),
            target_rows=binding.target_rows,
            seed=binding.seed,
        )
        if (
            _positions_sha256(expected_positions)
            != binding.selected_positions_sha256
        ):
            raise DevelopmentSampleError(
                "development sample row selection changed across resume"
            )
        expected_sample = parent_table.take(
            pa.array(expected_positions, type=pa.int64())
        )
        actual_sample = pq.read_table(binding.cohort_path)
        if (
            binding.selected_rows != len(expected_positions)
            or not actual_sample.equals(expected_sample)
        ):
            raise DevelopmentSampleError(
                "development sample is not the deterministic parent subset"
            )
        return binding
    if not parent_path.is_file() or parent_path.is_symlink():
        raise DevelopmentSampleError(
            "development sampling requires a locked, materialized analysis cohort; "
            "the Planner/QC phase did not produce cohort_analysis.parquet"
        )
    cohort_path = run_dir / DEVELOPMENT_COHORT_FILENAME
    trajectory_path = run_dir / DEVELOPMENT_TRAJECTORY_FILENAME
    if cohort_path.exists() or trajectory_path.exists():
        raise DevelopmentSampleError(
            "partial development sample exists without a committed manifest"
        )
    typed_parent = load_verified_materialized_cohort_authority(parent_path)
    table = pq.read_table(parent_path)
    identity_column = _identity_column(
        table,
        typed_parent=typed_parent,
        declared_id_columns=declared_id_columns,
    )
    identities = table.column(identity_column).combine_chunks().to_pylist()
    positions = _selected_positions(
        identities,
        target_rows=int(target_rows),
        seed=int(seed),
    )
    selected_ids = [identities[position] for position in positions]
    parent_sha = _sha256_file(parent_path)
    parent_size = int(parent_path.stat().st_size)
    positions_sha = _positions_sha256(positions)
    sample_definition = {
        "schema": DEVELOPMENT_SAMPLE_SCHEMA,
        "algorithm": "sha256_identity_rank_v1",
        "seed": int(seed),
        "target_rows": int(target_rows),
        "paper_authority": False,
    }
    sample_definition_sha = _canonical_sha256(sample_definition)
    cohort_authority_ref = None
    if typed_parent is None:
        _atomic_write_table(
            cohort_path,
            table.take(pa.array(positions, type=pa.int64())),
        )
    else:
        implementation_paths = (
            Path(__file__).resolve(),
            Path(__file__).resolve().parents[1] / "intake" / "materialized_metadata.py",
        )
        typed_child = publish_ordered_subset_materialized_cohort(
            parent_path,
            cohort_path,
            selected_row_positions=positions,
            semantic_provenance={
                "cohort_sha256": sample_definition_sha,
                "cohort_definition": sample_definition,
                "predicate_column_bindings": {},
                "n_universe": int(table.num_rows),
                "n_analysis_cohort": len(positions),
                "paper_authority": False,
            },
            producer_implementation_sha256=implementation_bundle_sha256(
                implementation_paths
            ),
            producer_parameters={
                "cohort_definition_sha256": sample_definition_sha,
                "cohort_definition": sample_definition,
                "predicate_column_bindings": {},
            },
            expected_parent_authority=typed_parent.reference,
        )
        if typed_child is None:  # pragma: no cover - typed parent required above
            raise DevelopmentSampleError("typed development sample lost authority")
        cohort_authority_ref = typed_child.reference
    sample_sha = _sha256_file(cohort_path)
    sample_size = int(cohort_path.stat().st_size)

    sampled_trajectory_binding = None
    trajectory_manifest = None
    trajectory_bound_ref = None
    if trajectory_binding is not None:
        trajectory_rows = _filter_trajectory(
            trajectory_binding.path,
            trajectory_path,
            selected_identities=selected_ids,
            identity_column=identity_column,
        )
        trajectory_authority_ref = None
        if trajectory_binding.authority_ref is not None:
            if cohort_authority_ref is None:
                raise DevelopmentSampleError(
                    "typed trajectory cannot bind an untyped development cohort"
                )
            typed_sample = load_verified_materialized_cohort_authority(
                cohort_path, expected_authority=cohort_authority_ref
            )
            source_trajectory = load_verified_materialized_trajectory_authority(
                trajectory_binding.path,
                expected_authority=trajectory_binding.authority_ref,
            )
            if typed_sample is None or source_trajectory is None:
                raise DevelopmentSampleError(
                    "typed development inputs lost their selected authority"
                )
            filtered_frame = pq.read_table(trajectory_path).to_pandas()
            trajectory_path.unlink()
            window = source_trajectory.authority.window
            source = source_trajectory.authority
            published = publish_materialized_trajectory_authority(
                filtered_frame,
                trajectory_path,
                bound_universe_path=cohort_path,
                bound_universe=typed_sample,
                requested_concepts=source.requested_concepts,
                materialized_concepts=source.materialized_concepts,
                available_unobserved_concepts=(source.available_unobserved_concepts),
                unavailable_concepts=source.unavailable_concepts,
                window=(
                    (window.start_hours, window.end_hours)
                    if window is not None
                    else None
                ),
                semantic_provenance={
                    "n_rows": int(trajectory_rows),
                    "n_stays": len(set(filtered_frame["stay_id"].tolist())),
                    "trajectory_concepts_materialized": list(
                        source.materialized_concepts
                    ),
                    "available_unobserved_concepts": list(
                        source.available_unobserved_concepts
                    ),
                    "unavailable_concepts": list(source.unavailable_concepts),
                    "paper_authority": False,
                },
                producer_implementation_sha256=implementation_bundle_sha256(
                    (
                        Path(__file__).resolve(),
                        Path(__file__).resolve().parents[1]
                        / "intake"
                        / "materialized_trajectory.py",
                    )
                ),
                producer_parameters={
                    "database": typed_sample.sidecar.source_database,
                    "requested_concepts": list(source.requested_concepts),
                    "materialized_concepts": list(source.materialized_concepts),
                    "available_unobserved_concepts": list(
                        source.available_unobserved_concepts
                    ),
                    "unavailable_concepts": list(source.unavailable_concepts),
                    "window": (
                        [window.start_hours, window.end_hours]
                        if window is not None
                        else None
                    ),
                    "bound_universe_authority_sha256": typed_sample.reference.sha256,
                },
            )
            trajectory_authority_ref = published.reference
            trajectory_bound_ref = typed_sample.reference
        trajectory_sha = _sha256_file(trajectory_path)
        trajectory_size = int(trajectory_path.stat().st_size)
        sampled_trajectory_binding = StagedTrajectoryBinding(
            path=trajectory_path,
            sha256=trajectory_sha,
            size=trajectory_size,
            authority_ref=trajectory_authority_ref,
        )
        trajectory_manifest = {
            "file": DEVELOPMENT_TRAJECTORY_FILENAME,
            "sha256": trajectory_sha,
            "size": trajectory_size,
            "rows": int(trajectory_rows),
            "authority_ref": (
                trajectory_authority_ref.to_dict()
                if trajectory_authority_ref is not None
                else None
            ),
            "bound_cohort_authority_ref": (
                trajectory_bound_ref.to_dict()
                if trajectory_bound_ref is not None
                else None
            ),
        }
    manifest = {
        "schema": DEVELOPMENT_SAMPLE_SCHEMA,
        "paper_authority": False,
        "algorithm": "sha256_identity_rank_v1",
        "seed": int(seed),
        "target_rows": int(target_rows),
        "parent": {
            "file": parent_path.name,
            "sha256": parent_sha,
            "size": parent_size,
            "rows": int(table.num_rows),
            "authority_ref": (
                typed_parent.reference.to_dict() if typed_parent is not None else None
            ),
        },
        "sample": {
            "file": DEVELOPMENT_COHORT_FILENAME,
            "sha256": sample_sha,
            "size": sample_size,
            "rows": len(positions),
            "identity_column": identity_column,
            "selected_positions_sha256": positions_sha,
            "authority_ref": (
                cohort_authority_ref.to_dict()
                if cohort_authority_ref is not None
                else None
            ),
        },
        "trajectory": trajectory_manifest,
    }
    _atomic_write_bytes(manifest_path, _canonical_bytes(manifest) + b"\n")
    return _manifest_binding(run_dir=run_dir, payload=manifest)


def record_development_sample_authority(
    *,
    binding: DevelopmentSampleBinding,
    evidence: Any,
    findings: list[ValidationFinding],
    emit_progress: Any,
    run_id: str,
) -> None:
    """Register one non-paper sample without adding orchestration closure state."""

    if evidence.get("development_execution_sample") is None:
        evidence.register_file(
            kind="log",
            description=(
                "Non-paper post-QC development sample authority. The Planner "
                "locked and materialized the full analysis cohort before this "
                "deterministic execution subset."
            ),
            source_path=binding.manifest_path,
            evidence_id="development_execution_sample",
            producer="runtime_supervisor",
            generation_mode="system",
            metadata={
                "paper_authority": False,
                "parent_cohort_sha256": binding.parent_sha256,
                "sample_cohort_sha256": binding.sample_sha256,
                "target_rows": binding.target_rows,
                "selected_rows": binding.selected_rows,
                "seed": binding.seed,
            },
        )
    findings[:] = [
        finding
        for finding in findings
        if finding.validator != "development_sample_authority"
    ]
    findings.append(
        ValidationFinding(
            validator="development_sample_authority",
            severity="error",
            message=(
                "This run executes a deterministic post-QC development sample "
                "and is not paper authority. It may validate workflow capability "
                "and runtime performance, but its results cannot enter a "
                "canonical scorecard or manuscript."
            ),
            detail={
                "paper_authority": False,
                "stage": "after_locked_cohort_materialization_and_qc",
                "target_rows": binding.target_rows,
                "selected_rows": binding.selected_rows,
                "seed": binding.seed,
                "manifest": binding.manifest_path.name,
            },
        )
    )
    emit_progress(
        "cohort",
        (
            "Selected deterministic post-QC development sample: "
            f"n={binding.selected_rows} of locked analysis cohort."
        ),
        run_id=run_id,
        paper_authority=False,
        path=str(binding.cohort_path),
    )


__all__ = [
    "DEVELOPMENT_COHORT_FILENAME",
    "DEVELOPMENT_SAMPLE_FILENAME",
    "DEVELOPMENT_SAMPLE_SCHEMA",
    "DEVELOPMENT_TRAJECTORY_FILENAME",
    "DevelopmentSampleBinding",
    "DevelopmentSampleError",
    "materialize_development_execution_sample",
    "record_development_sample_authority",
]
