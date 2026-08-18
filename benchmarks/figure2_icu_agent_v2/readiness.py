"""Development-only reachability audit for the Figure 2 Held-out27 bank.

This owner answers a deliberately narrow question: can each frozen item name a
registered scientific primary contract, catalogued concepts, and columns that
exist in the immutable 2026-07-17 development data vintage?  It inspects
Parquet footers only.  It does not authorize patient-data loading or a formal
run, and it cannot issue paper authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from easyicu.research_agent.concept_catalog import load_concept_catalog
from easyicu.research_agent.planning.scientific_action_catalog import (
    scientific_actions_for_analysis_type,
)

from .protocol import (
    EXPECTED_DATABASE_COUNTS,
    BenchmarkContractError,
    load_heldout_taskbank,
    validate_experiment_bundle,
)


_FORMAL_BLOCKERS = (
    "FORMAL_NATIVE_V2_INPUT_NOT_AUTHORIZED",
    "FORMAL_CLINICAL_REVIEW_PENDING",
    "FORMAL_METHODS_REVIEW_PENDING",
    "FORMAL_ENVIRONMENT_FREEZE_PENDING",
)


@dataclass(frozen=True)
class DevelopmentSchemaAudit:
    root: str
    run_manifest_sha256: str
    database_parquet_counts: tuple[tuple[str, int], ...]
    database_columns: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class TaskDevelopmentReadiness:
    task_id: str
    database: str
    analysis_type: str
    primary_contract_id: str | None
    primary_action_execution_modes: tuple[tuple[str, str], ...]
    missing_catalog_concepts: tuple[str, ...]
    missing_development_schema_concepts: tuple[str, ...]
    contract_reachable: bool
    concepts_catalogued: bool
    development_full6_schema_observed: bool
    development_ready: bool
    formal_native_v2_input_authorized: bool
    clinical_review_status: str
    methods_review_status: str
    environment_freeze_status: str
    formal_ready: bool
    blocking_reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class DevelopmentReadinessReceipt:
    schema_version: str
    protocol_ref: str
    protocol_sha256: str
    heldout_taskbank_sha256: str
    development_data_root: str
    development_run_manifest_sha256: str
    paper_authority: bool
    task_count: int
    contract_reachable_count: int
    concepts_catalogued_count: int
    development_schema_observed_count: int
    development_ready_count: int
    formal_ready_count: int
    database_parquet_counts: tuple[tuple[str, int], ...]
    tasks: tuple[TaskDevelopmentReadiness, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regular_file(path: Path, *, maximum_bytes: int) -> Path:
    if path.is_symlink():
        raise BenchmarkContractError(
            "DEVELOPMENT_SCHEMA_PATH_INVALID",
            f"development schema authority must not be a symlink: {path}",
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise BenchmarkContractError(
            "DEVELOPMENT_SCHEMA_PATH_INVALID",
            f"development schema authority is unavailable: {path}",
        ) from exc
    if not resolved.is_file():
        raise BenchmarkContractError(
            "DEVELOPMENT_SCHEMA_PATH_INVALID",
            f"development schema authority is not a regular file: {path}",
        )
    size = resolved.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise BenchmarkContractError(
            "DEVELOPMENT_SCHEMA_PATH_INVALID",
            f"development schema authority size is outside 1..{maximum_bytes}: {path}",
        )
    return resolved


def audit_development_full6_schema(root: Path) -> DevelopmentSchemaAudit:
    """Read only Parquet footers and return the development schema identity."""

    if root.is_symlink():
        raise BenchmarkContractError(
            "DEVELOPMENT_SCHEMA_PATH_INVALID",
            f"development full6 root must not be a symlink: {root}",
        )
    try:
        resolved_root = root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise BenchmarkContractError(
            "DEVELOPMENT_SCHEMA_PATH_INVALID",
            f"development full6 root is unavailable: {root}",
        ) from exc
    if not resolved_root.is_dir():
        raise BenchmarkContractError(
            "DEVELOPMENT_SCHEMA_PATH_INVALID",
            f"development full6 root is not a directory: {root}",
        )

    manifest = _regular_file(
        resolved_root / "run_manifest.json",
        maximum_bytes=4_000_000,
    )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise BenchmarkContractError(
            "DEVELOPMENT_MANIFEST_INVALID",
            "development run_manifest.json is not readable JSON",
        ) from exc
    if not isinstance(payload, dict):
        raise BenchmarkContractError(
            "DEVELOPMENT_MANIFEST_INVALID",
            "development run_manifest.json must be a JSON object",
        )

    import pyarrow.parquet as pq

    database_columns: list[tuple[str, tuple[str, ...]]] = []
    database_counts: list[tuple[str, int]] = []
    for database in EXPECTED_DATABASE_COUNTS:
        database_root = resolved_root / database
        if database_root.is_symlink() or not database_root.is_dir():
            raise BenchmarkContractError(
                "DEVELOPMENT_DATABASE_PATH_INVALID",
                f"development database directory is unavailable: {database}",
            )
        parquet_paths = tuple(sorted(database_root.glob("*.parquet")))
        if not parquet_paths:
            raise BenchmarkContractError(
                "DEVELOPMENT_DATABASE_SCHEMA_MISSING",
                f"development database has no Parquet modules: {database}",
            )
        columns: set[str] = set()
        for path in parquet_paths:
            if path.is_symlink() or not path.is_file():
                raise BenchmarkContractError(
                    "DEVELOPMENT_SCHEMA_PATH_INVALID",
                    f"development Parquet path is not a regular file: {path}",
                )
            try:
                columns.update(pq.ParquetFile(path).schema_arrow.names)
            except Exception as exc:
                raise BenchmarkContractError(
                    "DEVELOPMENT_SCHEMA_UNREADABLE",
                    f"unable to read Parquet footer for {database}/{path.name}",
                ) from exc
        database_counts.append((database, len(parquet_paths)))
        database_columns.append((database, tuple(sorted(columns))))

    return DevelopmentSchemaAudit(
        root=str(resolved_root),
        run_manifest_sha256=_sha256(manifest),
        database_parquet_counts=tuple(database_counts),
        database_columns=tuple(database_columns),
    )


def build_development_readiness(root: Path) -> DevelopmentReadinessReceipt:
    """Compile task-level development reachability without granting authority."""

    bundle = validate_experiment_bundle()
    taskbank = load_heldout_taskbank()
    schema = audit_development_full6_schema(root)
    columns_by_database = dict(schema.database_columns)
    catalogued = set(load_concept_catalog().available_concepts)

    rows: list[TaskDevelopmentReadiness] = []
    for task in taskbank.tasks:
        action_catalog = scientific_actions_for_analysis_type(task.analysis_type)
        action_index = {action.action_id: action for action in action_catalog.actions}
        missing_primary_actions = tuple(
            action_id
            for action_id in action_catalog.required_primary_action_ids
            if action_id not in action_index
        )
        unavailable_primary_actions = tuple(
            action_id
            for action_id in action_catalog.required_primary_action_ids
            if action_id in action_index
            and action_index[action_id].execution_mode == "not_available"
        )
        primary_modes = tuple(
            (action_id, action_index[action_id].execution_mode)
            for action_id in action_catalog.required_primary_action_ids
            if action_id in action_index
        )
        missing_catalog = tuple(sorted(set(task.required_concepts) - catalogued))
        missing_schema = tuple(
            sorted(
                set(task.required_concepts)
                - set(columns_by_database.get(task.database, ()))
            )
        )
        contract_reachable = bool(
            action_catalog.primary_contract_registered
            and action_catalog.primary_contract_id
            and not missing_primary_actions
            and not unavailable_primary_actions
        )
        concepts_complete = not missing_catalog
        schema_complete = not missing_schema
        development_ready = contract_reachable and concepts_complete and schema_complete
        blockers: list[str] = []
        if not action_catalog.primary_contract_registered:
            blockers.append("SCIENTIFIC_PRIMARY_CONTRACT_UNREGISTERED")
        if missing_primary_actions:
            blockers.append("SCIENTIFIC_PRIMARY_ACTION_MISSING")
        if unavailable_primary_actions:
            blockers.append("SCIENTIFIC_PRIMARY_ACTION_UNAVAILABLE")
        if missing_catalog:
            blockers.append("REQUIRED_CONCEPT_NOT_CATALOGUED")
        if missing_schema:
            blockers.append("DEVELOPMENT_SCHEMA_CONCEPT_MISSING")
        blockers.extend(_FORMAL_BLOCKERS)
        rows.append(
            TaskDevelopmentReadiness(
                task_id=task.task_id,
                database=task.database,
                analysis_type=task.analysis_type,
                primary_contract_id=action_catalog.primary_contract_id,
                primary_action_execution_modes=primary_modes,
                missing_catalog_concepts=missing_catalog,
                missing_development_schema_concepts=missing_schema,
                contract_reachable=contract_reachable,
                concepts_catalogued=concepts_complete,
                development_full6_schema_observed=schema_complete,
                development_ready=development_ready,
                formal_native_v2_input_authorized=False,
                clinical_review_status="pending",
                methods_review_status="pending",
                environment_freeze_status="pending",
                formal_ready=False,
                blocking_reason_codes=tuple(blockers),
            )
        )

    return DevelopmentReadinessReceipt(
        schema_version="easyicu.figure2_development_readiness/1",
        protocol_ref=bundle.protocol_ref,
        protocol_sha256=bundle.protocol_sha256,
        heldout_taskbank_sha256=bundle.heldout_taskbank_sha256,
        development_data_root=schema.root,
        development_run_manifest_sha256=schema.run_manifest_sha256,
        paper_authority=False,
        task_count=len(rows),
        contract_reachable_count=sum(row.contract_reachable for row in rows),
        concepts_catalogued_count=sum(row.concepts_catalogued for row in rows),
        development_schema_observed_count=sum(
            row.development_full6_schema_observed for row in rows
        ),
        development_ready_count=sum(row.development_ready for row in rows),
        formal_ready_count=sum(row.formal_ready for row in rows),
        database_parquet_counts=schema.database_parquet_counts,
        tasks=tuple(rows),
    )


__all__ = [
    "DevelopmentReadinessReceipt",
    "DevelopmentSchemaAudit",
    "TaskDevelopmentReadiness",
    "audit_development_full6_schema",
    "build_development_readiness",
]
