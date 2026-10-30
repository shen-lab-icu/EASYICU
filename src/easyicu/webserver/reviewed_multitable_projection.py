"""Host-owned partial Parquet projections of an exactly reviewed file roster.

This fixed operator copies a named roster in full and emits the selected columns
of every row of each file, preserving file-local row order. It neither joins
files nor constructs a cohort, computes a feature, or authorizes analysis.
The caller must have recorded an actual host approval of the exact scope.
"""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Callable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from easyicu.research_agent.authority.filesystem import AnchoredDirectory
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_json, canonical_sha256
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import sources
from easyicu.webserver.raw_case_snapshot import stage_reviewed_source_files, verify_staged_source_files
from easyicu.webserver.raw_preparation_attestation import implementation_identity, publish_attestation, verify_attestation
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding, consume_host_review, verify_host_review


class _ParquetProjectionImplementation:
    pass


def _scope_shape(scope: Mapping[str, Any]) -> dict:
    """Validate a scope without looking up or reading the original source."""
    checked = json.loads(canonical_json(dict(scope)))
    if (set(checked) != {"adapter_id", "source_identity", "inputs", "purpose", "plan_sha256", "dependencies_sha256"}
            or checked["adapter_id"] not in {"registered-parquet-projections/1", "registered-parquet-projections/2"}
            or checked["purpose"] != "prepare_multiple_tables_only"
            or any(not isinstance(checked[name], str) or not re.fullmatch(r"[0-9a-f]{64}", checked[name])
                   for name in ("plan_sha256", "dependencies_sha256"))):
        raise ValueError("multitable_projection_scope_invalid")
    identity = SourceIdentity.from_record(checked["source_identity"])
    if identity is None or identity.to_dict() != checked["source_identity"]:
        raise ValueError("multitable_projection_source_invalid")
    entries = checked["inputs"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 1000:
        raise ValueError("multitable_projection_roster_invalid")
    paths = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"table", "path", "columns"}:
            raise ValueError("multitable_projection_entry_invalid")
        table, name, columns = entry["table"], entry["path"], entry["columns"]
        path = PurePosixPath(name) if isinstance(name, str) else PurePosixPath()
        if (not isinstance(table, str) or not table.strip() or len(table) > 160
                or not isinstance(name, str) or not name or path.is_absolute() or path.as_posix() != name
                or any(part in {".", ".."} for part in path.parts) or "\\" in name or "\x00" in name
                or path.suffix.lower() != ".parquet" or name in paths):
            raise ValueError("multitable_projection_file_invalid")
        paths.add(name)
        if (not isinstance(columns, list) or not columns or len(columns) > 512
                or any(not isinstance(col, str) or not col or len(col) > 256 for col in columns)
                or len(columns) != len(set(columns))):
            raise ValueError("multitable_projection_columns_invalid")
    return checked


def _validated_scope(scope: Mapping[str, Any]) -> tuple[dict, Path]:
    checked = _scope_shape(scope)
    if checked["adapter_id"] == "registered-parquet-projections/2":
        from easyicu.webserver.reviewed_multitable_source import resolve_source_bound_multitable_scope
        source, _ = resolve_source_bound_multitable_scope(checked)
        return checked, source
    identity = SourceIdentity.from_record(checked["source_identity"])
    matches = [Path(row["path"]) for row in sources.load_registry().get("sources", [])
               if row.get("ok") and row.get("path")
               and sources.registered_source_identity(str(row["path"])) == identity]
    if len(matches) != 1 or not matches[0].is_absolute():
        raise ValueError("multitable_projection_source_registration_not_unique")
    return checked, matches[0]


def require_reviewable_multitable_scope(scope: Mapping[str, Any]) -> None:
    """Validate the named source and roster; this does not create authority."""
    _validated_scope(scope)


def run_reviewed_multitable_projection(*, checkpoint: RawPreparationCheckpoint,
                                       binding: RawPreparationHostBinding,
                                       scope: Mapping[str, Any], host_work_root: Path) -> Path:
    """Execute a fixed identity projection after consuming a host decision."""
    checked, source = _validated_scope(scope)
    if checkpoint.plan_sha256 != checked["plan_sha256"]:
        raise ValueError("multitable_projection_plan_mismatch")
    coordinates = binding.model_dump(mode="python")
    verify_host_review(checkpoint=checkpoint, scope=checked, **coordinates)
    implementation = implementation_identity(_ParquetProjectionImplementation(), Path(__file__))
    consume_host_review(checkpoint=checkpoint, scope=checked, **coordinates)
    if _validated_scope(checked)[1] != source:
        raise ValueError("multitable_projection_registration_changed")
    root = Path(tempfile.mkdtemp(prefix="reviewed-multitable-projection-", dir=host_work_root))
    output = root / "output"
    output.mkdir(mode=0o700)
    snapshot = stage_reviewed_source_files(source_root=source, staging_root=root / "snapshot",
        relative_files=(entry["path"] for entry in checked["inputs"]),
        source_identity_sha256=canonical_sha256(checked["source_identity"]))
    return _project_staged(root=root, snapshot=snapshot, checked=checked, checkpoint=checkpoint,
                           binding=binding, implementation=implementation)


def _project_staged(*, root: Path, snapshot: Any, checked: dict, checkpoint: RawPreparationCheckpoint,
                    binding: RawPreparationHostBinding, implementation: Mapping[str, str],
                    before_publish: Callable[[], None] | None = None) -> Path:
    """Project only pre-staged bytes, then sync and attest the output."""
    output = root / "output"
    staged = {item.relative_path: item for item in snapshot.files}
    lineage = []
    for index, entry in enumerate(checked["inputs"]):
        relative = PurePosixPath(entry["path"])
        parent = root / "snapshot" / Path(*relative.parts[:-1])
        output_name = f"part-{index:04d}.parquet"
        with AnchoredDirectory.open(parent) as directory:
            with directory.open_regular(relative.name) as stream:
                parquet = pq.ParquetFile(stream)
                schema = parquet.schema_arrow
                if len(schema.names) != len(set(schema.names)) or any(name not in schema.names for name in entry["columns"]):
                    raise ValueError("multitable_projection_schema_invalid")
                fields = [schema.field(name).with_metadata(None) for name in entry["columns"]]
                if any(pa.types.is_nested(field.type) for field in fields):
                    raise ValueError("multitable_projection_nested_columns_unsupported")
                selected_schema = pa.schema(fields)
                rows = 0
                with pq.ParquetWriter(output / output_name, selected_schema) as writer:
                    for batch in parquet.iter_batches(batch_size=65536, columns=entry["columns"]):
                        # Arrow may return source field order. Reorder explicitly to
                        # the approved order, and discard source schema metadata.
                        table = pa.Table.from_batches([batch]).select(entry["columns"]).cast(selected_schema)
                        writer.write_table(table)
                        rows += table.num_rows
                if rows != parquet.metadata.num_rows:
                    raise ValueError("multitable_projection_row_count_mismatch")
            directory.assert_still_selected()
        source_file = staged[entry["path"]]
        lineage.append({"declared_table": entry["table"], "table_identity_verified": False,
                        "source_path": entry["path"],
                        "source_sha256": source_file.sha256, "source_size_bytes": source_file.size_bytes,
                        "output": output_name, "rows": rows, "columns": entry["columns"],
                        "row_mapping": "same_one_based_data_record_ordinal_within_source_file",
                        "column_mapping": "same_named_input_column", "transformation": "identity"})
    verify_staged_source_files(root / "snapshot", snapshot)
    if implementation_identity(_ParquetProjectionImplementation(), Path(__file__)) != implementation:
        raise ValueError("multitable_projection_implementation_changed")
    if before_publish is not None:
        before_publish()
    with AnchoredDirectory.open(output) as directory:
        for name in sorted(os.listdir(output)):
            with directory.open_regular(name) as stream:
                os.fsync(stream.fileno())
        directory._directory.fsync()
        directory.assert_still_selected()
    publish_attestation(root=root, snapshot=snapshot, scope=checked, checkpoint_sha256=checkpoint.checkpoint_sha256,
        binding=binding, implementation=implementation,
        host_consumption={"method": "host_parquet_column_projection", "inputs": lineage,
                          "full_input_files_staged": True, "all_rows_projected": True,
                          "cohort_complete": False, "derived_features_computed": False})
    verify_attestation(root=root, scope=checked, checkpoint_sha256=checkpoint.checkpoint_sha256,
        binding=binding, adapter=_ParquetProjectionImplementation(), worker=Path(__file__))
    return output


def verify_reviewed_multitable_projection(*, root: Path, checkpoint_sha256: str,
                                         binding: RawPreparationHostBinding, scope: Mapping[str, Any]) -> dict:
    checked = _scope_shape(scope)
    return verify_attestation(root=root, scope=checked, checkpoint_sha256=checkpoint_sha256,
        binding=binding, adapter=_ParquetProjectionImplementation(), worker=Path(__file__))
