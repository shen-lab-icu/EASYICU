"""Metadata-only discovery of candidate Parquet partitions for a registered source.

A local conversion manifest can suggest a review roster. Matching local files and
schemas is not independent verification of a release, complete raw records, the
cohort, or a producer's actual reads. No preparation authority is created here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from easyicu.concept.provenance import metadata_digest
from easyicu.config import load_src_cfg
from easyicu.research_agent.authority.filesystem import AnchoredDirectory, AuthorityFilesystemError
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import sources
from easyicu.webserver.raw_source_authority import resolve_raw_source_identity_binding


def _registered_export(identity: SourceIdentity) -> Path:
    matches = [Path(row["path"]) for row in sources.load_registry().get("sources", [])
               if isinstance(row, Mapping) and row.get("ok") and row.get("path")
               and sources.registered_source_identity(str(row["path"])) == identity]
    if len(matches) != 1 or not matches[0].is_absolute():
        raise ValueError("partition_candidate_registration_invalid")
    return matches[0]


def _manifest(root: Path) -> tuple[dict, str]:
    with AnchoredDirectory.open(root) as directory:
        raw = directory.read_bytes("conversion_manifest.json", max_bytes=8 * 1024 * 1024)
        directory.assert_still_selected()
    try:
        result = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("partition_candidate_manifest_invalid") from exc
    if not isinstance(result, dict) or result.get("schema_version") != "easyicu.conversion_manifest/1":
        raise ValueError("partition_candidate_manifest_invalid")
    return result, hashlib.sha256(raw).hexdigest()


def _relative(value: Any) -> PurePosixPath:
    path = PurePosixPath(value) if isinstance(value, str) else PurePosixPath()
    if (not isinstance(value, str) or not value or "\\" in value or "\x00" in value
            or path.is_absolute() or path.as_posix() != value
            or any(part in {".", ".."} for part in path.parts)
            or path.suffix.lower() != ".parquet"):
        raise ValueError("partition_candidate_path_invalid")
    return path


def discover_source_partition_candidates(*, identity: SourceIdentity,
                                         requested_columns: Mapping[str, list[str]]) -> dict:
    """Return a bounded candidate roster from bound manifest and local schemas.

    The exact source files are *not* content-hashed by this helper; the existing
    identity binding may itself verify bytes of its designated identity table.
    No data rows are decoded and no cohort/analysis or execution grant is made.
    """
    identity = SourceIdentity.from_record(identity.to_dict()) if isinstance(identity, SourceIdentity) else None
    if identity is None or not isinstance(requested_columns, Mapping) or not 1 <= len(requested_columns) <= 100:
        raise ValueError("partition_candidate_request_invalid")
    export = _registered_export(identity)
    binding = resolve_raw_source_identity_binding(export_path=export, database=identity.database)
    if binding is None:
        raise ValueError("partition_candidate_source_binding_unavailable")
    root = Path(binding.source_root)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("partition_candidate_source_invalid")
    manifest, manifest_sha = _manifest(root)
    config = load_src_cfg(identity.database)
    if manifest.get("database") != config.name or config.name != identity.database:
        raise ValueError("partition_candidate_database_mismatch")
    tables = manifest.get("tables")
    if not isinstance(tables, list) or len(tables) > 10000:
        raise ValueError("partition_candidate_manifest_invalid")
    result: list[dict] = []
    recorded_directories: dict[str, set[str]] = {}
    path_owners: dict[str, str] = {}
    identical_duplicates = 0
    for name, columns in requested_columns.items():
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,160}", name)
                or name not in config.tables or not isinstance(columns, list) or not columns
                or len(columns) > 512 or any(not isinstance(col, str) or not col or len(col) > 256 for col in columns)
                or len(columns) != len(set(columns))):
            raise ValueError("partition_candidate_columns_invalid")
        config_columns = {col.get("name") if isinstance(col, Mapping) else getattr(col, "name", None)
                          for col in config.tables[name].columns.values()}
        if not set(columns).issubset(config_columns):
            raise ValueError("partition_candidate_columns_not_configured")
        records = [item for item in tables if isinstance(item, dict) and item.get("table") == name]
        if len(records) != 1 or records[0].get("status") != "completed" or not isinstance(records[0].get("outputs"), list):
            raise ValueError("partition_candidate_table_not_complete")
        if len(records[0]["outputs"]) > 2000 or any(not isinstance(item, dict) for item in records[0]["outputs"]):
            raise ValueError("partition_candidate_roster_invalid")
        unique: dict[str, dict] = {}
        for item in records[0]["outputs"]:
            path = _relative(item.get("relative_path")).as_posix()
            if path in unique:
                if item != unique[path]:
                    raise ValueError("partition_candidate_duplicate_conflict")
                identical_duplicates += 1
            else:
                unique[path] = item
            if path in path_owners and path_owners[path] != name:
                raise ValueError("partition_candidate_file_assigned_twice")
            path_owners[path] = name
        paths = sorted(unique)
        if not paths or len(paths) > 1000:
            raise ValueError("partition_candidate_roster_invalid")
        parents = {PurePosixPath(path).parent.as_posix() for path in paths}
        if len(parents) != 1:
            raise ValueError("partition_candidate_multiple_parents")
        parent_name = next(iter(parents))
        if parent_name != "." and parent_name != name:
            raise ValueError("partition_candidate_directory_mismatch")
        if parent_name != ".":
            expected = {PurePosixPath(path).name for path in paths}
            if parent_name in recorded_directories and recorded_directories[parent_name] != expected:
                raise ValueError("partition_candidate_directory_conflict")
            recorded_directories[parent_name] = expected
        for value in paths:
            path = _relative(value)
            directory_path = root.joinpath(*path.parts[:-1])
            try:
                with AnchoredDirectory.open(directory_path) as directory:
                    with directory.open_regular(path.name) as stream:
                        schema = pq.read_schema(stream)
                    directory.assert_still_selected()
            except (AuthorityFilesystemError, OSError, pa.ArrowException) as exc:
                raise ValueError("partition_candidate_schema_unavailable") from exc
            if len(schema.names) != len(set(schema.names)) or not set(columns).issubset(schema.names):
                raise ValueError("partition_candidate_schema_mismatch")
            result.append({"table": name, "path": value, "columns": list(columns),
                           "schema_sha256": canonical_sha256([(field.name, str(field.type)) for field in schema])})
    for name, expected in recorded_directories.items():
        with AnchoredDirectory.open(root / name) as directory:
            if set(os.listdir(root / name)) != expected:
                raise ValueError("partition_candidate_directory_roster_mismatch")
            if any(not stat.S_ISREG(directory.stat(item).st_mode) for item in expected):
                raise ValueError("partition_candidate_file_not_regular")
            directory.assert_still_selected()
    _, last_manifest_sha = _manifest(root)
    current = resolve_raw_source_identity_binding(export_path=export, database=identity.database)
    if (_registered_export(identity) != export or last_manifest_sha != manifest_sha or current is None
            or current.source_root != binding.source_root
            or current.export_manifest_sha256 != binding.export_manifest_sha256
            or current.table_sha256 != binding.table_sha256):
        raise ValueError("partition_candidate_source_changed")
    body = {"schema_version": "easyicu.source-partition-candidate/1",
            "source_identity_sha256": canonical_sha256(identity.to_dict()),
            "export_manifest_sha256": binding.export_manifest_sha256,
            "identity_table_sha256": dict(binding.table_sha256),
            "source_config_sha256": metadata_digest(config.model_dump(mode="json")),
            "conversion_manifest_sha256": manifest_sha,
            "identical_duplicate_entries_coalesced": identical_duplicates,
            "files": result, "source_rows_decoded": False,
            "files_content_verified": False, "independent_release_verified": False,
            "full_source_roster_verified": False, "cohort_complete": False,
            "analysis_authorized": False}
    return {**body, "candidate_sha256": canonical_sha256(body)}


def candidate_from_bound_research_context(*, study: Mapping[str, Any], run_dir: Path) -> dict:
    """Build a non-executable preview from the run's registered context evidence.

    Concept mappings require an explicit source concept. Source identifier
    candidates use configured table ID columns verified against the selected
    partition schemas. A research variable is offered as a match only for an
    exact cohort ID name. Neither proves a join or an exported/derived
    identifier, nor infers cohort or outcome dependencies.
    """
    from easyicu.concept.loader import load_dictionary
    from easyicu.concept.provenance import capture_data_provenance
    from easyicu.databases.research_plan_requirements import _constraints
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
    from easyicu.research_agent.research_context.typed import parse_research_context_json

    source = study.get("data_source")
    if not isinstance(source, Mapping) or not isinstance(source.get("path"), str):
        raise ValueError("partition_candidate_study_source_missing")
    identity = sources.registered_source_identity(source["path"])
    if identity is None or source.get("database") != identity.database:
        raise ValueError("partition_candidate_study_source_missing")
    evidence = EvidenceStore(run_dir).get("research_context")
    path = verified_run_evidence_path(run_dir, evidence) if evidence else None
    if path is None:
        raise ValueError("partition_candidate_context_unavailable")
    with AnchoredDirectory.open(path.parent) as directory:
        raw = directory.read_bytes(path.name, max_bytes=5 * 1024 * 1024)
        directory.assert_still_selected()
    if hashlib.sha256(raw).hexdigest() != evidence.sha256:
        raise ValueError("partition_candidate_context_unavailable")
    context = parse_research_context_json(raw.decode("utf-8"))
    recorded = SourceIdentity.from_record(_constraints(context).get("source_identity"))
    if recorded != identity or context.cohort.database != identity.database:
        raise ValueError("partition_candidate_context_source_mismatch")
    concepts = list(dict.fromkeys(str(item.source_concept) for item in context.variables if item.source_concept))
    unmapped = [item.name for item in context.variables if not item.source_concept]
    if not concepts:
        raise ValueError("partition_candidate_no_recorded_concepts")
    config = load_src_cfg(identity.database)
    snapshot = capture_data_provenance(load_dictionary(include_sofa2=True), concepts=concepts,
        source_database=identity.database, class_prefixes=tuple(config.class_prefix), source_config=config,
        source_config_sha256=metadata_digest(config.model_dump(mode="json")))
    tables: dict[str, set[str]] = {}
    features = []
    for concept in snapshot.concepts:
        if concept.status != "recorded" or not concept.sources:
            unmapped.append(concept.concept_id)
            continue
        for item in concept.sources:
            if not item.table or item.table not in config.tables:
                unmapped.append(concept.concept_id)
                continue
            default = config.tables[item.table].defaults
            columns = [default.id_var, item.selector_variable, item.value_variable,
                       item.unit_variable, item.time_variable, item.duration_variable]
            columns = list(dict.fromkeys(value for value in columns if isinstance(value, str) and value))
            if not columns:
                unmapped.append(concept.concept_id)
                continue
            tables.setdefault(item.table, set()).update(columns)
            features.append({"concept": concept.concept_id, "table": item.table,
                             "columns": columns, "item_ids": item.item_ids[:100],
                             "item_ids_truncated": len(item.item_ids) > 100,
                             "declared_unit": concept.canonical_unit,
                             "derivation_or_callback_present": bool(concept.derivation_callback or item.callback
                                 or item.selector_regex or item.codes or item.unit_conversions),
                             "actual_records_verified": False})
    if not tables:
        raise ValueError("partition_candidate_no_mapped_columns")
    candidate = discover_source_partition_candidates(identity=identity,
        requested_columns={table: sorted(columns) for table, columns in tables.items()})
    # Show configured source-table IDs after schema verification even when no
    # research variable is bound. An exact name is only a variable candidate
    # when the variable is also a cohort-declared ID. A generated identifier
    # is never equated by inference to a physical source ID.
    identifier_candidates = []
    for table in sorted(tables):
        column = config.tables[table].defaults.id_var
        files = [item for item in candidate["files"] if item["table"] == table]
        if not isinstance(column, str) or not column or not files or not all(column in item["columns"] for item in files):
            continue
        variable = next((item for item in context.variables if not item.source_concept
            and item.role == "id" and item.name in context.cohort.id_columns and item.name == column), None)
        identifier_candidates.append({"variable": variable.name if variable else None, "table": table,
            "column": column, "actual_records_verified": False,
            "join_or_derived_identifier_verified": False})
        if variable:
            unmapped = [name for name in unmapped if name != variable.name]
    candidate["research_context_evidence_sha256"] = evidence.sha256
    candidate["dictionary_sha256"] = snapshot.dictionary_sha256
    candidate["features"] = features
    candidate["identifier_candidates"] = identifier_candidates
    candidate["unmapped_variables"] = list(dict.fromkeys(unmapped))
    candidate["outcome_and_cohort_dependencies_complete"] = False
    candidate["candidate_sha256"] = canonical_sha256({k: v for k, v in candidate.items() if k != "candidate_sha256"})
    return candidate
