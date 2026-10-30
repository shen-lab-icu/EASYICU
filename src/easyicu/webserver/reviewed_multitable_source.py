"""Bind a candidate multi-table projection to a registered export's raw source.

This module creates or checks a partial metadata-only scope; neither operation
approves that scope or claims a complete cohort, exact source contents, or an
independently verified release. The executor must separately verify and consume
the host decision for the exact scope before reading patient rows.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import source_partition_candidates as discovery


def _normalized_columns(requested: Mapping[str, list[str]]) -> dict[str, list[str]]:
    if not isinstance(requested, Mapping) or not 1 <= len(requested) <= 100:
        raise ValueError("multitable_bound_request_invalid")
    result = {}
    for name, columns in requested.items():
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,160}", name)
                or not isinstance(columns, list) or not 1 <= len(columns) <= 512
                or any(not isinstance(col, str) or not col or len(col) > 256 for col in columns)
                or len(columns) != len(set(columns))):
            raise ValueError("multitable_bound_request_invalid")
        result[name] = sorted(columns)
    return {key: result[key] for key in sorted(result)}


def _inputs(candidate: dict) -> list[dict]:
    return [{"table": item["table"], "path": item["path"], "columns": item["columns"]}
            for item in candidate["files"]]


def build_source_bound_multitable_scope(*, identity: SourceIdentity, requested_columns: Mapping[str, list[str]],
                                        plan_sha256: str) -> tuple[dict, dict]:
    """Build a partial candidate. It is neither scientific review nor approval."""
    if (not isinstance(identity, SourceIdentity) or not isinstance(plan_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", plan_sha256)):
        raise ValueError("multitable_bound_request_invalid")
    requested = _normalized_columns(requested_columns)
    candidate = discovery.discover_source_partition_candidates(identity=identity, requested_columns=requested)
    scope = {"adapter_id": "registered-parquet-projections/2", "source_identity": identity.to_dict(),
             "inputs": _inputs(candidate), "purpose": "prepare_multiple_tables_only",
             "plan_sha256": plan_sha256, "dependencies_sha256": candidate["candidate_sha256"]}
    return scope, candidate


def resolve_source_bound_multitable_scope(scope: Mapping[str, Any]) -> tuple[Path, dict]:
    """Re-derive the exact candidate roster and return its verified raw root.

    The input must be the scope already bound to the original host decision.
    Checkpoints and decisions are verified independently by the caller. Raw
    source path and registered export path never come from the client scope.
    """
    if (not isinstance(scope, Mapping)
            or set(scope) != {"adapter_id", "source_identity", "inputs", "purpose", "plan_sha256", "dependencies_sha256"}
            or scope.get("adapter_id") != "registered-parquet-projections/2"
            or scope.get("purpose") != "prepare_multiple_tables_only"
            or not isinstance(scope.get("plan_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", scope["plan_sha256"])
            or not isinstance(scope.get("dependencies_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", scope["dependencies_sha256"])):
        raise ValueError("multitable_bound_scope_invalid")
    identity = SourceIdentity.from_record(scope.get("source_identity"))
    if identity is None or identity.to_dict() != scope["source_identity"]:
        raise ValueError("multitable_bound_source_invalid")
    entries = scope.get("inputs")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 1000:
        raise ValueError("multitable_bound_roster_invalid")
    columns_by_table: dict[str, list[str]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"table", "path", "columns"}:
            raise ValueError("multitable_bound_roster_invalid")
        table = entry.get("table")
        columns = entry.get("columns")
        # Validate each entry before using its table name as a dictionary key.
        valid = _normalized_columns({table: columns}) if isinstance(table, str) else None
        if valid is None:
            raise ValueError("multitable_bound_roster_invalid")
        normalized = valid[table]
        if table in columns_by_table and columns_by_table[table] != normalized:
            raise ValueError("multitable_bound_column_mismatch")
        columns_by_table[table] = normalized
    requested = _normalized_columns(columns_by_table)
    candidate = discovery.discover_source_partition_candidates(identity=identity, requested_columns=requested)
    if candidate["candidate_sha256"] != scope["dependencies_sha256"] or _inputs(candidate) != entries:
        raise ValueError("multitable_bound_candidate_changed")
    # The source is resolved again from the registered export; a replaced
    # manifest or binding must not silently redirect a reviewed scope.
    export = discovery._registered_export(identity)
    binding = discovery.resolve_raw_source_identity_binding(export_path=export, database=identity.database)
    if (binding is None or not Path(binding.source_root).is_absolute()
            or binding.export_manifest_sha256 != candidate["export_manifest_sha256"]
            or dict(binding.table_sha256) != candidate["identity_table_sha256"]
            or discovery._registered_export(identity) != export):
        raise ValueError("multitable_bound_source_changed")
    return Path(binding.source_root), candidate
