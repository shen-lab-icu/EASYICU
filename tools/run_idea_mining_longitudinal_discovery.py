#!/usr/bin/env python3
"""Run provider-free longitudinal Idea Mining on an existing EasyICU export.

This command never extracts database data.  It profiles prepared parquet
modules in-place, verifies repeated-measure coordinates, and writes bounded
cross-database trajectory candidates for human/prior-art review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from easyicu.concept.catalog import (  # noqa: E402
    CONCEPT_GROUPS_INTERNAL,
    SUPPORTED_DB_KEYS,
    TIME_SERIES_COMPATIBLE_MODULES,
)
from easyicu.research_agent.discovery.idea_mining_longitudinal import (  # noqa: E402
    LONGITUDINAL_DISCOVERY_SCHEMA_VERSION,
    generate_longitudinal_transportability_candidates,
    profile_longitudinal_table,
)

_ID_COLUMN_BY_DATABASE = {
    "aumc": "admissionid",
    "eicu": "patientunitstayid",
    "hirid": "patientid",
    "miiv": "stay_id",
    "mimic": "icustay_id",
    "sic": "CaseID",
}
_TIME_COLUMN = "charttime"


def _parse_csv(value: str | None) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _numeric_columns(schema: pa.Schema) -> set[str]:
    return {
        field.name
        for field in schema
        if pa.types.is_integer(field.type)
        or pa.types.is_floating(field.type)
        or pa.types.is_boolean(field.type)
    }


def _eligible_values(*, module: str, schemas: dict[str, pa.Schema]) -> list[str]:
    declared = list(CONCEPT_GROUPS_INTERNAL.get(module, []))
    if not declared or not schemas:
        return []
    common_numeric = set.intersection(
        *(_numeric_columns(schema) for schema in schemas.values())
    )
    return [value for value in declared if value in common_numeric]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--modules",
        default=None,
        help=(
            "Optional comma-separated prepared modules. By default all "
            "catalog-declared time-series-compatible modules are considered."
        ),
    )
    parser.add_argument("--databases", default=",".join(SUPPORTED_DB_KEYS))
    parser.add_argument("--min-ready-databases", type=int, default=4)
    parser.add_argument("--sample-rows", type=int, default=100_000)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise SystemExit(f"prepared data root not found: {data_root}")
    databases = _parse_csv(args.databases)
    unknown_databases = [db for db in databases if db not in _ID_COLUMN_BY_DATABASE]
    if unknown_databases:
        raise SystemExit(
            "database identity contract is unavailable for: "
            + ", ".join(unknown_databases)
        )
    modules = _parse_csv(args.modules) or sorted(TIME_SERIES_COMPATIBLE_MODULES)
    unknown_modules = [
        module
        for module in modules
        if module not in TIME_SERIES_COMPATIBLE_MODULES
        or module not in CONCEPT_GROUPS_INTERNAL
    ]
    if unknown_modules:
        raise SystemExit(
            "module is not catalog-declared longitudinal: " + ", ".join(unknown_modules)
        )

    all_profiles = []
    module_audit: list[dict[str, object]] = []
    for module in modules:
        paths = {
            database: data_root / database / f"{module}.parquet"
            for database in databases
        }
        schemas = {
            database: pq.read_schema(path)
            for database, path in paths.items()
            if path.is_file()
            and _ID_COLUMN_BY_DATABASE[database] in pq.read_schema(path).names
            and _TIME_COLUMN in pq.read_schema(path).names
        }
        values = _eligible_values(module=module, schemas=schemas)
        if len(schemas) < args.min_ready_databases or not values:
            module_audit.append(
                {
                    "module": module,
                    "status": "not_profiled",
                    "databases_with_coordinates": sorted(schemas),
                    "common_declared_numeric_values": values,
                }
            )
            continue
        before = len(all_profiles)
        for database in sorted(schemas):
            all_profiles.extend(
                profile_longitudinal_table(
                    path=paths[database],
                    database=database,
                    id_column=_ID_COLUMN_BY_DATABASE[database],
                    time_column=_TIME_COLUMN,
                    value_columns=values,
                    sample_rows=args.sample_rows,
                )
            )
        module_audit.append(
            {
                "module": module,
                "status": "profiled",
                "databases_with_coordinates": sorted(schemas),
                "common_declared_numeric_values": values,
                "profiles_emitted": len(all_profiles) - before,
            }
        )

    candidates = generate_longitudinal_transportability_candidates(
        profiles=all_profiles,
        min_ready_databases=args.min_ready_databases,
        limit=args.limit,
    )
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": LONGITUDINAL_DISCOVERY_SCHEMA_VERSION,
        "route": "provider_free_design_first_longitudinal",
        "prepared_data_root": str(data_root),
        "prepared_data_reextracted": False,
        "databases_considered": databases,
        "modules_considered": modules,
        "min_ready_databases": args.min_ready_databases,
        "sample_rows_per_artifact": args.sample_rows,
        "selection_scope": (
            "bounded longitudinal-readiness triage; human protocol and prior-art "
            "review required; not a novelty, result, or paper authorization"
        ),
        "module_audit": module_audit,
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
    manifest = out_dir / "longitudinal_discovery_manifest.json"
    manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "profile_count": len(all_profiles),
                "candidate_count": len(candidates),
                "candidate_concepts": [candidate.concept for candidate in candidates],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
