#!/usr/bin/env python3
"""Audit EasyICU's 19 module groups against the merged concept dictionary.

This is a top-level consistency audit. It does not validate raw item semantics;
it answers whether the public module catalog, full-export scope, and extraction
dictionary are structurally aligned by module, concept, and database.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "output" / "data_processing" / "module_group_coverage_audit"
DATABASES = ("aumc", "eicu", "eicu_demo", "hirid", "miiv", "mimic", "mimic_demo", "sic")

SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from easyicu.concept.catalog import (  # noqa: E402
    COMPOSITE_CONCEPT_OUTPUT_SOURCES,
    CONCEPT_DICTIONARY,
    CONCEPT_GROUP_NAMES,
    CONCEPT_GROUPS_INTERNAL,
    HIDDEN_DICTIONARY_CONCEPTS,
)
from easyicu.resources import load_data_sources, load_dictionary  # noqa: E402


@dataclass(frozen=True)
class CoverageRow:
    module: str
    module_en: str
    module_zh: str
    concept: str
    database: str
    status: str
    source_count: int
    source_tables: str
    source_basis: str
    dependencies: str
    note: str


def _csv_write(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _source_tables(sources: list[object]) -> str:
    tables = sorted({str(getattr(source, "table", "") or "") for source in sources})
    return "|".join(table for table in tables if table)


def _deps(concept_def: object | None) -> list[str]:
    if concept_def is None:
        return []
    deps = list(getattr(concept_def, "sub_concepts", []) or [])
    deps.extend(getattr(concept_def, "depends_on", []) or [])
    return sorted(dict.fromkeys(str(dep) for dep in deps if dep))


def _status_for(
    concept: str,
    database: str,
    dictionary: object,
    data_sources: object,
) -> tuple[str, int, str, str, str, str]:
    concept_def = dictionary.get(concept)
    labels = CONCEPT_DICTIONARY.get(concept)

    if concept in COMPOSITE_CONCEPT_OUTPUT_SOURCES:
        basis = COMPOSITE_CONCEPT_OUTPUT_SOURCES[concept]
        return ("composite_loader", 0, "", basis, "", "produced by dedicated loader/output alias")

    if concept_def is None:
        if labels is not None:
            return ("catalog_only", 0, "", "", "", "in public catalog but absent from merged dictionary")
        return ("missing", 0, "", "", "", "absent from public catalog and dictionary")

    try:
        config = data_sources.get(database)
    except Exception:
        config = None

    sources = []
    if config is not None:
        try:
            sources = list(concept_def.for_data_source(config))
        except Exception:
            sources = []
    if sources:
        return (
            "direct_source",
            len(sources),
            _source_tables(sources),
            "dictionary.sources",
            "|".join(_deps(concept_def)),
            "",
        )

    deps = _deps(concept_def)
    if deps or getattr(concept_def, "callback", None) or getattr(concept_def, "class_name", None):
        return (
            "derived_dictionary",
            0,
            "",
            "dictionary.derived",
            "|".join(deps),
            "no direct source for this database; resolved from dependencies/callback when available",
        )

    return ("unsupported", 0, "", "", "", "no source for this database")


def build_rows() -> tuple[list[CoverageRow], dict]:
    dictionary = load_dictionary(include_sofa2=True)
    data_sources = load_data_sources()

    rows: list[CoverageRow] = []
    for module, concepts in CONCEPT_GROUPS_INTERNAL.items():
        module_en, module_zh = CONCEPT_GROUP_NAMES.get(module, (module, module))
        for concept in concepts:
            for database in DATABASES:
                status, n_source, tables, basis, deps, note = _status_for(
                    concept, database, dictionary, data_sources
                )
                rows.append(
                    CoverageRow(
                        module=module,
                        module_en=module_en,
                        module_zh=module_zh,
                        concept=concept,
                        database=database,
                        status=status,
                        source_count=n_source,
                        source_tables=tables,
                        source_basis=basis,
                        dependencies=deps,
                        note=note,
                    )
                )

    grouped = [concept for concepts in CONCEPT_GROUPS_INTERNAL.values() for concept in concepts]
    duplicate_concepts = sorted(
        concept for concept, count in Counter(grouped).items() if count > 1
    )
    dictionary_concepts = set(dictionary.keys())
    public_concepts = set(CONCEPT_DICTIONARY)
    catalog_concepts = set(grouped)

    status_counts = Counter(row.status for row in rows)
    module_status = defaultdict(Counter)
    for row in rows:
        module_status[row.module][row.status] += 1

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "module_count": len(CONCEPT_GROUPS_INTERNAL),
        "catalog_concept_count": len(catalog_concepts),
        "public_concept_dictionary_count": len(CONCEPT_DICTIONARY),
        "merged_dictionary_count": len(dictionary_concepts),
        "database_count": len(DATABASES),
        "row_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "duplicate_grouped_concepts": duplicate_concepts,
        "missing_group_names": sorted(set(CONCEPT_GROUPS_INTERNAL) - set(CONCEPT_GROUP_NAMES)),
        "grouped_not_in_public_dictionary": sorted(catalog_concepts - public_concepts),
        "public_dictionary_not_grouped": sorted(public_concepts - catalog_concepts),
        "grouped_not_in_merged_dictionary_or_composite": sorted(
            catalog_concepts - dictionary_concepts - set(COMPOSITE_CONCEPT_OUTPUT_SOURCES)
        ),
        "merged_dictionary_hidden_or_unlisted": sorted(dictionary_concepts - catalog_concepts),
        "hidden_dictionary_concepts": sorted(HIDDEN_DICTIONARY_CONCEPTS),
        "module_status_counts": {
            module: dict(sorted(counts.items()))
            for module, counts in sorted(module_status.items())
        },
    }
    return rows, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows, metadata = build_rows()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _csv_write(out_dir / "concept_module_coverage.csv", [asdict(row) for row in rows])

    module_rows = []
    for module, counts in metadata["module_status_counts"].items():
        concepts = CONCEPT_GROUPS_INTERNAL[module]
        module_en, module_zh = CONCEPT_GROUP_NAMES.get(module, (module, module))
        module_rows.append(
            {
                "module": module,
                "module_en": module_en,
                "module_zh": module_zh,
                "concept_count": len(concepts),
                **{status: counts.get(status, 0) for status in sorted(metadata["status_counts"])},
            }
        )
    _csv_write(out_dir / "module_summary.csv", module_rows)

    (out_dir / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "modules": metadata["module_count"],
                "concepts": metadata["catalog_concept_count"],
                "status_counts": metadata["status_counts"],
                "alignment_issues": {
                    key: len(metadata[key])
                    for key in (
                        "duplicate_grouped_concepts",
                        "missing_group_names",
                        "grouped_not_in_public_dictionary",
                        "public_dictionary_not_grouped",
                        "grouped_not_in_merged_dictionary_or_composite",
                    )
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
