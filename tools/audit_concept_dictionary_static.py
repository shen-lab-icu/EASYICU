#!/usr/bin/env python3
"""Static EasyICU concept dictionary audit.

This audit does not need local ICU source extracts. It checks packaged
dictionary structure, large-table prefilter synchronization, high-risk unit
rules, selected source-catalog labels, and duplicate source-id review queues.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "src" / "easyicu" / "data"
OUT_DIR = REPO_ROOT / "output" / "data_processing" / "concept_dictionary_static_audit"
DICTIONARY_FILES = ("concept-dict.json", "sofa2-dict.json")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_ids(source_def: dict[str, Any]) -> set[int]:
    ids = source_def.get("ids")
    if isinstance(ids, int):
        return {ids}
    if isinstance(ids, list):
        return {item for item in ids if isinstance(item, int)}
    return set()


def _iter_concepts() -> Iterable[tuple[str, str, dict[str, Any]]]:
    for filename in DICTIONARY_FILES:
        dictionary = _load_json(DATA_DIR / filename)
        for concept, concept_def in dictionary.items():
            if isinstance(concept_def, dict):
                yield filename, concept, concept_def


def _iter_sources() -> Iterable[tuple[str, str, str, dict[str, Any]]]:
    for filename, concept, concept_def in _iter_concepts():
        sources = concept_def.get("sources")
        if not isinstance(sources, dict):
            continue
        for dataset, source_defs in sources.items():
            if not isinstance(source_defs, list):
                continue
            for source_def in source_defs:
                if isinstance(source_def, dict):
                    yield filename, concept, dataset, source_def


def _dictionary_source_ids(dataset: str, table: str, sub_var: str) -> set[int]:
    ids: set[int] = set()
    for _, _, db, source_def in _iter_sources():
        if (
            db == dataset
            and source_def.get("table") == table
            and source_def.get("sub_var") == sub_var
        ):
            ids.update(_source_ids(source_def))
    return ids


def _runtime_prefilters() -> dict[str, set[int]]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from easyicu.datasource import (  # pylint: disable=import-outside-toplevel
        AUMC_NUMERICITEMS_ITEMIDS,
        HIRID_OBSERVATIONS_VARIABLEIDS,
        MIIV_CHARTEVENTS_ITEMIDS,
        MIIV_LABEVENTS_ITEMIDS,
        MIMIC_DEMO_CHARTEVENTS_ITEMIDS,
        MIMIC_DEMO_LABEVENTS_ITEMIDS,
    )

    return {
        "aumc:numericitems:itemid": set(AUMC_NUMERICITEMS_ITEMIDS),
        "miiv:chartevents:itemid": set(MIIV_CHARTEVENTS_ITEMIDS),
        "miiv:labevents:itemid": set(MIIV_LABEVENTS_ITEMIDS),
        "mimic_demo:chartevents:itemid": set(MIMIC_DEMO_CHARTEVENTS_ITEMIDS),
        "mimic_demo:labevents:itemid": set(MIMIC_DEMO_LABEVENTS_ITEMIDS),
        "hirid:observations:variableid": set(HIRID_OBSERVATIONS_VARIABLEIDS),
    }


def check_prefilters() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, runtime_ids in _runtime_prefilters().items():
        dataset, table, sub_var = key.split(":")
        dictionary_ids = _dictionary_source_ids(dataset, table, sub_var)
        missing = sorted(dictionary_ids - runtime_ids)
        extra = sorted(runtime_ids - dictionary_ids)
        rows.append(
            {
                "check": "prefilter_sync",
                "severity": "critical" if missing or extra else "ok",
                "concept": "",
                "dataset": dataset,
                "table": table,
                "source": sub_var,
                "detail": json.dumps(
                    {
                        "dictionary_n": len(dictionary_ids),
                        "runtime_n": len(runtime_ids),
                        "missing": missing,
                        "extra": extra,
                    },
                    ensure_ascii=False,
                ),
            }
        )
    return rows


def check_table_columns() -> list[dict[str, Any]]:
    data_sources = _load_json(DATA_DIR / "data-sources.json")
    table_defs = {source["name"]: source["tables"] for source in data_sources}
    source_field_keys = {
        "amount_var",
        "auom_var",
        "aux_time",
        "dir_var",
        "dur_var",
        "end_var",
        "grp_var",
        "id_var",
        "index_var",
        "rate_var",
        "stop_var",
        "sub_var",
        "unit_var",
        "val_var",
        "value_var",
        "weight_var",
    }
    rows: list[dict[str, Any]] = []
    for filename, concept, dataset, source_def in _iter_sources():
        table = source_def.get("table")
        if not table:
            continue
        dataset_tables = table_defs.get(dataset)
        if dataset_tables is None:
            rows.append(
                {
                    "check": "source_schema",
                    "severity": "critical",
                    "concept": concept,
                    "dataset": dataset,
                    "table": str(table),
                    "source": filename,
                    "detail": "missing dataset in data-sources.json",
                }
            )
            continue
        table_def = dataset_tables.get(table)
        if table_def is None:
            rows.append(
                {
                    "check": "source_schema",
                    "severity": "critical",
                    "concept": concept,
                    "dataset": dataset,
                    "table": str(table),
                    "source": filename,
                    "detail": "missing source table in data-sources.json",
                }
            )
            continue
        columns = {str(column).lower() for column in table_def.get("cols", {})}
        for field_key in source_field_keys:
            field_name = source_def.get(field_key)
            if isinstance(field_name, str) and field_name.lower() not in columns:
                rows.append(
                    {
                        "check": "source_schema",
                        "severity": "critical",
                        "concept": concept,
                        "dataset": dataset,
                        "table": str(table),
                        "source": filename,
                        "detail": f"{field_key} references missing column {field_name}",
                    }
                )
    if not rows:
        rows.append(
            {
                "check": "source_schema",
                "severity": "ok",
                "concept": "",
                "dataset": "",
                "table": "",
                "source": "",
                "detail": "all dictionary source tables and columns exist",
            }
        )
    return rows


def check_high_risk_units() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filename, concept, dataset, source_def in _iter_sources():
        table = str(source_def.get("table", ""))
        callback = str(source_def.get("callback", ""))
        ids = _source_ids(source_def)
        severity = "warning"
        detail = ""
        if concept == "fio2" and table in {"chartevents", "respiratorycharting"}:
            if "percent_as_numeric" not in callback:
                detail = "FiO2 chart/respiratory source lacks percent_as_numeric callback"
        elif concept == "crea" and dataset in {"aumc", "hirid"}:
            if "0.011309" not in callback and "0.011312" not in callback:
                detail = f"{dataset} creatinine source lacks umol/L to mg/dL conversion"
        elif concept == "tidal_vol" and dataset == "aumc" and ({8871, 9669} & ids):
            if "convert_unit" not in callback or "1000" not in callback:
                detail = "AUMC liter-unit VTe source is mapped into ml tidal_vol without L-to-ml conversion"
        elif concept == "vent_rate" and dataset == "aumc" and 12287 in ids:
            detail = "AUMC 12287 has source-dictionary unit l/min, not /min"
        elif concept == "tidal_vol" and 224684 in ids:
            detail = "224684 is Tidal Volume (set), not measured tidal volume"
        elif concept == "resp" and ({224688, 619} & ids):
            detail = "resp should not include respiratory-rate setting itemids 224688 or 619"
        elif concept == "vent_rate" and ({224422, 224689, 224690} & ids):
            detail = "vent_rate should not include Spont RR or spontaneous/total respiratory rate itemids"
        elif concept == "minute_vol" and ({224688, 224690} & ids):
            detail = "minute_vol should not include respiratory-rate itemids"
        elif concept == "vent_end" and dataset in {"miiv", "mimic", "mimic_demo"} and 226732 in ids:
            detail = "226732 is O2 Delivery Device(s), not a ventilation-end event"
        elif concept == "urine" and ({227510, 227511} & ids):
            detail = "urine should not include enteral tube-feed residual output itemids"
        elif concept == "rrt" and 224270 in ids:
            detail = "rrt should not use Dialysis Catheter access placement as active RRT"
        elif concept == "ecmo_indication" and dataset == "miiv" and ids != {229268}:
            detail = "MIIV ecmo_indication should use 229268 Circuit Configuration, not flow/checkbox ECMO itemids"
        elif concept == "ecmo_indication" and table == "nursecharting":
            detail = "ecmo_indication should not infer VV/VA indication from ambiguous nursecharting ECMO text"
        elif concept == "ecmo" and table == "nursecharting" and source_def.get("sub_var") != "nursingchartvalue":
            detail = "eICU ECMO nursecharting source should scan nursingchartvalue, not label metadata"
        if detail:
            rows.append(
                {
                    "check": "high_risk_unit_rule",
                    "severity": severity,
                    "concept": concept,
                    "dataset": dataset,
                    "table": table,
                    "source": filename,
                    "detail": detail,
                }
            )
    if not rows:
        rows.append(
            {
                "check": "high_risk_unit_rule",
                "severity": "ok",
                "concept": "",
                "dataset": "",
                "table": "",
                "source": "",
                "detail": "no high-risk static unit/callback rule violations",
            }
        )
    return rows


def _load_miiv_catalog() -> dict[tuple[str, int], dict[str, Any]]:
    path = REPO_ROOT / "benchmarks" / "catalogs" / "source_item_catalog_miiv.json"
    if not path.exists():
        return {}
    payload = _load_json(path)
    items = payload.get("items", []) if isinstance(payload, dict) else []
    catalog: dict[tuple[str, int], dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        itemid = item.get("itemid")
        if not isinstance(itemid, int):
            continue
        table = str(item.get("table", ""))
        if table.endswith("/labevents"):
            table = "labevents"
        elif table.endswith("/chartevents"):
            table = "chartevents"
        catalog[(table, itemid)] = item
    return catalog


def check_miiv_catalog_labels() -> list[dict[str, Any]]:
    catalog = _load_miiv_catalog()
    rows: list[dict[str, Any]] = []
    if not catalog:
        return [
            {
                "check": "miiv_catalog_presence",
                "severity": "warning",
                "concept": "",
                "dataset": "miiv",
                "table": "",
                "source": "benchmarks/catalogs/source_item_catalog_miiv.json",
                "detail": "MIIV source catalog is unavailable",
            }
        ]
    for filename, concept, dataset, source_def in _iter_sources():
        if dataset != "miiv":
            continue
        table = str(source_def.get("table", ""))
        if table not in {"chartevents", "labevents"}:
            continue
        for itemid in sorted(_source_ids(source_def)):
            item = catalog.get((table, itemid))
            if item is None:
                rows.append(
                    {
                        "check": "miiv_catalog_label",
                        "severity": "warning",
                        "concept": concept,
                        "dataset": dataset,
                        "table": table,
                        "source": filename,
                        "detail": f"itemid {itemid} not found in packaged MIIV source catalog",
                    }
                )
            elif str(item.get("fluid", "")).lower() not in {"", "blood"} and concept in {
                "crea",
                "hgb",
                "lact",
                "pco2",
                "ph",
                "plt",
                "po2",
                "wbc",
            }:
                rows.append(
                    {
                        "check": "miiv_catalog_label",
                        "severity": "warning",
                        "concept": concept,
                        "dataset": dataset,
                        "table": table,
                        "source": filename,
                        "detail": (
                            f"itemid {itemid} has non-blood fluid "
                            f"{item.get('fluid')} label={item.get('label')}"
                        ),
                    }
                )
    if not rows:
        rows.append(
            {
                "check": "miiv_catalog_label",
                "severity": "ok",
                "concept": "",
                "dataset": "miiv",
                "table": "",
                "source": "benchmarks/catalogs/source_item_catalog_miiv.json",
                "detail": "all MIIV chartevents/labevents itemids exist in packaged source catalog",
            }
        )
    return rows


def _source_defs(
    dictionary: dict[str, Any],
    concept: str,
    dataset: str,
    *,
    table: str | None = None,
    sub_var: str | None = None,
) -> list[dict[str, Any]]:
    concept_def = dictionary.get(concept, {})
    sources = concept_def.get("sources", {}) if isinstance(concept_def, dict) else {}
    source_defs = sources.get(dataset, []) if isinstance(sources, dict) else []
    if not isinstance(source_defs, list):
        return []
    rows = [source for source in source_defs if isinstance(source, dict)]
    if table is not None:
        rows = [source for source in rows if source.get("table") == table]
    if sub_var is not None:
        rows = [source for source in rows if source.get("sub_var") == sub_var]
    return rows


def _combined_ids(source_defs: list[dict[str, Any]]) -> set[int]:
    ids: set[int] = set()
    for source_def in source_defs:
        ids.update(_source_ids(source_def))
    return ids


def check_shared_mechanism_alignment() -> list[dict[str, Any]]:
    """Guard top-level mechanism concepts shared by concept and SOFA-2 dictionaries."""
    concept = _load_json(DATA_DIR / "concept-dict.json")
    sofa2 = _load_json(DATA_DIR / "sofa2-dict.json")
    rows: list[dict[str, Any]] = []

    def add(detail: str, concept_name: str, dataset: str = "", table: str = "") -> None:
        rows.append(
            {
                "check": "shared_mechanism_alignment",
                "severity": "warning",
                "concept": concept_name,
                "dataset": dataset,
                "table": table,
                "source": "concept-dict.json/sofa2-dict.json",
                "detail": detail,
            }
        )

    concept_ecmo = _combined_ids(_source_defs(concept, "ecmo", "miiv", table="chartevents"))
    sofa2_ecmo = _combined_ids(_source_defs(sofa2, "ecmo", "miiv", table="chartevents"))
    missing_ecmo = sorted(sofa2_ecmo - concept_ecmo)
    if missing_ecmo:
        add(f"base ecmo is missing SOFA-2 MIIV ECMO ids {missing_ecmo}", "ecmo", "miiv", "chartevents")

    miiv_ecmo_indication = _source_defs(concept, "ecmo_indication", "miiv", table="chartevents")
    if _combined_ids(miiv_ecmo_indication) != {229268}:
        add("base MIIV ecmo_indication must use only 229268 Circuit Configuration", "ecmo_indication", "miiv", "chartevents")
    elif miiv_ecmo_indication and miiv_ecmo_indication[0].get("val_var") != "value":
        add("base MIIV ecmo_indication must map the value column", "ecmo_indication", "miiv", "chartevents")

    concept_mcs = _combined_ids(
        _source_defs(concept, "mech_circ_support", "miiv", table="chartevents")
    )
    sofa2_mcs = _combined_ids(
        _source_defs(sofa2, "mech_circ_support", "miiv", table="chartevents")
    )
    missing_mcs = sorted(sofa2_mcs - concept_mcs)
    if missing_mcs:
        add(
            f"base mech_circ_support is missing SOFA-2 MIIV chartevent ids {missing_mcs}",
            "mech_circ_support",
            "miiv",
            "chartevents",
        )

    active_rrt_ids = {225436, 225441, 225802, 225803, 225805, 225809, 225955}
    aumc_rrt_ids = {8805, 7666, 7667, 7668, 10736, 12444, 6684, 8806, 8808, 12091}
    for dictionary_name, dictionary in (("concept-dict.json", concept), ("sofa2-dict.json", sofa2)):
        miiv_procedure = _combined_ids(
            _source_defs(dictionary, "rrt", "miiv", table="procedureevents")
        )
        missing = sorted(active_rrt_ids - miiv_procedure)
        if missing:
            add(f"{dictionary_name} MIIV rrt missing active RRT ids {missing}", "rrt", "miiv", "procedureevents")
        if 224270 in miiv_procedure:
            add(f"{dictionary_name} MIIV rrt includes Dialysis Catheter access placement", "rrt", "miiv", "procedureevents")

        aumc_numeric = _combined_ids(
            _source_defs(dictionary, "rrt", "aumc", table="numericitems")
        )
        missing_aumc = sorted(aumc_rrt_ids - aumc_numeric)
        if missing_aumc:
            add(f"{dictionary_name} AUMC rrt missing RRT numericitems {missing_aumc}", "rrt", "aumc", "numericitems")

    for dataset in ("eicu", "eicu_demo"):
        ecmo_nurse = _source_defs(concept, "ecmo", dataset, table="nursecharting")
        if ecmo_nurse and any(source.get("sub_var") != "nursingchartvalue" for source in ecmo_nurse):
            add("base ecmo should scan eICU nursecharting value text for ECMO", "ecmo", dataset, "nursecharting")
        if _source_defs(concept, "ecmo_indication", dataset, table="nursecharting"):
            add(
                "base ecmo_indication should not infer VV/VA from ambiguous eICU nursecharting text",
                "ecmo_indication",
                dataset,
                "nursecharting",
            )

    eicu_mcs = _source_defs(concept, "mech_circ_support", "eicu", table="treatment")
    if eicu_mcs:
        regex = str(eicu_mcs[0].get("regex", ""))
        if "Tandem" not in regex or "ventricular assist" not in regex:
            add("base eICU mech_circ_support regex is narrower than SOFA-2 mechanism coverage", "mech_circ_support", "eicu", "treatment")

    eicu_rrt = _source_defs(concept, "rrt", "eicu", table="treatment")
    if eicu_rrt:
        regex = str(eicu_rrt[0].get("regex", ""))
        if "ultrafiltration" not in regex or "renal replacement" not in regex:
            add("base eICU rrt treatment regex is narrower than SOFA-2 RRT coverage", "rrt", "eicu", "treatment")
    if not _source_defs(concept, "rrt", "eicu", table="intakeoutput", sub_var="cellpath"):
        add("base eICU rrt is missing intakeoutput Dialysis cellpath source", "rrt", "eicu", "intakeoutput")

    if not rows:
        rows.append(
            {
                "check": "shared_mechanism_alignment",
                "severity": "ok",
                "concept": "",
                "dataset": "",
                "table": "",
                "source": "concept-dict.json/sofa2-dict.json",
                "detail": "top-level ECMO/MCS/RRT mechanism definitions pass static alignment rules",
            }
        )
    return rows


def check_mimic_demo_source_mirror() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    allowed_missing = {"samp"}
    allowed_different = {"samp"}
    for filename, concept, concept_def in _iter_concepts():
        sources = concept_def.get("sources")
        if not isinstance(sources, dict):
            continue
        if "mimic" in sources and "mimic_demo" not in sources and concept not in allowed_missing:
            rows.append(
                {
                    "check": "mimic_demo_source_mirror",
                    "severity": "warning",
                    "concept": concept,
                    "dataset": "mimic_demo",
                    "table": "",
                    "source": filename,
                    "detail": "concept has mimic source definitions but no explicit mimic_demo source",
                }
            )
        elif (
            "mimic" in sources
            and "mimic_demo" in sources
            and sources["mimic"] != sources["mimic_demo"]
            and concept not in allowed_different
        ):
            rows.append(
                {
                    "check": "mimic_demo_source_mirror",
                    "severity": "warning",
                    "concept": concept,
                    "dataset": "mimic_demo",
                    "table": "",
                    "source": filename,
                    "detail": "mimic_demo source definitions differ from mimic outside allowed exceptions",
                }
            )
    if not rows:
        rows.append(
            {
                "check": "mimic_demo_source_mirror",
                "severity": "ok",
                "concept": "",
                "dataset": "mimic_demo",
                "table": "",
                "source": "concept-dict.json/sofa2-dict.json",
                "detail": "all mimic-backed concepts have explicit mimic_demo source definitions",
            }
        )
    return rows


ALLOWED_DUPLICATE_CONCEPT_GROUPS = [
    {"bicar", "bicarb"},
    {"sao2", "o2sat", "spo2"},
]


def _is_allowed_duplicate(concepts: set[str]) -> bool:
    if any(concepts <= group for group in ALLOWED_DUPLICATE_CONCEPT_GROUPS if group):
        return True
    if concepts <= {"abx", "vancomycin", "meropenem"}:
        return True
    if concepts <= {"dobu_rate", "dobu_dur", "dopa_rate", "dopa_dur", "norepi_rate", "norepi_dur", "epi_rate", "epi_dur"}:
        return True
    if "total_input_ml" in concepts:
        return True
    return False


def duplicate_review_rows(limit: int = 200) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for _, concept, dataset, source_def in _iter_sources():
        table = str(source_def.get("table", ""))
        sub_var = str(source_def.get("sub_var", ""))
        for itemid in _source_ids(source_def):
            by_key[(dataset, table, sub_var, str(itemid))].add(concept)
    rows: list[dict[str, Any]] = []
    for (dataset, table, sub_var, itemid), concepts in sorted(by_key.items()):
        if len(concepts) <= 1 or _is_allowed_duplicate(concepts):
            continue
        rows.append(
            {
                "dataset": dataset,
                "table": table,
                "sub_var": sub_var,
                "item_id": itemid,
                "concepts": ";".join(sorted(concepts)),
            }
        )
    return rows[:limit]


def write_outputs(rows: list[dict[str, Any]], duplicate_rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "static_audit_findings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["check", "severity", "concept", "dataset", "table", "source", "detail"],
        )
        writer.writeheader()
        writer.writerows(rows)
    with (OUT_DIR / "duplicate_source_id_review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "table", "sub_var", "item_id", "concepts"])
        writer.writeheader()
        writer.writerows(duplicate_rows)

    severity_counts = Counter(row["severity"] for row in rows)
    source_counts = Counter()
    for _, _, dataset, source_def in _iter_sources():
        source_counts[f"{dataset}:{source_def.get('table', '')}"] += 1

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "concept_defs": sum(1 for _ in _iter_concepts()),
        "source_defs": sum(1 for _ in _iter_sources()),
        "severity_counts": dict(severity_counts),
        "source_defs_by_dataset": dict(Counter(dataset for _, _, dataset, _ in _iter_sources())),
        "top_source_tables": dict(source_counts.most_common(20)),
        "duplicate_review_rows": len(duplicate_rows),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    actionable = [row for row in rows if row["severity"] not in {"ok"}]
    md = [
        "# EasyICU Concept Dictionary Static Audit",
        "",
        f"_Generated at {summary['generated_at']}._",
        "",
        "## Summary",
        "",
        f"- Concept definitions: {summary['concept_defs']}",
        f"- Source definitions: {summary['source_defs']}",
        f"- Actionable findings: {len(actionable)}",
        f"- Duplicate source-id review rows: {len(duplicate_rows)}",
        "",
        "## Severity Counts",
        "",
        "| severity | n |",
        "| --- | ---: |",
    ]
    for severity, count in sorted(severity_counts.items()):
        md.append(f"| {severity} | {count} |")
    md.extend(
        [
            "",
            "## Actionable Findings",
            "",
        ]
    )
    if actionable:
        md.extend(["| check | concept | dataset | table | detail |", "| --- | --- | --- | --- | --- |"])
        for row in actionable[:80]:
            md.append(
                f"| {row['check']} | {row['concept']} | {row['dataset']} | "
                f"{row['table']} | {row['detail']} |"
            )
    else:
        md.append("No actionable static findings.")
    md.extend(
        [
            "",
            "## Files",
            "",
            "- `static_audit_findings.csv`: structured static checks.",
            "- `duplicate_source_id_review.csv`: non-allowed source-id reuse queue.",
            "- `summary.json`: machine-readable summary.",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    rows = []
    rows.extend(check_prefilters())
    rows.extend(check_table_columns())
    rows.extend(check_high_risk_units())
    rows.extend(check_miiv_catalog_labels())
    rows.extend(check_shared_mechanism_alignment())
    rows.extend(check_mimic_demo_source_mirror())
    duplicate_rows = duplicate_review_rows()
    write_outputs(rows, duplicate_rows)
    actionable = sum(1 for row in rows if row["severity"] not in {"ok"})
    print(OUT_DIR)
    print(f"findings={len(rows)} actionable={actionable} duplicate_review_rows={len(duplicate_rows)}")


if __name__ == "__main__":
    main()
