#!/usr/bin/env python3
"""Audit all EasyICU concept dictionary source mappings structurally.

This audit is deliberately narrower than semantic curation. It asks whether a
packaged concept mapping can be resolved against the configured database
schema and local source dictionaries: does the table exist, do referenced
columns exist, do exact ids appear in a local item catalog when such a catalog
exists, and do regexes compile.

It does not decide whether a plausible unmapped raw item should be added. Use
``audit_source_dictionary_coverage.py`` for that review queue.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "src" / "easyicu" / "data"
DEFAULT_DB_ROOT = Path(os.environ.get("EASYICU_DB_ROOT", "/Volumes/外置硬盘/databases"))
DEFAULT_OUT = REPO_ROOT / "output" / "data_processing" / "easyicu_dictionary_full_audit"
DICTIONARY_FILES = ("concept-dict.json", "sofa2-dict.json")
PUBLIC_DBS = ("aumc", "eicu", "mimic", "miiv", "hirid", "sic")
DEMO_DBS = ("eicu_demo", "mimic_demo")

SOURCE_COLUMN_KEYS = {
    "sub_var",
    "val_var",
    "value_var",
    "unit_var",
    "index_var",
    "id_var",
    "dur_var",
    "end_var",
    "stop_var",
    "rate_var",
    "rate_uom",
    "amount_var",
    "auom_var",
    "dir_var",
    "weight_var",
    "grp_var",
    "aux_time",
}

ID_LIKE_VARS = {
    "itemid",
    "variableid",
    "pharmaid",
    "laboratoryid",
    "dataid",
    "drugid",
    "fieldid",
    "refid",
}

LABEL_LIKE_VARS = {
    "labname",
    "respchartvaluelabel",
    "nursingchartcelltypevalname",
    "nursingchartcelltypevallabel",
    "nursingchartvalue",
    "celllabel",
    "cellpath",
    "treatmentstring",
    "drugname",
    "medication",
    "item",
    "value",
}


@dataclass(frozen=True)
class SourceRow:
    dictionary: str
    concept: str
    db: str
    source_index: int
    table: str
    sub_var: str
    ids: str
    regex: str
    class_name: str
    callback: str
    status: str
    severity: str
    issue: str
    evidence: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return pd.DataFrame()


def _read_parquet_dataset(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.exists() and path.is_file():
        return _read_parquet(path, columns=columns)
    if path.exists() and path.is_dir():
        frames = []
        for part in sorted(path.glob("*.parquet")):
            if part.name.startswith("._"):
                continue
            frame = _read_parquet(part, columns=columns)
            if not frame.empty:
                frames.append(frame)
        if frames:
            return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def _read_bucket_parquet_dataset(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists() or not path.is_dir():
        return pd.DataFrame()
    frames = []
    for part in sorted(path.glob("bucket_id=*/*.parquet")):
        if part.name.startswith("._"):
            continue
        frame = _read_parquet(part, columns=columns)
        if not frame.empty:
            frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_data_source_schema() -> dict[str, dict[str, dict[str, Any]]]:
    payload = _read_json(DATA_DIR / "data-sources.json")
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for db in payload:
        name = str(db.get("name", ""))
        tables = {}
        for table, table_def in (db.get("tables") or {}).items():
            cols = {
                _norm(col)
                for col in (table_def.get("cols") or {}).keys()
            }
            defaults = table_def.get("defaults") or {}
            if isinstance(defaults, dict):
                for value in defaults.values():
                    for item in _as_list(value):
                        if isinstance(item, str):
                            cols.add(_norm(item))
            tables[_norm(table)] = {
                "columns": cols,
                "raw_name": table,
            }
        out[name] = tables
    return out


def build_aumc_catalog(db_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    path = db_root / "aumc" / "aumc_item_dict.parquet"
    df = _read_parquet(path)
    cat: dict[tuple[str, str], dict[str, str]] = {}
    if df.empty:
        df = pd.DataFrame()
    if not df.empty:
        for row in df.itertuples(index=False):
            data = row._asdict()
            key = (_norm(data.get("table")), str(data.get("id")))
            cat[key] = {
                "label": str(data.get("label", "")),
                "unit": "" if pd.isna(data.get("unit", "")) else str(data.get("unit", "")),
            }
            cat[("*", str(data.get("id")))] = cat[key]

    # The compact AUMC item dictionary is useful but not exhaustive for all
    # event tables. Add distinct itemids from the actual converted tables as a
    # fallback so the structural audit does not misclassify valid mappings.
    table_specs = {
        "drugitems": ("itemid", "item", "doseunit"),
        "listitems": ("itemid", "item", None),
        "procedureorderitems": ("itemid", "item", None),
        "processitems": ("itemid", "item", None),
        "freetextitems": ("itemid", "item", None),
    }
    root = db_root / "aumc"
    for table, (id_col, label_col, unit_col) in table_specs.items():
        columns = [id_col, label_col] + ([unit_col] if unit_col else [])
        frame = _read_parquet_dataset(root / f"{table}.parquet", columns=columns)
        if frame.empty:
            frame = _read_parquet_dataset(root / table, columns=columns)
        if frame.empty:
            frame = _read_bucket_parquet_dataset(root / f"{table}_bucket", columns=columns)
        if frame.empty or id_col not in frame:
            continue
        keep = [col for col in columns if col and col in frame.columns]
        for item in frame[keep].drop_duplicates(subset=[id_col]).itertuples(index=False):
            data = item._asdict()
            item_id = str(data.get(id_col, ""))
            if not item_id:
                continue
            key = (_norm(table), item_id)
            cat.setdefault(
                key,
                {
                    "label": str(data.get(label_col, "")),
                    "unit": "" if not unit_col or pd.isna(data.get(unit_col, "")) else str(data.get(unit_col, "")),
                },
            )
            cat.setdefault(("*", item_id), cat[key])
    return cat


def build_mimic_catalog(db_root: Path, db: str) -> dict[tuple[str, str], dict[str, str]]:
    if db == "mimic":
        root = db_root / "mimiciii"
        item_paths = [root / "d_items.parquet"]
        lab_paths = [root / "d_labitems.parquet"]
    else:
        root = db_root / "mimiciv"
        item_paths = [root / "icu" / "d_items.parquet", root / "d_items.parquet"]
        lab_paths = [root / "hosp" / "d_labitems.parquet", root / "d_labitems.parquet"]

    cat: dict[tuple[str, str], dict[str, str]] = {}
    item_path = _first_existing(item_paths)
    if item_path:
        df = _read_parquet(item_path)
        if not df.empty:
            df.columns = [str(c).upper() for c in df.columns]
            for row in df.itertuples(index=False):
                data = row._asdict()
                itemid = str(data.get("ITEMID", ""))
                table = _norm(data.get("LINKSTO", ""))
                if itemid:
                    cat[(table, itemid)] = {
                        "label": str(data.get("LABEL", "")),
                        "unit": "" if pd.isna(data.get("UNITNAME", "")) else str(data.get("UNITNAME", "")),
                    }
                    cat[("*", itemid)] = cat[(table, itemid)]
    lab_path = _first_existing(lab_paths)
    if lab_path:
        df = _read_parquet(lab_path)
        if not df.empty:
            df.columns = [str(c).upper() for c in df.columns]
            for row in df.itertuples(index=False):
                data = row._asdict()
                itemid = str(data.get("ITEMID", ""))
                if itemid:
                    cat[("labevents", itemid)] = {
                        "label": str(data.get("LABEL", "")),
                        "unit": "",
                    }
                    cat[("*", itemid)] = cat[("labevents", itemid)]
    return cat


def build_hirid_catalog(db_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    candidates = [
        db_root / "hirid" / "hirid_variable_reference.parquet",
        db_root / "hirid" / "hirid_variable_reference_preprocessed.parquet",
    ]
    cat: dict[tuple[str, str], dict[str, str]] = {}
    for path in candidates:
        df = _read_parquet(path)
        if df.empty:
            continue
        for _, data in df.iterrows():
            label = str(data.get("Variable Name", data.get("Name", data.get("variable_name", ""))))
            unit = data.get("Unit", data.get("unit", ""))
            unit = "" if pd.isna(unit) else str(unit)
            raw_ids = str(data.get("raw variable ids", ""))
            ids = re.findall(r"\d+", raw_ids)
            var_id = data.get("ID", data.get("Variable id", data.get("id")))
            if var_id is not None and not pd.isna(var_id):
                ids.append(str(int(var_id)) if isinstance(var_id, float) and var_id.is_integer() else str(var_id))
            source_table = _norm(data.get("Source Table", "observations"))
            if source_table in {"observation", "observations"}:
                tables = ("observations",)
            elif source_table in {"pharma", "pharmaceutical", "pharma records"}:
                tables = ("pharma",)
            else:
                tables = ("observations", "pharma")
            for item_id in sorted(set(ids)):
                payload = {"label": label, "unit": unit}
                for table in tables:
                    cat[(table, item_id)] = payload
                cat[("*", item_id)] = payload
    return cat


def build_sic_catalog(db_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    path = db_root / "sic" / "d_references.parquet"
    df = _read_parquet(path)
    if df.empty:
        return {}
    cat: dict[tuple[str, str], dict[str, str]] = {}
    for row in df.itertuples(index=False):
        data = row._asdict()
        item_id = str(data.get("ReferenceGlobalID", data.get("referenceglobalid", "")))
        if not item_id:
            continue
        label = str(data.get("ReferenceName", data.get("referencename", "")))
        desc = str(data.get("ReferenceDescription", data.get("referencedescription", "")))
        unit = data.get("ReferenceUnit", data.get("referenceunit", ""))
        unit = "" if pd.isna(unit) else str(unit)
        payload = {"label": " | ".join(x for x in [label, desc] if x), "unit": unit}
        for table in ("data_float_h", "laboratory", "medication", "data_ref", "data_range", "*"):
            cat[(table, item_id)] = payload
    return cat


def build_eicu_label_catalog(db_root: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    """Return (table, column, value) -> metadata for exact label checks."""
    from audit_source_dictionary_coverage import _distinct_eicu_labels, _eicu_paths

    root = db_root / "eicu"
    specs = [
        ("lab", "labname", "labmeasurenamesystem"),
        ("respiratorycharting", "respchartvaluelabel", None),
        ("nursecharting", "nursingchartcelltypevalname", None),
        ("nursecharting", "nursingchartcelltypevallabel", None),
        ("nursecharting", "nursingchartvalue", None),
        ("intakeoutput", "celllabel", None),
        ("intakeoutput", "cellpath", None),
        ("treatment", "treatmentstring", None),
        ("medication", "drugname", None),
        ("infusiondrug", "drugname", None),
    ]
    cat: dict[tuple[str, str, str], dict[str, str]] = {}
    for table, label_col, unit_col in specs:
        for item in _distinct_eicu_labels(_eicu_paths(root, table), table, label_col, unit_col):
            cat[(_norm(table), _norm(label_col), str(item.item_id))] = {
                "label": item.label,
                "unit": item.unit,
            }
    return cat


def build_catalogs(db_root: Path, scan_eicu_labels: bool) -> tuple[
    dict[str, dict[tuple[str, str], dict[str, str]]],
    dict[tuple[str, str, str], dict[str, str]],
]:
    id_catalogs = {
        "aumc": build_aumc_catalog(db_root),
        "mimic": build_mimic_catalog(db_root, "mimic"),
        "miiv": build_mimic_catalog(db_root, "miiv"),
        "hirid": build_hirid_catalog(db_root),
        "sic": build_sic_catalog(db_root),
    }
    eicu_labels = build_eicu_label_catalog(db_root) if scan_eicu_labels else {}
    return id_catalogs, eicu_labels


def _column_status(source: dict[str, Any], table_columns: set[str]) -> tuple[str, str]:
    missing: list[str] = []
    checked: list[str] = []
    for key in sorted(SOURCE_COLUMN_KEYS):
        value = source.get(key)
        if value is None or isinstance(value, bool):
            continue
        for col in _as_list(value):
            if not isinstance(col, str) or not col:
                continue
            col_norm = _norm(col)
            checked.append(f"{key}={col}")
            if col_norm not in table_columns:
                missing.append(f"{key}={col}")
    if missing:
        return "missing_column", "; ".join(missing)
    return "ok", "; ".join(checked)


def _regex_status(pattern: Any) -> tuple[str, str]:
    if not pattern:
        return "not_applicable", ""
    try:
        re.compile(str(pattern))
    except re.error as exc:
        return "invalid_regex", str(exc)
    return "ok", "regex_compiles"


def _id_status(
    *,
    db: str,
    table: str,
    sub_var: str,
    ids: list[Any],
    id_catalogs: dict[str, dict[tuple[str, str], dict[str, str]]],
    eicu_labels: dict[tuple[str, str, str], dict[str, str]],
) -> tuple[str, str]:
    if not ids:
        return "not_applicable", ""
    sub_norm = _norm(sub_var)
    table_norm = _norm(table)
    string_ids = [str(item) for item in ids]

    if db == "eicu":
        if sub_norm not in LABEL_LIKE_VARS:
            return "not_cataloged", f"no exact label catalog for {db}.{table}.{sub_var}"
        missing = [
            item for item in string_ids
            if (table_norm, sub_norm, item) not in eicu_labels
        ]
        if missing and eicu_labels:
            return "id_missing", "missing labels: " + ", ".join(missing[:20])
        if missing and not eicu_labels:
            return "not_cataloged", "eICU label scan disabled"
        return "ok", f"{len(string_ids)} label(s) found"

    if sub_norm not in ID_LIKE_VARS:
        return "not_cataloged", f"sub_var={sub_var} is not an item-id field"

    catalog = id_catalogs.get(db, {})
    if not catalog:
        return "not_cataloged", f"no local id catalog loaded for {db}"
    missing: list[str] = []
    table_mismatch: list[str] = []
    for item in string_ids:
        if (table_norm, item) in catalog:
            continue
        if ("*", item) in catalog:
            table_mismatch.append(item)
        else:
            missing.append(item)
    if missing:
        return "id_missing", "missing ids: " + ", ".join(missing[:20])
    if table_mismatch:
        return "table_mismatch", "id exists in catalog but not table: " + ", ".join(table_mismatch[:20])
    return "ok", f"{len(string_ids)} id(s) found"


def _severity(statuses: list[str]) -> tuple[str, str, str]:
    if "missing_table" in statuses:
        return "error", "high", "table not configured"
    if "missing_column" in statuses:
        return "error", "high", "column not configured"
    if "invalid_regex" in statuses:
        return "error", "high", "regex does not compile"
    if "id_missing" in statuses:
        return "error", "high", "exact ids not found in local source dictionary"
    if "table_mismatch" in statuses:
        return "warning", "medium", "ids exist but source table may be wrong"
    if "not_cataloged" in statuses:
        return "warning", "low", "mapping is structurally valid but no exact id catalog check was available"
    return "ok", "none", "passed structural checks"


def audit_dictionary(
    *,
    dictionary_name: str,
    payload: dict[str, Any],
    schema: dict[str, dict[str, dict[str, Any]]],
    id_catalogs: dict[str, dict[tuple[str, str], dict[str, str]]],
    eicu_labels: dict[tuple[str, str, str], dict[str, str]],
    dbs: set[str],
    include_demo: bool,
) -> list[SourceRow]:
    rows: list[SourceRow] = []
    for concept, concept_def in sorted(payload.items()):
        if not isinstance(concept_def, dict):
            continue
        sources = concept_def.get("sources") or {}
        if not isinstance(sources, dict):
            continue
        for db, source_defs in sources.items():
            if db not in dbs and not (include_demo and db in DEMO_DBS):
                continue
            for idx, source in enumerate(_as_list(source_defs)):
                if not isinstance(source, dict):
                    continue
                table = str(source.get("table") or "")
                sub_var = str(source.get("sub_var") or "")
                ids = _as_list(source.get("ids"))
                regex = str(source.get("regex") or "")
                statuses: list[str] = []
                evidence: list[str] = []

                db_schema = schema.get(db)
                if not db_schema:
                    statuses.append("missing_db")
                    evidence.append(f"db {db} absent from data-sources.json")
                elif table:
                    table_schema = db_schema.get(_norm(table))
                    if not table_schema:
                        statuses.append("missing_table")
                        evidence.append(f"table {db}.{table} absent from data-sources.json")
                    else:
                        col_status, col_evidence = _column_status(source, table_schema["columns"])
                        statuses.append(col_status)
                        if col_evidence:
                            evidence.append(col_evidence)
                elif source.get("class_name") not in {"fun_itm", "rec_cncpt"} and not concept_def.get("concepts"):
                    statuses.append("missing_table")
                    evidence.append("source has no table")

                regex_status, regex_evidence = _regex_status(source.get("regex"))
                statuses.append(regex_status)
                if regex_evidence:
                    evidence.append(regex_evidence)

                id_status, id_evidence = _id_status(
                    db=db,
                    table=table,
                    sub_var=sub_var,
                    ids=ids,
                    id_catalogs=id_catalogs,
                    eicu_labels=eicu_labels,
                )
                statuses.append(id_status)
                if id_evidence:
                    evidence.append(id_evidence)

                status, severity, issue = _severity(statuses)
                rows.append(
                    SourceRow(
                        dictionary=dictionary_name,
                        concept=concept,
                        db=db,
                        source_index=idx,
                        table=table,
                        sub_var=sub_var,
                        ids=json.dumps(ids, ensure_ascii=False),
                        regex=regex,
                        class_name=str(source.get("class_name") or source.get("class") or ""),
                        callback=str(source.get("callback") or concept_def.get("callback") or ""),
                        status=status,
                        severity=severity,
                        issue=issue,
                        evidence=" | ".join(evidence),
                    )
                )
    return rows


def write_outputs(rows: list[SourceRow], out_dir: Path, dbs: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "all_source_mapping_structural_audit.csv"
    findings_path = out_dir / "structural_findings.csv"
    fields = list(SourceRow.__dataclass_fields__)
    with all_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([row.__dict__ for row in rows])

    findings = [row for row in rows if row.status != "ok"]
    with findings_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([row.__dict__ for row in findings])

    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "databases": dbs,
        "n_source_mappings": len(rows),
        "n_findings": len(findings),
        "status_counts": {},
        "severity_counts": {},
        "by_db_status": {},
        "by_concept_findings": {},
    }
    for row in rows:
        summary["status_counts"][row.status] = summary["status_counts"].get(row.status, 0) + 1
        summary["severity_counts"][row.severity] = summary["severity_counts"].get(row.severity, 0) + 1
        db_key = f"{row.db}:{row.status}"
        summary["by_db_status"][db_key] = summary["by_db_status"].get(db_key, 0) + 1
        if row.status != "ok":
            c_key = f"{row.concept}:{row.status}"
            summary["by_concept_findings"][c_key] = summary["by_concept_findings"].get(c_key, 0) + 1
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    status_lines = ["| status | n |", "| --- | ---: |"]
    for status, count in sorted(summary["status_counts"].items()):
        status_lines.append(f"| {status} | {count} |")
    sev_lines = ["| severity | n |", "| --- | ---: |"]
    for severity, count in sorted(summary["severity_counts"].items()):
        sev_lines.append(f"| {severity} | {count} |")
    top_findings = sorted(
        summary["by_concept_findings"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:30]
    top_lines = ["| concept:status | n |", "| --- | ---: |"]
    for key, count in top_findings:
        top_lines.append(f"| `{key}` | {count} |")

    md = [
        "# EasyICU Full Concept Dictionary Structural Audit",
        "",
        f"_Generated at {summary['generated_at']}._",
        "",
        "## Scope",
        "",
        "This is a structural audit of all packaged EasyICU concept source mappings for the selected public databases. It checks configured tables, columns, local source-dictionary ids, and regex compilation. It does not decide whether unmapped raw candidates should be added.",
        "",
        "## Summary",
        "",
        f"- Databases: {', '.join(dbs)}",
        f"- Source mappings audited: {summary['n_source_mappings']}",
        f"- Non-OK findings: {summary['n_findings']}",
        "",
        "## Status Counts",
        "",
        *status_lines,
        "",
        "## Severity Counts",
        "",
        *sev_lines,
        "",
        "## Top Concept Findings",
        "",
        *top_lines,
        "",
        "## Files",
        "",
        "- `all_source_mapping_structural_audit.csv`: one row per dictionary source mapping.",
        "- `structural_findings.csv`: non-OK rows requiring review or documentation.",
        "- `summary.json`: machine-readable counts.",
        "",
        "## Interpretation",
        "",
        "`error` rows are likely executable dictionary problems. `warning` rows are often mappings that cannot be exact-id validated from available local catalogs, especially free-text regex or non-item-id sources. Semantic correctness still requires targeted review for high-impact concepts.",
    ]
    (out_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dbs", nargs="*", default=list(PUBLIC_DBS))
    parser.add_argument("--include-demo", action="store_true")
    parser.add_argument(
        "--scan-eicu-labels",
        action="store_true",
        help="Scan eICU local label columns to exact-check string ids. Slower, but more complete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dbs = list(args.dbs)
    schema = load_data_source_schema()
    id_catalogs, eicu_labels = build_catalogs(args.db_root, args.scan_eicu_labels)
    rows: list[SourceRow] = []
    for filename in DICTIONARY_FILES:
        rows.extend(
            audit_dictionary(
                dictionary_name=filename,
                payload=_read_json(DATA_DIR / filename),
                schema=schema,
                id_catalogs=id_catalogs,
                eicu_labels=eicu_labels,
                dbs=set(dbs),
                include_demo=args.include_demo,
            )
        )
    write_outputs(rows, args.out_dir, dbs)
    errors = sum(1 for row in rows if row.status == "error")
    warnings = sum(1 for row in rows if row.status == "warning")
    print(args.out_dir)
    print(f"source_mappings={len(rows)} findings={errors + warnings} errors={errors} warnings={warnings}")


if __name__ == "__main__":
    main()
