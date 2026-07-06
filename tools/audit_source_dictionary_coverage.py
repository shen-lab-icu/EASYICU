#!/usr/bin/env python3
"""Audit EasyICU concept mappings against raw source dictionaries.

This is a maintenance audit for the packaged concept dictionaries, not a
patient-data analysis. It compares selected high-value concepts against local
source item catalogs such as AUMC ``aumc_item_dict.parquet``, MIMIC
``d_items``/``d_labitems``, HiRID variable references, and eICU label columns.

The audit is intentionally conservative: candidates are review queues. Only
items with direct clinical/semantic equivalence should be added to
``concept-dict.json`` or ``sofa2-dict.json``.
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
DEFAULT_OUT = REPO_ROOT / "output" / "data_processing" / "source_dictionary_coverage_audit"

DICTIONARY_FILES = ("concept-dict.json", "sofa2-dict.json")


@dataclass(frozen=True)
class CatalogItem:
    db: str
    table: str
    item_id: str
    label: str
    unit: str = ""
    category: str = ""
    extra: str = ""


CONCEPT_PATTERNS: dict[str, list[str]] = {
    "hr": [r"\bheart\s*rate\b", r"\bhartfrequentie\b", r"\bhr\b"],
    "sbp": [r"systolic\s+bp", r"systolic.*blood", r"\bsbp\b", r"bloeddruk\s+systolisch"],
    "dbp": [r"diastolic\s+bp", r"diastolic.*blood", r"\bdbp\b", r"bloeddruk\s+diastolisch"],
    "map": [
        r"mean\s+arterial",
        r"systemic\s+mean",
        r"\bmap\b",
        r"bloeddruk\s+gemiddeld",
        r"arterial\s+pressure\s+mean",
    ],
    "resp": [r"resp(iratory)?\s*rate", r"ademfrequentie", r"\bresp\b"],
    "spo2": [r"\bspo2\b", r"\bo2\s*sat", r"oxygen\s+saturation", r"saturatie"],
    "o2sat": [r"\bspo2\b", r"\bo2\s*sat", r"oxygen\s+saturation", r"saturatie"],
    "fio2": [r"\bfio2\b", r"inspired\s+oxygen", r"oxygen.*fraction", r"\bo2.*insp"],
    "peep": [r"\bpeep\b", r"positive\s+end[\s-]?expiratory", r"eind\s*exp.*druk"],
    "pip": [r"\bpip\b", r"peak\s+insp", r"piek\s*druk", r"\bppeak\b"],
    "plateau_pres": [r"plateau.*pres", r"plateau.*druk", r"\bpplat\b"],
    "mean_airway_pres": [r"mean\s+airway", r"mean\s+air\s+pres", r"gemiddelde.*luchtweg"],
    "minute_vol": [r"minute\s+vent", r"minute\s+vol", r"minuut.*volume", r"\bmv\b", r"\bve\b"],
    "tidal_vol": [r"tidal\s+vol", r"\bvte?\b", r"teugvolume", r"ademvolume"],
    "tidal_vol_set": [r"tidal.*set", r"\bvt\s*set\b", r"set\s*tidal", r"ingestelde.*teugvolume"],
    "vent_rate": [r"vent.*rate", r"vent.*freq", r"set.*resp.*rate", r"resp.*rate.*set", r"frequentie"],
    "ps": [r"pressure\s+support", r"\bps\b", r"support\s+above\s+peep"],
    "crea": [r"\bcreatinine\b", r"kreatinine", r"\bcrea\b"],
    "bili": [r"bilirubin", r"bilirubine"],
    "plt": [r"\bplatelet", r"thrombocyte", r"trombocyt"],
    "wbc": [r"white\s+blood", r"\bwbc\b", r"leukocyt"],
    "hgb": [r"hemoglobin", r"haemoglobin", r"\bhgb\b", r"\bhb\b"],
    "lact": [r"lactate", r"lactaat"],
    "po2": [r"\bpao2\b", r"\bpo2\b", r"oxygen\s+partial"],
    "pco2": [r"\bpaco2\b", r"\bpco2\b", r"carbon\s+dioxide\s+partial"],
    "ph": [r"^ph$", r"\bph\b"],
    "urine": [r"\burine\b", r"urine.*output", r"diures", r"urineproductie"],
    "rrt": [r"\bcrrt\b", r"dialysis", r"hemodialysis", r"haemodialysis", r"\bcvvh", r"hemofiltrat"],
}

GLOBAL_EXCLUDES = [
    r"\balarm\b",
    r"\bgoal\b",
    r"\bscore\b",
    r"\bapache\b",
    r"\bsofa\b",
    r"\bingredient\b",
    r"\bdose\b",
    r"\border\b",
    r"upper\s*limit",
    r"lower\s*limit",
    r"set\s*high",
    r"set\s*low",
]

CONCEPT_EXCLUDES: dict[str, list[str]] = {
    "hr": [
        r"fetal",
        r"target",
        r"\b(?:mgm|mg|mcg|gm|g|ml|cc|u|units?|meq)\s*/?\s*(?:kg|k)?\s*/?\s*hr\b",
        r"\d+\s*(?:mgm|mg|gm|mcg).*hr",
        r"\bhr\s*(?:limit|>|<)",
        r"infusion",
        r"replace\s+rate",
        r"orthostat",
        r"walking",
        r"aerobic",
        r"activity",
        r"recovery",
        r"lowest\s+heart\s+rate",
        r"\brest\s+hr\b",
        r"\brdos\b",
        r"\bcfs\b",
        r"\b24\s*hr\b",
        r"\b72\s*hr\b",
        r"change\s+q\s+\d+\s*hr",
        r"urine.*hr",
    ],
    "map": [r"airway", r"mean\s+air", r"\bhfo\b", r"\btcpcv\b"],
    "resp": [r"vent", r"set", r"mandatory", r"\bhigh\b", r"quotient"],
    "spo2": [
        r"fio2",
        r"mixed\s+venous",
        r"venous",
        r"arterial",
        r"\bpa\s+line\b",
        r"\bpvr\b",
        r"\bsvr\b",
        r"blood\s+gas",
        r"probe",
        r"seizure",
        r"\[pre\]",
        r"\[post\]",
        r"\bjv\b",
        r"preductal",
        r"postductal",
        r"walking",
        r"aerobic",
        r"activity",
        r"recovery",
        r"\brest\s+o2",
    ],
    "o2sat": [
        r"fio2",
        r"mixed\s+venous",
        r"venous",
        r"\bpa\s+line\b",
        r"\bpvr\b",
        r"\bsvr\b",
        r"probe",
        r"seizure",
        r"\[pre\]",
        r"\[post\]",
        r"\bjv\b",
        r"preductal",
        r"postductal",
        r"walking",
        r"aerobic",
        r"activity",
        r"recovery",
        r"\brest\s+o2",
    ],
    "fio2": [r"challenge", r"\becmo\b", r"flow", r"lpm", r"\bo2\s+sat"],
    "peep": [
        r"peak",
        r"piek",
        r"pip",
        r"plateau",
        r"support",
        r"total|tot",
        r"\bhigh\b",
        r"\blow\b",
        r"sigh",
        r"auto",
        r"intrinsic",
        r"blood\s+gas",
    ],
    "pip": [r"peep", r"plateau", r"support"],
    "mean_airway_pres": [r"arterial", r"map\s+arterial", r"peep", r"peak", r"plateau"],
    "minute_vol": [
        r"\bhi(gh)?\b",
        r"\blow\b",
        r"limit",
        r"pacer",
        r"pacemaker",
        r"sens",
        r"temporary",
        r"tidal",
        r"per\s+kg",
        r"%",
    ],
    "tidal_vol": [
        r"\bhi(gh)?\b",
        r"\blow\b",
        r"alarm",
        r"\bset\b",
        r"ratio",
        r"vd/vt",
        r"rsbi",
        r"pulse\s+ox",
        r"per\s+kg",
    ],
    "vent_rate": [r"heart", r"pulse", r"spont", r"hfo", r"dose"],
    "ps": [r"population", r"score"],
    "crea": [r"urine", r"clearance", r"klar"],
    "bili": [
        r"urine",
        r"direct",
        r"conjugated",
        r"ascites",
        r"body\s+fluid",
        r"pleural",
        r"neonatal",
        r"\bcsf\b",
        r"joint\s+fluid",
        r"stool",
    ],
    "plt": [
        r"transfusion",
        r"pheresis",
        r"smear",
        r"clump",
        r"mean\s+platelet",
        r"aggregation",
        r"intake",
        r"blood\s+products",
        r"suspension",
        r"large\s+platelets",
    ],
    "hgb": [
        r"urine",
        r"plasma\s+free",
        r"carboxy",
        r"methemoglobin|met-hb",
        r"a1c",
        r"absolute",
        r"fetal",
        r"hemoglobin\s+a2?",
        r"hemoglobin\s+[cfs]\b",
        r"reticulocyte",
        r"glycated",
        r"\bp50\b",
        r"plasma\s+hemoglobin",
        r"sulf",
        r"other\s+body\s+fluid",
        r"ascites",
        r"pleur",
        r"streef",
        r"\bso2\b",
    ],
    "lact": [
        r"lactated\s+ringer",
        r"ringers\s+lactate",
        r"dextrose.*lactate",
        r"hetastarch.*lactate",
        r"dehydrogenase|\bld\b",
        r"ingredient",
        r"inamrinone",
        r"fluid",
        r"ascites",
        r"pleural",
        r"\bcsf\b",
        r"stool",
    ],
    "ph": [
        r"urine",
        r"stool",
        r"gastric",
        r"phosph",
        r"phenyt",
        r"phone",
        r"pleural",
        r"emesis",
        r"\bgi\b",
        r"stomach",
        r"\bogt\b",
        r"semen",
        r"faeces",
        r"dipstick",
        r"\bsoft\b",
        r"body\s+fluid",
        r"\bfluid\b",
    ],
    "wbc": [
        r"urine",
        r"csf",
        r"stool",
        r"joint\s+fluid",
        r"other\s+fluid",
        r"pleural",
        r"ascites",
        r"alkaline\s+phosphatase",
    ],
    "urine": [
        r"cc/k/hr",
        r"kg/hr",
        r"per\s+kg",
        r"\b24\s*hr\b",
        r"irrigant",
        r"specimen",
        r"urinezuur",
        r"creatinine",
        r"kreatinine",
        r"protein",
        r"eiwit",
        r"osmol",
        r"glucose",
        r"ph",
        r"screening",
        r"specific\s+gravity",
    ],
    "rrt": [
        r"access",
        r"catheter",
        r"shunt",
        r"machine",
        r"\btype\b",
        r"site",
        r"appear",
        r"system\s+integrity",
        r"not\s+removed",
        r"lumen",
        r"medication",
        r"solution",
        r"heparin",
        r"\bkcl\b",
        r"calcium",
        r"ca\s+gluc",
        r"crystalloid",
        r"irrigation",
        r"rescue\s+line",
        r"fluid",
        r"\boff\b",
        r"\bin\b",
        r"indwelling",
        r"specimen",
        r"\bebl\b",
        r"lab\s+afnemen",
        r"filter\s+change|filter.*wissel",
        r"\breset|resetten",
        r"\bflush\b",
        r"\burine\b",
        r"\bopdr\b",
        r"calcium.*crrt|crrt.*calcium",
        r"citrate",
        r"last\s+dialysis",
        r"dialysis\s+patient",
    ],
}

DEFAULT_CONCEPTS = sorted(CONCEPT_PATTERNS)


def _load_json(filename: str) -> dict[str, Any]:
    return json.loads((DATA_DIR / filename).read_text(encoding="utf-8"))


def _iter_source_defs(dictionary: dict[str, Any]) -> Iterable[tuple[str, str, dict[str, Any]]]:
    for concept, concept_def in dictionary.items():
        if not isinstance(concept_def, dict):
            continue
        sources = concept_def.get("sources")
        if not isinstance(sources, dict):
            continue
        for db, source_defs in sources.items():
            if not isinstance(source_defs, list):
                continue
            for source_def in source_defs:
                if isinstance(source_def, dict):
                    yield concept, db, source_def


def load_mappings() -> dict[tuple[str, str, str, str], dict[str, set[str]]]:
    """Return mapping keys -> exact ids and regexes.

    Key shape is ``(concept, db, table, sub_var)``.
    """

    mappings: dict[tuple[str, str, str, str], dict[str, set[str]]] = {}
    for filename in DICTIONARY_FILES:
        dictionary = _load_json(filename)
        for concept, db, source_def in _iter_source_defs(dictionary):
            table = str(source_def.get("table", ""))
            sub_var = str(source_def.get("sub_var", ""))
            if not table or not sub_var:
                continue
            key = (concept, db, table, sub_var)
            entry = mappings.setdefault(key, {"ids": set(), "regex": set()})
            ids = source_def.get("ids")
            if isinstance(ids, list):
                entry["ids"].update(str(item) for item in ids)
            elif ids is not None:
                entry["ids"].add(str(ids))
            regex = source_def.get("regex")
            if isinstance(regex, str) and regex:
                entry["regex"].add(regex)
    return mappings


def _read_parquet_if_exists(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return pd.DataFrame()


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def catalog_aumc(db_root: Path) -> list[CatalogItem]:
    path = db_root / "aumc" / "aumc_item_dict.parquet"
    df = _read_parquet_if_exists(path)
    if df.empty:
        return []
    return [
        CatalogItem(
            db="aumc",
            table=str(row.table),
            item_id=str(row.id),
            label=str(row.label),
            unit="" if pd.isna(row.unit) else str(row.unit),
        )
        for row in df.itertuples(index=False)
    ]


def catalog_mimic_like(db_root: Path, db: str) -> list[CatalogItem]:
    if db == "mimic":
        root = db_root / "mimiciii"
        item_paths = [root / "d_items.parquet"]
        lab_paths = [root / "d_labitems.parquet"]
    elif db == "miiv":
        root = db_root / "mimiciv"
        item_paths = [root / "icu" / "d_items.parquet", root / "d_items.parquet"]
        lab_paths = [root / "hosp" / "d_labitems.parquet", root / "d_labitems.parquet"]
    else:
        return []

    rows: list[CatalogItem] = []
    item_path = _first_existing(item_paths)
    if item_path:
        df = _read_parquet_if_exists(item_path)
        if not df.empty:
            df.columns = [str(c).upper() for c in df.columns]
            for row in df.itertuples(index=False):
                data = row._asdict()
                rows.append(
                    CatalogItem(
                        db=db,
                        table=str(data.get("LINKSTO", "")),
                        item_id=str(data.get("ITEMID", "")),
                        label=str(data.get("LABEL", "")),
                        unit="" if pd.isna(data.get("UNITNAME", "")) else str(data.get("UNITNAME", "")),
                        category="" if pd.isna(data.get("CATEGORY", "")) else str(data.get("CATEGORY", "")),
                    )
                )
    lab_path = _first_existing(lab_paths)
    if lab_path:
        df = _read_parquet_if_exists(lab_path)
        if not df.empty:
            df.columns = [str(c).upper() for c in df.columns]
            for row in df.itertuples(index=False):
                data = row._asdict()
                rows.append(
                    CatalogItem(
                        db=db,
                        table="labevents",
                        item_id=str(data.get("ITEMID", "")),
                        label=str(data.get("LABEL", "")),
                        unit="",
                        category="" if pd.isna(data.get("CATEGORY", "")) else str(data.get("CATEGORY", "")),
                        extra="" if pd.isna(data.get("FLUID", "")) else str(data.get("FLUID", "")),
                    )
                )
    return rows


def catalog_hirid(db_root: Path) -> list[CatalogItem]:
    rows: list[CatalogItem] = []
    seen: set[tuple[str, str]] = set()

    reference = _read_parquet_if_exists(db_root / "hirid" / "hirid_variable_reference.parquet")
    if not reference.empty:
        for _, data in reference.iterrows():
            item_id = data.get("ID")
            if item_id is None or pd.isna(item_id):
                continue
            item_id = str(int(item_id)) if isinstance(item_id, float) and item_id.is_integer() else str(item_id)
            source_table = str(data.get("Source Table", "Observation")).strip().lower()
            table = "pharma" if source_table.startswith("pharma") else "observations"
            key = (table, item_id)
            seen.add(key)
            rows.append(
                CatalogItem(
                    db="hirid",
                    table=table,
                    item_id=item_id,
                    label=str(data.get("Variable Name", "")),
                    unit="" if pd.isna(data.get("Unit", "")) else str(data.get("Unit", "")),
                    category=str(data.get("Source Table", "")),
                )
            )

    path = db_root / "hirid" / "hirid_variable_reference_preprocessed.parquet"
    df = _read_parquet_if_exists(path)
    if df.empty:
        return rows
    for _, data in df.iterrows():
        raw_ids = str(data.get("raw variable ids", ""))
        for raw_id in re.findall(r"\d+", raw_ids) or [str(data.get("Variable id", ""))]:
            key = ("observations", str(raw_id))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                CatalogItem(
                    db="hirid",
                    table="observations",
                    item_id=str(raw_id),
                    label=str(data.get("Name", "")),
                    unit="" if pd.isna(data.get("Unit", "")) else str(data.get("Unit", "")),
                    category="" if pd.isna(data.get("Type", "")) else str(data.get("Type", "")),
                )
            )
    return rows


def catalog_sic(db_root: Path) -> list[CatalogItem]:
    path = db_root / "sic" / "d_references.parquet"
    df = _read_parquet_if_exists(path)
    if df.empty:
        return []

    table_source_vars = {
        "data_float_h": "DataID",
        "data_range": "DataID",
        "laboratory": "LaboratoryID",
        "medication": "DrugID",
        "data_ref": "RefID",
    }
    table_membership: dict[str, set[str]] = {}
    for table, source_var in table_source_vars.items():
        frame = _read_parquet_if_exists(db_root / "sic" / table, columns=[source_var])
        if frame.empty:
            frame = _read_parquet_if_exists(db_root / "sic" / f"{table}.parquet", columns=[source_var])
        if frame.empty:
            frame = _read_parquet_if_exists(db_root / "sic" / table, columns=[source_var.lower()])
        if frame.empty:
            frame = _read_parquet_if_exists(db_root / "sic" / f"{table}.parquet", columns=[source_var.lower()])
        if frame.empty:
            table_membership[table] = set()
            continue
        column = source_var if source_var in frame.columns else source_var.lower()
        table_membership[table] = {
            str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
            for value in frame[column].dropna().unique()
        }

    rows: list[CatalogItem] = []
    for _, data in df.iterrows():
        item_id = data.get("ReferenceGlobalID")
        if item_id is None or pd.isna(item_id):
            continue
        item_id = str(int(item_id)) if isinstance(item_id, float) and item_id.is_integer() else str(item_id)
        label = " | ".join(
            item
            for item in [
                str(data.get("ReferenceName", "")),
                str(data.get("ReferenceDescription", "")),
                str(data.get("ReferenceValue", "")),
            ]
            if item and item.lower() != "nan"
        )
        unit = "" if pd.isna(data.get("ReferenceUnit", "")) else str(data.get("ReferenceUnit", ""))
        for table, source_var in table_source_vars.items():
            members = table_membership.get(table, set())
            if not members or item_id not in members:
                continue
            rows.append(
                CatalogItem(
                    db="sic",
                    table=table,
                    item_id=item_id,
                    label=label,
                    unit=unit,
                    category=source_var,
                    extra="" if pd.isna(data.get("ReferenceType", "")) else str(data.get("ReferenceType", "")),
                )
            )
    return rows


def _eicu_paths(root: Path, stem: str) -> list[Path]:
    directory = root / stem
    if directory.exists():
        paths = sorted(p for p in directory.glob("*.parquet") if not p.name.startswith("._"))
        if paths:
            return paths
    single = root / f"{stem}.parquet"
    return [single] if single.exists() else []


def _distinct_eicu_labels(paths: Iterable[Path], table: str, label_col: str, unit_col: str | None = None) -> list[CatalogItem]:
    frames = []
    columns = [c for c in [label_col, unit_col] if c]
    for path in paths:
        df = _read_parquet_if_exists(path, columns=columns)
        if not df.empty and label_col in df:
            frames.append(df)
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    if unit_col and unit_col in df:
        grouped = df[[label_col, unit_col]].drop_duplicates()
    else:
        grouped = df[[label_col]].drop_duplicates()
        grouped["unit"] = ""
        unit_col = "unit"
    rows = []
    for row in grouped.itertuples(index=False):
        data = row._asdict()
        label = data.get(label_col)
        if pd.isna(label):
            continue
        unit = data.get(unit_col, "")
        rows.append(
            CatalogItem(
                db="eicu",
                table=table,
                item_id=str(label),
                label=str(label),
                unit="" if pd.isna(unit) else str(unit),
            )
        )
    return rows


def catalog_eicu(db_root: Path) -> list[CatalogItem]:
    root = db_root / "eicu"
    rows: list[CatalogItem] = []
    rows += _distinct_eicu_labels(_eicu_paths(root, "lab"), "lab", "labname", "labmeasurenamesystem")
    rows += _distinct_eicu_labels(_eicu_paths(root, "respiratorycharting"), "respiratorycharting", "respchartvaluelabel")
    rows += _distinct_eicu_labels(
        _eicu_paths(root, "nursecharting"),
        "nursecharting",
        "nursingchartcelltypevalname",
    )
    rows += _distinct_eicu_labels(_eicu_paths(root, "intakeoutput"), "intakeoutput", "celllabel")
    return rows


def build_catalog(db_root: Path, dbs: Iterable[str]) -> list[CatalogItem]:
    rows: list[CatalogItem] = []
    for db in dbs:
        if db == "aumc":
            rows += catalog_aumc(db_root)
        elif db in {"mimic", "miiv"}:
            rows += catalog_mimic_like(db_root, db)
        elif db == "hirid":
            rows += catalog_hirid(db_root)
        elif db == "eicu":
            rows += catalog_eicu(db_root)
        elif db == "sic":
            rows += catalog_sic(db_root)
    return rows


def _compile(patterns: list[str]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{pattern})" for pattern in patterns), re.IGNORECASE)


def _is_candidate(item: CatalogItem, concept: str) -> bool:
    text = " ".join([item.label, item.category, item.extra]).strip()
    if not text:
        return False
    include = _compile(CONCEPT_PATTERNS[concept]).search(text)
    if not include:
        return False
    if concept == "urine" and not re.search(
        r"output|volume|hourly\s+urine|urine\s+catheter|foley|void|diures",
        text,
        flags=re.IGNORECASE,
    ):
        return False
    excludes = GLOBAL_EXCLUDES + CONCEPT_EXCLUDES.get(concept, [])
    return _compile(excludes).search(text) is None


def _source_vars_for_item(item: CatalogItem) -> list[str]:
    if item.db in {"mimic", "miiv", "aumc", "hirid"}:
        return ["itemid", "variableid", "pharmaid", "LaboratoryID", "DataID"]
    if item.db == "sic":
        return [item.category] if item.category else ["DataID", "LaboratoryID", "DrugID", "RefID"]
    if item.db == "eicu":
        if item.table == "lab":
            return ["labname"]
        if item.table == "respiratorycharting":
            return ["respchartvaluelabel"]
        if item.table == "nursecharting":
            return ["nursingchartcelltypevalname"]
        if item.table == "intakeoutput":
            return ["celllabel", "cellpath"]
    return []


def _mapped_status(
    item: CatalogItem,
    concept: str,
    mappings: dict[tuple[str, str, str, str], dict[str, set[str]]],
) -> str:
    for sub_var in _source_vars_for_item(item):
        key = (concept, item.db, item.table, sub_var)
        mapped = mappings.get(key)
        if not mapped:
            continue
        if item.item_id in mapped["ids"]:
            return "mapped"
        if any(re.search(pattern, item.item_id, flags=re.IGNORECASE) for pattern in mapped["regex"]):
            return "covered_by_regex"
        if any(re.search(pattern, item.label, flags=re.IGNORECASE) for pattern in mapped["regex"]):
            return "covered_by_regex"
    return "unmapped_candidate"


def audit(
    catalog: list[CatalogItem],
    concepts: Iterable[str],
    mappings: dict[tuple[str, str, str, str], dict[str, set[str]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concept in concepts:
        if concept not in CONCEPT_PATTERNS:
            continue
        for item in catalog:
            if not _is_candidate(item, concept):
                continue
            status = _mapped_status(item, concept, mappings)
            rows.append(
                {
                    "concept": concept,
                    "db": item.db,
                    "table": item.table,
                    "item_id": item.item_id,
                    "label": item.label,
                    "unit": item.unit,
                    "category": item.category,
                    "extra": item.extra,
                    "status": status,
                }
            )
    return rows


def write_outputs(rows: list[dict[str, Any]], out_dir: Path, concepts: list[str], dbs: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "source_dictionary_coverage_candidates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["concept", "db", "table", "item_id", "label", "unit", "category", "extra", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "concepts": concepts,
        "dbs": dbs,
        "n_rows": len(rows),
        "status_counts": {},
        "by_db_status": {},
        "by_concept_status": {},
    }
    for row in rows:
        status = row["status"]
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
        db_key = f"{row['db']}:{status}"
        summary["by_db_status"][db_key] = summary["by_db_status"].get(db_key, 0) + 1
        concept_key = f"{row['concept']}:{status}"
        summary["by_concept_status"][concept_key] = summary["by_concept_status"].get(concept_key, 0) + 1

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    review = [row for row in rows if row["status"] == "unmapped_candidate"]
    review_path = out_dir / "unmapped_candidate_review.csv"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["concept", "db", "table", "item_id", "label", "unit", "category", "extra", "status"],
        )
        writer.writeheader()
        writer.writerows(review)

    md = [
        "# EasyICU Source Dictionary Coverage Audit",
        "",
        f"_Generated at {summary['generated_at']}._",
        "",
        "## Scope",
        "",
        "This audit compares selected EasyICU concepts against local raw source dictionaries and label catalogs. It produces a review queue, not automatic mapping decisions.",
        "",
        "## Summary",
        "",
        f"- Concepts audited: {len(concepts)}",
        f"- Databases audited: {', '.join(dbs)}",
        f"- Candidate rows: {len(rows)}",
        f"- Unmapped candidates: {len(review)}",
        "",
        "## Status Counts",
        "",
        "| status | n |",
        "| --- | ---: |",
    ]
    for status, count in sorted(summary["status_counts"].items()):
        md.append(f"| {status} | {count} |")
    md.extend(
        [
            "",
            "## Files",
            "",
            "- `source_dictionary_coverage_candidates.csv`: all matched candidates.",
            "- `unmapped_candidate_review.csv`: candidates not covered by exact ids or source regexes.",
            "- `summary.json`: machine-readable counts.",
            "",
            "## Interpretation",
            "",
            "A clean structural dictionary does not prove semantic completeness. Review `unmapped_candidate_review.csv`; only direct equivalents with compatible units and table semantics should be added to the packaged dictionaries.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--concepts", nargs="*", default=DEFAULT_CONCEPTS)
    parser.add_argument("--dbs", nargs="*", default=["aumc", "mimic", "miiv", "hirid", "eicu", "sic"])
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write empty outputs even when no source catalog items are available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    concepts = [concept for concept in args.concepts if concept in CONCEPT_PATTERNS]
    dbs = list(args.dbs)
    mappings = load_mappings()
    catalog = build_catalog(args.db_root, dbs)
    if not catalog and not args.allow_empty:
        raise SystemExit(
            f"No source catalog items found under {args.db_root}. "
            "Mount the source databases or pass --allow-empty explicitly."
        )
    rows = audit(catalog, concepts, mappings)
    write_outputs(rows, args.out_dir, concepts, dbs)
    n_unmapped = sum(1 for row in rows if row["status"] == "unmapped_candidate")
    print(args.out_dir)
    print(f"catalog_items={len(catalog)} candidate_rows={len(rows)} unmapped_candidates={n_unmapped}")


if __name__ == "__main__":
    main()
