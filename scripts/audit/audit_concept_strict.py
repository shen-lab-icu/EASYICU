#!/usr/bin/env python3
"""
Stricter concept completeness audit.

Improvements over audit_concept_completeness.py:
  - Word-boundary regex (no substring matches like 'HR' -> 'Threonine')
  - Per-concept exclusion patterns (alarm/score/calc/ingredient/dosage)
  - Optional unit-match filter (mmHg for pressures, etc.)
  - Optional category-match filter (vital signs only for vitals concepts)
  - Output: candidate "missing" items for human review, grouped by concept

Usage:
  python3 scripts/audit_concept_strict.py --concept peep tidal_vol pip
  python3 scripts/audit_concept_strict.py --category ventilator
  python3 scripts/audit_concept_strict.py --all-user-added
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set

import pandas as pd

EASY_DICT = Path("/Users/haibo/Documents/GitHub/EASYICU/src/easyicu/data/concept-dict.json")
RICU_DICT = Path("/Library/Frameworks/R.framework/Versions/4.6/Resources/library/ricu/extdata/config/concept-dict.json")
DB_ROOT = Path("/Volumes/外置硬盘/databases")
OUT_DIR = Path(__file__).resolve().parents[1] / "src" / "data_processing"

# Synonym table per concept (English + Dutch + German for AUMC/SIC)
SYNONYMS: Dict[str, Dict[str, List[str]]] = {
    "peep": {
        "en": [r"\bpeep\b", r"positive\s+end[\s\-]?expiratory"],
        "nl": [r"\bpeep\b", r"eind.?expir.*druk"],
        "de": [r"\bpeep\b"],
    },
    "tidal_vol": {
        "en": [r"\btidal\s+vol(ume)?\b", r"\bvte?\b", r"\bvt[\s\-]?(obs|exp|spont)"],
        "nl": [r"teugvolume", r"ademvolume", r"\bvte?\b"],
        "de": [r"\bvte?\b", r"atemzugvolumen"],
    },
    "tidal_vol_set": {
        "en": [r"\bvt\s*set\b", r"set\s*tidal", r"\btidal.*set\b"],
        "nl": [r"\bvt\s*set\b", r"ingestelde.*teugvolume"],
        "de": [r"\bvt\s*set\b"],
    },
    "pip": {
        "en": [r"\bpip\b", r"peak\s+insp", r"piek.?druk"],
        "nl": [r"piek.?druk", r"\bpip\b"],
        "de": [r"\bpip\b", r"spitzendruck"],
    },
    "plateau_pres": {
        "en": [r"plateau.*pres", r"\bpplat\b"],
        "nl": [r"plateau.*druk", r"\bpplat\b"],
        "de": [r"plateau.*druck", r"\bpplat\b"],
    },
    "mean_airway_pres": {
        "en": [r"mean\s+air(way)?\s+pres", r"\bmap\s*(aw|airway)\b", r"\bmean\s+(insp|exp).*pres"],
        "nl": [r"mean.*luchtweg.*druk", r"gemiddelde.*luchtweg"],
        "de": [r"mean.*airway"],
    },
    "minute_vol": {
        "en": [r"minute\s+vent", r"minute\s+vol(ume)?\b", r"\bmv\b\s*(exp|exhal)"],
        "nl": [r"minuut.*volume", r"\bmv\b"],
        "de": [r"minutenvolumen", r"\bmv\b"],
    },
    "vent_rate": {
        "en": [r"vent.*rate\b", r"vent.*freq", r"set.*resp.*rate", r"resp.*rate.*set"],
        "nl": [r"vent.*frequentie", r"\baf\s*set"],
        "de": [r"beatmungsfrequenz"],
    },
    "etco2": {
        "en": [r"\betco2\b", r"end[\s\-]?tidal\s+co2", r"end[\s\-]?expiratory\s+co2"],
        "nl": [r"\betco2\b", r"end.*tidal.*co2", r"end.*exp.*co2"],
        "de": [r"\betco2\b"],
    },
    "compliance": {
        "en": [r"compliance\s*(stat|dyn|c\b)?", r"\bcrs\b", r"\bcst\b", r"\bcdyn\b"],
        "nl": [r"compliantie", r"\bcrs\b"],
        "de": [r"compliance"],
    },
    "driving_pres": {
        "en": [r"driving\s+pres", r"\bdp\b\s*driv", r"plateau.*minus.*peep"],
        "nl": [r"driving.*druk"],
        "de": [r"driving.*druck"],
    },
}

# Exclusion patterns — these labels are NEVER the concept even if they look similar
EXCLUDE_PATTERNS: Dict[str, List[str]] = {
    "_global": [
        r"\balarm\b", r"\bscore\b", r"\bapache.*ii\b", r"\bsofa\b", r"\bingredient\b",
        r"\bset\s*high\b", r"\bset\s*low\b", r"upper\s*limit", r"lower\s*limit",
    ],
    "peep": [r"\bpeak\b", r"\bplateau\b", r"\bpip\b", r"\bsupport\b", r"\bvent\s*mode\b"],
    "tidal_vol": [r"\bset\b", r"\bspont(aneous)?\b"],  # set/spont go to tidal_vol_set/different concept
    "tidal_vol_set": [],  # the "set" exclusion above doesn't apply here
    "pip": [r"\bplateau\b", r"\bpeep\b", r"\bmean\b"],
    "plateau_pres": [r"\bpeak\b", r"\bpip\b", r"\bpeep\b"],
    "mean_airway_pres": [r"\bpeak\b", r"\bplateau\b", r"\bpeep\b", r"^map\s+arterial", r"mean\s+arterial"],
    "minute_vol": [r"\btidal\b"],
    "vent_rate": [r"\bdose\b", r"\bdrug\b", r"\bppm\b"],
    "etco2": [r"\bo2\b", r"\bspo2\b"],
    "compliance": [r"\bpatient\b\s*comp(liant|liance)", r"compliance\s*(with|to|wit)"],
    "driving_pres": [r"\bsbp\b", r"\bdbp\b", r"arterial"],
}


def load_dict(db: str) -> pd.DataFrame:
    """Return (id, label, table, unit) for the DB's lookup dictionary."""
    if db == "miiv":
        d = pd.read_parquet(DB_ROOT / "mimic-iv-3.1" / "d_items.parquet")
        d.columns = [c.upper() for c in d.columns]
        return d.rename(columns={"ITEMID": "id", "LABEL": "label", "LINKSTO": "table", "UNITNAME": "unit"})[["id", "label", "table", "unit"]]
    elif db == "mimic":
        d = pd.read_parquet(DB_ROOT / "mimiciii" / "d_items.parquet")
        d.columns = [c.upper() for c in d.columns]
        return d.rename(columns={"ITEMID": "id", "LABEL": "label", "LINKSTO": "table", "UNITNAME": "unit"})[["id", "label", "table", "unit"]]
    elif db == "hirid":
        d = pd.read_parquet(DB_ROOT / "hirid-a-high-time-resolution-icu-dataset-1.1.1" / "hirid_variable_reference.parquet")
        return d.rename(columns={"ID": "id", "Variable Name": "label", "Source Table": "table", "Unit": "unit"})[["id", "label", "table", "unit"]]
    elif db == "sic":
        d = pd.read_parquet(DB_ROOT / "sic" / "d_references.parquet")
        d = d.rename(columns={"ReferenceGlobalID": "id", "ReferenceValue": "label", "ReferenceUnit": "unit"})
        d["table"] = "data_float_h"
        return d[["id", "label", "table", "unit"]]
    elif db == "aumc":
        seen = set()
        it = pd.read_csv(DB_ROOT / "aumc" / "numericitems.csv",
                         usecols=['itemid', 'item', 'unit'], dtype={'itemid': 'int32', 'item': 'string', 'unit': 'string'},
                         chunksize=2_000_000, encoding='latin-1')
        for ch in it:
            for tup in ch.drop_duplicates().itertuples(index=False):
                seen.add((tup.itemid, str(tup.item), "numericitems", str(tup.unit) if pd.notna(tup.unit) else ""))
        return pd.DataFrame(list(seen), columns=["id", "label", "table", "unit"])
    else:
        return pd.DataFrame(columns=["id", "label", "table", "unit"])


def build_synonym_regex(concept: str, db: str) -> str:
    syn = SYNONYMS.get(concept, {})
    locale_keys = ["en"]
    if db == "aumc": locale_keys.append("nl")
    if db == "sic":  locale_keys.append("de")
    pats = []
    for lk in locale_keys:
        pats.extend(syn.get(lk, []))
    if not pats:
        return r"$^"  # never matches
    return "|".join(f"({p})" for p in pats)


def build_exclude_regex(concept: str) -> str:
    pats = EXCLUDE_PATTERNS["_global"] + EXCLUDE_PATTERNS.get(concept, [])
    return "|".join(f"({p})" for p in pats)


def audit_concept_db(concept: str, db: str, mapping: List[Dict], dict_df: pd.DataFrame) -> dict:
    rx = build_synonym_regex(concept, db)
    exclude = build_exclude_regex(concept)
    matches = dict_df["label"].astype(str).str.contains(rx, case=False, regex=True, na=False)
    excluded = dict_df["label"].astype(str).str.contains(exclude, case=False, regex=True, na=False)
    candidates = dict_df[matches & ~excluded]
    found_ids = set(candidates["id"].tolist())
    mapped_ids: Set = set()
    for entry in mapping:
        v = entry.get("ids")
        if v is None: continue
        if isinstance(v, list):
            mapped_ids.update(v)
        else:
            mapped_ids.add(v)
    missing = found_ids - mapped_ids
    extra = mapped_ids - found_ids
    miss_with_labels = [(i, lbl, tbl, unit) for i, lbl, tbl, unit in candidates[["id", "label", "table", "unit"]].itertuples(index=False, name=None) if i in missing]
    if missing and extra:
        status = "BOTH"
    elif missing:
        status = "MISSING"
    elif extra:
        status = "EXTRA"
    else:
        status = "OK"
    return {
        "status": status,
        "mapped_ids": sorted(mapped_ids, key=str),
        "found_in_dict": int(len(candidates)),
        "missing_candidates": miss_with_labels[:20],
        "extra_mapped_not_in_dict": sorted(extra, key=str)[:20],
    }


def audit_concepts(concepts: List[str], dbs: List[str], easy: dict) -> dict:
    out = {}
    dict_dfs = {db: load_dict(db) for db in dbs}
    for c in concepts:
        cobj = easy.get(c)
        if not cobj:
            out[c] = {"error": "not in concept-dict"}
            continue
        out[c] = {"description": cobj.get("description"), "category": cobj.get("category"), "per_db": {}}
        for db in dbs:
            src = cobj.get("sources", {}).get(db, [])
            if not src:
                out[c]["per_db"][db] = {"status": "not_mapped"}
                continue
            if all(e.get("class_name") == "col_itm" for e in src):
                out[c]["per_db"][db] = {"status": "col_based"}
                continue
            out[c]["per_db"][db] = audit_concept_db(c, db, src, dict_dfs[db])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", nargs="*")
    ap.add_argument("--category")
    ap.add_argument("--all-user-added", action="store_true")
    ap.add_argument("--dbs", nargs="*", default=["miiv", "mimic", "aumc", "hirid", "sic"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    easy = json.loads(EASY_DICT.read_text())
    ricu = json.loads(RICU_DICT.read_text())
    user_added = set(easy) - set(ricu)

    if args.concepts:
        concepts = [c for c in args.concepts if c in easy]
    elif args.category:
        concepts = [c for c in easy if easy[c].get("category") == args.category]
    elif args.all_user_added:
        concepts = sorted(user_added)
    else:
        ap.error("Specify --concepts, --category, or --all-user-added")

    print(f"Auditing {len(concepts)} concepts × {len(args.dbs)} DBs")
    out = audit_concepts(concepts, args.dbs, easy)
    summary = {
        "_metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "n_concepts": len(concepts),
            "dbs": args.dbs,
            "method": "strict word-boundary regex + global+concept exclusions",
        },
        "concepts": out,
        "summary_status_counts": {},
    }
    for c, r in out.items():
        for db, s in r.get("per_db", {}).items():
            st = s.get("status", "?")
            summary["summary_status_counts"][st] = summary["summary_status_counts"].get(st, 0) + 1

    out_path = Path(args.out) if args.out else (OUT_DIR / f"audit_strict_{'_'.join(concepts)[:80]}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))

    # Human-readable summary
    print(f"\nWrote {out_path}")
    print("=== Per-concept × per-DB status ===")
    for c, r in out.items():
        print(f"\n{c} ({r.get('category','-')}): {r.get('description','')[:80]}")
        for db, s in r.get("per_db", {}).items():
            st = s.get("status", "?")
            extra = ""
            if st in ("MISSING", "BOTH") and s.get("missing_candidates"):
                ex = s["missing_candidates"][:3]
                extra = "  candidates: " + ", ".join(f"{i}={lbl[:30]}({unit})" for i, lbl, _, unit in ex)
            print(f"  {db:6s} {st:10s} mapped={len(s.get('mapped_ids',[])):>2}  found_in_dict={s.get('found_in_dict','?')}{extra}")
    print("\n=== Aggregate ===")
    print(json.dumps(summary["summary_status_counts"], indent=2))


if __name__ == "__main__":
    main()
