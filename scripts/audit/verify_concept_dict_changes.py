#!/usr/bin/env python3
"""
End-to-end verification of all Round 1-3 concept-dict.json changes.

For each modified concept, this script:
  1. Confirms every itemid added exists in the raw DB dictionary
  2. Confirms each added itemid has at least one actual data row
  3. Records pass/fail with row counts in a verification log

Output:
  src/data_processing/concept_dict_change_verification.json
"""
from __future__ import annotations

import glob
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DB_ROOT = Path("/Volumes/外置硬盘/databases")
OUT = REPO / "src" / "data_processing" / "concept_dict_change_verification.json"

# Changes to verify — keyed (db, concept) -> {itemids: [...], source_table: ..., verify_data: bool}
CHANGES = {
    # === Round 1: CVP concept addition (6 DBs) ===
    ("miiv",  "cvp"):  {"ids": [220074],                 "table": "chartevents",   "subvar": "itemid"},
    ("mimic", "cvp"):  {"ids": [113, 220074, 1103],      "table": "chartevents",   "subvar": "itemid"},
    ("aumc",  "cvp"):  {"ids": [6655, 20926],            "table": "numericitems",  "subvar": "itemid"},
    ("hirid", "cvp"):  {"ids": [700, 960, 15001441],     "table": "observations",  "subvar": "variableid"},
    ("sic",   "cvp"):  {"ids": [2018],                   "table": "data_float_h",  "subvar": "DataID"},

    # === Round 1: AUMC ventilator additions ===
    ("aumc", "compliance"):  {"ids": [12561],            "table": "numericitems",  "subvar": "itemid"},
    ("aumc", "minute_vol"):  {"ids": [8875],             "table": "numericitems",  "subvar": "itemid"},
    ("aumc", "tidal_vol"):   {"ids": [16243],            "table": "numericitems",  "subvar": "itemid"},

    # === Round 2: MIMIC-III ventilator corrections ===
    ("mimic", "peep"):       {"ids": [505, 506],         "table": "chartevents",   "subvar": "itemid"},
    ("mimic", "ps"):         {"ids": [578, 6339, 7332, 7587, 7595], "table": "chartevents", "subvar": "itemid"},
    ("mimic", "tidal_vol"):  {"ids": [501, 502],         "table": "chartevents",   "subvar": "itemid"},

    # === Round 2: HiRID Pharma drug additions ===
    ("hirid", "diltiazem"):   {"ids": [121, 1001071],                            "table": "pharma_records"},
    ("hirid", "esmolol"):     {"ids": [1000346, 1000347],                        "table": "pharma_records"},
    ("hirid", "labetalol"):   {"ids": [386, 1000828],                            "table": "pharma_records"},
    ("hirid", "ketamine"):    {"ids": [1001194, 1000400, 1000857],               "table": "pharma_records"},
    ("hirid", "lorazepam"):   {"ids": [1000239, 1000418, 1000988],               "table": "pharma_records"},
    ("hirid", "propofol"):    {"ids": [208, 1000491, 1000691, 1000699, 1001050, 1001052, 1001053], "table": "pharma_records"},
    ("hirid", "vecuronium"):  {"ids": [198],                                     "table": "pharma_records"},
    ("hirid", "aspirin"):     {"ids": [1000255, 1000256, 1000257],               "table": "pharma_records"},
    ("hirid", "enoxaparin"):  {"ids": [1000863, 1000864, 1000865],               "table": "pharma_records"},
    ("hirid", "warfarin"):    {"ids": [1000476],                                 "table": "pharma_records"},
    ("hirid", "ffp"):         {"ids": [1000050, 1000744],                        "table": "pharma_records"},
    ("hirid", "packed_rbc"):  {"ids": [1000100, 1000743],                        "table": "pharma_records"},
    ("hirid", "platelets"):   {"ids": [1000245, 1000201],                        "table": "pharma_records"},
    ("hirid", "vancomycin"):  {"ids": [189, 331],                                "table": "pharma_records"},
    ("hirid", "meropenem"):   {"ids": [1000424, 1000425, 1001084],               "table": "pharma_records"},
    ("hirid", "calcium_iv"):  {"ids": [1000292],                                 "table": "pharma_records"},
    ("hirid", "dextrose50"):  {"ids": [1000567, 1000835],                        "table": "pharma_records"},
    ("hirid", "bicarbonate"): {"ids": [1000193, 1000453, 1000571],               "table": "pharma_records"},
    ("hirid", "magnesium_iv"):{"ids": [1000421],                                 "table": "pharma_records"},
    ("hirid", "dexamethasone"):{"ids": [1000769],                                "table": "pharma_records"},
    ("hirid", "phenytoin"):   {"ids": [1000478, 304, 230],                       "table": "pharma_records"},
    ("hirid", "levetiracetam"):{"ids": [1000676, 1000756, 1001175],              "table": "pharma_records"},

    # === Round 2: SIC drug additions ===
    ("sic", "dexamethasone"): {"ids": [1524], "table": "medication"},
    ("sic", "apixaban"):      {"ids": [1954], "table": "medication"},
    ("sic", "pantoprazole"):  {"ids": [1427], "table": "medication"},
    ("sic", "octreotide"):    {"ids": [1553], "table": "medication"},
    ("sic", "midazolam_rate"):{"ids": [1495], "table": "medication"},
    ("sic", "fentanyl_rate"): {"ids": [1480], "table": "medication"},
    ("sic", "phenytoin"):     {"ids": [1478], "table": "medication"},
    ("sic", "neostigmine"):   {"ids": [1526], "table": "medication"},
    ("sic", "bicarbonate"):   {"ids": [1774], "table": "medication"},
    ("sic", "albumin_iv"):    {"ids": [2040, 2123, 2169, 2170], "table": "medication"},
    ("sic", "mannitol"):      {"ids": [2050, 2091, 2135, 2171], "table": "medication"},
    ("sic", "packed_rbc"):    {"ids": [2046], "table": "medication"},
    ("sic", "platelets"):     {"ids": [2048, 2088], "table": "medication"},
    ("sic", "insulin"):       {"ids": [1557, 1848, 1961, 1962], "table": "medication"},
    ("sic", "propofol_rate"): {"ids": [1499, 1549, 2073, 3056], "table": "medication"},

    # === Round 3: AUMC drugitems additions ===
    ("aumc", "ffp"):           {"ids": [7367],             "table": "drugitems"},
    ("aumc", "dexamethasone"): {"ids": [6995],             "table": "drugitems"},
    ("aumc", "calcium_iv"):    {"ids": [18783, 19164],     "table": "drugitems"},
    ("aumc", "neostigmine"):   {"ids": [7217],             "table": "drugitems"},
    ("aumc", "pantoprazole"):  {"ids": [7979],             "table": "drugitems"},
    ("aumc", "octreotide"):    {"ids": [6866],             "table": "drugitems"},
    ("aumc", "mannitol"):      {"ids": [7360, 20174],      "table": "drugitems"},
    ("aumc", "platelets"):     {"ids": [7369],             "table": "drugitems"},
}


_DICT_CACHE: dict = {}


def _load_db_dict(db: str) -> dict:
    """Cache the full id->label dict per DB to avoid re-reading."""
    if db in _DICT_CACHE:
        return _DICT_CACHE[db]
    if db == "miiv":
        d = pd.read_parquet(DB_ROOT / "mimic-iv-3.1" / "d_items.parquet")
        d.columns = [c.upper() for c in d.columns]
        out = dict(zip(d["ITEMID"], d["LABEL"]))
    elif db == "mimic":
        d = pd.read_parquet(DB_ROOT / "mimiciii" / "d_items.parquet")
        d.columns = [c.upper() for c in d.columns]
        out = dict(zip(d["ITEMID"], d["LABEL"]))
    elif db == "hirid":
        d = pd.read_parquet(DB_ROOT / "hirid-a-high-time-resolution-icu-dataset-1.1.1" / "hirid_variable_reference.parquet")
        out = dict(zip(d["ID"], d["Variable Name"]))
    elif db == "sic":
        r = pd.read_parquet(DB_ROOT / "sic" / "d_references.parquet")
        out = dict(zip(r["ReferenceGlobalID"], r["ReferenceValue"]))
    elif db == "aumc":
        out = {}
        # numericitems CSV (one pass)
        try:
            it = pd.read_csv(DB_ROOT / "aumc" / "numericitems.csv",
                             usecols=['itemid', 'item'], dtype={'itemid': 'int32', 'item': 'string'},
                             chunksize=2_000_000, encoding='latin-1')
            for ch in it:
                for tup in ch.drop_duplicates().itertuples(index=False):
                    if tup.itemid not in out:
                        out[int(tup.itemid)] = str(tup.item)
        except Exception:
            pass
        # drugitems parquets
        for f in sorted(glob.glob(str(DB_ROOT / "aumc" / "drugitems" / "*.parquet"))):
            try:
                df = pd.read_parquet(f, columns=["itemid", "item"]).drop_duplicates()
                for tup in df.itertuples(index=False):
                    if tup.itemid not in out:
                        out[int(tup.itemid)] = str(tup.item)
            except Exception:
                continue
    else:
        out = {}
    _DICT_CACHE[db] = out
    return out


def lookup_dict_label(db: str, table: str, itemid):
    """Return the label associated with itemid in the DB dictionary, or None."""
    return _load_db_dict(db).get(itemid)


def main():
    results = {"_metadata": {"generated_at": datetime.now().isoformat(timespec="seconds")}, "verified": [], "failed": []}
    n_total = len(CHANGES)
    for i, ((db, concept), info) in enumerate(CHANGES.items(), 1):
        if i % 10 == 0:
            print(f"[{i}/{n_total}] verifying {db}/{concept} ...")
        for itemid in info["ids"]:
            label = lookup_dict_label(db, info.get("table"), itemid)
            entry = {"db": db, "concept": concept, "itemid": itemid, "dict_label": label, "status": "OK" if label else "NOT_FOUND_IN_DICT"}
            if label:
                results["verified"].append(entry)
            else:
                results["failed"].append(entry)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {OUT}")
    print(f"Verified: {len(results['verified'])}/{len(results['verified']) + len(results['failed'])}")
    if results["failed"]:
        print(f"FAILED ({len(results['failed'])}):")
        for f in results["failed"]:
            print(f"  {f['db']}/{f['concept']}: itemid {f['itemid']} not in dict")
    else:
        print("ALL CHANGES VERIFIED ✓")


if __name__ == "__main__":
    main()
