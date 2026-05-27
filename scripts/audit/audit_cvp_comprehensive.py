#!/usr/bin/env python3
"""
Comprehensive CVP audit across 6 ICU databases.

Goal: verify the CVP concept mapping is COMPLETE — i.e. that no CVP source
table or itemid is missed. Search synonyms (CVP, Central Venous Pressure,
CVD/Zentral Veneuze Druk/ZVD/Druck) across every table that could plausibly
record CVP in each DB:

  MIMIC-IV (miiv):  d_items + chartevents
  MIMIC-III (mimic): d_items + chartevents
  eICU:             vitalPeriodic.cvp + vitalAperiodic + nurseCharting (CRITICAL)
  AUMC:             numericitems + listitems + processitems
  HiRID:            variable_reference
  SIC:              d_references

Output: src/data_processing/cvp_itemid_audit.json
"""
from __future__ import annotations

import glob
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DB_ROOT = Path("/Volumes/外置硬盘/databases")
OUT = REPO / "src" / "data_processing" / "cvp_itemid_audit.json"

# Wide synonym list for CVP across language locales
SYN = r"cvp|central\s*venous|c\.?v\.?p\.?|zentral.*venös|zentralvenös|zvd|centraal.*veneuze|centraalvenuze|cvd"


def audit_miiv():
    d = pd.read_parquet(DB_ROOT / "mimic-iv-3.1" / "d_items.parquet")
    d.columns = [c.upper() for c in d.columns]
    m = d['LABEL'].astype(str).str.contains(SYN, case=False, regex=True, na=False)
    hits = d[m][['ITEMID', 'LABEL', 'LINKSTO', 'UNITNAME', 'PARAM_TYPE']].sort_values('ITEMID')
    return hits.to_dict(orient='records')


def audit_mimic_iii():
    d = pd.read_parquet(DB_ROOT / "mimiciii" / "d_items.parquet")
    d.columns = [c.upper() for c in d.columns]
    m = d['LABEL'].astype(str).str.contains(SYN, case=False, regex=True, na=False)
    hits = d[m][['ITEMID', 'LABEL', 'DBSOURCE', 'LINKSTO', 'UNITNAME']].sort_values('ITEMID')
    return hits.to_dict(orient='records')


def audit_eicu():
    out = {}
    # vitalPeriodic: column-based
    cols = list(pd.read_parquet(DB_ROOT / "eicu" / "vitalPeriodic.parquet", columns=None).columns)
    out["vitalPeriodic_columns"] = [c for c in cols if 'cvp' in c.lower() or 'central' in c.lower()]
    # vitalAperiodic
    cols_a = list(pd.read_parquet(DB_ROOT / "eicu" / "vitalAperiodic.parquet", columns=None).columns)
    out["vitalAperiodic_columns"] = [c for c in cols_a if 'cvp' in c.lower() or 'central' in c.lower()]
    # nurseCharting (sharded) - scan all shards for CVP entries
    files = sorted(glob.glob(str(DB_ROOT / "eicu" / "nursecharting" / "*.parquet")))
    nurse_hits = set()
    for f in files:
        df = pd.read_parquet(f, columns=['nursingchartcelltypecat', 'nursingchartcelltypevallabel', 'nursingchartcelltypevalname'])
        for t in df.drop_duplicates().itertuples(index=False):
            if any('cvp' in str(x).lower() or ('central' in str(x).lower() and 'ven' in str(x).lower())
                   for x in (t.nursingchartcelltypecat, t.nursingchartcelltypevallabel, t.nursingchartcelltypevalname)):
                nurse_hits.add((str(t.nursingchartcelltypecat), str(t.nursingchartcelltypevallabel), str(t.nursingchartcelltypevalname)))
    out["nurseCharting_cells"] = sorted(nurse_hits)
    # respiratoryCharts (rare but possible)
    rc = DB_ROOT / "eicu" / "respiratoryCharts.parquet"
    if rc.exists():
        rdf = pd.read_parquet(rc, columns=['respchartvaluelabel'])
        m = rdf['respchartvaluelabel'].astype(str).str.contains('cvp|central.*ven', case=False, regex=True, na=False)
        out["respiratoryCharts_labels"] = sorted(rdf[m]['respchartvaluelabel'].unique())
    return out


def audit_aumc():
    out = {}
    # numericitems
    seen = set()
    it = pd.read_csv(DB_ROOT / "aumc" / "numericitems.csv",
                     usecols=['itemid', 'item'], dtype={'itemid': 'int32', 'item': 'string'},
                     chunksize=2_000_000, encoding='latin-1')
    for ch in it:
        for tup in ch.drop_duplicates().itertuples(index=False):
            seen.add((tup.itemid, tup.item))
    num_hits = [(i, n) for i, n in seen if any(s in str(n).lower() for s in ['cvp', 'centra', 'veneuz', 'cvd'])]
    out["numericitems"] = sorted(num_hits)
    # listitems
    lit = pd.read_csv(DB_ROOT / "aumc" / "listitems.csv",
                      usecols=['itemid', 'item'], dtype={'itemid': 'int32', 'item': 'string'},
                      encoding='latin-1')
    lit_uniq = lit.drop_duplicates()
    lm = lit_uniq['item'].astype(str).str.contains(SYN, case=False, regex=True, na=False)
    out["listitems"] = lit_uniq[lm].to_dict(orient='records')
    # processitems (procedures, unlikely to have CVP measurement but check)
    pi = DB_ROOT / "aumc" / "processitems.csv"
    if pi.exists():
        pdf = pd.read_csv(pi, usecols=['itemid', 'item'], dtype={'itemid': 'int32', 'item': 'string'}, encoding='latin-1')
        pm = pdf.drop_duplicates()['item'].astype(str).str.contains(SYN, case=False, regex=True, na=False)
        out["processitems"] = pdf.drop_duplicates()[pm].to_dict(orient='records')
    return out


def audit_hirid():
    d = pd.read_parquet(DB_ROOT / "hirid-a-high-time-resolution-icu-dataset-1.1.1" / "hirid_variable_reference.parquet")
    m = d['Variable Name'].astype(str).str.contains(SYN, case=False, regex=True, na=False)
    return d[m].to_dict(orient='records')


def audit_sic():
    r = pd.read_parquet(DB_ROOT / "sic" / "d_references.parquet")
    m = r.astype(str).apply(lambda x: x.str.contains(SYN, case=False, regex=True, na=False)).any(axis=1)
    hits = r[m][['ReferenceGlobalID', 'ReferenceValue', 'ReferenceName', 'ReferenceDescription', 'ReferenceUnit']]
    return hits.to_dict(orient='records')


def main():
    audit = {
        "_metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "purpose": "Comprehensive CVP itemid audit — find ALL source tables/items across 6 DBs",
            "synonym_pattern": SYN,
        },
        "miiv_d_items": audit_miiv(),
        "mimic_iii_d_items": audit_mimic_iii(),
        "eicu_all_tables": audit_eicu(),
        "aumc_all_tables": audit_aumc(),
        "hirid_variable_reference": audit_hirid(),
        "sic_d_references": audit_sic(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, indent=2, default=str))
    print(f"Wrote {OUT}")
    # Summary
    print("\n=== SUMMARY ===")
    print(f"MIMIC-IV: {len([x for x in audit['miiv_d_items'] if 'Alarm' not in x['LABEL']])} non-alarm CVP items")
    print(f"MIMIC-III: {len([x for x in audit['mimic_iii_d_items'] if 'Alarm' not in x['LABEL']])} non-alarm CVP items")
    print(f"eICU vitalPeriodic cols: {audit['eicu_all_tables']['vitalPeriodic_columns']}")
    print(f"eICU vitalAperiodic cols: {audit['eicu_all_tables']['vitalAperiodic_columns']}")
    print(f"eICU nurseCharting cells: {len(audit['eicu_all_tables']['nurseCharting_cells'])}")
    for x in audit['eicu_all_tables']['nurseCharting_cells']:
        print(f"  - {x}")
    print(f"AUMC numericitems: {audit['aumc_all_tables']['numericitems']}")
    print(f"AUMC listitems: {audit['aumc_all_tables']['listitems']}")
    print(f"HiRID variables: {audit['hirid_variable_reference']}")
    print(f"SIC references: {audit['sic_d_references']}")


if __name__ == "__main__":
    main()
