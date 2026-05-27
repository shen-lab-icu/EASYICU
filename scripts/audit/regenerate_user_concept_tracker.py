#!/usr/bin/env python3
"""
Regenerate the user-added concepts tracker file.

Diffs EasyICU's concept-dict.json against ricu's installed concept-dict.json
to identify user-added concepts, then writes/updates the persistent tracker
at /Users/haibo/Documents/GitHub/EASYICU/docs/user_added_concepts_tracker.md.

To mark a concept as audited:
  1. Add an entry to the AUDIT dict below
  2. Re-run this script

Output:
  /Users/haibo/Documents/GitHub/EASYICU/docs/user_added_concepts_tracker.md
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

RICU_DICT = Path("/Library/Frameworks/R.framework/Versions/4.6/Resources/library/ricu/extdata/config/concept-dict.json")
EASY_DICT = Path("/Users/haibo/Documents/GitHub/EASYICU/src/easyicu/data/concept-dict.json")
OUT = Path("/Users/haibo/Documents/GitHub/EASYICU/docs/user_added_concepts_tracker.md")

DBS = ["miiv", "mimic", "eicu", "aumc", "hirid", "sic"]
DB_LABEL = {"miiv": "MIIV", "mimic": "MIIIv", "eicu": "eICU", "aumc": "AUMC", "hirid": "HiRID", "sic": "SIC"}

# Per-concept audit state — keyed by concept name.
# Values: dict with per-DB status ('✅' | '🟡' | '⚠️' | '❌' | '—') + 'notes' field.
# Add to this dict when a concept has been audited.
AUDIT: dict = {
    "cvp": {
        "miiv": "✅", "mimic": "✅", "eicu": "⚠️", "aumc": "✅", "hirid": "✅", "sic": "✅",
        "notes": "2026-05-27. AUMC: +20926 CVDm-gekoppeld (10× rows). eICU: nurseCharting source added (50/801 nc-only patients reached; vallabel limitation upstream). MIIV/MIIIv/HiRID/SIC verified complete.",
    },
    # ============= Phase 1: sofa2-dict layer (21 composite concepts) =============
    # These live in sofa2-dict.json (not concept-dict.json) — composite via callback.
    # All 21 verified structurally: callbacks exist, dependencies resolvable.
    # 8 overlap with main concept-dict (uo_6h/12h/24h, ecmo, ecmo_indication, mech_circ_support, rrt, rrt_criteria)
    # ============= Phase 2: ventilator (11 concepts) =============
    "peep": {
        "miiv": "✅", "mimic": "⚠️", "eicu": "✅", "aumc": "✅", "hirid": "✅", "sic": "✅",
        "notes": "2026-05-27. MIIV ZAuto Peep Level 224699 is Auto-PEEP (distinct concept). MIMIC-III has 535/543 mismapped (are PIP/Plateau); 505/506 (true PEEP) missing. Non-blocking since MIMIC-III not in primary thesis analysis.",
    },
    "tidal_vol": {
        "miiv": "✅", "mimic": "⚠️", "eicu": "—", "aumc": "✅", "hirid": "✅", "sic": "✅",
        "notes": "2026-05-27. AUMC +16243 Zephyros Vte added. AUMC 12360 Insp.tidal (2) skipped (secondary recording). MIMIC-III 36 candidates vs 5 mapped — most are PCV Exh/Insp Vt variants; needs decision.",
    },
    "tidal_vol_set": {
        "miiv": "✅", "mimic": "✅", "eicu": "—", "aumc": "✅", "hirid": "✅", "sic": "🟡",
        "notes": "2026-05-27. All DBs verified except SIC (audit shows EXTRA — synonym list may be too narrow).",
    },
    "pip": {
        "miiv": "✅", "mimic": "⚠️", "eicu": "—", "aumc": "✅", "hirid": "✅", "sic": "🟡",
        "notes": "2026-05-27. MIMIC-III includes 506 (PEEP Set) by mistake. AUMC complete. SIC needs verification.",
    },
    "plateau_pres": {
        "miiv": "✅", "mimic": "✅", "eicu": "—", "aumc": "✅", "hirid": "✅", "sic": "❌",
        "notes": "2026-05-27. MIIV 228866 candidate is IABP-specific (not airway plateau). SIC not mapped.",
    },
    "mean_airway_pres": {
        "miiv": "✅", "mimic": "⚠️", "eicu": "—", "aumc": "✅", "hirid": "✅", "sic": "✅",
        "notes": "2026-05-27. MIMIC-III missing 1209/1672 (HFO MAP / MEAN AIRWAY PRESS). AUMC 12362 (2) skipped.",
    },
    "minute_vol": {
        "miiv": "✅", "mimic": "⚠️", "eicu": "—", "aumc": "⚠️", "hirid": "❌", "sic": "⚠️",
        "notes": "2026-05-27. AUMC +8875 Mv Spontaan added (12276/12357 skipped, insp+secondary variants). MIMIC-III major gap (32 candidates vs 3 mapped). SIC 2019 MV(L) missing. HiRID not mapped.",
    },
    "vent_rate": {
        "miiv": "✅", "mimic": "✅", "eicu": "—", "aumc": "✅", "hirid": "✅", "sic": "✅",
        "notes": "2026-05-27. EXTRA flags from audit reflect narrow synonym list; manual review confirms mappings are correct.",
    },
    "etco2": {
        "miiv": "✅", "mimic": "✅", "eicu": "—", "aumc": "✅", "hirid": "✅", "sic": "✅",
        "notes": "2026-05-27. MIIV/MIMIC 228641 candidate is 'Clinical indication' (not measurement) — correctly excluded.",
    },
    "compliance": {
        "miiv": "✅", "mimic": "✅", "eicu": "—", "aumc": "✅", "hirid": "❌", "sic": "✅",
        "notes": "2026-05-27. AUMC +12561 Cdyn (dynamic compliance) added. HiRID has no compliance variable in reference.",
    },
    "driving_pres": {
        "miiv": "—", "mimic": "—", "eicu": "—", "aumc": "—", "hirid": "—", "sic": "—",
        "notes": "2026-05-27. Derived concept (Plateau − PEEP); not mapped to raw itemids. Compute downstream from plateau_pres + peep.",
    },
    "ps": {
        "miiv": "🟡", "mimic": "🟡", "eicu": "—", "aumc": "🟡", "hirid": "❌", "sic": "🟡",
        "notes": "2026-05-27. EXTRA flags reflect narrow regex; manual review needed. HiRID not mapped.",
    },
    # ============= Phase 3: renal (8 derived) =============
    "kdigo_aki":     {"miiv": "—","mimic": "—","eicu": "—","aumc": "—","hirid": "—","sic": "—", "notes": "2026-05-27. Derived via _callback_kdigo_aki (concept_callbacks.py:7379). Deps: crea/urine/weight/rrt all OK across DBs."},
    "kdigo_creat":   {"miiv": "—","mimic": "—","eicu": "—","aumc": "—","hirid": "—","sic": "—", "notes": "2026-05-27. Derived via kdigo_creatinine (kdigo_aki.py:41). Dep: crea OK across DBs."},
    "kdigo_uo":      {"miiv": "—","mimic": "—","eicu": "—","aumc": "—","hirid": "—","sic": "—", "notes": "2026-05-27. Derived via kdigo_uo (kdigo_aki.py:197). Deps: urine/weight OK."},
    "uo_6h":         {"miiv": "—","mimic": "—","eicu": "—","aumc": "—","hirid": "—","sic": "—", "notes": "2026-05-27. Derived via uo_6h callback (callbacks.py:1472)."},
    "uo_12h":        {"miiv": "—","mimic": "—","eicu": "—","aumc": "—","hirid": "—","sic": "—", "notes": "2026-05-27. Derived via uo_12h callback."},
    "uo_24h":        {"miiv": "—","mimic": "—","eicu": "—","aumc": "—","hirid": "—","sic": "—", "notes": "2026-05-27. Derived via uo_24h callback."},
    "rrt":           {"miiv": "✅","mimic": "✅","eicu": "✅","aumc": "✅","hirid": "✅","sic": "✅", "notes": "2026-05-27. Raw RRT-active itemids across all 6 DBs."},
    "rrt_criteria":  {"miiv": "—","mimic": "—","eicu": "—","aumc": "—","hirid": "—","sic": "—", "notes": "2026-05-27. Derived via rrt_criteria callback (callbacks_missing.py:14). Composite of crea/uo_*h/potassium/ph/bicarb/rrt."},
    # ============= Phase 4: vitals + chemistry + output =============
    "pulse_pressure":            {"miiv":"—","mimic":"—","eicu":"—","aumc":"—","hirid":"—","sic":"—", "notes": "2026-05-27. Derived (SBP−DBP) via _callback_pulse_pressure."},
    "anion_gap":                 {"miiv":"—","mimic":"—","eicu":"—","aumc":"—","hirid":"—","sic":"—", "notes": "2026-05-27. Derived (Na−Cl−HCO3) via _callback_anion_gap. Deps: na/cl/bicar OK."},
    "bicarb":                    {"miiv":"✅","mimic":"✅","eicu":"✅","aumc":"✅","hirid":"✅","sic":"✅", "notes": "2026-05-27. Alias of bicar — sources pulled directly from bicar."},
    "potassium":                 {"miiv":"✅","mimic":"✅","eicu":"✅","aumc":"✅","hirid":"✅","sic":"✅", "notes": "2026-05-27. Alias of k — sources direct."},
    "fluid_balance":             {"miiv":"—","mimic":"—","eicu":"—","aumc":"—","hirid":"—","sic":"—", "notes": "2026-05-27. Derived (total_input_ml − urine) via _callback_fluid_balance_hourly. See docs/fluid_balance_design.md."},
    "fluid_balance_cumulative":  {"miiv":"—","mimic":"—","eicu":"—","aumc":"—","hirid":"—","sic":"—", "notes": "2026-05-27. Derived (cumsum of fluid_balance) via _callback_fluid_balance_cumulative."},
    "total_input_ml":            {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"❌","sic":"❌", "notes": "2026-05-27. By design HiRID/SIC pending (see description). MIIV/MIIIv/eICU/AUMC have raw sources; not yet itemid-completeness audited."},
    # ============= Phase 5: medications — coverage scan, not full audit =============
    # (45 drugs; coverage summary below. Per-drug itemid-completeness audit not performed.)
    "apixaban":      {"miiv":"❌","mimic":"❌","eicu":"❌","aumc":"❌","hirid":"—","sic":"✅", "notes": "2026-05-27. DOAC genuinely absent in MIIV/MIMIC inputevents and HiRID Pharma (newer drug). SIC has DrugID 1954."},
    "dexamethasone": {"miiv":"✅","mimic":"✅","eicu":"🟡","aumc":"❌","hirid":"✅","sic":"✅", "notes": "2026-05-27 Round 2. MIIV/MIIIv added via prescriptions table (inputevents has none; only dextrose/dexmedetomidine). HiRID 1000769 Fortecortin Tbl. SIC 1524. AUMC remains unmapped."},
    "aspirin":       {"miiv":"❌","mimic":"❌","eicu":"🟡","aumc":"🟡","hirid":"✅","sic":"🟡", "notes": "2026-05-27. HiRID added (3 Aspirin Tbl). MIIV/MIIIv still missing — would need prescriptions table (similar to dexamethasone pattern)."},
    "ffp":           {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"❌","hirid":"✅","sic":"❌", "notes": "2026-05-27 Round 2. HiRID added (Transfusion of plasma FFP, 2 IDs). AUMC and SIC confirmed absent from d_references / drugs."},
    "phenytoin":     {"miiv":"🟡","mimic":"🟡","eicu":"❌","aumc":"❌","hirid":"✅","sic":"✅", "notes": "2026-05-27 Round 2. HiRID added (Phenhydan Inf Lsg + tabl + inj). SIC 1478. eICU needs medication-table regex add."},
    "nicardipine":   {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"—","sic":"—", "notes": "2026-05-27. Confirmed absent in HiRID Pharma (Nimotop=nimodipine, different drug) and SIC d_references."},
    "ketamine":      {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"✅","sic":"🟡", "notes": "2026-05-27 Round 2. HiRID added (3 Ketalar items)."},
    "cisatracurium": {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"❌","hirid":"—","sic":"🟡", "notes": "2026-05-27. HiRID has only Tracrium (=atracurium, different drug)."},
    "neostigmine":   {"miiv":"🟡","mimic":"❌","eicu":"🟡","aumc":"🟡","hirid":"—","sic":"✅", "notes": "2026-05-27 Round 2. SIC 1526 added. HiRID confirmed absent."},
    "albumin_iv":    {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"❌","hirid":"—","sic":"✅", "notes": "2026-05-27 Round 2. SIC: 4 Humanalbumin forms (2040/2123/2169/2170). HiRID confirmed absent."},
    "mannitol":      {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"—","sic":"✅", "notes": "2026-05-27 Round 2. SIC: 4 Mannit forms. HiRID confirmed absent."},
    "pantoprazole":  {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"—","sic":"✅", "notes": "2026-05-27 Round 2. SIC 1427. HiRID absent (PPI family genuinely missing from Pharma)."},
    "octreotide":    {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"—","sic":"✅", "notes": "2026-05-27 Round 2. SIC 1553. HiRID absent."},
    "diltiazem":     {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"✅","sic":"🟡", "notes": "2026-05-27 Round 2. HiRID: 2 Diltiazem Tbl forms."},
    "esmolol":       {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"✅","sic":"🟡", "notes": "2026-05-27 Round 2. HiRID: Esmolol Inj + Brevibloc Perfusor."},
    "labetalol":     {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"✅","sic":"🟡", "notes": "2026-05-27 Round 2. HiRID: Trandate inj + Perfusor."},
    "nitroglycerin": {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"✅","sic":"🟡", "notes": "2026-05-27 Round 2. HiRID: 1 PO capsule (no IV form in HiRID Pharma)."},
    "lorazepam":     {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"✅","sic":"🟡", "notes": "2026-05-27 Round 2. HiRID: 3 Temesta forms."},
    "propofol":      {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"✅","sic":"🟡", "notes": "2026-05-27 Round 2. HiRID: 7 Disoprivan/Propofol forms."},
    "propofol_rate": {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"✅","sic":"✅", "notes": "2026-05-27 Round 2. HiRID + SIC added (4 forms each)."},
    "vecuronium":    {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"✅","sic":"❌", "notes": "2026-05-27 Round 2. HiRID Norcuron inj."},
    "enoxaparin":    {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"❌","hirid":"✅","sic":"🟡", "notes": "2026-05-27 Round 2. HiRID: 5 Clexane SC forms."},
    "warfarin":      {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"❌","hirid":"✅","sic":"🟡", "notes": "2026-05-27 Round 2. HiRID Marcoumar (warfarin equivalent)."},
    "packed_rbc":    {"miiv":"🟡","mimic":"🟡","eicu":"❌","aumc":"🟡","hirid":"✅","sic":"✅", "notes": "2026-05-27 Round 2. HiRID + SIC added."},
    "platelets":     {"miiv":"🟡","mimic":"🟡","eicu":"❌","aumc":"🟡","hirid":"✅","sic":"✅", "notes": "2026-05-27 Round 2. HiRID + SIC added."},
    "vancomycin":    {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"✅","sic":"🟡", "notes": "2026-05-27 Round 2. HiRID: Vancocin Amp + oral Kps."},
    "meropenem":     {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"✅","sic":"🟡", "notes": "2026-05-27 Round 2. HiRID: 3 Meronem/Meropenem forms."},
    "calcium_iv":    {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"❌","hirid":"✅","sic":"🟡", "notes": "2026-05-27 Round 2. HiRID: Calcium Sandoz Lsg 10% IV only (excluded PO Brausetabl; Calciparine=heparin filtered)."},
    "dextrose50":    {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"❌","hirid":"✅","sic":"🟡", "notes": "2026-05-27 Round 2. HiRID: Glucose 50% + 20%."},
    "bicarbonate":   {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"✅","sic":"✅", "notes": "2026-05-27 Round 2. HiRID: 3 Na-Bicarbonat forms. SIC 1774 Natriumhydrogencarbonat."},
    "magnesium_iv":  {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"✅","sic":"🟡", "notes": "2026-05-27 Round 2. HiRID: Magnesium Sulfat 50%."},
    "levetiracetam": {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"✅","sic":"🟡", "notes": "2026-05-27 Round 2. HiRID: 3 Keppra/Levetiracetam forms."},
    "insulin":       {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"🟡","sic":"✅", "notes": "2026-05-27 Round 2. SIC: 4 insulin forms (regular, glargine, Aspart-Protamin mixtures)."},
    "midazolam_rate":{"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"🟡","sic":"✅", "notes": "2026-05-27 Round 2. SIC: 1495 Midazolam (rate via AmountPerMinute)."},
    "fentanyl_rate": {"miiv":"🟡","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"🟡","sic":"✅", "notes": "2026-05-27 Round 2. SIC: 1480 FentaNYL only (remifentanil/sufentanil/alfentanil are different drugs, not included)."},
    # ============= Round 3: well-covered medications completeness audits =============
    "amiodarone":       {"miiv":"✅","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"🟡","sic":"🟡", "notes": "2026-05-27 Round 3. MIIV verified complete (4/4 in inputevents)."},
    "dexmedetomidine":  {"miiv":"✅","mimic":"🟡","eicu":"🟡","aumc":"❌","hirid":"🟡","sic":"❌", "notes": "2026-05-27 Round 3. MIIV verified (2/2 Precedex). AUMC has only research-protocol Dexmedetomidine/Placebo (excluded). SIC absent."},
    "dextrose50":       {"miiv":"✅","mimic":"🟡","eicu":"🟡","aumc":"❌","hirid":"✅","sic":"🟡", "notes": "2026-05-27 Round 3. MIIV verified (1/1). AUMC has Glucose 10% only (not 50%)."},
    "fentanyl":         {"miiv":"✅","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"🟡","sic":"🟡", "notes": "2026-05-27 Round 3. MIIV verified (3/3, excludes sufentanil/alfentanil)."},
    "furosemide":       {"miiv":"✅","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"🟡","sic":"🟡", "notes": "2026-05-27 Round 3. MIIV verified complete (2/2 Lasix)."},
    "heparin":          {"miiv":"✅","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"🟡","sic":"🟡", "notes": "2026-05-27 Round 3. MIIV verified — Impella/CRRT-Prefilter heparin items excluded by design (concept is systemic anticoagulation, not circuit flushes)."},
    "midazolam":        {"miiv":"✅","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"🟡","sic":"🟡", "notes": "2026-05-27 Round 3. MIIV verified complete (1/1)."},
    "milrinone":        {"miiv":"✅","mimic":"🟡","eicu":"🟡","aumc":"❌","hirid":"🟡","sic":"🟡", "notes": "2026-05-27 Round 3. MIIV verified (1/1 Primacor). AUMC confirmed absent."},
    "morphine":         {"miiv":"✅","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"🟡","sic":"🟡", "notes": "2026-05-27 Round 3. MIIV verified complete (1/1)."},
    "potassium_iv":     {"miiv":"✅","mimic":"🟡","eicu":"🟡","aumc":"🟡","hirid":"❌","sic":"🟡", "notes": "2026-05-27 Round 3. MIIV verified (4/4 KCl + K-phosphate). HiRID has only PO retard tabs (excluded — concept is IV)."},
    "rocuronium":       {"miiv":"✅","mimic":"❌","eicu":"🟡","aumc":"🟡","hirid":"🟡","sic":"🟡", "notes": "2026-05-27 Round 3. MIIV verified (1/1). MIIIv absent."},
    "uo_6h":   {"miiv":"—","mimic":"—","eicu":"—","aumc":"—","hirid":"—","sic":"—", "notes": "Derived via uo_6h callback (callbacks.py:1472). Deps: urine, weight — all OK."},
    "uo_12h":  {"miiv":"—","mimic":"—","eicu":"—","aumc":"—","hirid":"—","sic":"—", "notes": "Derived via uo_12h callback. Deps: urine, weight."},
    "uo_24h":  {"miiv":"—","mimic":"—","eicu":"—","aumc":"—","hirid":"—","sic":"—", "notes": "Derived via uo_24h callback. Deps: urine, weight."},
    # ============= Phase 6: remaining 5 =============
    "ecmo":              {"miiv":"🟡","mimic":"❌","eicu":"🟡","aumc":"🟡","hirid":"—","sic":"❌", "notes": "2026-05-27. HiRID has no ECMO variable in reference (structural). MIMIC-III gap real. SIC gap unknown."},
    "ecmo_indication":   {"miiv":"🟡","mimic":"❌","eicu":"🟡","aumc":"🟡","hirid":"—","sic":"❌", "notes": "2026-05-27. Same as ecmo — HiRID structural absence."},
    "infection_icd":     {"miiv":"—","mimic":"—","eicu":"🟡","aumc":"—","hirid":"—","sic":"—", "notes": "2026-05-27. By design eICU-only — diagnosis-text infection proxy (Angus 2001 ICD)."},
    "mech_circ_support": {"miiv":"🟡","mimic":"❌","eicu":"🟡","aumc":"🟡","hirid":"—","sic":"❌", "notes": "2026-05-27. HiRID has no IABP/LVAD/Impella in reference (structural)."},
    "sedated_gcs":       {"miiv":"❌","mimic":"❌","eicu":"🟡","aumc":"🟡","hirid":"❌","sic":"❌", "notes": "2026-05-27. By design: only eICU/AUMC record GCS-before-sedation separately."},
}

# Audit history log entries (append-only).
AUDIT_HISTORY: list[tuple[str, str, str, str]] = [
    ("2026-05-27", "cvp", "initial mapping + comprehensive 6-DB audit",
     "AUMC: +20926 (CVDm-gekoppeld, 10× rows) added. eICU: nurseCharting source added but loader limitation = 50/801 nc-only patients. MIIV/MIIIv/HiRID/SIC verified complete."),
    ("2026-05-27", "Phase 1: sofa2-dict (21)", "structural verification",
     "All callbacks registered (sofa2.py, callbacks.py, sepsis_sofa2.py, callbacks_missing.py). All dependencies resolvable. 8 concepts overlap with main dict (uo_*h, ecmo*, mech_circ, rrt*)."),
    ("2026-05-27", "Phase 2: ventilator (11)", "strict 6-DB audit",
     "AUMC: +12561 Cdyn / +8875 Mv Spontaan / +16243 Zephyros Vte added. MIMIC-III multiple itemid mismatches found (peep 535/543 are PIP/Plateau; pip 506 is PEEP-Set; ps 502 is PCV Vt) — non-blocking, MIMIC-III not in primary analyses. driving_pres confirmed derived."),
    ("2026-05-27", "Phase 3: renal (8)", "callback registration verification",
     "All 8 callbacks resolve via concept_callbacks.py registry. Dependencies (crea, urine, weight, potassium, ph, bicarb, rrt) all verified in concept-dict."),
    ("2026-05-27", "Phase 4: vitals+chem+output (7)", "callback + alias verification",
     "anion_gap, pulse_pressure, fluid_balance_hourly, fluid_balance_cumulative callbacks all registered. bicarb/potassium are aliases (verified). total_input_ml HiRID/SIC by-design pending."),
    ("2026-05-27", "Phase 5: medications (45)", "coverage scan (not full itemid audit)",
     "Per-DB coverage: 6/6 = 6 drugs (amiodarone/fentanyl/midazolam/morphine/furosemide/heparin); 5/6 = 18; 4/6 = 16; 3/6 = 3 (aspirin/ffp/phenytoin); 2/6 = 1 (dexamethasone); 1/6 = 1 (apixaban). HiRID systematic gap: 29/45 missing (Pharma table needs per-drug mapping). Per-drug itemid completeness audit deferred."),
    ("2026-05-27", "Phase 6: remaining (5)", "structural verification",
     "infection_icd by-design eICU-only. sedated_gcs by-design eICU+AUMC. ecmo/ecmo_indication/mech_circ_support: HiRID has no corresponding variables in reference (structural). MIMIC-III/SIC gaps real but lower priority."),
    ("2026-05-27", "Round 2: HiRID Pharma batch (20 drugs)", "systematic Pharma table mapping",
     "HiRID Pharma reference has 565 items; matched 20/29 missing drugs (diltiazem/esmolol/labetalol/nitroglycerin/ketamine/lorazepam/propofol*/vecuronium/aspirin/enoxaparin/warfarin/ffp/packed_rbc/platelets/vancomycin/meropenem/calcium_iv/dextrose50/bicarbonate/magnesium_iv/dexamethasone/phenytoin/levetiracetam). Confirmed structural absence: nicardipine (Nimotop=nimodipine, different drug), cisatracurium (Tracrium=atracurium), neostigmine, apixaban, albumin_iv, mannitol, pantoprazole, octreotide. HiRID meds coverage 16→36/45."),
    ("2026-05-27", "Round 2: MIMIC-III ventilator itemid corrections", "audit-driven mismatch fix",
     "peep: removed 535 (=PIP) and 543 (=Plateau); added 505/506 (true PEEP/PEEP Set). pip: removed 506 (=PEEP Set). ps: removed 502 (=PCV Insp Vt); added 578/6339/7332/7587/7595 (real PS items). tidal_vol: added 501/502 (PCV Exh/Insp Vt — actual measurements)."),
    ("2026-05-27", "Round 2: SIC drugs batch (15 additions)", "d_references drug lookup",
     "Added: dexamethasone (1524), apixaban (1954), pantoprazole (1427), octreotide (1553), midazolam_rate (1495), fentanyl_rate (1480 only), phenytoin (1478), neostigmine (1526), bicarbonate (1774), albumin_iv (4 forms), mannitol (4 forms), packed_rbc (2046), platelets (2 forms), insulin (4 forms), propofol_rate (4 forms). Confirmed absent in SIC: ffp, nicardipine, plateau_pres drug (different concept). SIC meds 26→41/45."),
    ("2026-05-27", "Round 2: dexamethasone MIIV prescriptions", "table-level fix",
     "MIIV inputevents has NO dexamethasone (only dextrose/dexmedetomidine). Added prescriptions.drug regex source excluding ophthalmic/topical preparations. MIIV meds 28→43/45."),
    ("2026-05-27", "Round 3: AUMC drugitems batch (8 additions)", "drugitems table lookup",
     "AUMC drugitems has 1117 unique drugs; added: ffp (7367), dexamethasone (6995), calcium_iv (18783/19164), neostigmine (7217), pantoprazole (7979 — excluded Esomep/Omep), octreotide (6866), mannitol (7360/20174), platelets (7369). Confirmed absent in AUMC: milrinone, enoxaparin, cisatracurium, apixaban, albumin_iv, packed_rbc, phenytoin. Warfarin: only Acenocoumarol/Sintrom in AUMC (different VKA, not added). AUMC meds 33→36/45."),
    ("2026-05-27", "Round 3: eICU phenytoin + MIIV/MIMIC aspirin", "alternative-table sources",
     "eICU phenytoin: added admissiondrug regex (7 matches: CEREBYX/DILANTIN/FOSPHENYTOIN/PHENYTOIN variants). MIIV/MIIIv aspirin: added prescriptions regex (46 matches in MIIV prescriptions). MIIV meds 43→44/45; eICU meds 41→42/45."),
    ("2026-05-27", "Round 3: 11 pending meds + 3 uo_*h", "MIIV completeness audit",
     "MIIV inputevents itemid completeness verified for amiodarone/dexmedetomidine/dextrose50/fentanyl/furosemide/heparin/midazolam/milrinone/morphine/potassium_iv/rocuronium — all complete (heparin Impella/CRRT-circuit items intentionally excluded). uo_6/12/24h marked as derived (— for all DBs, callback-based by design)."),
]


def build():
    ricu = json.loads(RICU_DICT.read_text())
    easy = json.loads(EASY_DICT.read_text())
    user_added = sorted(set(easy) - set(ricu))

    lines = []
    lines.append("# User-added EasyICU concepts (not in ricu)\n")
    lines.append(f"> Tracker file. Maintained continuously. Last regenerated: {date.today().isoformat()}\n")
    lines.append(f"> Source-of-truth diff: `concept-dict.json` (EasyICU, {len(easy)} concepts) vs ricu R package ({len(ricu)} concepts).\n")
    lines.append(f"> User-added count: **{len(user_added)}**.\n")

    lines.append("\n## Legend\n")
    for sym, meaning in [
        ("✅", "mapped & audited (itemids complete)"),
        ("🟡", "mapped, not audited (correctness unverified)"),
        ("⚠️", "mapped but known incomplete (audit found missing itemids)"),
        ("❌", "NOT mapped (DB not covered for this concept)"),
        ("—",  "DB not applicable (e.g. derived concept, eICU column-based)"),
    ]:
        lines.append(f"- {sym} {meaning}\n")

    lines.append("\n## Status table\n")
    lines.append("| Concept | Category | " + " | ".join(DB_LABEL[d] for d in DBS) + " | Audit notes |\n")
    lines.append("|---|---|" + "|".join("---" for _ in DBS) + "|---|\n")
    for c in user_added:
        cobj = easy[c]
        cat = cobj.get("category", "?")
        audit = AUDIT.get(c, {})
        cells = []
        for db in DBS:
            if db in audit:
                cells.append(audit[db])
            else:
                src = cobj.get("sources", {}).get(db)
                cells.append("🟡" if src else "❌")
        notes = audit.get("notes", "")
        lines.append(f"| `{c}` | {cat} | " + " | ".join(cells) + f" | {notes} |\n")

    lines.append("\n## Per-DB mapping coverage summary\n")
    lines.append("| DB | mapped | unmapped | coverage |\n|---|---|---|---|\n")
    for db in DBS:
        n = sum(1 for c in user_added if easy[c].get("sources", {}).get(db))
        lines.append(f"| {DB_LABEL[db]} | {n}/{len(user_added)} | {len(user_added)-n}/{len(user_added)} | {100*n/len(user_added):.0f}% |\n")

    lines.append("\n## Category breakdown\n")
    cats = Counter(easy[c].get("category", "?") for c in user_added)
    for cat, n in cats.most_common():
        lines.append(f"- **{cat}**: {n}\n")

    lines.append("\n## High-priority items needing attention\n\n")
    lines.append("These 77 user-added concepts have **no validation from ricu's upstream** — every mapping below needs human review. Priority order (impact × ease):\n\n")
    lines.append("1. **`anion_gap`** — 0/6 DBs mapped. Derived concept (Na − Cl − HCO3); decide derive-vs-extract.\n")
    lines.append("2. **`driving_pres`** — 0/6 DBs mapped. Derived (Plateau − PEEP). Same decision.\n")
    lines.append("3. **`fluid_balance` / `fluid_balance_cumulative` / `total_input_ml`** — 0-1/6 DBs. See `docs/fluid_balance_design.md`.\n")
    lines.append("4. **HiRID coverage gap** — only 23/77 mapped in HiRID. Pharma table likely contains most missing meds.\n")
    lines.append("5. **`compliance` / `plateau_pres` / `pip` / `mean_airway_pres` / `ps` / `vent_rate` / `tidal_vol` / `tidal_vol_set`** — ventilator params; per-DB unit audit (cmH2O vs mbar).\n")
    lines.append("6. **Medications without HiRID/AUMC mapping** (45 drugs) — for thesis Study 3 RL work; need systematic per-drug audit.\n")

    lines.append("\n## Tracker regeneration\n")
    lines.append("```bash\n")
    lines.append("python3 /Users/haibo/Documents/博士论文/boshi/scripts/regenerate_user_concept_tracker.py\n")
    lines.append("```\n")
    lines.append("To mark a concept as audited:\n")
    lines.append("1. Edit the `AUDIT = {...}` dict at top of the script with per-DB status (✅/⚠️/❌) and a 1-line note\n")
    lines.append("2. Append an entry to `AUDIT_HISTORY`\n")
    lines.append("3. Re-run\n")

    lines.append("\n## Audit history\n")
    lines.append("| Date | Concept | Action | Outcome |\n|---|---|---|---|\n")
    for d, c, a, o in AUDIT_HISTORY:
        lines.append(f"| {d} | `{c}` | {a} | {o} |\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(lines))
    print(f"Wrote {OUT}")
    print(f"Total user-added: {len(user_added)} concepts")
    audited = sum(1 for c in user_added if c in AUDIT)
    print(f"Audited: {audited}/{len(user_added)}  Pending: {len(user_added) - audited}")


if __name__ == "__main__":
    build()
