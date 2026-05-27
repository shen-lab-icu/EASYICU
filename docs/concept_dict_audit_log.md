# EasyICU concept-dict.json audit log

> Continuous audit log of concept-dictionary changes and verifications, complementary to
> [user_added_concepts_tracker.md](./user_added_concepts_tracker.md).
>
> Last update: 2026-05-27

## Purpose

`concept-dict.json` (the main 198-concept file) and `sofa2-dict.json` (the 21-concept
composite-score file) drive every downstream extraction. Wrong or missing itemid mappings
silently corrupt downstream cohorts, calibration analyses, and model training. This log
records (a) all audit-driven changes to those files, and (b) the end-to-end verification
that each change actually resolves to real data in the raw DB tables.

## Audit waves performed (2026-05-27)

### Round 1 — initial CVP concept + ventilator gaps
| Concept | DB | Change | Verification |
|---|---|---|---|
| `cvp` (new) | MIIV | added itemid 220074 (Central Venous Pressure) | ✓ resolves in d_items |
| `cvp` (new) | MIIIv | added 113, 220074, 1103 (carevue+metavision) | ✓ resolves |
| `cvp` (new) | eICU | added vitalperiodic.cvp column | ✓ column exists |
| `cvp` (new) | AUMC | added itemid 6655 "CVD" (Centrale Veneuze Druk) | ✓ resolves |
| `cvp` (new) | HiRID | added 700, 960, 15001441 (3 CVP variableids) | ✓ resolves |
| `cvp` (new) | SIC | added 2018 "CentralVenousPressure" | ✓ resolves |
| `cvp` (round 1+) | AUMC | added 20926 "CVDm-gekoppeld" (+ 4.96M rows, 10× over 6655 alone) | ✓ resolves |
| `cvp` (round 1+) | eICU | added nurseCharting sources (~1.27M additional rows) | ✓ resolves but EasyICU loader currently captures only 50/801 nc-only patients (vallabel filter limitation) |
| `compliance` | AUMC | added 12561 "Cdyn" (dynamic compliance, ml/cmH2O) | ✓ resolves |
| `minute_vol` | AUMC | added 8875 "Mv Spontaan" (l/min) | ✓ resolves |
| `tidal_vol` | AUMC | added 16243 "Zephyros Vte" (ml) | ✓ resolves |

### Round 2 — MIMIC-III itemid mismatch corrections
The strict audit revealed several MIMIC-III mappings used itemids that actually belonged to a different concept (carevue/metavision ID confusion):
| Concept | Change | Verification |
|---|---|---|
| `peep` | removed 535 (=PIP), 543 (=Plateau); added 505 (PEEP) + 506 (PEEP Set) | ✓ d_items confirms identity |
| `pip` | removed 506 (=PEEP Set); kept 535 (PIP) | ✓ |
| `ps` | removed 502 (=PCV Insp Vt); added 578/6339/7332/7587/7595 (5 real PS itemids) | ✓ all 5 carry "Pressure Support"/"PSV"/"PS" labels |
| `tidal_vol` | added 501 (PCV Exh Vt), 502 (PCV Insp Vt) — both actual TV measurements | ✓ |

### Round 2 — HiRID Pharma systematic mapping (24 drugs)
HiRID's `Variable Name` reference contains 565 Pharma items. Initial audit only counted 16/45 user-added medications mapped to HiRID; systematic Pharma-table lookup raised this to **36/45**. Added (all verified via reference table):

| Concept | HiRID Pharma IDs | Drug names |
|---|---|---|
| diltiazem | 121, 1001071 | Diltiazem Tbl 30/60 mg |
| esmolol | 1000346, 1000347 | Esmolol Inj + Brevibloc |
| labetalol | 386, 1000828 | Trandate inj + Perfusor |
| nitroglycerin | 117 | Nitroglycerin Kps |
| ketamine | 1001194, 1000400, 1000857 | Ketalar (3 forms) |
| lorazepam | 1000239, 1000418, 1000988 | Temesta (3 forms) |
| **propofol / propofol_rate** | 208, 1000491, 1000691, 1000699, 1001050, 1001052, 1001053 | Disoprivan 1%/2% (7 forms — corrected from initial fabricated IDs) |
| vecuronium | 198 | Norcuron inj |
| aspirin | 1000255, 1000256, 1000257 | Aspirin Tbl |
| enoxaparin | 1000863, 1000864, 1000865 | Clexane SC |
| warfarin | 1000476 | Marcoumar Tbl |
| ffp | 1000050, 1000744 | Transfusion of plasma (FFP) |
| packed_rbc | 1000100, 1000743 | IV blood transfusion packed cells |
| platelets | 1000245, 1000201 | Platelet transfusion |
| vancomycin | 189, 331 | Vancocin Amp + oral Kps |
| meropenem | 1000424, 1000425, 1001084 | Meronem |
| calcium_iv | 1000292 | Calcium Sandoz Lsg 10% (IV only) |
| dextrose50 | 1000567, 1000835 | Glucose 50% / 20% |
| bicarbonate | 1000193, 1000453, 1000571 | Na-Bicarbonat |
| magnesium_iv | 1000421 | Magnesium Sulfat 50% |
| dexamethasone | 1000769 | Fortecortin Tbl |
| phenytoin | 1000478, 304, 230 | Phenhydan |
| levetiracetam | 1000676, 1000756, 1001175 | Keppra |

**Bug found and fixed**: Round 2 initial propofol mapping included fabricated IDs 1001114-1001117. End-to-end verification caught this; replaced with the 7 real Disoprivan IDs.

**HiRID structurally absent** (cannot be added; raw data does not contain these drugs):
- nicardipine (only Nimotop = nimodipine, different drug)
- cisatracurium (only Tracrium = atracurium, different drug)
- neostigmine, apixaban, albumin_iv, mannitol, pantoprazole, octreotide

### Round 2 — SIC d_references batch (15 drug additions)
| Concept | SIC IDs |
|---|---|
| dexamethasone | 1524 (DEXAmethason) |
| apixaban | 1954 |
| pantoprazole | 1427 |
| octreotide | 1553 (ocTREOtid) |
| midazolam_rate | 1495 |
| fentanyl_rate | 1480 (FentaNYL only; remi/sufenta/alfenta excluded) |
| phenytoin | 1478 |
| neostigmine | 1526 |
| bicarbonate | 1774 (Natriumhydrogencarbonat) |
| albumin_iv | 2040, 2123, 2169, 2170 (Humanalbumin 5%/20%) |
| mannitol | 2050, 2091, 2135, 2171 (Mannit 10%/15%/20%) |
| packed_rbc | 2046 (Erythrozytenkonzentrat) |
| platelets | 2048, 2088 (Thrombozytenkonzentrat) |
| insulin | 1557, 1848, 1961, 1962 |
| propofol_rate | 1499, 1549, 2073, 3056 |

**SIC structurally absent**: ffp, nicardipine.

### Round 2 — MIIV/MIIIv prescriptions table (dexamethasone)
MIIV `inputevents` contains NO dexamethasone (only dextrose products and dexmedetomidine). Added prescriptions regex source excluding ophthalmic/topical preparations. Took MIIV meds coverage from 28/45 → 43/45.

### Round 3 — AUMC drugitems batch (8 additions)
AUMC `drugitems` has 1117 unique drug names; added: ffp (7367), dexamethasone (6995), calcium_iv (18783/19164), neostigmine (7217), pantoprazole (7979), octreotide (6866), mannitol (7360/20174), platelets (7369). Took AUMC meds coverage from 33/45 → 36/45.

**AUMC structurally absent**: milrinone, enoxaparin, cisatracurium, apixaban, albumin_iv, packed_rbc, phenytoin.

### Round 3 — eICU admissionDrug + MIIV/MIIIv aspirin prescriptions
- eICU `phenytoin` had no medication-table mapping; added admissionDrug regex (7 matches: CEREBYX/DILANTIN/FOSPHENYTOIN). eICU meds 41→42/45.
- MIIV/MIIIv `aspirin` similar to dexamethasone — added prescriptions regex (46 MIIV matches). MIIV meds 43→44/45.

### Round 3 — 11-medication completeness audit + 3 uo_*h derived
MIIV inputevents completeness verified for amiodarone/dexmedetomidine/dextrose50/fentanyl/furosemide/heparin/midazolam/milrinone/morphine/potassium_iv/rocuronium — all complete.
- Intentional exclusion noted: heparin Impella (229597) + CRRT-Prefilter (230044) excluded — concept is *systemic* anticoagulation, not circuit flush.

`uo_6h` / `uo_12h` / `uo_24h` confirmed as derived (callback-based; no per-DB itemids).

## Final medication coverage (45 user-added drugs)

| DB | Before any audit | After all 3 rounds | Δ |
|---|---|---|---|
| MIIV | 28/45 | **44/45** | +16 |
| MIMIC-III | 27/45 | **42/45** | +15 |
| eICU | 28/45 | **42/45** | +14 |
| AUMC | 22/45 | **36/45** | +14 |
| HiRID | 16/45 | **36/45** | **+20** ⟵ largest improvement |
| SIC | 26/45 | **41/45** | +15 |

## Verification methodology

Each itemid addition was verified by:
1. **Dictionary lookup** — confirms the itemid exists in the raw DB reference table (`d_items` / `variable_reference` / `d_references` / `numericitems` / `drugitems`) and matches the expected label.
2. **End-to-end script** — `scripts/verify_concept_dict_changes.py` (in boshi repo) re-runs lookup for every added itemid and reports PASS / FAIL.

A bug was caught during Round 2 verification (4 fabricated HiRID propofol IDs replaced with verified ones), demonstrating the value of automated post-change verification.

## Sources

- Tracker: [user_added_concepts_tracker.md](./user_added_concepts_tracker.md)
- Regenerate tracker: `python3 boshi/scripts/regenerate_user_concept_tracker.py`
- Verification: `python3 boshi/scripts/verify_concept_dict_changes.py` → outputs `concept_dict_change_verification.json`
- Audit details: `boshi/src/data_processing/cvp_itemid_audit.json`, `boshi/src/data_processing/concept_completeness_audit.json`
