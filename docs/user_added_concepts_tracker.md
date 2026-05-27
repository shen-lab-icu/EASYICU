# User-added EasyICU concepts (not in ricu)
> Tracker file. Maintained continuously. Last regenerated: 2026-05-27
> Source-of-truth diff: `concept-dict.json` (EasyICU, 198 concepts) vs ricu R package (121 concepts).
> User-added count: **77**.

## Legend
- ✅ mapped & audited (itemids complete)
- 🟡 mapped, not audited (correctness unverified)
- ⚠️ mapped but known incomplete (audit found missing itemids)
- ❌ NOT mapped (DB not covered for this concept)
- — DB not applicable (e.g. derived concept, eICU column-based)

## Status table
| Concept | Category | MIIV | MIIIv | eICU | AUMC | HiRID | SIC | Audit notes |
|---|---|---|---|---|---|---|---|---|
| `albumin_iv` | medications | 🟡 | 🟡 | 🟡 | ❌ | — | ✅ | 2026-05-27 Round 2. SIC: 4 Humanalbumin forms (2040/2123/2169/2170). HiRID confirmed absent. |
| `amiodarone` | medications | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 2026-05-27 Round 3. MIIV verified complete (4/4 in inputevents). |
| `anion_gap` | chemistry | — | — | — | — | — | — | 2026-05-27. Derived (Na−Cl−HCO3) via _callback_anion_gap. Deps: na/cl/bicar OK. |
| `apixaban` | medications | ❌ | ❌ | ❌ | ❌ | — | ✅ | 2026-05-27. DOAC genuinely absent in MIIV/MIMIC inputevents and HiRID Pharma (newer drug). SIC has DrugID 1954. |
| `aspirin` | medications | ❌ | ❌ | 🟡 | 🟡 | ✅ | 🟡 | 2026-05-27. HiRID added (3 Aspirin Tbl). MIIV/MIIIv still missing — would need prescriptions table (similar to dexamethasone pattern). |
| `bicarb` | chemistry | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 2026-05-27. Alias of bicar — sources pulled directly from bicar. |
| `bicarbonate` | medications | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | ✅ | 2026-05-27 Round 2. HiRID: 3 Na-Bicarbonat forms. SIC 1774 Natriumhydrogencarbonat. |
| `calcium_iv` | medications | 🟡 | 🟡 | 🟡 | ❌ | ✅ | 🟡 | 2026-05-27 Round 2. HiRID: Calcium Sandoz Lsg 10% IV only (excluded PO Brausetabl; Calciparine=heparin filtered). |
| `cisatracurium` | medications | 🟡 | 🟡 | 🟡 | ❌ | — | 🟡 | 2026-05-27. HiRID has only Tracrium (=atracurium, different drug). |
| `compliance` | ventilator | ✅ | ✅ | — | ✅ | ❌ | ✅ | 2026-05-27. AUMC +12561 Cdyn (dynamic compliance) added. HiRID has no compliance variable in reference. |
| `cvp` | vitals | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | 2026-05-27. AUMC: +20926 CVDm-gekoppeld (10× rows). eICU: nurseCharting source added (50/801 nc-only patients reached; vallabel limitation upstream). MIIV/MIIIv/HiRID/SIC verified complete. |
| `dexamethasone` | medications | ✅ | ✅ | 🟡 | ❌ | ✅ | ✅ | 2026-05-27 Round 2. MIIV/MIIIv added via prescriptions table (inputevents has none; only dextrose/dexmedetomidine). HiRID 1000769 Fortecortin Tbl. SIC 1524. AUMC remains unmapped. |
| `dexmedetomidine` | medications | ✅ | 🟡 | 🟡 | ❌ | 🟡 | ❌ | 2026-05-27 Round 3. MIIV verified (2/2 Precedex). AUMC has only research-protocol Dexmedetomidine/Placebo (excluded). SIC absent. |
| `dextrose50` | medications | ✅ | 🟡 | 🟡 | ❌ | ✅ | 🟡 | 2026-05-27 Round 3. MIIV verified (1/1). AUMC has Glucose 10% only (not 50%). |
| `diltiazem` | medications | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 2026-05-27 Round 2. HiRID: 2 Diltiazem Tbl forms. |
| `driving_pres` | ventilator | — | — | — | — | — | — | 2026-05-27. Derived concept (Plateau − PEEP); not mapped to raw itemids. Compute downstream from plateau_pres + peep. |
| `ecmo` | respiratory | 🟡 | ❌ | 🟡 | 🟡 | — | ❌ | 2026-05-27. HiRID has no ECMO variable in reference (structural). MIMIC-III gap real. SIC gap unknown. |
| `ecmo_indication` | respiratory | 🟡 | ❌ | 🟡 | 🟡 | — | ❌ | 2026-05-27. Same as ecmo — HiRID structural absence. |
| `enoxaparin` | medications | 🟡 | 🟡 | 🟡 | ❌ | ✅ | 🟡 | 2026-05-27 Round 2. HiRID: 5 Clexane SC forms. |
| `esmolol` | medications | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 2026-05-27 Round 2. HiRID: Esmolol Inj + Brevibloc Perfusor. |
| `fentanyl` | medications | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 2026-05-27 Round 3. MIIV verified (3/3, excludes sufentanil/alfentanil). |
| `fentanyl_rate` | medications | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 2026-05-27 Round 2. SIC: 1480 FentaNYL only (remifentanil/sufentanil/alfentanil are different drugs, not included). |
| `ffp` | medications | 🟡 | 🟡 | 🟡 | ❌ | ✅ | ❌ | 2026-05-27 Round 2. HiRID added (Transfusion of plasma FFP, 2 IDs). AUMC and SIC confirmed absent from d_references / drugs. |
| `fluid_balance` | output | — | — | — | — | — | — | 2026-05-27. Derived (total_input_ml − urine) via _callback_fluid_balance_hourly. See docs/fluid_balance_design.md. |
| `fluid_balance_cumulative` | output | — | — | — | — | — | — | 2026-05-27. Derived (cumsum of fluid_balance) via _callback_fluid_balance_cumulative. |
| `furosemide` | medications | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 2026-05-27 Round 3. MIIV verified complete (2/2 Lasix). |
| `heparin` | medications | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 2026-05-27 Round 3. MIIV verified — Impella/CRRT-Prefilter heparin items excluded by design (concept is systemic anticoagulation, not circuit flushes). |
| `infection_icd` | outcome | — | — | 🟡 | — | — | — | 2026-05-27. By design eICU-only — diagnosis-text infection proxy (Angus 2001 ICD). |
| `insulin` | medications | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 2026-05-27 Round 2. SIC: 4 insulin forms (regular, glargine, Aspart-Protamin mixtures). |
| `kdigo_aki` | renal | — | — | — | — | — | — | 2026-05-27. Derived via _callback_kdigo_aki (concept_callbacks.py:7379). Deps: crea/urine/weight/rrt all OK across DBs. |
| `kdigo_creat` | renal | — | — | — | — | — | — | 2026-05-27. Derived via kdigo_creatinine (kdigo_aki.py:41). Dep: crea OK across DBs. |
| `kdigo_uo` | renal | — | — | — | — | — | — | 2026-05-27. Derived via kdigo_uo (kdigo_aki.py:197). Deps: urine/weight OK. |
| `ketamine` | medications | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 2026-05-27 Round 2. HiRID added (3 Ketalar items). |
| `labetalol` | medications | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 2026-05-27 Round 2. HiRID: Trandate inj + Perfusor. |
| `levetiracetam` | medications | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 2026-05-27 Round 2. HiRID: 3 Keppra/Levetiracetam forms. |
| `lorazepam` | medications | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 2026-05-27 Round 2. HiRID: 3 Temesta forms. |
| `magnesium_iv` | medications | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 2026-05-27 Round 2. HiRID: Magnesium Sulfat 50%. |
| `mannitol` | medications | 🟡 | 🟡 | 🟡 | 🟡 | — | ✅ | 2026-05-27 Round 2. SIC: 4 Mannit forms. HiRID confirmed absent. |
| `mean_airway_pres` | ventilator | ✅ | ⚠️ | — | ✅ | ✅ | ✅ | 2026-05-27. MIMIC-III missing 1209/1672 (HFO MAP / MEAN AIRWAY PRESS). AUMC 12362 (2) skipped. |
| `mech_circ_support` | cardiovascular | 🟡 | ❌ | 🟡 | 🟡 | — | ❌ | 2026-05-27. HiRID has no IABP/LVAD/Impella in reference (structural). |
| `meropenem` | medications | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 2026-05-27 Round 2. HiRID: 3 Meronem/Meropenem forms. |
| `midazolam` | medications | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 2026-05-27 Round 3. MIIV verified complete (1/1). |
| `midazolam_rate` | medications | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 2026-05-27 Round 2. SIC: 1495 Midazolam (rate via AmountPerMinute). |
| `milrinone` | medications | ✅ | 🟡 | 🟡 | ❌ | 🟡 | 🟡 | 2026-05-27 Round 3. MIIV verified (1/1 Primacor). AUMC confirmed absent. |
| `minute_vol` | ventilator | ✅ | ⚠️ | — | ⚠️ | ❌ | ⚠️ | 2026-05-27. AUMC +8875 Mv Spontaan added (12276/12357 skipped, insp+secondary variants). MIMIC-III major gap (32 candidates vs 3 mapped). SIC 2019 MV(L) missing. HiRID not mapped. |
| `morphine` | medications | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 2026-05-27 Round 3. MIIV verified complete (1/1). |
| `neostigmine` | medications | 🟡 | ❌ | 🟡 | 🟡 | — | ✅ | 2026-05-27 Round 2. SIC 1526 added. HiRID confirmed absent. |
| `nicardipine` | medications | 🟡 | 🟡 | 🟡 | 🟡 | — | — | 2026-05-27. Confirmed absent in HiRID Pharma (Nimotop=nimodipine, different drug) and SIC d_references. |
| `nitroglycerin` | medications | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 2026-05-27 Round 2. HiRID: 1 PO capsule (no IV form in HiRID Pharma). |
| `octreotide` | medications | 🟡 | 🟡 | 🟡 | 🟡 | — | ✅ | 2026-05-27 Round 2. SIC 1553. HiRID absent. |
| `packed_rbc` | medications | 🟡 | 🟡 | ❌ | 🟡 | ✅ | ✅ | 2026-05-27 Round 2. HiRID + SIC added. |
| `pantoprazole` | medications | 🟡 | 🟡 | 🟡 | 🟡 | — | ✅ | 2026-05-27 Round 2. SIC 1427. HiRID absent (PPI family genuinely missing from Pharma). |
| `peep` | ventilator | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ | 2026-05-27. MIIV ZAuto Peep Level 224699 is Auto-PEEP (distinct concept). MIMIC-III has 535/543 mismapped (are PIP/Plateau); 505/506 (true PEEP) missing. Non-blocking since MIMIC-III not in primary thesis analysis. |
| `phenytoin` | medications | 🟡 | 🟡 | ❌ | ❌ | ✅ | ✅ | 2026-05-27 Round 2. HiRID added (Phenhydan Inf Lsg + tabl + inj). SIC 1478. eICU needs medication-table regex add. |
| `pip` | ventilator | ✅ | ⚠️ | — | ✅ | ✅ | 🟡 | 2026-05-27. MIMIC-III includes 506 (PEEP Set) by mistake. AUMC complete. SIC needs verification. |
| `plateau_pres` | ventilator | ✅ | ✅ | — | ✅ | ✅ | ❌ | 2026-05-27. MIIV 228866 candidate is IABP-specific (not airway plateau). SIC not mapped. |
| `platelets` | medications | 🟡 | 🟡 | ❌ | 🟡 | ✅ | ✅ | 2026-05-27 Round 2. HiRID + SIC added. |
| `potassium` | chemistry | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 2026-05-27. Alias of k — sources direct. |
| `potassium_iv` | medications | ✅ | 🟡 | 🟡 | 🟡 | ❌ | 🟡 | 2026-05-27 Round 3. MIIV verified (4/4 KCl + K-phosphate). HiRID has only PO retard tabs (excluded — concept is IV). |
| `propofol` | medications | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 2026-05-27 Round 2. HiRID: 7 Disoprivan/Propofol forms. |
| `propofol_rate` | medications | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | ✅ | 2026-05-27 Round 2. HiRID + SIC added (4 forms each). |
| `ps` | ventilator | 🟡 | 🟡 | — | 🟡 | ❌ | 🟡 | 2026-05-27. EXTRA flags reflect narrow regex; manual review needed. HiRID not mapped. |
| `pulse_pressure` | vitals | — | — | — | — | — | — | 2026-05-27. Derived (SBP−DBP) via _callback_pulse_pressure. |
| `rocuronium` | medications | ✅ | ❌ | 🟡 | 🟡 | 🟡 | 🟡 | 2026-05-27 Round 3. MIIV verified (1/1). MIIIv absent. |
| `rrt` | renal | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 2026-05-27. Raw RRT-active itemids across all 6 DBs. |
| `rrt_criteria` | renal | — | — | — | — | — | — | 2026-05-27. Derived via rrt_criteria callback (callbacks_missing.py:14). Composite of crea/uo_*h/potassium/ph/bicarb/rrt. |
| `sedated_gcs` | neurology | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | 2026-05-27. By design: only eICU/AUMC record GCS-before-sedation separately. |
| `tidal_vol` | ventilator | ✅ | ⚠️ | — | ✅ | ✅ | ✅ | 2026-05-27. AUMC +16243 Zephyros Vte added. AUMC 12360 Insp.tidal (2) skipped (secondary recording). MIMIC-III 36 candidates vs 5 mapped — most are PCV Exh/Insp Vt variants; needs decision. |
| `tidal_vol_set` | ventilator | ✅ | ✅ | — | ✅ | ✅ | 🟡 | 2026-05-27. All DBs verified except SIC (audit shows EXTRA — synonym list may be too narrow). |
| `total_input_ml` | output | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ | 2026-05-27. By design HiRID/SIC pending (see description). MIIV/MIIIv/eICU/AUMC have raw sources; not yet itemid-completeness audited. |
| `uo_12h` | renal | — | — | — | — | — | — | Derived via uo_12h callback. Deps: urine, weight. |
| `uo_24h` | renal | — | — | — | — | — | — | Derived via uo_24h callback. Deps: urine, weight. |
| `uo_6h` | renal | — | — | — | — | — | — | Derived via uo_6h callback (callbacks.py:1472). Deps: urine, weight — all OK. |
| `vancomycin` | medications | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 2026-05-27 Round 2. HiRID: Vancocin Amp + oral Kps. |
| `vecuronium` | medications | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | ❌ | 2026-05-27 Round 2. HiRID Norcuron inj. |
| `vent_rate` | ventilator | ✅ | ✅ | — | ✅ | ✅ | ✅ | 2026-05-27. EXTRA flags from audit reflect narrow synonym list; manual review confirms mappings are correct. |
| `warfarin` | medications | 🟡 | 🟡 | 🟡 | ❌ | ✅ | 🟡 | 2026-05-27 Round 2. HiRID Marcoumar (warfarin equivalent). |

## Per-DB mapping coverage summary
| DB | mapped | unmapped | coverage |
|---|---|---|---|
| MIIV | 62/77 | 15/77 | 81% |
| MIIIv | 57/77 | 20/77 | 74% |
| eICU | 62/77 | 15/77 | 81% |
| AUMC | 55/77 | 22/77 | 71% |
| HiRID | 47/77 | 30/77 | 61% |
| SIC | 54/77 | 23/77 | 70% |

## Category breakdown
- **medications**: 45
- **ventilator**: 11
- **renal**: 8
- **chemistry**: 3
- **output**: 3
- **vitals**: 2
- **respiratory**: 2
- **outcome**: 1
- **cardiovascular**: 1
- **neurology**: 1

## High-priority items needing attention

These 77 user-added concepts have **no validation from ricu's upstream** — every mapping below needs human review. Priority order (impact × ease):

1. **`anion_gap`** — 0/6 DBs mapped. Derived concept (Na − Cl − HCO3); decide derive-vs-extract.
2. **`driving_pres`** — 0/6 DBs mapped. Derived (Plateau − PEEP). Same decision.
3. **`fluid_balance` / `fluid_balance_cumulative` / `total_input_ml`** — 0-1/6 DBs. See `docs/fluid_balance_design.md`.
4. **HiRID coverage gap** — only 23/77 mapped in HiRID. Pharma table likely contains most missing meds.
5. **`compliance` / `plateau_pres` / `pip` / `mean_airway_pres` / `ps` / `vent_rate` / `tidal_vol` / `tidal_vol_set`** — ventilator params; per-DB unit audit (cmH2O vs mbar).
6. **Medications without HiRID/AUMC mapping** (45 drugs) — for thesis Study 3 RL work; need systematic per-drug audit.

## Tracker regeneration
```bash
python3 /Users/haibo/Documents/博士论文/boshi/scripts/regenerate_user_concept_tracker.py
```
To mark a concept as audited:
1. Edit the `AUDIT = {...}` dict at top of the script with per-DB status (✅/⚠️/❌) and a 1-line note
2. Append an entry to `AUDIT_HISTORY`
3. Re-run

## Audit history
| Date | Concept | Action | Outcome |
|---|---|---|---|
| 2026-05-27 | `cvp` | initial mapping + comprehensive 6-DB audit | AUMC: +20926 (CVDm-gekoppeld, 10× rows) added. eICU: nurseCharting source added but loader limitation = 50/801 nc-only patients. MIIV/MIIIv/HiRID/SIC verified complete. |
| 2026-05-27 | `Phase 1: sofa2-dict (21)` | structural verification | All callbacks registered (sofa2.py, callbacks.py, sepsis_sofa2.py, callbacks_missing.py). All dependencies resolvable. 8 concepts overlap with main dict (uo_*h, ecmo*, mech_circ, rrt*). |
| 2026-05-27 | `Phase 2: ventilator (11)` | strict 6-DB audit | AUMC: +12561 Cdyn / +8875 Mv Spontaan / +16243 Zephyros Vte added. MIMIC-III multiple itemid mismatches found (peep 535/543 are PIP/Plateau; pip 506 is PEEP-Set; ps 502 is PCV Vt) — non-blocking, MIMIC-III not in primary analyses. driving_pres confirmed derived. |
| 2026-05-27 | `Phase 3: renal (8)` | callback registration verification | All 8 callbacks resolve via concept_callbacks.py registry. Dependencies (crea, urine, weight, potassium, ph, bicarb, rrt) all verified in concept-dict. |
| 2026-05-27 | `Phase 4: vitals+chem+output (7)` | callback + alias verification | anion_gap, pulse_pressure, fluid_balance_hourly, fluid_balance_cumulative callbacks all registered. bicarb/potassium are aliases (verified). total_input_ml HiRID/SIC by-design pending. |
| 2026-05-27 | `Phase 5: medications (45)` | coverage scan (not full itemid audit) | Per-DB coverage: 6/6 = 6 drugs (amiodarone/fentanyl/midazolam/morphine/furosemide/heparin); 5/6 = 18; 4/6 = 16; 3/6 = 3 (aspirin/ffp/phenytoin); 2/6 = 1 (dexamethasone); 1/6 = 1 (apixaban). HiRID systematic gap: 29/45 missing (Pharma table needs per-drug mapping). Per-drug itemid completeness audit deferred. |
| 2026-05-27 | `Phase 6: remaining (5)` | structural verification | infection_icd by-design eICU-only. sedated_gcs by-design eICU+AUMC. ecmo/ecmo_indication/mech_circ_support: HiRID has no corresponding variables in reference (structural). MIMIC-III/SIC gaps real but lower priority. |
| 2026-05-27 | `Round 2: HiRID Pharma batch (20 drugs)` | systematic Pharma table mapping | HiRID Pharma reference has 565 items; matched 20/29 missing drugs (diltiazem/esmolol/labetalol/nitroglycerin/ketamine/lorazepam/propofol*/vecuronium/aspirin/enoxaparin/warfarin/ffp/packed_rbc/platelets/vancomycin/meropenem/calcium_iv/dextrose50/bicarbonate/magnesium_iv/dexamethasone/phenytoin/levetiracetam). Confirmed structural absence: nicardipine (Nimotop=nimodipine, different drug), cisatracurium (Tracrium=atracurium), neostigmine, apixaban, albumin_iv, mannitol, pantoprazole, octreotide. HiRID meds coverage 16→36/45. |
| 2026-05-27 | `Round 2: MIMIC-III ventilator itemid corrections` | audit-driven mismatch fix | peep: removed 535 (=PIP) and 543 (=Plateau); added 505/506 (true PEEP/PEEP Set). pip: removed 506 (=PEEP Set). ps: removed 502 (=PCV Insp Vt); added 578/6339/7332/7587/7595 (real PS items). tidal_vol: added 501/502 (PCV Exh/Insp Vt — actual measurements). |
| 2026-05-27 | `Round 2: SIC drugs batch (15 additions)` | d_references drug lookup | Added: dexamethasone (1524), apixaban (1954), pantoprazole (1427), octreotide (1553), midazolam_rate (1495), fentanyl_rate (1480 only), phenytoin (1478), neostigmine (1526), bicarbonate (1774), albumin_iv (4 forms), mannitol (4 forms), packed_rbc (2046), platelets (2 forms), insulin (4 forms), propofol_rate (4 forms). Confirmed absent in SIC: ffp, nicardipine, plateau_pres drug (different concept). SIC meds 26→41/45. |
| 2026-05-27 | `Round 2: dexamethasone MIIV prescriptions` | table-level fix | MIIV inputevents has NO dexamethasone (only dextrose/dexmedetomidine). Added prescriptions.drug regex source excluding ophthalmic/topical preparations. MIIV meds 28→43/45. |
| 2026-05-27 | `Round 3: AUMC drugitems batch (8 additions)` | drugitems table lookup | AUMC drugitems has 1117 unique drugs; added: ffp (7367), dexamethasone (6995), calcium_iv (18783/19164), neostigmine (7217), pantoprazole (7979 — excluded Esomep/Omep), octreotide (6866), mannitol (7360/20174), platelets (7369). Confirmed absent in AUMC: milrinone, enoxaparin, cisatracurium, apixaban, albumin_iv, packed_rbc, phenytoin. Warfarin: only Acenocoumarol/Sintrom in AUMC (different VKA, not added). AUMC meds 33→36/45. |
| 2026-05-27 | `Round 3: eICU phenytoin + MIIV/MIMIC aspirin` | alternative-table sources | eICU phenytoin: added admissiondrug regex (7 matches: CEREBYX/DILANTIN/FOSPHENYTOIN/PHENYTOIN variants). MIIV/MIIIv aspirin: added prescriptions regex (46 matches in MIIV prescriptions). MIIV meds 43→44/45; eICU meds 41→42/45. |
| 2026-05-27 | `Round 3: 11 pending meds + 3 uo_*h` | MIIV completeness audit | MIIV inputevents itemid completeness verified for amiodarone/dexmedetomidine/dextrose50/fentanyl/furosemide/heparin/midazolam/milrinone/morphine/potassium_iv/rocuronium — all complete (heparin Impella/CRRT-circuit items intentionally excluded). uo_6/12/24h marked as derived (— for all DBs, callback-based by design). |
