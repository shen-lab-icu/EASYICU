# Patient quality module missingness audit

- Date: 2026-07-30
- Task: `PATIENT-QUALITY-MODULE-MISSINGNESS-AUDIT`
- Branch: `codex/web-copilot-cockpit-lite-20260729`
- Scope: Native Patient Review → Data Quality
- Source: official local MIMIC-IV Clinical Database Demo v2.2 export

## User-facing problem

The quality page prioritized a short “highest missing features” chart, then
rendered a dense 281-feature catalog. More importantly, sparse event exports
were interpreted as measurements: non-events became “missing” values. This made
in-hospital mortality appear 89.3% missing and Sepsis-3 (SOFA-2) appear 52.9%
missing even though those values were the complements of the real event rates.

## Corrected metric contract

- Measurement missingness is patient-level:
  `patients with no non-null observation / 140 review entities`.
- A module bar is the equal-weight mean of its measurement-feature missingness
  values, calculated from raw entity counts before the final one-decimal round.
- The diamond is the highest-missing measurement feature in that module.
- Sparse Boolean events and exposure-presence modules report event/exposure
  rates and are excluded from all missingness denominators.
- Entity identifiers are normalized before counting, preventing integer and
  float representations of the same stay from doubling the denominator.
- The backend loads all 19 registered modules for this aggregate audit.

## Independent raw-Parquet check

The values below were independently recomputed from
`/Users/haibo/.easyicu/demo_sources/mimic_iv_demo_v2_2/export`, not copied from
the web response.

| Module | Measurement features | Mean missing | Highest feature |
|---|---:|---:|---|
| Circulatory | 7 | 92.2% | PAWP 98.6% |
| Chemistry | 49 | 63.9% | TRI 100.0% |
| Blood gas | 9 | 49.4% | MetHb 100.0% |
| Respiratory | 9 | 40.2% | Oxygenation index 67.9% |
| Neurological | 12 | 35.6% | Total GCS 100.0% |
| Hematology | 25 | 33.7% | MPV 100.0% |
| Other scores | 9 | 33.3% | APACHE-IV 100.0% |
| Demographics | 6 | 18.2% | BMI 52.1% |
| SOFA-1 | 7 | 10.2% | Liver component 36.4% |
| Renal | 15 | 9.7% | Rolling 24h urine output 31.4% |
| Vitals | 12 | 6.5% | CVP 72.1% |
| SOFA-2 | 7 | 5.3% | Liver component 36.4% |
| Sepsis shared | 2 | 0.0% | 0.0% |
| Outcome measurements | 3 | 0.0% | 0.0% |
| Sepsis-3 SOFA-1 / SOFA-2 | 0 | N/A | event rate |
| Medications / vasopressors / ventilator | 0 | N/A | exposure rate |

Raw event checks:

- In-hospital death: 15 true, 125 null/non-event, 140 denominator → 10.7%.
- Sepsis-3 (SOFA-2): 66 positive entities, 140 denominator → 47.1%.
- The former 89.3% and 52.9% “missing” values were therefore false inversions.

## Implemented interaction

- Replaced the top-feature ranking with one ECharts view containing all 19
  modules.
- Removed the duplicate module-coverage block from the rendered page.
- Collapsed the complete 281-feature audit and exact audit tables by default.
- Kept module, status, and search filters inside the optional detailed audit.
- Feature rows now label event/exposure rates explicitly instead of showing them
  under the missingness column.

## Browser evidence

- Official MIMIC-IV Demo opened through the normal local source workflow.
- ECharts SVG mounted with all 19 modules and no horizontal page overflow.
- Feature and exact audit sections start collapsed.
- Opening the detailed audit and filtering to Outcome shows in-hospital death as
  `事件率 10.7%`; no `缺失 89.3%` value remains.
- Screenshots:
  - `output/ui-qa/20260730_patient_quality_module_missingness_audit/module_missingness_overview.png`
  - `output/ui-qa/20260730_patient_quality_module_missingness_audit/event_rate_audit.png`

## Verification

- Backend Patient Review and presence-rate contracts: `12 passed`.
- Patient frontend + native static route suite: `78 passed, 1 deselected`.
- The deselected test is the known worktree-name provenance assertion
  (`easyicu-copilot-cockpit-lite` vs `EASYICU`), unrelated to this patch.
- Patient quality and ECharts executable owner contracts: passed.
- JavaScript syntax, Python compile, CSS brace/comment balance, and
  `git diff --check`: passed.
- `patient-insights.css` remains route-pure and was reduced to 563 lines by
  removing the unused duplicate module-coverage renderer and styles.
