# EasyICU full6 v6 candidate audit (2026-09-20)

## Decision

Reject the current candidate for sealing and downstream thesis refresh. Rebuild
the affected modules from one clean EasyICU commit after the defects below are
closed. The candidate remains useful as diagnostic evidence and must not be
deleted or relabelled as a formal release.

Audited candidate:

`00-data-foundation/easyicu_full6_runs/candidates/full6_native_v6_dictfix_aumc_miiv_fe382262_20260918`

Reference release:

`00-data-foundation/easyicu_full6_runs/releases/full6_native_v5_hirid_aki_rate_e0621aa1_20260906`

## Release-contract failures

- The concept-foundation lock is explicitly `finalized: false`.
- Six database `_manifest.json` files still identify the v5 runtime and old
  creatinine/PaO2 bounds. They do not describe the candidate Parquet files.
- Actual Parquet row counts disagree with the root manifests for 42 of 114
  files.
- Final `run_metadata.json` and `module_refresh_provenance.json` are absent.
- The candidate combines different extraction commits. MIMIC-IV was refreshed
  at `bcb5032f`; eICU, HiRID, MIMIC-III and SIC were subsequently refreshed
  after `abae4d1b`; AUMC has no equivalent final refresh receipt. Consequently,
  `rrt_criteria` does not have one common time semantics across all six
  databases.
- The pending service-category dictionary correction affects demographics, but
  demographics was not included in the recorded v6 refresh scope.

## Confirmed MIMIC-IV urine/KDIGO defect

The v6 MIMIC-IV renal table contains raw urine observations but no calculated
urine-output rates:

| Measure | v5 | v6 candidate |
|---|---:|---:|
| Non-null `urine` rows | 4,058,470 | 4,058,469 |
| Non-null `uo_6h` rows | 3,569,417 | 0 |
| Non-null `uo_12h` rows | 3,341,765 | 0 |
| Non-null `uo_24h` rows | 2,953,681 | 0 |
| UO-positive KDIGO rows | 860,547 | 0 |
| Any-reference-AKI stays | 60,249 | 29,028 |

The immediate cause was reproduced against the raw MIMIC-IV source. The
MIMIC-family ICU-episode quarantine added in `bcb5032f` treated an identity
table's numeric `stay_id` as though it were a relative event hour. It therefore
dropped all admission-weight rows. Without weight, the UO windows and UO KDIGO
stage were unassessable. Raw source records were present: the two reproduced
stays had valid kilogram weights and urine observations.

The source fix now returns identity-only tables before time alignment and adds a
regression test. A real-data replay for those two stays restored weight and
produced non-null 6/12/24-hour UO rates. The formal sealer now also rejects a
renal module when `uo_6h`, `uo_12h`, `uo_24h`, or
`aki_stage_uo_reference` is missing or entirely NULL.

## KDIGO definition review

The implemented creatinine, urine-output and active-RRT stage thresholds follow
the published KDIGO 2012 structure and the MIMIC-IV reference implementation:

- creatinine rise of at least 0.3 mg/dL within 48 hours or at least 1.5-fold
  within 7 days for stage 1;
- 2-fold and 3-fold thresholds for stages 2 and 3;
- UO below 0.5 mL/kg/h for 6 and 12 hours, below 0.3 mL/kg/h for 24 hours, or
  12-hour anuria;
- documented RRT initiation contributes stage 3.

Primary references:

- KDIGO 2012 guideline:
  <https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf>
- MIT-LCP MIMIC-IV KDIGO reference SQL:
  <https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/organfailure/kdigo_stages.sql>

As of 2026-09-20, KDIGO describes the 2026 AKI/AKD guideline as a public-review
draft still being prepared for publication. The 2012 definition therefore
remains the finalized reference for this release:
<https://kdigo.org/guidelines/acute-kidney-injury/>.

`rrt_criteria` is not the KDIGO RRT-initiation component. It is a separate
derived variable for physiological indications in a patient not receiving RRT.
It must not replace observed active RRT in KDIGO staging.

The concept-level creatinine ceiling of 25 mg/dL is a data-cleaning rule, not a
KDIGO threshold. It correctly restores the 15--25 mg/dL records removed by v5,
but a MIMIC-IV raw audit found values just above 25 as well as obvious sentinel
artifacts. The release should preserve an explicit exclusion receipt and run a
stage-level sensitivity analysis with a wider defensible ceiling rather than
describe 25 mg/dL as part of KDIGO.

## SOFA reconciliation

The SOFA filling/window algorithm itself did **not** change between v5 and v6.
The v5 runtime commit (`e0621aa1`) and the audited source commit (`a3191f74`)
contain byte-identical implementations of `_callback_sofa_score`,
`_compose_fill_limits`, and `_merge_tables`. Both implement the R ricu sequence:

1. `fill_gaps()` inserts empty rows on the hourly grid between the first and
   last observed component time. It does not interpolate or carry a component
   value forward.
2. `slide(..., before = 24 hours)` takes each organ's worst observed score in
   the trailing window.
3. The total is the row sum of the six rolled organs, with an entirely missing
   organ contributing zero to that sum.

The large SOFA-1 increase is instead explained by a correction to the formal
module extraction after v5. The old grouped export requested total SOFA and the
six organ concepts separately. The separately loaded concepts are sparse
point-in-time states; during the output merge they replaced the rolled organ
states returned by the total-score callback. The native publisher then rebuilt
the total from those point states, so both the published organs and total lost
the trailing-window meaning. Commit `147f52d3` changed the module loader to
request `sofa` once with `keep_components=True`, preserving the total and all
six rolled organs from the same callback result.

For five reproduced MIMIC-IV stays, the current implementation and the v6
candidate had identical row counts and component summaries (`n=294`, mean total
SOFA 4.0034). The v5 file for the same stays had `n=385`, mean SOFA 0.6494 and
mostly sparse point states.

This interpretation was checked directly against R ricu 0.6.3, rather than
only against EasyICU's comments or tests. The retained R reference contains 50
MIMIC-IV stays and 12,743 hourly SOFA rows. Feeding its six point-component
tables into the R `sofa_score()` callback and the EasyICU callback produced the
same 12,743 keys, six rolled components, and totals with zero mismatches. Against
the independently extracted R total, the formal exports compared as follows on
exactly aligned `stay_id + charttime` rows:

| Export | Aligned rows | Exact total agreement | Mean absolute error | Correlation |
|---|---:|---:|---:|---:|
| v5 | 4,384 | 21.67% | 2.999 | 0.606 |
| v6 candidate | 3,929 | 89.11% | 0.227 | 0.978 |

In hours 0--24, v5 exact agreement was 20.31% with MAE 3.032, whereas the v6
candidate reached 83.48% with MAE 0.281. The residual v6 disagreements are
mainly respiratory component differences (366 of the 428 unequal totals),
consistent with the candidate's intentional source/dictionary changes such as
the lower PaO2 acceptance bound. They are not caused by a different SOFA
filling algorithm. Across databases, first-24-hour maximum SOFA rises by
roughly one point after preserving the rolled components. This is an expected
material version change and must be reported in the release notes. It does not
make the current v6 candidate sealable because the independent urine/KDIGO and
provenance failures above remain release-blocking.

Traditional SOFA provenance:
<https://pubmed.ncbi.nlm.nih.gov/8844239/>.

R ricu implementation used for the parity audit:
<https://github.com/eth-mds/ricu/blob/main/R/callback-sofa.R>.

Machine-readable input hashes and comparison metrics are recorded in
`docs/qa/v6_sofa_ricu_parity_receipt_20260920.json`.

## Vasopressor reconciliation

Summing simultaneously administered, converted vasopressor doses is consistent
with the additive norepinephrine-equivalent construction. The MIMIC reference
also sums converted agents:
<https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/medication/norepinephrine_equivalent_dose.sql>.

The lower eICU and MIMIC-III v6 means are largely explained by removing
implausible vasopressin and phenylephrine rates. In v5, eICU contained 35,347
vasopressin rows above 0.15 U/min and MIMIC-III contained 33,431; maxima were
190 and 50 U/min. These are incompatible with the normalized unit and dominated
the old norepinephrine-equivalent values.

The dopamine factor is not universal: this repository uses `/150`, whereas the
Goradia review and current MIMIC reference use `/100`. The release must name the
chosen formula and retain a formula sensitivity analysis for analyses where
norepinephrine equivalent is central.

Primary formula review:
<https://pubmed.ncbi.nlm.nih.gov/33220576/>.

## Raw-source spot checks

MIMIC-IV spot checks traced recovered v6 measurements back to raw source rows:

- 12 low-PaO2 candidate rows were sampled; all were found in raw item 50821.
  Eleven matched the simple hourly join directly, and one matched after using
  EasyICU's minute-flooring then hourly-bucketing convention.
- 12 creatinine rows above 15 mg/dL were sampled; all matched raw lab/chart
  item IDs and units, including duplicated lab/chart observations with the same
  value.
- The two-stay urine/weight replay confirmed that the raw data were present and
  that the all-NULL UO result came from the identity/time-axis bug.

These are targeted causal checks, not a claim of complete six-database
row-by-row validation.

## Required rebuild and promotion sequence

The executable scope, resource-profile prerequisites, commands, and acceptance
gates are recorded in `docs/qa/v6_rebuild_runbook_20260920.md`.

1. Re-extract all changed modules for all six databases from one clean commit
   containing the identity-table fix and longitudinal `rrt_criteria` change.
2. Include demographics in the refresh closure and record the exact per-module
   scope.
3. Republish all six native manifests from the rebuilt Parquet files; regenerate
   hashes, row counts, runtime provenance and module-refresh provenance.
4. Run v5-to-v6 reconciliation at row, stay and component level. Require non-null
   UO coverage, stable observed-RRT stay counts, explainable SOFA changes and
   bounded vasoactive distributions.
5. Run the creatinine-ceiling and norepinephrine-formula sensitivity checks.
6. Finalize the concept foundation only after the evidence is reviewed, then run
   `EX-A01_seal_full6_release.py` without bypasses.
7. Build downstream derived releases from the sealed v6 and refresh only the
   study analyses whose inputs changed.
