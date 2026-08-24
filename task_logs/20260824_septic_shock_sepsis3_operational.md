# Sepsis-3 septic-shock operational phenotype

- Date: 2026-08-24
- Branch: `codex/septic-shock-sepsis3-20260823`
- Base: `origin/main@8115f933f54b`
- Concept: `septic_shock_sepsis3_2016`
- Clinical contract: `septic_shock_sepsis3_2016_operational`

## Implemented contract

The owner is `easyicu.scores.septic_shock.septic_shock_sepsis3_2016`.
It emits one nullable-Boolean assessment per Sepsis-3 input event plus stable
reason and ascertainment receipts.

The v1 operational positive requires:

1. positive canonical `sep3` input;
2. positive norepinephrine, epinephrine, dopamine, vasopressin (`adh_rate`),
   or phenylephrine (`phn_rate`) evidence from sepsis onset through +24 hours;
3. lactate strictly `>2 mmol/L` within +/-6 hours of the vasopressor evidence.

Dobutamine and other inotropes from the circEWS phenotype are not accepted as
Sepsis-3 vasopressor evidence. A direct positive drug event can confirm the
operational drug clause even when another drug stream is unmapped. A negative
drug clause becomes `NA` when any eligible stream is structurally unavailable.

The flag is deliberately non-canonical. The database inputs do not establish
that vasopressors were required specifically to maintain MAP >=65 mmHg, nor do
they establish adequate volume resuscitation. Both limitations are exported as
`not_observed` receipts and `clinical_definition_complete=False`.

## Production binding

- dictionary: recursive concept using `sep3`, `lact`, and five eligible drug-rate concepts;
- callback: `_callback_septic_shock_sepsis3_2016`;
- catalog/export: Circulatory module;
- governance: source-bound golden contract, explicit opt-in metadata, independent clinical review pending;
- generated artifacts: clinical conformance matrix and native Web data catalog.

## Verification

- Focused and adjacent tests: `154 passed`.
- Dedicated owner/callback/golden/resolver tests: `11 passed` after the final receipt update.
- Clinical registry validation with SOFA-2 dictionary: zero findings.
- `ruff check`, `compileall`, JSON parsing, and `git diff --check`: passed.

Read-only real-data smoke used the pre-existing frozen export
`/Volumes/外置硬盘/easyicu_data/full6_20260717`. It evaluated only the first
200 stored `sepsis3_sofa1` events per available database; these counts are an
engineering oracle, not a prevalence estimate or manuscript result.

| Database | Smoke state counts | Interpretation |
| --- | --- | --- |
| MIMIC-III | 16 true / 176 false / 8 NA | executable; missing lactate stayed unknown |
| MIMIC-IV | 51 true / 141 false / 8 NA | executable; missing lactate stayed unknown |
| eICU | 0 true / 200 false / 0 NA | executable in this bounded source-order sample; not a prevalence claim |
| AUMC | 100 true / 56 false / 44 NA | direct positives retained; missing vasopressin/phenylephrine mappings prevented unsafe negatives |
| HiRID | unavailable | frozen export has no `sepsis3_sofa1` input |
| SICdb | unavailable | frozen export has no `sepsis3_sofa1` input |

No source dataset or frozen export was modified.

## Remaining gates

- Independent clinician review is still pending.
- Database conformance is mapping-only, not chart-reviewed phenotype validation.
- HiRID and SICdb require a governed strict Sepsis-3 input before this phenotype
  can be reported there.
- The current v1 time windows are explicit operational choices and should be
  sensitivity-tested before a study promotes the phenotype to a primary claim.
