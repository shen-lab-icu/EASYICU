# AKI Source Contract

EasyICU's canonical KDIGO phenotype consumes three explicit inputs:
`kdigo_creatinine_input`, `kdigo_urine_input`, and `acute_rrt_input` (plus
measured body weight). These names form a versioned boundary between the AKI
phenotype and broader descriptive concepts.

## Official implementation references

- MIMIC-IV: [MIT-LCP urine output](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/measurement/urine_output.sql), [KDIGO creatinine](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/organfailure/kdigo_creatinine.sql), [KDIGO urine output](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/organfailure/kdigo_uo.sql), and [KDIGO stages](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/organfailure/kdigo_stages.sql).
- eICU-CRD: [MIT-LCP pivoted urine output](https://github.com/MIT-LCP/eicu-code/blob/master/concepts/pivoted/pivoted-uo.sql). EasyICU uses current `cellvaluenumeric`, not cumulative `outputtotal`, and restricts matches to the official `I&O|Output (ml)` namespace.
- AmsterdamUMCdb: [official urine item IDs](https://github.com/AmsterdamUMC/AmsterdamUMCdb/blob/master/amsterdamumcdb/sql/common/legacy/urine_output.sql) and [official outlier repair](https://github.com/AmsterdamUMC/AmsterdamUMCdb/blob/master/amsterdamumcdb/scores.py). Values above 2500 mL are divided by 10 before values still above 4500 mL are removed.
- HiRID: [official variable reference](https://github.com/ratschlab/HIRID-ICU-Benchmark/blob/master/preprocessing/resources/varref.tsv). Variable `10020000` is `OUTurine/h` (mL/h); `30005110` is cumulative total fluid output and is excluded.
- SICdb: [official repository](https://github.com/CITI-USZ/SICdb). Event `Offset` is anchored to ICU admission by subtracting `cases.ICUOffset`; urine and CRRT use DataIDs 725 and 723 respectively.

## Creatinine history window

`kdigo_creatinine_input` has a phenotype-specific 168-hour pre-ICU lookback.
This requirement propagates to its `crea` dependency before AUMC admission-time
cropping and is included in memory and disk cache keys. Generic `crea` and the
published chemistry module retain the standard 24-hour pre-ICU boundary.

At each measurement, the canonical baseline is the lowest *prior* creatinine
in `[t-168 h, t)`; the current and future values are excluded. EasyICU does not
use a whole-stay minimum or eGFR=75 back-calculation in the canonical phenotype.
AUMC can therefore use genuine hospital measurements from the preceding seven
days. HiRID usually has no pre-ICU history, so its first ICU creatinine remains
indeterminate and later measurements use only observed prior ICU values. The
prior-only implementation was also cross-checked against
[eth-mds/icu-features](https://github.com/eth-mds/icu-features).

## Ascertainment rule

A positive component establishes AKI immediately. A negative AKI result is
published only when creatinine and urine criteria are both assessable and
negative and the dedicated positive-event RRT source was successfully searched
for the cohort. An empty successful RRT query is negative evidence; a failed or
unavailable query remains indeterminate.
