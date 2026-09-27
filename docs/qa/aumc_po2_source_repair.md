# AUMC PaO₂ clock and labelled-unit repair

The high-frequency fallback rounded already admission-binned PaO₂ data on the
absolute clock. For non-hour admissions this moved observations to the preceding
patient hour and changed downstream P/F pairs. Keep `po2` alongside AUMC
`tidal_vol` and `fio2` in the no-second-resampling contract.

The built-in AUMC PaO₂ dictionary also lacked labelled kPa conversion. Apply
``convert_unit(binary_op(`*`, 7.50061683), 'mmHg', '^kpa$')`` before numerical bounds
and pooling. The coefficient follows the [official AUMC legacy SQL at a pinned
commit](https://github.com/AmsterdamUMC/AmsterdamUMCdb/blob/5422b94dbaba6c1d9b7c301f059c425ca1fc13ce/amsterdamumcdb/sql/common/legacy/pO2_FiO2_estimated.sql).
Unlike that SQL, the callback retains unrounded conversion precision. This is not
an adoption of its arterial-specimen or matching-window definitions.

Two regressions fail under the preceding source and pass after repair: admission
at minute20 with and without unrelated high-frequency rows; mixed mmHg100 and
kPa10 values converted before the hourly minimum and bounds. The four relevant
core modules pass47 tests. Changed Python files pass Ruff; repository Ruff reports
16 pre-existing findings in unchanged scripts. The broad suite was stopped by this
task after approximately18 minutes at32%; no complete-suite pass is claimed.

A source-bound Study2 audit used isolated sealed-source snapshots
`97f508c6c94751a596481efafe68b41232e6bf42` with the FiO₂ repair, then this clock
repair and unit conversion as three explicit arms. All12,978 fixed original AUMC
stays were loaded in one batch per arm. The three native outputs exactly match
independent raw SQL over the original first48h, including missingness. Independent
pandas comparisons authenticate all outputs. The complete repair changes values
or observation availability in11,383 fixed stays compared with immutable sealed
v6. It therefore requires affected analysis inputs to be rebuilt; near-unchanged
cohort eligibility is insufficient to waive that check.

The authoritative lightweight receipts are in the owning Study2 repository under
`results/evidence/16_第二代亚型计划/M2-A143_AHRF核心轨迹多库运输/pf_source/`
run `m2-a143-pf-source-v6-20260927c`, with failed reference/report attempts `a` and
`b` retained. Final reference retries reused exact authenticated native executions;
they did not repeatedly extract the cohort. No patient-level outputs are in Git.

This source repair does not mutate sealed exports, certify all P/F clinical
definitions, recalculate onset eligibility, refit models or publish a new release.
Other databases and pCO₂ mapping are outside this change.
