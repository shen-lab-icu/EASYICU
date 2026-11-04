# Round 6 mapping and channel semantics

The respiratory output retains the existing AUMC `fio2_chart` pooled value and
adds seven source-specific percentage columns:
`fio2_chart_aumc_6699`, `fio2_chart_aumc_12279`, `fio2_chart_aumc_12282`,
`fio2_chart_aumc_12329`, `fio2_chart_aumc_12369`, `fio2_chart_aumc_16246` and
`fio2_chart_aumc_20134`. Each column uses the same 21–100% bounds and the same
admission-relative aggregation as the pooled concept. Its column metadata
identifies the exact `numericitems.itemid`. These columns expose contributing
devices without changing the pooled calculation or imposing a device priority.

The available audit evidence establishes bedside FiO₂ measurements but does
not establish each device's set/analyzed role. The pool and each source column
therefore declare `chart_channel=unknown` with
`chart_channel_reason=device_channel_definition_not_verified`. Numeric magnitude
does not identify a device's channel. A verified device catalogue is needed
before assigning set/analyzed roles or introducing a channel priority.

HiRID `fio2` and `fio2_chart` both select observation 2010, normalize fractions
and percentage encodings using the existing percentage converter, and retain
the shared percentage bounds. HiRID has no invented laboratory FiO₂ channel.

HiRID `ins` records administered insulin amounts. `units/hr` expresses total U
in a **one hour bin**: multiple administrations in that bin are summed. It is
not an instantaneous continuous IV infusion rate, and does not establish the
route or continuity of administration. Source metadata records
`time_basis=one_hour_bin_total`, `quantity_kind=administered_amount` and
`continuous_iv_rate=false`. The existing values, bounds and summation remain.

MIMIC-III nephrostomy urine item 41184 joins the existing volume and signed
irrigation calculation. Repeated equal volumes at the same native timestamp
and source item count once; distinct drainage items remain additive. SIC
creatinine items 367 and 368 share mg/dL units; equal same-time values count
once across the two assays before hourly aggregation. Unverified urine item
41868 and creatinine HPM item 369 remain excluded.

Legacy `easyicu.callbacks.gcs` applies its inclusive six-hour validity window
independently to original eye, verbal, motor and total observations. A fresh
component does not refresh another component's observation time. Missing-value
imputation follows expiration; the native GCS callback retains its own owner.

Validation uses invented source tables only. Net new real-data hours, source
totals, urine-window effects and SIC baseline/staging effects require the lead's
approved final-SHA sample run. Small SIC counts must be reported as less than 10.
