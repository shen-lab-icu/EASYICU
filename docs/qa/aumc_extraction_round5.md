# AUMC extraction audit, round 5

This change uses source definitions supplied in the audit and small invented
fixtures. It does not re-read patient records, reproduce the audit's aggregate
counts, or establish changes in clinical scores.

AUMC source milliseconds now retain fractional minutes. Infusion rates and
explicit process windows subtract the exact admission origin before intersecting
ICU hours with `[start, stop)`. A stop exactly on an hour does not occupy that
hour. Boolean and categorical windows are unioned by their value before
expansion; a short first record cannot discard a longer record starting in the
same hour. Point airway assessments occupy only their recorded hour, without
an artificial one-minute window spilling across a boundary. The AUMC
`vent_ind` callback consumes the resulting invasive hours directly, rather than
turning each into a six-hour default window; native-time windows retain their
exact duration, and noninvasive ventilation does not become invasive. AUMC
vasopressor duration starts retain their precision, including through recursive
loads; rate, exposure and `*60` paths use the same hourly clock.

Dextrose uses D10-equivalent amounts after filtering flushes and checking volume
units, rate-time units and concentration. An explicitly labelled intermittent
bag amount is distinct from a continuous pump rate. Otherwise a usable volume
rate integrated over the exact duration supplies the interval amount. Allocation
weights each hour by actual overlap, then adds concurrent contributions. Hourly
`ml/hr` is the average delivered rate over the whole hour, so summing one-hour
bins conserves each bag and the stay total. Neither start nor stop is stretched.

Item 8851 keeps 100–1500 mL values, converts 0.1–1.5 L values to mL, and counts
rejected values outside those two populations. The parked 2 L maximum remains
excluded. Item 12291 and expired tidal volume remain distinct quantities.
Midazolam exposure requires a finite positive dose and excludes non-IV order
categories. Rocuronium and VKA exposure also require a finite positive dose
(R5C-AUMC-027); zero-dose and negative placeholders are not administration.
RRT uses finite nonzero circuit pressures, including negative access pressure,
and positive flow/removal evidence. Without positive evidence it stays unknown.

AUMC `cai` uses actual-pH item 9560. Corrected items 8915/9561/10267 are separately
loadable as `cai_ph74`, an auxiliary dictionary concept omitted from the default
web/module selection. They are never pooled into, or silently substituted for,
`cai`. This additional dictionary key requires the lead to regenerate the
protected community registry and catalog summary after merging the lanes.

D-dimer items 10393/12163 remain unmapped (R5C-AUMC-011). The repository and audit
packet identify blood D-dimer and an mg/L conversion, but establish neither the
historical FEU/DDU basis nor an assay-specific conversion to the concept's FEU
scale. The review removed the worker's speculative mapping and synthetic FEU
labels. MIMIC's documented reference-range evidence for its own assays cannot
establish the AUMC assay basis. This is an explicit non-fix, not a lane blocker.

ScvO2 requires unanimous nonempty `Centrale lijn` sites on item 7432, with no
nonempty simultaneous 11646 auxiliary specimen description. Item 12311 is
fractional and item 8903 is percent. Missing, mixed, peripheral, pulmonary-artery
and conflicting specimen records do not qualify. `fio2_chart` uses the audited
bedside source pool and explicitly labels its source metadata `set_or_analyzed`.
Individual device labels were not supplied, so no per-device set-versus-analyzed
classification is invented. ECMO sweep-gas FiO2 and oxygen flow stay excluded.

ECMO process 20664 supplies exact interval evidence in both dictionaries. Item
20662 remains the type source; the process does not establish VA versus VV.
ECMO procedure-order regex evidence is retained pending independent specificity
review: an aPTT order without a complete process record cannot establish that an
ECMO label is false. CPAP process 9671 remains unmapped pending joint airway and
device classification. Vasopressor bounds are unchanged pending unit, weight and
neighboring-infusion verification of proposed high values.

Advanced respiratory support uses mode evidence and explicit ventilation
processes. PEEP settings alone do not establish active support. Standby suppresses
mode evidence from the same device at the original timestamp. Independent active
device/process evidence survives, distinguishing Servo, Evita 4 and Evita 1,
with conflicts counted in logging before hour
aggregation. Secondary arterial SBP/DBP channels 8841/8842 rank below 6641/6643 and
above cuff channels. Nonpositive pressures do not qualify. Total PEEP 12364 is a
fallback to 12301, separate from set PEEP and routine measured PEEP 8879.

For R5C-AUMC-026, `plateau_pres_raw` remains the charted quantity. The PIP/PEEP
consistency filter is not validation of an end-inspiratory hold or absence of
patient effort. Interpretation under assisted modes needs those independent
conditions; the filter is not evidence of a controlled-breath plateau.

The urine-window reminders belong with the parallel urine lane: a descriptive
partly observed window does not establish a complete assessment window. This
lane does not alter urine thresholds, coverage publication or scoring policy.
The remaining low-priority metadata/source reminders do not supply independent
clinical verification and therefore do not change extraction values or
scientific status. See the lane receipt for each disposition.

Scores depending on the changed inputs require recomputation by the lead:
SOFA/SOFA-2 components and totals, Sepsis-3 variants, GCS/AVPU/qSOFA, supplemental
oxygen, norepinephrine equivalents and the RRT-based AKI reference. The present
synthetic checks establish input semantics, not measured score changes.
The dense-point fallback also keeps raw timestamps for chart FiO2 and both
calcium channels until admission-relative grouping; crossing its 1,000-row
threshold cannot change the hourly values.

Review path checks exercise flat and bucketed source tables, single-concept and
multi-concept loads, and native-v2 publication. An AUMC partial numeric batch
must merge its callback-backed concepts too; returning only the DuckDB columns
silently lost infusion/window results in `merge=True, r_compatible=False`.
