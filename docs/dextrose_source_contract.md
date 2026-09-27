# D10-equivalent glucose source contract

`dex` represents a D10-equivalent glucose rate in mL/h. It is not
dexmedetomidine and is not an hourly administered-volume measure.

The MetaVision `mimv_rate` callback excludes `Rewritten` records before using
either recorded rates or amount/duration fallback. The dictionary declares
`statusdescription` in `extra_vars` for MIMIC-III, its demo and MIMIC-IV.
Missing required columns raise an error instead of returning unprocessed
amounts as rates. A missing status cell is retained; it is not evidence that
an order was rewritten. Paused, Changed, Stopped and FinishedRunning records
are not discarded merely because their delivery subsequently ended.

The AUMC D10 source requires a recorded `ml` dose unit before minute-to-hour
and concentration conversion. `druppel` (drops) has no source-supported mL
conversion here and is excluded with a count warning. Unit-column absence
raises an error. This dictionary constraint does not alter other AUMC rate
concepts. CareVue unitless observed-zero groups retain their existing
semantics; the unit-completeness sensitivity is not silently made a repair.

The MIMIC documentation identifies Rewritten rates/amounts as undelivered:
[MetaVision source](https://mimic.mit.edu/docs/iii/tables/inputevents_mv.html).
MIMIC-IV documentation describes removal of these historical audit trails:
[MIMIC-IV changes](https://mimic.mit.edu/docs/iv/about/whatsnew.html).
The local full-data verification nevertheless checks MIMIC-IV explicitly.
AUMC distinguishes prescribed dose/rate from administered dose:
[drugitems](https://github.com/AmsterdamUMC/AmsterdamUMCdb/wiki/drugitems).

Verification against sealed source `97f508c6` used three actual loader arms:
the frozen code, that code with only this patch, and maintained base
`11944cff` with the same patch. Each arm covered the full sealed cohort of
AUMC 23,106, MIMIC-III 61,532 and MIMIC-IV 94,458 ICU stays. Non-null rates:

| Database | Sealed hours | Corrected hours | Removed / changed hours |
| --- | ---: | ---: | ---: |
| AUMC | 7,025 | 7,024 | 1 / 0 |
| MIMIC-III | 34,931 | 34,131 | 800 / 2,682 |
| MIMIC-IV | 67,477 | 67,477 | 0 / 0 |

All actual outputs matched independently reconstructed raw SQL at float32
precision. A second verifier reread raw files with Arrow and reconstructed
all nine arm/database combinations in pandas. No sealed file or current
pointer was changed. The repaired source and candidate field outputs do
not constitute a new scientific release or a rerun of downstream studies.

Evidence and executable replay:
[QC-A02 repair](../../../00-data-foundation/preprocessing/docs/publication_qc/dextrose-repair-v6-20260927b/README.md).

Remaining interpretation limits: inclusive hourly expansion and overlap
medians do not conserve administered volume. CareVue grouped-rate
construction can use later records from the order. These representations
need a separate temporal/volume contract before use as a prospective
treatment exposure or cumulative administered glucose dose.
