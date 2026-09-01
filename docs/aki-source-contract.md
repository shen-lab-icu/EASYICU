# AKI Source Contract

Current EasyICU renal exports publish two separate AKI layers plus evidence
quality receipts:

1. `MIT_LCP_KDIGO_REFERENCE_PORT_V1` is the pinned public-reference phenotype
   used for cross-database analyses.
2. One versioned source-native profile is selected for each database. It is
   either evaluated under its own semantics or accompanied by an explicit
   unavailability receipt.
3. The historical `EASYICU_KDIGO_STRICT_PRIOR_V1` implementation remains
   callable only for sealed-release reproduction and for deriving evidence
   receipts. Its disease labels are not emitted by the current renal contract.

The five explicit callback inputs are `kdigo_creatinine_input`,
`kdigo_urine_input`, measured `weight`, `acute_rrt_input`, and
`crrt_mode_input`. The CRRT-specific input is required to reproduce the pinned
MIMIC-IV source-native treatment component; the cross-database reference layer
uses active RRT because CRRT modality is not uniformly available across all
six databases.

## Public-reference semantics

The reference layer is anchored to MIT-LCP mimic-code commit
`d20b49a71ebb8cafc6febb0821432778592192d5`. It uses prior 48-hour and
7-day creatinine minima, 6/12/24-hour normalized urine windows, the maximum of
the available component stages, missing-component-to-zero combination
semantics, and a past-only six-hour rolling maximum. Its active-RRT treatment
port is deliberately broader than the CRRT-only component in the pinned
MIMIC-IV SQL and is labelled as such in every export.

This layer is a public-reference semantic port outside MIMIC-IV. It must not be
described as an official native phenotype for eICU, HiRID, AUMC, or SICdb.

## Source-native semantics

- MIMIC-IV reproduces the pinned MIT-LCP dynamic KDIGO profile and uses the
  dedicated CRRT-mode input for the treatment component.
- MIMIC-III reproduces the pinned legacy MIT-LCP profile, including its reduced
  urine documentation-span thresholds and absence of an RRT component.
- eICU exposes the pinned official urine component but no official complete AKI
  stage.
- HiRID fails closed when the HiRID-II author endpoint and publication-only
  auxiliaries are unavailable.
- AUMC's registered official legacy profile is a future-looking,
  stage-3-like case-level creatinine endpoint rather than complete dynamic
  KDIGO.
- SICdb's native `KDIGO_AKI_168` is a future-looking case-level maximum over
  the first 168 hours.

Future-looking AUMC and SICdb native endpoints are never broadcast onto early
hourly renal rows. The dynamic export records their profile, time scale,
future-information flag, and non-embedding reason. Their explicit profile APIs
remain available for appropriately timed case-level compatibility analyses.

## Component source contracts

- MIMIC-IV follows the pinned MIT-LCP urine-output mapping, including GU
  irrigant netting.
- eICU uses current `cellvaluenumeric`, excludes cumulative `outputtotal`, and
  restricts matches to the official `I&O|Output (ml)` namespace.
- AUMC uses the official urine item IDs and outlier repair: values above
  2500 mL are divided by 10 before values still above 4500 mL are removed.
- HiRID variable `10020000` is recorded `OUTurine/h` in mL/h; cumulative total
  fluid output `30005110` is excluded.
- SICdb event offsets are anchored to ICU admission; urine and CRRT use the
  pinned native identifiers.

`kdigo_creatinine_input` carries a phenotype-specific 168-hour pre-ICU
lookback. Generic creatinine and the published chemistry module retain their
standard 24-hour pre-ICU boundary.

## Evidence quality and downstream events

Creatinine, urine, RRT, baseline, and observation-window receipts are published
as a quality layer. They may define sensitivity cohorts, but they do not create
a second `strict` AKI disease label and do not alter the public-reference
stage.

Incident AKI, incident severe AKI, persistent AKI, component-specific events,
and renal-SOFA worsening are downstream study outcomes. Their landmark,
lookback, horizon, persistence, and recovery-gap parameters do not belong in
the reusable EasyICU renal phenotype.
