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

### Signed irrigation aggregation correction

The maintained loader now applies `SIGNED_SUM_THEN_BOUNDS_V1` to sources using
`mimic_urine_output` (MIMIC-IV and MIMIC-III, including inherited/demo sources).
Positive item 227488 values become negative summands. All selected channels
are summed within the same patient and requested time bin **before** applying
the existing urine range of 0–5000mL. Without a requested grid, simultaneous
native timestamps are summed. Observed zero remains zero; negative,
nonfinite or excessive totals become unavailable, never clipped to zero or
moved to another time. A non-sum aggregation request is rejected.

This is an extraction correction, not a new KDIGO threshold. Individual
irrigation channels may legitimately exceed 5000mL while their net is within
range. Other numeric concepts retain their existing pre-aggregation bounds.
The source definition records the new bounds contract, changing the concept
dictionary cache signature; old disk-cache results must not be reused.

The sealed `full6_native_v6_clean_rebuild_97f508c6_20260921` predates this repair.
A raw-source audit confirmed that its published irrigation-hour urine values
omit the negative irrigant term. Fixing the maintained loader does not repair
that release in place or establish corrected KDIGO, fluid balance or clinical
outcomes. Raw negative net, collection duration and downstream window
semantics still require explicit study-level review.

Full-path regression tests cover Python/DuckDB, MIMIC-IV/MIMIC-III,
timestamp/string storage, native/hourly/half-hour grids, multiple patients,
large cancelling channels, genuine zero and invalid totals. Fractional time
keys now bypass the integer-packing optimization to avoid merging adjacent
half-hour bins; integer packing also checks overflow. MIMIC DuckDB timestamps
are explicitly parsed, supporting the original string-valued Parquet source.

HiRID's rate-source flag and extraction-bin width must reach **both** the
public-reference phenotype and the quality receipts. The rate is integrated
over its preceding observed chart interval before normalization by weight and
covered clock time; it is not a volume event to divide again by the charting
gap. Both paths use the existing full 6/12/24-hour rate-window support. A
constant 80 mL/h in an 80-kg patient must remain urine stage 0 whether recorded
hourly or every four hours. The renal bundle infers this source type from the
database when its flag is omitted; direct reference-profile calls must supply
`urine_source_is_rate=True` for rate inputs. Other databases and frozen native
MIMIC profiles retain volume-event semantics. Stage-0 combination, missing
components, creatinine and RRT policies are unchanged.

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
