# Vasopressor duration contract

This document defines the extraction semantics for `dobu_dur`, `dopa_dur`,
`epi_dur`, and `norepi_dur`. All exported durations use floating-point **hours**.
The rules apply identically to the four drug concepts, but the source evidence
differs between interval-based and point-based databases.

## MIMIC MetaVision (MIMIC-III and MIMIC-IV)

MetaVision supplies explicit `starttime` and `endtime` intervals. EasyICU:

1. excludes `Rewritten`, `Cancelled`/`Canceled`, `Flushed`, and `Bolus` rows;
2. additionally excludes non-zero `cancelreason` rows where that field exists;
3. drops missing, zero-length, and negative intervals;
4. clips every interval end to the matching ICU `outtime`;
5. uses `intime + los` as a deterministic outtime fallback and drops duration
   episodes for an affected stay if neither boundary is available;
6. merges overlapping intervals and intervals separated by at most 5 minutes
   within stay and `linkorderid`, using the previous running maximum end; and
7. reports `(episode_end - episode_start)` in exact hours without flooring an
   absolute clock.

The five-minute tolerance preserves pump/rate-change continuity while splitting
long idle periods inside a reused order. In the local full raw tables, 1,213
MIMIC-III and 7,963 MIMIC-IV four-drug order groups contained a gap over five
minutes after invalid-status filtering. MIMIC-III also contained 25,953
`Rewritten` four-drug rows; the historical 8,379-hour extreme was one such row.
MIMIC-IV contained 28 `Bolus` rows among the four duration drugs.
Nine MIMIC-IV stays (687 matching raw rows) have neither `outtime` nor a usable
`los`; their duration episodes are therefore unavailable rather than extrapolated
from medication timestamps. MIMIC-III has no four-drug events in its ten stays
with missing outtime.

## MIMIC-III CareVue

CareVue records rate-set points rather than reliable interval ends. Its duration
is therefore an **inferred observed episode span**, not exact pump-on time.
EasyICU follows the stay-plus-drug event sequence and deliberately does not use
`linkorderid` as an episode boundary because that identifier can change during a
continuous administration:

1. only a positive numeric `rate` establishes an active observation;
2. `Stopped`, `D/C*`, or `rate == 0` terminates the current episode;
3. `Restart` forces a boundary but cannot create duration without a subsequent
   positive rate; `NotStopd` and `NEWBOTTLE` do not terminate an episode;
4. absent an explicit stop, adjacent active observations at most 5 hours apart
   remain in one episode, while a larger gap starts a new episode;
5. a single positive observation without a later explicit stop has unknown
   duration (`NULL`), never a fabricated zero; and
6. the inferred endpoint is clipped to ICU outtime, and an episode beginning at
   or after outtime is removed.

Five minutes is not a defensible CareVue point-gap threshold: the local four-drug
table is predominantly documented every 15, 30, or 60 minutes. Five hours is a
conservative missing-documentation tolerance, while explicit stop semantics
remain authoritative. There were 18,166 explicit stop points; 7,238 had another
point within five hours and 3,129 had a subsequent positive rate, so a gap-only
algorithm would incorrectly bridge real stops. The implementation follows the
event semantics in the official MIMIC-III
[`norepinephrine_durations.sql`](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/durations/norepinephrine_durations.sql),
with the additional audited five-hour fail-safe gap.

The full raw four-drug sensitivity run supports five hours as a stable upper
tolerance rather than an arbitrary precision claim:

| gap tolerance | inferred episodes | known spans | singleton/unknown | spans >168 h | maximum span (h) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 h | 30,657 | 25,548 | 5,109 | 61 | 367.83 |
| 2 h | 20,088 | 18,461 | 1,627 | 161 | 445.75 |
| 3 h | 19,093 | 17,589 | 1,504 | 209 | 668.50 |
| 5 h | 18,697 | 17,245 | 1,452 | 217 | 856.00 |
| 6 h | 18,634 | 17,191 | 1,443 | 217 | 856.00 |

The large change at one hour reflects over-splitting around ordinary CareVue
documentation gaps; five and six hours are nearly identical. Long inferred
spans remain visible for QC rather than being silently capped. They describe a
chain of supported observations, not a claim of uninterrupted pump operation.

## Other databases and release boundary

- AmsterdamUMCdb uses explicit continuous-infusion intervals, excludes
  bolus/flush/push rows, and merges overlap or gaps up to five minutes.
- eICU uses a five-hour point-gap span and represents singleton duration as
  unknown.
- HiRID and SICdb retain their source-specific interval callbacks, normalized to
  hours.

Because duration feeds the `*60` concepts and vasopressor indicators, any change
to this contract requires a clean six-database, 19-module re-extraction. Old
exports cannot be patched or promoted as current.

## Hourly dose-rate summary (S732TW)

The product rule for hourly **dose rates** is the time-weighted mean of the
active rate: integral(rate dt) / active time in that ICU hour. Concurrent
infusions of the same drug are summed at each instant before either the mean
or the maximum is taken. The denominator is the union of active intervals,
not 60 minutes and not the sum of individual infusion durations. Intervals
are half-open; no active time gives no row, and a stop never creates a zero
rate observation. Exact overlap integration avoids rounding partial minutes.

Coverage is `norepi_rate`, `epi_rate`, `dopa_rate`, `dobu_rate`, `adh_rate`,
`phn_rate`, `angiotensin_ii_rate`, `milrinone_rate`, `enoximone_rate`,
`propofol_rate`, `midazolam_rate`, `fentanyl_rate` and `remifentanil_rate`.
These are every dictionary dose-rate concept using `mimic_rate_cv`,
`mimic_rate_mv`, `eicu_rate_kg`/`eicu_rate_units`/`eicu_rate_mass`,
`aumc_rate_kg`/`aumc_rate_mass`, `hirid_rate_kg`/`hirid_rate`/`hirid_rate_mass`,
`sic_rate_kg`/`sic_rate_units` or `sicdb_amount_rate`; angiotensin's dedicated
callback delegates to MetaVision's interval owner. Weight selection, unit
conversion, bounds on source doses, route/continuous filters and excluded
statuses remain with their source owners.

`ins` is excluded as a **mixed administered-dose concept**. Its MetaVision
`distribute_amount` sources include boluses and timed total units, HiRID
explicitly declares one-hour-bin totals, CareVue supplies amounts, and its
AUMC/eICU/SICdb continuous sources coexist with those quantities under the
same name. Converting it to an active-time rate would change its quantity
and require separate infusion/bolus concepts. Its existing dose allocation,
including the `sicdb_amount_rate` source, is retained. Dextrose and fluid
volume/amount callbacks do not enter the dose-rate summary.

CareVue and eICU settings hold to the next observation of the same infusion
where its gap is at most five hours. Larger gaps break the episode; zero,
explicit CareVue stop/discontinuation/restart and invalid eICU settings close
rather than renew a setting. Amount-only CareVue rows do not change a setting.
Different infusion/link/order ids remain separate and add concurrently; an
eICU event-row id (`infusiondrugid`) is not an infusion identity. Without an
infusion id a drug has one setting sequence, including source-label changes.
At an exact-time conflict a stop wins, otherwise the largest setting wins,
so table order cannot select a different rate. The final positive setting of
each sequence remains active for **one hour from its exact charttime**, as in
ricu 0.6.3 `create_intervals(overhang = 1 h)` and EasyICU's previous rate path.
A CareVue positive setting followed by a gap greater than five hours receives
the same one-hour overhang. eICU retains dev9's carry to the next setting,
capped at six hours, including every hour whose start is at most five hours
after the positive setting. Its values use the active-time mean over that
carry, including partial hours and subsequent within-hour changes. The final
eICU setting still receives only a one-hour overhang. The existing episode/stay end rules
still truncate the result. Zero, explicit stop and invalid settings add no
active time. This declared rate overhang does **not** supply observed episode
duration: singleton duration stays NULL, duration episode inference and `*60`
qualification stay unchanged.

Coverage follows dev9 except for zero/stop rows, permitted point-carry hours
more than five hours after the setting, and exact-clock positive/stop ties.
Stops win those ties deterministically; dev9 selected by source row order.

MetaVision, AUMC and SICdb retain their explicit interval endpoints. A newer
setting of the same recorded order supersedes its predecessor at its start;
independent orders add. MetaVision retains its optional `orderid` for this
purpose, falling back to `linkorderid` when no order id is supplied. CareVue
continues to identify its point sequence by `linkorderid`. HiRID givendose is an amount over the **preceding**
same-infusion interval (DEV-RICUC-04), not a forward setting: retain that
source interpretation and its six-hour valid-amount-interval limit, but
recover its constant rate as dose / exact interval minutes before applying
the shared active-time mean. Untimed HiRID amounts retain the existing
one-hour-bin attribution. **SICdb propofol is an hourly summary source**:
`data_float_h` DataID 3057 (`PropofolPerHourWeight`, g/h/kg), converted by
1e6 / 60 to mcg/kg/min. Its dose and coverage retain dev9's native hourly
summary; an active-time denominator cannot be recovered from that summary.
The proposed medication 1499/1549 replacement is not used. Other weight
sources and unit conversions remain unchanged. Minute opt-in paths retain their existing source
representation; ordinary hourly loads use the same exact
start/stop spans directly, without materialising a parallel minute table.

The nine vasopressor/inotrope concepts additionally declare opt-in/internal
`<concept>_max` companions. They share all source declarations with their mean
counterparts and take the maximum of the concurrent summed active dose.
They are hidden from default export modules. SOFA-1 and SOFA-2 cardiovascular
load the four dopamine/dobutamine/epinephrine/norepinephrine maxima and use
`*60` only as the unchanged administration-duration gate. Existing `vaso60`
and `*60` outputs remain unchanged. If a maximum companion is absent for an
hour, the existing `*60` dose is retained. Nonpositive qualification rows are
also retained and cannot be activated by a maximum. `norepi_equiv` sums its **hourly mean**
components with the existing factors (including dopamine); no potency factor
changes in S732TW.

Clean re-extraction is required for affected rate and derived score exports;
the dictionary is sealed under reseal 14, including the S732COV2 SICdb source restoration.
