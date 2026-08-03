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
