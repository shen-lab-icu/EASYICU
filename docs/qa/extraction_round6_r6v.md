# Round-6 unverified extraction claims: r6v

Base: `87c685f181b93a9a71e84d59fe195cedf30c70e2`. All examples are invented.
The executable evidence is `tests/core/test_extraction_audit_round5b_r6v.py`.
Passing examples establish the stated behavior on these inputs; they do not
close the auditor's real-cohort residuals from `a7a30bf98`.

## Findings and evidence boundaries

| Item | Executable evidence, prefixed `test_r6_` | Conclusion |
| --- | --- | --- |
| F005 | `f005_rejected_gap_has_no_volume_but_prior_overlap_can_remain`; `f005_left_edge_invalid_channel_same_time_and_stage_evidence` | The daily owner's segment forward-fill does not add the rejected collection's amount. Earlier accepted collections can still overlap an intermediate window. Exact KDIGO integration, daily event-grid publication, and descriptive assessment must be compared on their respective input clocks. Real residuals remain open. |
| F027 | `f027_chart_state_machine_and_explicit_intervals_keep_distinct_evidence`; `f027_public_selector_preserves_distinct_ventilation_intervals` | The chart state machine accepts an eight-hour setting gap and splits at 8.01 h. Oxygen therapy closes at the last setting; extubation closes at its own timestamp. Supplementary procedure intervals have separate IMV/NIV categories. A mixed hour can contain both. The real selector also retains both categories at a nonzero ICU-minute origin, using minute-aligned raw timestamps. This tests raw interval construction, not final native-hour union parity; the ri5/ru5 landing still needs the requested replay. |
| F028 | `f028_real_selector_resolves_same_time_native_modes_before_hour_pooling` | The real MIIV selector and resolver choose the native PCV+ record over a simultaneous VOL/AC record, then retain that earlier mode over a later VOL/AC record. Reversing source row order gives the same pressure/controlled pair. Standby is excluded by the shipped source selector. No mapping defect was reproduced. |
| F029 | `f029_duration_helper_is_not_rate_owner_and_cv_mv_use_different_pooling`; `f029_public_rate_selectors_and_equivalent_share_source_clock` | `mimic_dur_incv` infers duration; `mimic_rate_cv` constructs and pools rates independently. CV floors relative to ICU admission, caps carry at six hours and takes a median across overlapping groups. MV repeats a rate in positive-overlap cells. Published rate multiplied by full cell width is not an exact delivered dose for partial intervals. Flat and bucketed public loads independently reproduce all four rates and the equivalent sum, with single/batch parity. Equivalent potency remains additive. Real source/unit residuals remain open. |
| F030 | `f030_owner_point_scores_differ_from_twenty_four_cell_worst_values`; `f030_public_parent_request_uses_twenty_four_cell_worst` | A point respiratory score and an aggregate-owned respiratory score answer different requests. A high score at hour zero persists through hour 23 and expires at hour 24. Both SOFA owners satisfy this constructed 24-cell case through their actual parent requests. The public fixture supplies each owner's declared support inputs, including positive PEEP for SOFA-2; a SOFA-1 ventilation interval alone is not that same dependency request. This does not close the reported renal residual or establish source parity. |
| F039 | `f039_hourly_first_preserves_chronology_instead_of_forcing_imv` | The standalone `change_interval(first)` helper selects earlier exact-time evidence, with deterministic simultaneous ties. The public eICU selector retains both IMV and NIV rows in a mixed hour; applying another `first` to already floored rows is a different request and can select by value order. Explicit NIV with shared PEEP remains NIV through the public selector. This refutes mandatory IMV reassignment from hourly co-occurrence, not the unverified native category residual. |
| F040 | `f040_support_metabolic_crisis_and_owner_availability_are_separate`; `f040_public_owners_resolve_support_and_metabolic_inputs`; `f040_native_respiratory_retains_explicit_sic_airway` | **Fixed a confirmed SIC support-loss defect:** an empty settings source returned window-schema `dur_var`; concatenating that schema with already materialized airway points made the resolver discard those points. Empty settings now return the same hourly schema as nonempty settings. The public raw-input regression changes the supported P/F=60 score from 2 to 4, preserves unsupported/normal scores 2/0, and reproduces renal metabolic scores 0/4/0. Per-concept, multi-concept, flat/bucket and native respiratory publication are checked. Reviewer triage is now **A/M**, because a component score error is reproduced; possible H severity requires the lead's real-cohort impact review. This does not close the historical real-cohort or eICU residuals. |
| F041 | `f041_public_smoothing_is_trailing_and_mimic3_has_no_native_smoother`; `f041_all_null_components_remain_unavailable_to_smoothing`; `f041_all_null_source_calculation_keeps_keyed_missing_smoothing`; `f041_active_rrt_supports_smoothing_without_creatinine_or_urine` | Fixed empty-component joins and all-null display smoothing; review additionally fixes unassessable urine rows renewing a normal zero. Short coverage, absent weight and rejected >12 h collections cannot renew display evidence. Supported stages are smoothed over the closed interval [t−6 h,t], without future rows; unsupported rows cannot renew that evidence. Active RRT is supporting evidence. Hour, minute and datetime axes are tested. The new unassessable-window regression also traverses the public native-v2 renal loader and the real resolver on both MIMIC sources. III has no native smoothed field. |
| F042 | `f042_charted_plateau_checks_pip_peep_without_using_niv_as_a_veto` | HiRID 3200 remains a charted plateau channel. The callback drops values above PIP+1 or below PEEP when those comparators exist, and keeps an otherwise plausible value when comparators are absent. It does not certify passive inspiratory hold, device operation or an invasive-only measurement. NIV is not a veto. |
| F043 | `f043_sic_hpm_369_cannot_enter_creatinine_selector_or_reference_stage` | The shipped SIC creatinine selector admits 367 only. Synthetic simultaneous 369 values do not enter the main creatinine series or its AKI reference result. Method, specimen and unit evidence are still required before admitting 369 or selecting a conversion. |
| F044 | `f044_counter_trajectory_alone_cannot_disambiguate_near_return_restart` | The established isolated spike is removed with its timestamp; sustained increases and clear restarts survive. [1000,10000,990,1200] produces 11200 mL under the declared restart rule. The same observed trajectory can represent a bad peak or a genuine large increment followed by a restart; its true total is not identifiable from four numbers. Such audit candidates are not evaluable without corroboration. No blanket clipping or peak removal is introduced. |
| F045 | `f045_io_raw_identity_classification_and_rate_overlap_conserve_volume` | Raw identity excludes the physical row key, but retains entry time and all other fields. One duplicate is removed; an equal value with a different entry time survives. Daily totals, irrigation and drug mass are excluded, ordinary volume and Nutrition Total survive, and a stopped mL/h record integrates with exact overlap. Expected hourly volumes are 250 and 30 mL. Real residuals remain open. |

## F005: collection-to-window ledger

The daily event-grid fixture has records at `0.25,1.25,…,20.25` h with 10 mL
each, then a rejected record at 34.25 h, followed by 60 mL at
`40.25,46.25,52.25,58.25` h. The rejected amount is independently changed from
1 to 9999 mL: every daily output and receipt stays identical. This covers all
four event-volume sources named in F005: MIIV, III, eICU and SIC.
`test_r6_f005_public_selector_rejected_collection_is_inert` repeats the
comparison through their real selectors/loaders, in flat and bucketed storage,
and checks single/batch parity. It uses 1 versus 4999 mL so both candidate
amounts survive the raw 5000 mL bound; the comparison therefore isolates gap
rejection rather than value clipping. Descriptive 6/24 h outputs are also
invariant.

This is the existing daily *event-grid* convention: end bin `k=floor(t)`, first
collection one bin, subsequent accepted collections spread over bins
`previous_k+1…k`; the source receipt retains `.25`. It is not a claim that these
bin endpoints equal the raw continuous collection endpoints. Exact KDIGO and
descriptive outputs use `collection_windows` instead. In that owner the
nonzero collection `(16,20]` intersects `(18,24]` for two hours, so half its
200 mL plus the 240 mL from `(20,24]` gives 340 mL over six hours. An explicitly
invalid amount breaks that owner's segment; simultaneous valid channels add
volume while sharing one duration. The KDIGO regression checks the resulting
rates, coverage and stage independently of the daily event-grid calculation.

| Daily bin | Accepted cells in its 24-cell window | Observed mL | Covered h | Full daily | Estimated daily | Latest contributing source h |
| --- | --- | ---: | ---: | --- | --- | ---: |
| 20 | 0…20 | 210 | 21 | unavailable | 240 | 20.25 |
| 24 | 1…20 | 200 | 20 | unavailable | 240 | 20.25 |
| 26 | 3…20 | 180 | 18 | unavailable | 240 | 20.25 |
| 27 | 4…20 | 170 | 17 | unavailable | unavailable | 20.25 for observed volume only |
| 34 | none, new segment | unavailable | 0 | unavailable | unavailable | unavailable |
| 52 | 35…52 | 180 | 18 | unavailable | 240 | 52.25 |
| 58 | 35…58 | 240 | 24 | 240 | unavailable | 58.25 |

The unchanged *estimate* at bins 24 and 26 comes from a constant rate on
different shrinking observed subsets, not from an inherited rejected volume.
The source time 20.25 before publication at 26 is legitimate; it must not be
rewritten to 26. The available-time columns for unavailable daily values remain
missing. Future-source flags remain meaningful on retrospective back-allocation
(for example bin 52 uses a source at 52.25).

## F041: limited display correction

The pinned point-stage contract continues to coalesce unavailable component
stages to zero, with `component_negative_or_unobserved_coalesced` ascertainment.
This change does not recode that point stage or its disease flags. The
smoothed fields are engineering displays, not independent KDIGO criteria or
substitutes for a raw-time stage or its evidence receipts.

The display now uses surviving creatinine measurements, **assessable urine
stages**, or positive RRT evidence. A urine calculation row whose stage is
missing is not evidence, even though its key survives the calculation. A valid first measurement still supports the
profile's zero even if no prior baseline exists. Missing records do not refresh
the six-hour evidence window. A fully unsupported window is missing. Empty
creatinine/urine calculations keep the original keyed spine instead of raising
`KeyError` or concatenating an empty list. MIMIC-III's absent native smoothed
field remains structurally absent.

The smoother has no separate per-value `available_at` column. Its computation
does not look beyond t; the published profile-level `uses_future` flag is
false. A retrospective urine source's receipt remains the authority for when
its contributing collection was actually available. Do not infer prospective
availability solely from the smoother's publication timestamp or its flag.

## Required real-cohort replay by the lead

Perform this in the authorized data environment after ri5, ru5, rr5 and rv5
land. Use one final SHA for both sides, the auditor's original stay/hour keys,
and identical requested concepts, interval, aggregation, `r_compatible`,
`keep_components`, time limits and patient scope. Never compare an old export
to new source callbacks. Preserve the source dictionary/config identities and
all raw identity fields before projection. Report aggregate counts only.

For each item compare the *union* of output keys: paired values, left-only,
right-only and paired missingness separately. Classify every residual by
source selection/status, unit, source clock, deduplication, interval construction,
hour pooling, owner availability or downstream formula. Agreement of paired
rows alone is insufficient. All numerical comparisons should use an explicitly
reported tolerance (for example absolute 1e−9 plus relative 1e−9); category and
missingness comparisons must be exact. Do not force unsupported hours to zero.

| Item | Inputs, entry points and required comparison |
| --- | --- |
| F005 | Replay `urine_raw_and_exact_replay`, R5-EICU-018/033 and `eicu_urine_interior_replay` keys. Preserve effective urine channels, invalid/void exclusions, signed irrigation, same-time aggregation, source clock/lag/gap receipts, weight and requested origin. At `source_availability` → `daily_volume_batch`/`collection_windows` → `kdigo_uo`/`build_renal_aki_bundle`, emit effective record → collection bounds/rejection reason → hour overlap → 6/12/24 h ledger. Compare full/estimated daily values, observed volume, covered hours, partial flags, descriptive/assessment rates, availability/future receipts and UO stage. Changing only a rejected long-gap amount must have zero effect; legal old-overlap residual must be itemized separately. Classify the MIIV 35/82/12, III 37/37 and eICU 277/35/1208 historical candidates without assuming the same final-version counts. |
| F027 | Replay `vent_panel_evidence.json` keys from III chartevents plus extubation procedures and explicit procedure intervals, and MIIV procedureevents with void filtering. Compare `_apply_callback` chart-state/explicit-interval output before rounding, `_callback_vent_ind` with actual parameters, and native positive-overlap hourly IMV/NIV unions after the interval fixes. Preserve non-integer ICU minute origin. Report differences in `vent_ind`, each category and one-sided grids; permit both categories within a mixed hour. |
| F028 | Replay `closing_evidence.json::miiv/vent_mode_replay` keys using raw itemids 223849/229314/227577, native labels, exact times and source selectors. Run the real `ConceptResolver` request; compare the surviving native row before `apply_vent_mode_frame`, the paired mapped axes and `change_interval(first)` hour selection. Preserve the parent request and same-time native-label ordering. Explain both the value residual and 56 reconstruction-only hours. |
| F029 | Replay `drugs_vent_evidence.json::mimic/rates` and `cv_aggregation_diagnostic` keys independently for each CV/MV item and unit. Retain CV linkorderid, terminal/missing-rate rows, dose/weight conversions and ICU origin; retain MV exact ends/status and normalized rates. Compare `mimic_rate_cv` and `mimic_rate_mv` before cross-source hourly pooling, then each rate and `_callback_norepi_equiv` on the same final key union. Separately integrate actual rates over raw overlaps for dose conservation; do not substitute `mimic_dur_incv` episodes or sum full-width published partial cells. |
| F030 | Replay `followup_evidence.json::mimic/sofa_replay` keys after dependency fixes. Preserve exact pafi/vent/adv_resp and renal component inputs, their owner-issued observed/available receipts, requested fill limits and point-versus-aggregate projection. Run component callbacks, then `_callback_sofa_score`/`_callback_sofa2_score` using exactly t−23…t hourly cells. Compare every component, receipt and total; trace each changed renal source through both scores. |
| F039 | Replay record237 with exact airway/device/treatment/settings timestamps, source order and IMV/NIV intervals before hour pooling. Run source callbacks → cross-source merge → `change_interval(first)` → `vent_ind`. Distinguish exact-time conflicts from within-hour transitions and settings that are already invasive anchors by the declared rule. Compare against independently corroborated exact intervals and report net misclassification counts only for hours with an evaluable ground truth. |
| F040 | Replay both `*_sofa2_{resp,renal}_owner_locf_replay` cohorts on the final SHA. Supply all raw pafi/SF/persistence, advanced support/ECMO, creatinine, urine assessment/coverage, RRT episode/indication/criteria and potassium/pH/bicarbonate events; include missing/invalid/restore transitions. Run `sofa2_component_evidence`, the component callbacks and `_callback_sofa2_score`. Independently rebuild availability before LOCF and 24-cell maxima; compare component/total values and missingness. SIC formal results require this new SHA. |
| F041 | Recompute each available smoothed field from supported point records on [t−6 h,t]; check the exact six-hour edge, recovery, long unsupported stretches and all-null stays. Compare MIIV native/reference separately, and confirm III has no native smoother. Preserve raw point-stage coalescing and its ascertainment label. Any manuscript use requires an input/formula validation separate from this engineering-display check. |
| F042 | Revisit the 109 hourly and fixed 20 exact-NIV candidates with raw HiRID 3200, airway/mode, PIP/PEEP times and official device/channel documentation. Compare `_callback_plateau_pressure` to its PIP/PEEP rule and separately adjudicate whether the raw measurement is a valid passive plateau. Without device/method evidence, sensitivity-only/unknown; do not delete NIV plateaus or certify them as invasive support. |
| F043 | For the 569/497 candidate panel and 172 simultaneous pairs, obtain the original 369 method, specimen and unit definitions plus 367/369 raw values. Confirm the shipped 367-only selector and no 369 contribution before proposing a new mapping. Compare a justified conversion against simultaneous pairs only after that evidence exists; neither pooling mg/dL labels nor dividing by 88.4 is an admissible substitute. |
| F044 | Replay the original 14 candidates with several readings before/after, absolute noon boundaries, reset events and other fluid evidence. Run `_isolated_counter_spikes` → `_restart_aware_increments` → `counter_increments` → `hourly_counter_volume`; compare interval and hourly total against each independently adjudicated trajectory. Classify isolated peak, sustained rise, real restart or not evaluable. Retain 38181.423 mL as an unresolved historical residual until the trajectories establish what it represents. |
| F045 | Replay `input_evidence.json::{eicu_input_replay_a7a30bf98,eicu_input_mismatch_classification}` residual keys with complete raw I/O rows, including key, bucket, cellpath/label, offset, entry offset, all totals and text. Compare `deduplicate_io` with DuckDB `io_relation_sql` before projection; apply the shipped selector and `systemic_intake` identically on each side; then compare hourly sums and conservation. Equal charted values with distinct entry times remain independent. Require zero unexplained paired or one-sided differences before closure. |


## Reviewer replay contract and remaining boundary checks

The two confirmed code fixes are F041's missing/unassessable smoothing and
F040's SIC empty-settings schema. They are not a claim that all historical
residuals in those items have disappeared. No patient-level data was accessed.

For every remaining real replay, the lead must supply a manifest containing
`sha`, dictionary/config digests, database, raw-root binding, original stay IDs,
original requested concept list, `interval`, `aggregate`, `r_compatible`,
`keep_components`, `align_to_admission`, and pre/post-ICU limits. Recover these
from the original request; an omitted parameter is not permission to substitute
a current default. The cohort is the union of the original residual keys and
paired controls. Raw input must include the necessary lookback, not just the
flagged publication hour: 24 h of urine/score history, 168 h of creatinine
history, and the preceding/following source records for any collection,
infusion or ventilation interval crossing the comparison boundary.

Use `ICUDataSource(load_src_cfg(database), base_path=raw_root)` with the shipped
`ConceptResolver(load_dictionary(include_sofa2=True))`. Call `load_concepts`
with the manifest parameters and `concept_workers=1`. Make a separate fresh
resolver request per concept (`merge=False`), the original batch request, and
`extract_database(..., native_export_v2=True)` for the corresponding modules;
include `outcome` for the native publisher's ICU bounds. Do not run those
clinical replays in this worker. Compare all three paths against the independent
record ledger over the union of stay/time keys, retaining original time-column
metadata until every clock has been converted to ICU-relative hours.

| Items | Exact raw fields and comparison boundary |
| --- | --- |
| F005 | MIIV/III `outputevents`: native stay ID, subject/admission IDs, `itemid`, `charttime`, `value`, `valueuom`; eICU `intakeoutput`: `patientunitstayid`, full raw identity, `cellpath`, `celllabel`, `cellvaluenumeric`, both offsets; SIC `data_float_h`: `CaseID`, `DataID`, `Offset`, `Val`. Preserve `icustays.intime`, eICU ICU origin, or `cases.ICUOffset`, plus actual selected weight and source availability/gap sidecars. At each owner input keep one row per effective record before collection inference. Independently sum each accepted interval's exact overlap with `(t-w,t]`; list daily event-bin differences separately from exact KDIGO integration, never silently round the independent ledger to obtain agreement. Require equal value/null/coverage/partial/availability/stage tuples or an explicit owner-contract difference for each residual. |
| F027/F039 | MIIV procedures: stay ID, `itemid`, `starttime`, `endtime`, `value`, `valueuom`, cancellation/status fields. III adds `chartevents` exact time, `itemid`, `value`, `valuenum`, `error` and explicit extubation procedures. eICU adds all respiratorycare, treatment and respiratorycharting event/entry offsets and labels. Capture `mech_vent` with `interval=None, aggregate=False, merge=False` in addition to the original request. Construct IMV/NIV sets by positive overlap with `[h,h+1)` on the original minute origin; retain both in mixed hours. Compare `vent_ind` to their union. When a publication has one category, record the actual consuming aggregation and source row order; do not apply a second lexical `first` to a floored table and call it the original request. |
| F028 | MIIV `chartevents` columns: native IDs, `charttime`, `itemid`, `value`, `valuenum`, `valueuom`, `warning`. There is no MIIV `error` field. Preserve selected rows for 223849/229314/227577 and same-time ordering before mapping. Compare mode and breath sequence together before and after the original hourly pooling; count unsupported-label exclusions separately from one-sided hours. |
| F029 | III CV: native IDs, exact `charttime`, `itemid`, `rate`, `rateuom`, `amount`, `amountuom`, `linkorderid`, `stopped`; MV adds exact `starttime`/`endtime`, `orderid`, `patientweight`, `cancelreason`, `statusdescription`. Load weight through its shipped priority selector. Reproduce each configured CV channel separately, including the mcg/min-to-per-kg channels, before median pooling; integrate MV raw overlaps separately. Require zero rate/missingness differences at this boundary before comparing the unchanged equivalent formula. |
| F030/F040 | Capture the exact requested `pafi`, `safi`/`spo2`/`fio2`, `vent_ind`, `adv_resp`, ECMO/persistence, creatinine, UO/coverage, RRT/indication and potassium/pH/bicarbonate frames. Retain owner `*_observed`/`*_available` columns before parent LOCF. Recompute the exact point score, then availability-aware LOCF and the maximum over the parent's 24 cells. Compare point requests separately from retained parent components and totals. For the confirmed SIC defect, replay a `data_range` 720 interval with no eligible settings rows: `mech_vent`/`adv_resp` must retain every positive-overlap hour and the P/F=60 supported respiratory score must be 4. Check both genuinely missing settings tables and existing tables with only unrelated DataIDs. |
| F041 | Request `kdigo_aki` through the resolver and `load_renal_aki_bundle` through the export owner. Enumerate supported stage observations in `[t-6,t]` per stay; a urine row with `aki_stage_uo` missing cannot renew a zero. Keep a valid measured creatinine's pinned point-coalescing contract, and positive RRT support. Compare reference/native smoothing nulls and values, verify no later timestamp contributes, and trace urine's separate availability receipt. III native smoothing is structurally absent, not a zero-valued column. |
| F042 | HiRID `observations`: `patientid`, `datetime`, `variableid`, `value`, raw status; IDs 3200, 3110, 2600/2610 and airway 15001552, plus `general.admissiontime`. Compare selected plateau values to simultaneous/owner-aligned PIP and PEEP independently of airway classification. Device method and passive-hold evidence must come from the channel documentation, not co-occurrence. |
| F043 | SIC `laboratory`: `CaseID`, `LaboratoryID`, `Offset`, `LaboratoryValue`, `LaboratoryType`, with original method/unit records for 367 and 369. Under the current selector, adding or changing only 369 must leave `crea` and reference AKI unchanged. A proposal to admit 369 requires external method/specimen/unit evidence first. |
| F044 | HiRID counter IDs 30005010/30005110: several raw readings on both sides of each candidate, absolute timestamps, noon boundaries and reset/device records, plus independent input/output evidence. Replay `counter_increments` and `hourly_counter_volume` on identical rows, then public `total_input_ml`, `total_output_ml`, `fluid_balance`. Adjudicate each of the 14 trajectories before deleting a candidate; ambiguous peak/restart paths stay not evaluable. Compare amounts and their temporal placement, not merely a cohort sum. |
| F045 | Retain **all** configured raw `intakeoutput` columns before projection: physical row key, patient ID, both offsets, cell path/label/value/text and cumulative totals. Physical key/storage bucket are excluded from duplicate identity; entry time is included. Compare `deduplicate_io` versus `io_relation_sql` on that complete identity, then selector and `systemic_intake`, then per-hour sums. Expected volume is ordinary accepted entries plus positive rate × exact overlap until the next same-channel record within 6 h. Final unbounded rate rows add no guessed volume. |

The public source fixtures assert their columns against `data-sources.json`;
all normalized callback-only examples remain supplementary, not substitutes
for those selector tests. The F030 synthetic comparison restricts itself to
the 27 explicitly sampled hours; auxiliary rows generated outside that range
must still be audited by the final interval/publication replay.


## Validation interpretation

The read-only base run uses the exact copied lane test bytes. Its 25 expected
failures are 22 F041 regressions and 3 F040 regressions, including native SIC
publication. The 40 passing cases are refutation/characterization and
non-regression controls; they are not presented as proof of a new fix. Native
checks explicitly reject nullable all-missing support columns rather than
allowing `Series.all(skipna=True)` to pass vacuously. Final-tree counts, mutation
results, restored source hashes and exact commands are in the lane handoff.
