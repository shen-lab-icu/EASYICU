# MIMIC-III streamed patient-partition boundary repair

Date: 2026-08-27 EDT  
Dataset: `/Volumes/外置硬盘/databases/mimiciii`  
Evidence root: `/Volumes/外置硬盘/tmp/easyicu-6db-resource-profile-30f5228-ak9lbI/mimiciii/`

## Problem

A 31,000-stay streamed extraction failed closed because the first MIMIC-III
`outcome` batch returned 574 `icustay_id` values outside its requested patient
partition. An isolated replay using source-order IDs found 865 outside stays;
all had values only from the admission-keyed `death` concept. Resolving an
admission/subject to ICU stays legitimately loaded sibling stays as internal
context, and the later outer merge allowed those sibling rows to escape.

## Repair

`easyicu.api.concepts` now enforces the caller's exact one-column ID dictionary
after standard and special concepts are merged. Frames containing that exact
ID column are restricted to requested values; frames keyed only by another ID
are not guessed across. This keeps broad dependency context internal while the
public output remains partition-pure.

## Evidence

- Unit and adjacent boundary/filter/resource tests: `172 passed`.
- Same isolated 31,000-stay outcome replay after repair: 31,000 rows, 31,000
  unique stays, outside rows/stays `0/0`.
- Fresh two-batch native-v2 package: outcome 61,532 rows; one-shot versus
  streamed outcome has bidirectional `EXCEPT ALL=0/0`.
- The dependent MIMIC-III medications run completed at 31,000 stays per batch:
  383.955 seconds, process-tree peak 7,236.8 MiB, stopped-for-RSS false. Its
  manifest contains two batches of 31,000 and 30,532 stays with no errors.

## Limits

This closes the exact streamed output boundary and validates outcome
partition-invariance on full MIMIC-III. It does not yet prove that 31,000 is the
fastest safe medications batch, establish all 19 MIMIC-III module profiles, or
provide Qualification12, Held-out27, clinical, causal or publication authority.
