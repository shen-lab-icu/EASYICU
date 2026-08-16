# DeepSeek round-2 remediation completion (2026-08-16)

## Scope and baseline

- Branch: `feat/figure2-dev9-heldout27-20260815`
- Reviewed baseline: `b6ee8f22e82f407efd1db5e3bb46f31bf81778fb`
- Active task owners: `DATA-FIX1` and `IDEA-MINING-DISCOVERY-MODULE`
- Scope: verify the second-round DeepSeek findings and implementation, repair residual correctness gaps, and add owner-level regressions.
- Out of scope: formal Figure 2/Held-out execution, real-data re-extraction, publication claims, pushing the branch, and concurrent Copilot/Agent UI work.

## Adjudication

The DeepSeek patch was directionally useful, but its `1400 passed` report did not prove every stated contract. Several fixes were incomplete or introduced a new semantic edge case. The completion patch therefore keeps the valid portions and closes the following residuals:

### Clinical scoring

- Treat MIMIC `patients.dod` as a calendar date. Subtracting an exact ICU admission timestamp from midnight makes same-day deaths negative; mortality horizons now use the ICU admission date while absent `dod` remains censored.
- Distinguish eICU ICD-9 V/external-cause E codes from true ICD-10 letter codes instead of assigning every alphabetic token to ICD-9.
- Require an explicit unit for generic numeric KDIGO `time`/`charttime` axes; normalized concept callers pass hours and legacy minute-offset tests declare minutes.
- Mask the SOFA-2 total unless all six owner availability receipts are positive. Component-completeness counts remain available for QC.
- Preserve row-level unknowns in circulatory-failure evidence, load levosimendan/theophylline, keep first-event time and level paired, avoid DataFrame-index identity, and do not promote a partially unknown rolling window.

### Converter and IO

- Parse mapping filters before the generic iterable branch and materialize FilterSpec iterables exactly once.
- Preserve the original parquet read failure when callers pass string paths.
- Make off-grid `fill_gaps` observations reach the preserving slow path; reject scalar and Index bare-numeric rounding axes.
- Verify HiRID archive extraction by safe, exact member size (including valid zero-byte files) and reject stale AUMC shard-count mismatches in either direction.
- Pre-scan the complete pandas CSV stream before opening immutable Parquet writers so late fractional values cannot be truncated by an early integer schema.
- Keep NOTEEVENTS identifiers numeric and timestamp/date fields temporal in both Arrow and pandas conversion paths, with invalid temporal values failing closed.

### Idea Mining and Web handoff

- Remove first-idea fallback from run-scoped actions; an exact non-empty run/idea identity is required.
- Accept a retained pre-ledger prior-art directory only when its receipt proves the exact run and idea identity.
- Reject an existing handoff with a different identity instead of silently regenerating it.
- Unwrap current `idea_plan.json` envelopes consistently for handoff refresh and stale-plan comparison, while retaining legacy direct-plan support.
- Preserve URL-fetch failure status when a DOI-shaped URL exists but neither HTML nor DOI metadata was retrieved.

## Verification

All commands used the repository Python 3.11 environment with `PYTHONPATH=src`.

- Clinical owner/adjacent gate: `189 passed, 13 skipped`.
  - The 13 skips are real-data tests; no claim is made about live six-database prevalence or extraction behavior.
- IO/converter owner/adjacent gate: `127 passed`.
- Web Idea source/handoff gate: `44 passed`.
- Discovery/Idea Mining/source-status/package boundary gate: `257 passed`.
- `ruff check` on every changed source/test path: passed.
- `python -m compileall -q src/easyicu/io src/easyicu/scores src/easyicu/webserver/ideas`: passed.
- `git diff --check`: passed.

The full exact-head CI matrix was not run because this is a scoped development remediation rather than a freeze/merge/release checkpoint.

## Concurrency boundary

Another session modified Copilot/Agent backend and frontend files in the same worktree while this review was running. Those files were not inspected as part of this task and must not be staged into this remediation commit. The commit is assembled from an explicit path allowlist; no `git add -A` is used.

## Remaining boundary

- The reviewed defects are closed at code and synthetic contract-test level.
- Real six-database validation remains separate because the real-data tests were unavailable/skipped.
- The canonical LOCK remains an independent publication gate.
- No formal Figure 2 run or manuscript result was produced.
