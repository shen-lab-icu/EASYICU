# StepResultEnvelope M3 registered-output dual-read

> Date: 2026-07-23 22:24 EDT  
> Task: `AGENT-STEP-RESULT-ENVELOPE-CONVERGENCE / M3`  
> Baseline: `a47c1ed` (`refactor/agent-control-plane`)  
> Commit: `1f932bd`  
> Scope: first opt-in Validator consumer, fail-closed comparison, and bounded legacy-parser deletion

## Outcome

M3 adds the first opt-in consumer of `StepResultEnvelope`:
`CrossStepRegisteredOutputEnvelopeDualReader`. It wraps the existing
`CrossStepRegisteredOutputValidator`, preserves the legacy finding only when
the envelope digest, source-summary digest, ledger status, normalization
result, and table-presence decision all agree, and otherwise emits one
fail-closed migration finding.

This adapter is used only by the offline replay tool and tests. `phase.py` and
`pipeline.py` do not import or instantiate it, so live execution, Writer,
readiness, scorer, repair routing, and paper authority remain unchanged.

The existing Validator no longer recursively treats arbitrary nested
`step_summary.outputs` strings ending in `.csv`/`.parquet` as registered table
authority. Its compatibility view now accepts only:

- current ledger `evidence_ids` beginning with `table_`; or
- a flat machine-readable `output_files` registration.

The typed envelope remains stricter: its table product must exist, stay inside
the output directory, verify by bytes/SHA, and be explicitly typed or compile
to a bounded table profile.

## Negative controls

The dual-read path is blocked when:

- the upstream envelope is missing;
- the envelope was compiled from a different summary;
- the current ledger status differs;
- the envelope digest fails;
- canonical normalization reports any error;
- legacy and canonical views disagree about whether a table exists.

A two-step replay fixture proves a non-trivial exact match: step 1 registers a
real table, step 2 falsely reports it unavailable, and the dual-reader finding
is byte-for-byte/model-dump equal to the legacy finding. Tampering the bound
input authority causes the envelope and registered-output comparisons to block.
An additional regression proves that an arbitrary nested `outputs` filename no
longer gains registered-table authority.

## Archived E1/E2 replay

Final diagnostic shadow outputs:

- `/Volumes/外置硬盘/easyicu_data/canonical9_shadow_envelopes/m3_dual_read_final/e1_run_20260723T211020_5733af`
- `/Volumes/外置硬盘/easyicu_data/canonical9_shadow_envelopes/m3_dual_read_final/e2_run_20260723T235937_f4d63c`

| Run | Envelopes | Normalization errors | Base Validator mismatches | Registered-output claims | Registered-output mismatches |
|---|---:|---:|---:|---:|---:|
| E1 | 8 | 0 | 0 | 0 | 0 |
| E2 | 9 | 0 | 0 | 0 | 0 |

The E1/E2 replay confirms the M2 artifact/status equivalence remains stable
under the M3 tool. It does **not** prove a real Canonical9
registered-output-claim hit, because these archived manifests contain no
`source_table_available` / `registered_output_readable` / `upstream_step`
availability block. That gate is covered non-trivially by the two-step replay
fixture above; no claim is inflated from a zero-hit archive.

## Honest code-size accounting

Gross M3 change before documentation:

- production: `+214/-16` (net `+198`);
- tests/tooling: `+242/-3` (net `+239`);
- total: `+456/-19`.

The large existing owner `audits/validators.py` is `+7/-15` (net `-8`) and
gains no import edge. The new 100-line opt-in adapter is isolated in
`audits/envelope_consumers.py`; comparison models/helpers stay in
`audits/envelope_shadow.py`.

Therefore M3 has begun deleting duplicated legacy parsing, but the migration is
still net additive overall. It does **not** close repair proliferation. No
point-repair module was deleted: direct inspection showed
`nullable_validation.py`, `rendering_role.py`, and `rendering_summary.py`
change generated-code semantics, while `percentage_identity.py` inserts a
runtime guard before output exists. Current envelope equivalence is
insufficient to remove them safely.

## Verification

Focused and adjacent regression matrix:

```text
430 passed in 10.94s
```

It covers the envelope/replay attacks, all Validator tests, percentage
identity, golden run, missingness, primary-model contract, stale authority,
Table 1, and Writer digest behavior.

Additional checks:

- Ruff: passed.
- Black: passed.
- `py_compile`: passed.
- `git diff --check`: passed.
- module graph / zero-cycle diff: passed.
- architecture comparison still reports the same 12 pre-existing branch
  regressions. M3 reduces `audits/validators.py` from 13,160 to 13,152 lines
  relative to its `a47c1ed` input and adds no Validator import edge; baselines
  were not rewritten.
- resource-context comparison retains the pre-existing branch drift; M3 does
  not alter Planner/resource selection and did not rewrite the baseline.

No Provider, Docker, extraction, raw patient table, new Canonical9 question, or
formal authority was invoked. The only real-run access was read-only replay of
already archived diagnostic step summaries and registered products; all new
outputs were written to the external disk.

## Next bounded increment (M4)

1. Add an envelope-native bounded fraction/percentage view and compare it
   against `StepSummaryFractionValidator` across its full adversarial corpus.
2. Require exact finding equality before any live consumer switch.
3. Keep `percentage_identity.py` until post-execution canonical checks prove
   the same runtime mismatch coverage and failure timing.
4. Continue moving migration logic into small envelope consumer modules; do
   not add to `validators.py`, `phase.py`, or `pipeline.py`.
5. Do not wire Writer/readiness/scorer or start online experiments until at
   least two Validator consumers have exact dual-read evidence and the
   production migration begins net deletion overall.
