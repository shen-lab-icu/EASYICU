# A2 batch 3a — EvidenceRegistrar promotion seam

Date: 2026-07-16

Base: `refactor/agent-control-plane@4b57162`

Scope: behavior-preserving extraction only; E3 was not resumed

## Outcome

The success-only alias promotion path is now behind a small, state-free
`EvidenceRegistrar` boundary:

- `EvidenceStore` remains the only durable evidence/alias authority.
- The registrar does not inspect statistical or clinical findings, select a
  current checkpoint, scan for a newer candidate, or retain its own current
  state.
- The caller still owns the exact sequence:
  `final gates -> quarantine cleanup -> promotion -> numeric claims -> terminal
  checkpoint`.
- Promotion now receives the exact evidence-id index for the current attempt
  and rejects any pending alias binding outside that index before calling the
  store.
- The prior 145-line alias-selection implementation moved out of
  `pipeline_execute.py`; a compatibility import preserves existing focused
  tests without leaving a duplicate implementation.

This is intentionally **batch 3a**, not completion of A2 batch 3. Numeric-claim
registration, evidence-store health, and recoverable cross-file promotion are
not hidden inside this extraction.

## Files

- `src/easyicu/research_agent/evidence_registration.py` (new)
- `src/easyicu/research_agent/pipeline_execute.py`
- `tests/research_agent/test_evidence_registration.py` (new)
- `tests/research_agent/test_pipeline_execute_contract.py`

## New adversarial coverage

1. A record outside the exact attempt index is rejected before the publisher
   is called.
2. A child figure cannot steal a parent analysis role.
3. A same-step retry may replace its own role alias.
4. Store publication exceptions propagate; the registrar has no second current
   state to expose.
5. Source-order guard proves final status resolution precedes promotion, and
   promotion precedes numeric authority and the terminal checkpoint.

## Verification

- Registrar/pipeline/evidence/authority focused suite: `113 passed`.
- Characterization + meta + resume revalidation + capsule suites:
  `110 passed`.
- Golden characterization rerun: `2 passed`.
- Targeted Ruff, Black (Python 3.13 target), and `git diff --check`: passed.

## Confirmed blockers for batch 3b

Read-only adversarial reconnaissance found two pre-existing durability gaps;
neither was introduced by this extraction:

1. **Corrupt store fail-open.** `_load_records`, `_load_aliases`, and
   `_load_numeric_claims` quarantine a damaged authority file and return an
   empty collection. A corrupt alias ledger can therefore look like an unused
   namespace and let another step claim a prior role. Production promotion
   needs a durable store-health signal and must fail closed on quarantined or
   generation-inconsistent authority state.
2. **No cross-file transaction.** `EvidenceStore._save()` replaces index,
   aliases, and numeric claims as three separate files. A failure after the
   first two replacements can leave aliases durably visible after in-memory
   rollback. Current non-success checkpoint filtering prevents immediate
   manuscript consumption, but residue can affect a later promotion.

A third policy decision remains explicit: numeric-claim persistence exceptions
are currently logged while the step may remain `ok`, and per-claim saves can
leave a partial numeric registry. Batch 3b must distinguish storage/integrity
failure (fail closed) from a rejected declared derived formula (warning) before
moving numeric registration into the registrar.

## Next reviewable batch

Add a versioned evidence-store generation/health contract and a recoverable
promotion-pending descriptor, with crash/reopen tests. Do not begin
`StepExecutor` extraction and do not resume E3 until batch 3 is closed and the
milestone performance/regression gate is run.
