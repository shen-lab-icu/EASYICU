# A2 batch 3b — transactional evidence authority hardening

Date: 2026-07-16

Base: `refactor/agent-control-plane@6ce7263`

Scope: evidence durability, current-authority selection, and terminal success
publication. E3 was not resumed. No planner-owned cohort, exposure, outcome,
method, or estimand decision changed, and no benchmark-specific rule was added.

## Outcome

`EvidenceStore` now publishes records, semantic aliases, base numeric claims,
and derived numeric claims as one logical generation. The historical flat files
(`evidence_index.json`, `evidence_aliases.json`, and `numeric_claims.json`) remain
compatibility projections only; they are never accepted as a second modern
authority.

The durable authority protocol has one explicit commit point:

1. write a persistent transaction receipt in `prepared` state, bound to the
   predecessor and candidate digests;
2. stage the previous authority, compatibility projections, current authority,
   head, and root high-water coordinate;
3. replace the same receipt with `state="committed"` last.

A verified `committed` receipt requires the receipt candidate, root marker,
head, and current full-state authority to agree exactly. A `prepared` receipt
selects only the verified predecessor (or the verified legacy baseline during
first migration), so a partially staged candidate is never current. The receipt
is persistent rather than deleted after success; consequently rolling back any
single selector cannot masquerade as an interrupted transaction.

The full-state payload has a strict schema, monotonic generation, predecessor
digest, and self digest. The root format marker permanently prevents a modern
store from being downgraded to mutable legacy files by deleting or damaging an
inner selector. Valid legacy stores migrate only on first mutation; incomplete,
corrupt, or symlinked layouts fail closed.

## Atomic success publication

The success path now stages, in one `success_publication_transaction`:

- output evidence records;
- base numeric claims from `step_summary`;
- accepted derived claims;
- current semantic aliases.

The transaction commits once after all staging succeeds. A stale writer,
artifact digest mismatch, projection write failure, selector write failure, or
commit-receipt failure leaves the predecessor selected and prevents aliases or
numeric claims from laundering a failed attempt into current evidence.
Post-replace acknowledgement errors are reconciled under the same process lock:
they count as success only when a strict reload selects the exact candidate.

Additional integrity rules in this batch:

- process-wide descriptor-anchored locking and compare-and-swap reject stale
  `EvidenceStore` handles;
- immutable evidence blobs are digest checked against the bytes actually copied;
- exact retries are idempotent without advancing the generation;
- nested publication rollback restores the whole staged state;
- an existing empty ledger is still modern authority and cannot be treated as
  an unanchored legacy run;
- store failure is propagated to the current step rather than followed by a
  second best-effort evidence write.

## Shared strict readers

Five production consumers now read the same selected snapshot rather than the
flat projections:

1. resume-plan candidate discovery;
2. `RunInputCapsule` evidence closure;
3. prior-code reuse in `ResumeController`;
4. run-lock authority anchoring;
5. discovery/manuscript-package evidence loading.

Modern resume therefore cannot fall back to an unregistered script or mutable
manifest after evidence authority exists. Corruption has one behavior across
these paths: structured fail-close, without scanning for an older candidate.

## Verification

- transactional authority adversarial suite: `52 passed`;
- evidence/derived/discovery/corruption/registration/resume-selector suite:
  `173 passed`;
- golden/capsule/execute-contract/meta-generalization suite: `112 passed`;
- run-lock/runtime-artifact/resume characterization suite: `68 passed`, plus
  audit-cache characterization `1 passed`;
- full `tests/research_agent/test_resume.py`: `79 passed` in 139.02 seconds;
- targeted Ruff: passed;
- Black checks on changed Black-owned files: passed after formatting
  `evidence.py`;
- both independent adversarial reviewers accepted the final prepared/committed
  state machine and reported no blocker.

## Deliberate boundaries

1. This protects against partial writes and rollback of individual authority
   coordinates on the local mutable filesystem. A privileged actor that
   coherently rolls back every anchor and every content blob is outside this
   local-store threat model; external transparency or remote append-only storage
   would be required for that guarantee.
2. The physical terminal run checkpoint remains separate from the evidence
   transaction. Consumers still require a current successful checkpoint, so a
   crash before terminal checkpoint publication cannot make staged evidence a
   current scientific result.
3. This batch establishes the EvidenceRegistrar/authority seam; it does not yet
   finish the StepExecutor/RunCoordinator extraction or prove the P0-5 runtime
   performance targets.

## Next action

Commit this batch as a Claude review point. After read-only acceptance, proceed
to the smallest StepExecutor/RunCoordinator extraction and then the milestone
regression/performance gate. Do not resume E3 until batch 4 and that gate are
complete.
