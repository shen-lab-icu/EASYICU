# StepAuthorityCapsule storage checkpoint

Date: 2026-07-16  
Branch: `refactor/agent-control-plane`  
Base: `main@30addf4`

## Scope

This checkpoint adds only the immutable storage primitive for future step resume:

- strict, path-free capsule and content-reference schemas;
- content-addressed blobs and capsule JSON under `.step_authority/`;
- atomic no-overwrite publication with digest, size, regular-file, containment,
  and symlink checks;
- candidate → concept-audited → executed stage closure;
- independent execution/audit seals, including failed execution before audit;
- candidate-origin authority for initial generation, repair, deterministic mutation,
  and explicit legacy adoption;
- explicit parent binding and code/input execution closure;
- concept-audit findings and cache coordinates bound to the exact candidate/input
  authority (cross-code audit reuse fails closed);
- no EvidenceStore registration, semantic alias, or step-status mutation.

The newest monotonic run checkpoint remains the only selector of current authority.
An orphan blob or capsule is deliberately undiscoverable by this module.

## Authority order

1. `RunInputCapsule` proves study identity.
2. The newest monotonic run checkpoint selects the current step record.
3. The provider receipt owns paid calls and logical repair attempts.
4. `StepAuthorityCapsule` recovers exact bytes and completed runner results.
5. A successful current checkpoint plus EvidenceStore owns publication authority.

No lower layer may promote itself into a higher layer.

## Verification

- `tests/research_agent/test_step_authority_capsule.py`: 17 passed, including
  blocking-findings mismatch, ancestor deletion, nested/swap symlink attacks,
  conflicting objects, and concurrent blob/capsule publication.
- Ruff, Black check, and `git diff --check`: passed.
- The prior provider-receipt docstring now describes attempt-owned calls, matching
  the schema-v2 accounting Claude independently approved.

## Deliberately not implemented yet

- initial-generation transport and provider-result persistence;
- RepairCoordinator persist-before-seal callback;
- checkpoint `capsule_ref` selection and legacy boundary;
- synthetic runner-result replay and the zero-repeat resume fast path;
- quarantine migration;
- E3 resume.

Those integrations require separate reviewable commits. This storage checkpoint
does not claim any performance improvement by itself.
