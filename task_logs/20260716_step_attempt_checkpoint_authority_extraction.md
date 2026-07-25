# A2 checkpoint authority extraction — StepAttemptState / CheckpointAuthority

Date: 2026-07-16

Base: `refactor/agent-control-plane@9814bc0`

Scope: behavior-preserving control-plane extraction plus atomic rollback
hardening. E3 was not resumed and no scientific routing, prompt, validator,
cohort, exposure, outcome, method, estimand, or evidence-publication policy was
changed.

## Outcome

`step_attempt_authority.py` now owns the mutable checkpoint/capsule mechanics
for one already-planned step attempt:

- authority coordinates and the checkpoint-selected current capsule;
- selected resume capsule and execution-replay consumption state;
- digest-bound concept-audit cache entries;
- exact candidate reuse/sealing;
- repair child sealing and exact-parent restoration after host rejection;
- initial-generation pending/completed checkpoint transitions; and
- terminal failed-repair pending-marker cleanup.

The execution loop injects all storage operations through
`StepAuthorityOperations`. This preserves the existing monkeypatch/test seams
and leaves the run checkpoint as the only selector of current capsule
authority. The extracted owner does not scan capsule storage, select scientific
methods, inspect validator prose, publish evidence, or create a second current
authority.

## Atomicity hardening

The old closures mutated `step_record`, `per_step_records`, repair-parent
latches, and the current capsule reference before a checkpoint write was known
to be durable. The new owner stages these mutations as one checkpoint
transaction:

1. snapshot the exact caller-owned mutable objects;
2. validate and stage the selected capsule plus extra/delete fields;
3. update the current checkpoint and flush while holding `shared_lock`;
4. on any exception, restore the same dict/list objects in place and restore
   the previous capsule selector;
5. expose a repair-child latch only after its child checkpoint succeeds; and
6. clear that latch only after exact-parent restoration is checkpointed.

Content-addressed blobs/capsules may remain as undiscoverable orphans after an
interrupted write. They never become current because only the checkpoint
selector grants authority.

## Structural delta

AST measurements against `9814bc0`:

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `_execute_one_step` lines | 6,859 | 6,686 | -173 |
| stored local names | 416 | 401 | -15 |
| direct nested functions | 33 | 26 | -7 |

`pipeline_execute.py` is 14,568 lines after this batch. The new owner is 351
lines with 323 lines of focused adversarial tests. This is a real state-owner
extraction, not a claim that the remaining god-function split or Track A
performance gate is complete.

## Adversarial coverage

The new tests lock five failure/authority cases:

1. checkpoint failure restores `step_record`, current records, attempt history,
   and current capsule ref in place;
2. identical code reuses the exact selected capsule with no new checkpoint;
3. a rejected repair restores its exact parent and clears its latch only after
   success;
4. parent-restoration checkpoint failure retains the child selector and latch;
5. repair-child checkpoint failure retains the parent and all pending markers.

Two pre-existing runtime-repair tests had stale provider-call assertions. Their
fixtures deliberately return a complete script to a patch-first transport, so
one logical repair consumes a patch call plus an authorized full-rewrite call.
The tests now distinguish logical attempts from provider calls and explicitly
assert that resume adds zero duplicate calls. The same old assertions fail on
the untouched `9814bc0` baseline, so this is test-contract synchronization, not
a behavior relaxation.

## Verification

- New authority tests: `5 passed`.
- Capsule/storage/pipeline-contract group: `101 passed`.
- Core characterization/meta/capsule/pipeline-contract/resume-revalidation
  group: `180 passed`.
- Provider-budget plus deferred runtime-repair files: `51 passed` in the
  canonical outer environment.
- Full `tests/research_agent/test_resume.py`: `79 passed` in 143.48 s.
- Cross-seed characterization at `PYTHONHASHSEED=42`: `48 passed` in 2.63 s.
- Four runner-backed capsule-resume cases that return macOS nested-sandbox code
  71 inside the tool sandbox passed `4/4` when run in the canonical outer
  environment; production was not changed to accommodate that false red.
- Black, Ruff, `py_compile`, and `git diff --check`: passed.
- Shared production module scan contains no H2/E2/E3, database, outcome, or
  benchmark-specific routing literal.
- Independent read-only adversarial review: ACCEPT, with no blocking, major,
  or medium finding; the only non-blocking note is that initial-transition
  methods rely mostly on the existing integration/provider tests.

## Honest boundary / next action

This closes one reviewable checkpoint-authority batch. Claude should perform a
read-only adversarial review of `9814bc0..<this commit>`, focusing on rollback,
receipt-to-capsule ordering, selector uniqueness, and crash/resume behavior.
Track A is not yet declared complete; E3 remains paused until the remaining
milestone regression and P0-5 performance gates are explicitly satisfied.
