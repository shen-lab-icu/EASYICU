# Post-Dev9 exact-head freeze preflight

- Date: 2026-08-23
- Parent checkpoint: `529591875175c55db74f43ea3ff40079797d1e13`
- Worktree: `/private/tmp/easyicu-merge-final`
- Scope: freeze-gate remediation only; no E1/E2/Qualification12/Held-out27 execution
- Provider calls / tokens / cost: `0 / 0 / 0`

## Root causes found before freeze

The first local full-suite attempt was not exact-head evidence: the command did
not set `PYTHONPATH=src`, while the machine's editable EasyICU install pointed
at the separate 17-entry dirty primary worktree. Its `14541 passed / 111 failed`
result mixed this clean checkout's tests with another checkout's package and is
therefore retained only as a checkout-contamination diagnostic.

Owner-level reproduction with `PYTHONPATH=src` reduced the actual freeze issues
to two intentional authority moves plus one prompt-budget regression:

- the optional research-design selection changed scorer-core `schema.py`, so
  the frozen v3 paper scorer tree digest required an exact implementation-hash
  refresh; rubric dimensions, task projection, safety protocol, and scoring
  semantics did not change;
- the provider-free resource/context baseline required remeasurement after the
  schema and Planner action-surface change;
- the first Method Adapter projection repeated adapter ids already represented
  by action identity and `execution_mode`, putting the fixed Planner directive
  over its byte guard. The projection now shows only partial/subcontract and
  support-only exceptions while full host-owned actions remain visible through
  the existing execution-mode groups. No prompt budget was raised.

A Coder prompt test that is 361 bytes over its target on local Python 3.13 was
also reproduced unchanged at frozen baseline `8115f93`. It is outside the
supported 3.10-3.12 CI matrix and was not treated as a new code defect.

## Frozen authority values

- Figure 2 scorer tree SHA-256:
  `5678d61a8de71132b4d3894cfb7879024caab3837b207677c08cb1bc14bd1844`
- Figure 2 suite projection SHA-256 remains:
  `11c39afac69c9a0b560c6aa92be19f05725f04c66b9d2c499cfdb353c40295ab`
- Figure 2 safety protocol SHA-256 remains:
  `76b4a20b39c76ce785d73fc9405954ed450bd2e6954b571621370699b3e9eb73`
- Offline Canonical9 maximum Planner envelope:
  `60,774 / 120,000` bytes (`67` bytes lower than the prior baseline)
- Resource selection / Coder resource envelope / Provider calls / patient-data
  reads: unchanged; Provider calls and patient-data reads remain zero.

## Verification before exact-head full suite

- Python 3.10 isolated environment, complete `dev + webapp + agentic` extras:
  - root-cause matrix: `7 passed`
  - adapter/prompt/resource/scorer matrix: `26 passed`
  - Anthropic native Messages transport imported and passed
- Correct-checkout scorer/evaluator and adjacent owner matrix:
  `342 passed` before the final adapter-test spelling correction; final focused
  adapter/resource/scorer matrix: `24 passed`
- Architecture gates: all 5 green; size/budget guards `141 passed`
- scorer manifest digest equals the live scorer tree digest
- provider-free resource baseline diff: green
- Ruff and `git diff --check`: green

## Claim boundary

This is a local freeze-preflight checkpoint. It is not a fresh E1 result, not a
formal benchmark result, and not yet full exact-head CI. No case-specific or
Sepsis-specific rule was introduced.
