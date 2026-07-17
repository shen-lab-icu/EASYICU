# Lossy-coercion repair anchor and pytest isolation

Date: 2026-07-16

Branch: `refactor/agent-control-plane`

Baseline: `be17f40`

## Scope

This small follow-up addresses Claude's M1 review finding and two pre-existing
pytest isolation defects. It does not change the research-agent scientific
policy, evidence authority, resume semantics, provider budget, or E3 artifacts.

## Changes

1. The deterministic `LOSSY_NUMERIC_COERCION` repair no longer requires the
   generated audit dictionary to use the E3-derived literal key
   `newly_invalid_or_coerced_n`.
2. The repair derives the key from exactly one structurally verified
   `original.notna() & coerced.isna()` loss-count entry and emits the guard with
   `repr(key)`. Multiple structural candidates, duplicate keys, dynamic keys,
   `**mapping`, mismatched finding lines, or unsafe `int` bindings still refuse
   deterministic mutation.
3. Mechanical preflight also treats a loss-count dictionary with any computed
   key as runtime-unstable. A computed key can equal and overwrite the literal
   count key, so an apparent handwritten guard must not make that script pass.
4. The research-agent session fixtures moved from the subdirectory conftest to
   the repository-level `tests/conftest.py`. This avoids pytest 9.1 losing the
   fixture when test paths enter, leave, and re-enter `tests/research_agent`.
5. The lightweight package loader now restores the normal
   `easyicu.research_agent` parent-child module binding on both fresh-load and
   already-loaded paths.

The key was intentionally not added to the global coder prompt: the validator
already recognizes the structural calculation independently of its display
label, and turning a model-generated label into a new host protocol would add
prompt and authority drift without improving safety.

## Verification

- Lossy-coercion and loader targeted tests: 36 passed, including renamed and
  escaped keys, ambiguous structural keys, literal/dynamic overwrite refusal,
  and rejection of a handwritten guard defeated by a computed key.
- Extended repair/preflight/meta suite: 558 passed; five figure-policy tests
  failed identically on an untouched `be17f40` clone and are baseline debt,
  not this patch.
- Deferred-audit -> anti-pipeline order regression: 53 passed.
- Runner/synthetic-fixture/cache/loader group: 50 passed.
- Characterization suite: 48 passed.
- The combined resume/characterization collection ended 115 passed / 12 resume
  failures after cache invalidation. A representative clean-run failure
  (`test_partial_manifest_is_written_after_run`) reproduces on an untouched
  `be17f40` clone; an earlier cache-hit collection passed 127/127. This
  cache-sensitive baseline debt is recorded for the milestone regression batch
  and was not changed here.
- The exact pytest 9.1 cross-directory sandwich reproducer now resolves `ra`.
- Black, Ruff, `py_compile`, and `git diff --check` passed.
- Production source contains no `newly_invalid_or_coerced_n` literal.

No E3 run, network call, or archived experiment mutation was performed.
