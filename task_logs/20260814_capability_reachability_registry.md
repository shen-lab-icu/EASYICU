# Capability reachability registry — 2026-08-14

## Scope

Close review finding 10 without wiring dormant methods or deleting tested code.
The zero-inbound capability inventory now has one closed runtime vocabulary:
`production_reachable`, `experimental`, and `disabled`.

## Result

- Commit: `456db25` (`fix(agent): govern production capability reachability`).
- Production rows must declare a public-API-to-executor route and an exact
  `tests/...py::test_name` integration test.
- The audit verifies that the referenced test file and function exist.
- RMST, decision-curve analysis, DeLong, conformal prediction, unused survival
  inputs, optional evaluators, adapters, and plugin surfaces remain available
  for development but are no longer represented as production capabilities.
- The public `easyicu-research-agent` CLI now has a focused integration test
  proving that `cli.main` reaches `ResearchAgentPipeline.run`.

## Verification

- Capability registry, convergence, and scientific-assessment checks: 53 passed.
- Inventory governance and CLI checks, including two fail-closed negative
  cases: 9 passed.
- `tools/audit_capability_inventory.py`: OK.
- Ruff, `git diff --check`, and architecture ratchet: clean; no baseline refresh.
- No full exact-head CI was run because this remains E1 development iteration.

## Next

Restart the Web process at the new exact HEAD and run a fresh ordinary-dialog
E1 through Plan review, execution, evidence, figures, interpretation, and
manuscript gates. Do not start the formal Canonical9 Provider batch yet.
