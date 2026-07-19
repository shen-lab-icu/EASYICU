# Research-agent pre-v1 compatibility-facade retirement

Date: 2026-07-19

## Decision

The user explicitly chose a breaking pre-v1 cleanup. Historical diagnostic runs remain archived as evidence, but the next E3/H2/E2 and Figure 2 runs will be fresh runs against the final architecture. The new package therefore does not preserve obsolete Python import paths solely to resume historical experiments.

The last compatibility-preserving tree is tagged:

`archive/pre-v1-agent-compat-20260719` (`dd9ea16`)

## Change

- Removed 65 exact top-level compatibility facades and 3 archive-only shims.
- Repointed production, generated adapters, tools, tests, and generated documentation to canonical responsibility packages.
- Replaced facade-identity tests with canonical-ownership, package-laziness, import-direction, and retired-path absence tests.
- Added release-archive assertions that the 68 retired modules are absent from both sdist and wheel.
- Rebased the module-graph baseline from the compatibility tree to the canonical tree.

Structural result:

| Metric | Before | After |
|---|---:|---:|
| top-level research-agent `.py` files | 160 | 92 |
| modules | 295 | 227 |
| import edges | 822 | 754 |
| cyclic modules / SCCs | 0 / 0 | 0 / 0 |

This is the first stage only: the remaining 92 files are real implementations and public entry points. They are being moved into responsibility packages in subsequent bundles; the target is a small public root, not 92 permanently flat implementation files.

## Behavioral and release evidence

- Canonical ownership + retired-path + graph + clean sdist/wheel: `261 passed`.
- Golden/meta/capability/method/execute/resume-revalidation: `132 passed`.
- Earlier broader critical shard: `445` tests with one expected golden mismatch; the mismatch was independently compared against the archived tag and was limited to generated deterministic-adapter import-path SHA propagation.
- After updating only the two derived authority hashes, the normalized golden passes. Numeric values, tables, product files, statuses, and gates were unchanged.
- Ruff, compileall, module-graph diff, and `git diff --check` passed.

## Golden boundary

The generated deterministic adapter now imports its runner from the canonical package rather than a deleted facade. That intentionally changes the generated code SHA. Numeric-claim evidence IDs and current-evidence mapping hashes change transitively because those seals bind the code SHA. No scientific result, denominator, model output, table, figure value, or gate verdict changed.

## Next

Move real implementations by responsibility (agents/providers, data foundation, planning/contracts, gates/figures/reporting, authority/evidence, execution/orchestration), shrink the root public API, then run segmented regression and fresh E3/H2/E2/Figure 2 experiments.
