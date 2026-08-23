# High-frequency Method Adapter catalog batch 1

Date: 2026-08-23
Task: `CAPABILITY-INCUBATOR-V1`
Authority ceiling: `analysis_only`

## Outcome

EasyICU now publishes one dependency-neutral, Planner-visible catalog for 19
high-frequency ICU method actions. The catalog maps each action to its real host
owner, entrypoint, selection kind, required declarations, positive/fail-closed
tests, scope, and claim ceiling.

- 14 adapters own a complete typed action.
- 5 adapters are explicitly `typed_subcontract`: ordinal stratification,
  KM/log-rank, PH diagnostics, cluster sizes, and outcome-by-cluster.
- 5 pre-existing `host_owned` actions with only a helper or renderer are now
  exposed to the Planner as `supporting_only`, not as full adapters.
- Catalog receipt: `easyicu.method_adapter_catalog/1`
- Adapter count: `19`
- Catalog SHA-256:
  `9b3fa036525bcf8641adacd115f96e5d0474998cacc4a3a4405dfba641b3d39a`

The adapter layer cannot promote an analysis: every entry is capped at
`analysis_only`, and all ordinary capability, evidence, scientific-validation,
figure, and reportability gates remain authoritative.

## Why this shape

The official [Biomni repository](https://github.com/snap-stanford/Biomni) uses a
thin registry built around tool name, description, and required parameters.
EasyICU reuses that efficient registry idea but keeps stricter clinical-research
boundaries: exact action IDs, typed declarations, real owner coordinates,
negative tests, deterministic catalog digest, and a non-promoting claim ceiling.

No statistical algorithm was reimplemented. Existing reviewed executors remain
the sole computation owners, and `scientific_action_catalog` only projects their
contracts to the Planner.

## Validation

- Focused adapter/Planner/executor matrix: `158 passed`.
- Architecture gates: `5/5` green.
  - arch-measure ratchet: green
  - module graph: acyclic; intentional `575 -> 576` owner-module baseline update
  - import-linter boundaries: `7 kept, 0 broken`
  - Ruff: green
  - size/budget guards: `141 passed`
- `git diff --check`: green.

No E1-H3, Qualification12, or Held-out27 run was started. Provider calls,
tokens, and cost for this batch were all zero.

## Next boundary

The next efficient batch is the research-design/novelty layer: 2-4 candidate
designs with selection/rejection reasons, reliable literature retrieval and
novelty positioning, and per-figure support/non-support claim boundaries. The
five `supporting_only` actions must not be relabelled as complete adapters unless
they gain exact input/output ownership and execution-selection tests.
