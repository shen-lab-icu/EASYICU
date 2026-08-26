# Progressive design selection + novelty binding (batch 2)

- Date: 2026-08-23
- Parent checkpoint: `b5421e0e7f6bfe3d1d56436546fab5f79f032197`
- Worktree: `/private/tmp/easyicu-merge-final`
- Scope: post-E1 architecture optimization only; no E1/E2/Qualification12/Held-out27 execution
- Provider calls / tokens / cost: `0 / 0 / 0`

## Outcome

Fresh Progressive Planner v2 outlines now carry one typed
`easyicu.research_design_selection/1` authority. It requires 2–4 scientifically
distinct candidate designs, exactly one selected design, and an explicit
scientific rejection reason for every alternative. Each candidate records the
estimand, time zero, observation window, primary method, required variables,
assumptions, sealed literature keys, novelty positioning, figure role, what the
design supports, and what it cannot prove. The claim ceiling is fixed at
`analysis_only`.

Selection based on observed significance, p-values, AIC/BIC, AUC, or observed
effects fails closed before a checkpoint is emitted. Run-bound validators also
reject unavailable analysis families, variables, and literature keys, and
require the selected design to match the outline family and bind a question
anchor.

The selection is produced in the existing outline call, so it adds no Provider
round-trip. It is carried through the host-assembled skeleton into
`AnalysisPlan`, where it participates in resume/scientific-scope identity.
Historical checkpoints without the optional field remain digest-compatible;
only a fresh Progressive Planner outline is required to supply it. The classic
one-shot Planner and legacy one-shot skeleton transports omit the repeated
high-entropy schema to preserve their existing payload budgets.

## Literature and novelty route reused

No literature database or retrieval framework was added. The existing route
already provides:

- PubMed and optional Tavily retrieval into one `LiteratureBundle`;
- exact search provenance and record-to-query binding;
- record-level screening decisions and direct-comparator roles;
- sealed literature citation keys and typed article-to-design bindings;
- a digest-bound unsigned `novelty_positioning_audit.json` that cannot claim
  novelty without independent review.

The selected design is now projected into that existing novelty packet:
selected estimand, time zero/window, method, Planner positioning, supported
claim, and cannot-prove boundary are exposed for comparator review. The packet
remains `review_required` or `not_established` until a digest-bound independent
review completes every prescribed comparison dimension.

## Verification

- Focused contract/integration matrix: `342 passed, 1 deselected, 1 warning`
  - design-selection contract and fail-closed cases
  - Progressive Planner generation, checkpoint replay, and compile path
  - classic Planner payload budget and structured output
  - literature bindings and method literature
  - novelty positioning and scientific maturity
  - write-phase boundary, pipeline resume, and execution plan scope
- Architecture gates: all 5 green
  - architecture ratchet: no lower-is-better metric regression
  - module graph: acyclic; intentional `576 -> 577` modules and `2318 -> 2321` edges
  - import contracts: 7 kept, 0 broken
  - Ruff: green
  - size/budget guards: `141 passed`
- `git diff --check`: green

## Claim boundary

This is a local architecture checkpoint, not a fresh E1 result, not full
exact-head CI, and not manuscript/benchmark readiness. No case-specific or
Sepsis-specific logic was introduced.
