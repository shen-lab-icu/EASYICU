# Step-Level Typed-Input Receipt Completeness Handoff

Branch: `window2/helper-bool-highs-20260720`  
Base: `dace357`  
Scope: host-owned aggregate verification only; no central pipeline wiring.

## What Was Added

`easyicu.research_agent.authority.typed_input_receipt.verify_step_typed_input_receipts`
verifies the complete set of durable typed-input consumption receipts for one
candidate consumer.

Inputs:

- Planner current-step typed input declarations (`planner_declared_inputs`)
- host resolved bindings for those inputs (`resolved_input_bindings`)
- current `resolved_inputs_sha256`
- current `consumer_step_id`
- current `consumer_code_sha256`
- durable typed-input consumption receipt list
- optional host-only `row_identity_not_applicable` input keys

Outputs:

- `StepTypedInputReceiptVerification.findings`: structured
  `ValidationFinding` entries from validator
  `typed_input_receipt_completeness`
- `StepTypedInputReceiptVerification.verified_inputs`: immutable mapping from
  input key to verified receipt, populated only when the whole set passes

The function never returns a boolean and never guesses a row key.

## Enforced Invariants

- Every Planner-declared typed file input has exactly one receipt.
- Missing, duplicate, and extra receipts are `ERROR` findings.
- Extra or missing host-resolved bindings are `ERROR` findings.
- Each receipt must match the current input key, evidence id, artifact SHA,
  resolved-input manifest SHA, binding digest, artifact relative path,
  consumer step id, and consumer code SHA.
- Receipts from old code or old steps cannot replay into a new candidate.
- A receipt for one logical input cannot be interchanged with another.
- The same consumed file identity cannot authorize multiple logical inputs.
- Row identity is accepted only from an explicit product contract.
- If row identity is absent, the host must explicitly list the input in
  `row_identity_not_applicable`.
- `row_identity_not_applicable` is accepted only when the receipt itself carries
  a `not_applicable` row-identity marker.
- Inputs needing row alignment but lacking an explicit row-identity contract
  produce an `ERROR`; no column-name heuristic is used.

## Required Central Wiring

This branch deliberately does not edit central orchestration files.

Wire boundary 1: after candidate code is final and before execution.

- At this point the host knows the Planner step inputs, resolved bindings,
  `resolved_inputs_sha256`, `consumer_step_id`, and candidate code SHA.
- The gate should require an empty receipt set only when the step has no
  Planner-declared typed file inputs.
- If findings contain any `ERROR`, do not execute the candidate and do not spend
  a Docker run.

Wire boundary 2: before result, table, model, or figure sealing.

- After host-owned consumers/loaders/renderers have sealed their actual
  typed-input receipts, call the same aggregate verifier against the final
  receipt list.
- Do this before any result/figure/evidence becomes current authority.
- If the aggregate verifier emits any `ERROR`, fail the candidate before seal.

Host row-identity policy:

- For row-aligned scientific tables and typed products, require explicit
  `row_identity_column`, `row_count`, and `row_identity_sha256` in the product
  contract.
- For pure audit/dictionary/reference inputs that do not align rows, the host
  may pass `row_identity_not_applicable=[...]`.
- The central layer must not infer this from product names or column names.

## Files Intentionally Not Touched

- `src/easyicu/research_agent/repair_registry.py`
- `src/easyicu/research_agent/execution/phase.py`
- `src/easyicu/research_agent/pipeline.py`
- `src/easyicu/research_agent/authority/figure_renderer.py`
- online run directories
- prompts or benchmark specs

## Verification

Focused tests:

- `tests/research_agent/test_typed_input_consumption_receipt.py`

The new aggregate test cases cover exact success, missing receipt, duplicate
receipt, extra receipt, identity interchange, old step/code replay, duplicate
logical use of the same file, missing/extra resolved bindings, missing
row-identity contract, accepted host-declared row-identity not applicable,
missing not-applicable receipt marker, conflicting not-applicable declaration,
unknown not-applicable input, and invalid receipt schema.
