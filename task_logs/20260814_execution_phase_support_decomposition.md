# 2026-08-14 — execution/phase.py decomposition batch (P3 structure debt, batch 4)

## Scope

One owner, one batch: the pre-loop helper block (L564–2524) of
`research_agent/execution/phase.py` was split. `run_execute_phase`
(4,987 lines) is untouched — splitting a giant function body is real
refactoring requiring characterization tests and is explicitly out of scope.

## Split

- `execution/phase_support.py` (1,613 lines): 72 self-contained helpers
  (terminal-record/capsule bookkeeping, runner-owns-step matchers, ordinal /
  missingness / robustness vocabularies, digest seal + provenance helpers,
  concept-constraint merges, code-reuse checks, replan directives...).
- `phase.py` (5,944 lines): the execute loop, six seam-adjacent stayers
  (`_selectively_revalidate_resume_successes`,
  `_execution_input_authority_integrity_finding`,
  `_planner_materialized_cohort_execution_receipt/prompt_payload`,
  `_repair_publication_figure_in_staging`,
  `_should_attempt_detached_figure_binding`), and a facade import that
  re-exports every moved name so `phase.X` lookups (including
  `monkeypatch.setattr(pipeline_execute, ...)`) keep resolving through phase
  globals.

Same fixpoint-ejection rule as the pipeline batch: any helper referencing a
test-patched name (`_evaluate_final_deterministic_gates`,
`trajectory_bundle_findings`, `sha256_of_file`, `write_run_checkpoint`,
`CriticAgent`, ...) or a non-header outside name stays.

## Verification

- Adjacent suites (step execution routing, attempt bootstrap, capability
  registry, quarantine state, candidate-loop decomposition, resume
  revalidation): 69 passed.
- 2 failures verified pre-existing via path-scoped stash controls:
  `test_execute_phase_routes_runner_and_gates_to_the_bound_step_cohort`,
  `test_replay_uses_shared_gates_and_never_constructs_llm_auditor` (the
  latter asserts a source string absent from the un-split `run_execute_phase`
  as well — it belongs to the concept-audit lane).
- End-to-end pipeline mock smoke (6 tests): passed.
- Module graph zero SCC; `phase_support` has no edge back to `phase`. ruff
  clean. Arch ratchet: new owner appended; baseline re-emitted with reason.

## Splitter note

The Assign-name scanner needed a `ctx=Store` filter — without it, bare Name
loads inside `frozenset({...})` values (e.g. `MISSINGNESS_SOURCE_AVAILABILITY_AUDIT`)
were misread as top-level definitions and leaked into the facade import.
Fixed before commit; recorded so the next splitter inherits the check.

## Follow-ups (not this batch)

- `run_execute_phase` (4,987 lines) and `ResearchAgentPipeline` (4,776 lines)
  are the remaining god-units; they need loop-body/method extraction with
  characterization tests, one owner per batch.
