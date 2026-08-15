# 2026-08-14 — pre-existing test-failure remediation (concept-audit + source-inspection seams)

## Root causes and fixes

All failures predate today's decomposition batches (verified via path-scoped
stash controls during batches 2–5) and trace to two concurrent workstreams:
the owner-issued concept-metadata contract (`0d59ad2`) and the candidate-loop
extraction (`1e5182a`).

1. **Outcome-ambiguity downgrade dead after owner metadata** (real code gap,
   `audits/validators.py`): `_downgrade_metadata_supported_outcome_findings`
   still required `descriptor.source_concept` to literally be
   `icu_mortality`/`hospital_mortality`/... . Owner metadata now keeps
   `source_concept` as the physical concept id (`death`) and carries the
   endpoint semantics in the description, so the downgrade never fired and
   optional-audit false positives blocked runs as errors. Fix: project the
   descriptor through `_descriptor_endpoint_semantic_key` (the same helper
   the context builder uses) before the membership test. This also closes a
   latent KeyError: `_script_has_conflicting_mortality_semantics` indexes
   `conflicting_labels[source]`, which requires the semantic key.
2. **Stale source-inspection seams** (three tests kept parsing
   `run_execute_phase` for code that moved to the candidate loop in
   `1e5182a`):
   - `test_execute_phase_routes_runner_and_gates_to_the_bound_step_cohort`:
     per-step `pipeline._build_runner(cohort_path=attempt.step_execution_cohort_path)`
     now asserted in `_candidate_execute_transition`; the phase-level
     `_bind_step_execution_cohort` assertion unchanged.
   - `test_replay_uses_shared_gates_and_never_constructs_llm_auditor`:
     `concept_audit.findings_for_code(` now asserted in
     `_candidate_concept_audit_transition`, with a no-fresh-LLMConceptAuditor
     guard on that source too.
   - `test_numeric_replay_is_wired_before_the_existing_in_run_repair_gate`:
     replay-before-repair-gate ordering now asserted inside
     `_candidate_contract_setup_transition`; the typed repair ticket
     consumption asserted in `_candidate_contract_repair_transition`.
3. **Question-text outcome override retired** (test fixture): the prompt test
   expected the question "…ICU mortality" to rebind `death` to
   `icu_mortality`. Under the owner-metadata contract the question may not
   redefine a physical concept; assertions updated to the owner-issued
   definition (`in hospital mortality`) present via `outcome_semantics`.

## Verification

- `test_validators.py` 217 passed (was 159 + 3–4 failures).
- `test_resume.py` full file 84 passed (the two quarantine-supersession
  failures resolve with the downgrade fix — they route through the concept
  audit).
- Combined regression across all touched suites: 396 passed.
- ruff clean on all edited files.

## Owners' note

The metadata semantics change and the candidate-loop extraction landed
without updating these tests; the source-inspection pattern is fragile by
design (it pins architecture). `test_candidate_loop_decomposition.py` now
covers the loop's own seams — new moves should extend that file in the same
commit that moves the code.
