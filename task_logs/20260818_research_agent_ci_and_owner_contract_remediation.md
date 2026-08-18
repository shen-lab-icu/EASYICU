# Research Agent CI and owner-contract remediation

Date: 2026-08-18
Task: `FIG2-DEV9-HELDOUT27` engineering gate repair
Base: `main@3eb28ef`
Scope: repository/CI truth, Research Agent maintainability, and fail-closed regressions only. No E1 canary or scientific result was produced.

## Why this checkpoint was needed

The 2026-08-17 handoff said the Research Agent gates were green, but the pushed workflows invoked `pytest -n --dist` without declaring `pytest-xdist`. A fresh CI environment therefore exited before collecting its first test. A local full run also exposed stale post-refactor assertions, architecture ownership gaps, and two real tuple-contract defects in rarely exercised pipeline branches.

## Repairs

1. Declared `pytest-xdist>=3.0` in the development dependencies and the explicit Research Agent workflow install list. The repository contract now checks that workflows cannot add xdist flags without the matching dependency.
2. Updated provider-boundary tests to the current `providers/clients.py` owner and preserved the tightened Codex subprocess secret policy: parent `OPENAI_API_KEY` is deliberately not inherited.
3. Moved mixed responsibilities behind small owner contracts:
   - Coder result-envelope validation → `agents/coder_output_contract.py`.
   - Replanner context/suffix compilation → `agents/replanner_context.py` and `agents/runtime_suffix_replanner.py`.
   - Evidence record resolution → `authority/evidence_record_resolution.py`.
   - Capability and association-result findings → `contracts/capability_declaration.py` and `contracts/association_result_findings.py`.
   - Figure-step, article-design, robustness mutation, and figure-promotion policy → dedicated `planning/` and `figures/` owners.
4. Consolidated byte-equivalent finite-float coercion into `numeric_scalars.py`. The duplicate-helper inventory fell from 31 named definitions to 19. The remaining helpers were not mechanically merged because they encode different contracts such as p-value ranges, non-negative counts, boolean predicates, typed-only acceptance, or raising validation.
5. Bound the resource baseline to Planner source, schema, and system-prompt digests; updated the frozen Figure 2 scorer digest and the characterization golden only where current deterministic authority order was independently reproduced.
6. Closed three full-suite defects without weakening gates:
   - `_step_run_concept_repair_phase` no longer returns `AgentRuntimeState` in the terminal-record slot.
   - a blocked hypothesis blueprint now preserves the pre-plan helper's fixed five-value return contract.
   - checkpoint I/O failure tests inject at the current `execution.phase_support` owner, so they exercise the real reservation/candidate checkpoint and still require `execution_raised` plus `diagnostic_only`.
7. Kept the architecture ratchet honest. The final `pipeline.py` is two physical lines below the accepted baseline; no baseline refresh was used to hide the terminal-return correction.

## Verification

| Gate | Result |
|---|---|
| Scoped Ruff over Research Agent, affected benchmark/tests/tools | pass |
| Candidate-loop, step-attempt, experience-bank, analysis-type, blocked-blueprint, and checkpoint regressions | `63 passed` |
| Original audit failure files plus repository hygiene | `72 passed` |
| Repository/benchmark/resource/golden/module-graph gate batch | `210 passed` after one LOC ratchet was correctly detected and corrected during iteration |
| Final architecture + affected pipeline regressions | `34 passed` |
| Full `tests/research_agent` with real xdist collection | `10,978 passed, 13 skipped, 0 failed` in `1,904.69 s` |
| Current module graph | `558 modules / 2,226 edges / 0 cyclic SCCs` |

The 13 skips are environment/data-dependent skips already declared by the suite. The full run emitted 484 warnings but no failures.

## Scientific boundary

This checkpoint repairs engineering and evidence-governance gates. It does not answer E1, does not establish E1 11/11 completion, does not authorize E2, and does not change Held-out27 readiness. Formal readiness remains `0/27`; the next scientific action is one fresh bounded E1 canary from the clean pushed commit.
