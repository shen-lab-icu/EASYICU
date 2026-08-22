# H3 Dev9 3/3 deterministic trajectory closure

Date: 2026-08-22 EDT

## Outcome

- Exact committed execution HEAD: `bb590ab9de40df04df88341d08c0966ee2f0d11c` (`main`, clean detached execution checkout).
- Exact runner image: `easyicu-research-agent:bb590ab`, immutable digest `sha256:c3d777d90f97dd622c0f716485b3a6a8fddbcb9770d16f9aca4a1dc513081ecd`; runtime check returned `status=ready`, `network=none`.
- Run: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_bb590ab_h3_execution_only_r6_20260822/h3_trajectory_clustering/aware/run_20260822T220237_eb986d`.
- Required/completed steps: `3/3`; missing `[]`; failed `[]`; `execution_complete=true` and step-level `step_scientific_requirements_complete=true`.
- Provider calls/tokens/cost: `0 / 0 / $0.00`; repair ledger: empty.
- The run remains forced diagnostic / `analysis_only`: run-level `artifact_valid=false`, `scientific_requirement_complete=false`, `analysis_validated=false`, and `paper_authorized=false`. This is a completed development execution, not a paper-ready result.

## Input authority

- Official native-v2 projection: `/Volumes/外置硬盘/easyicu_data/figure2_h3_typed_projection_d6c3bf7_20260822`.
- Materialized Canonical9 binding: `/Volumes/外置硬盘/easyicu_data/canonical9_miiv_h3_c95f345_receipted_20260822`, explicitly `paper_authority=false`.
- Canonical9 JSONL SHA-256: `41a62d21e59b151fbfff66dd407b43d921ffa263e520de7260cdff60a271b9ae`.
- Development binding receipt SHA-256: `962acf7ef42c5df35e008fa988c65efcd5f995176573b89e6bfc2b46f88a12a2`.
- Materialization receipt SHA-256: `e23758d469f89c4a34e92c4e14f1a79e2bc0902586511c6ef2c68558dfa32473`.
- The binding contains 94,458 ICU stays, 8,666,830 trajectory rows, six SOFA-2 component owner-receipt concepts and lactate.

## Owner-local closure

1. The signed trajectory representation now emits a typed `manifest:trajectory_window_manifest` and preserves the full typed subject universe, including all-unavailable members, with cohort-flow accounting.
2. The generic trajectory method contract supports the closed representation -> candidate selection -> stability-characterization chain without Planner/Coder execution.
3. Candidate selection records a prespecified scientific rejection as a successful execution with `scientific_status=failed_closed`, rather than converting it into a code-repair request.
4. The downstream stability owner consumes that rejection and emits a typed no-solution state: zero stability refits, no frozen assignments, and empty profile/size/stability tables.
5. The candidate owner registers all declared typed output products so execution-only resume can validate the products without asking a model to reconstruct them.

No conditional branch on H3, Sepsis, one database, or one manuscript result was added. The changes are generic representation, method-runtime, candidate-settlement, stability and declared-product contracts.

## Scientific result and claim boundary

- Candidate selection reached the upper prespecified grid boundary (`k=6`) without an interior BIC optimum.
- The signed reason is `H3_NO_INTERIOR_BIC_OPTIMUM`; candidate `reportable_result=no_interior_solution_in_prespecified_candidate_range`.
- Stability characterization therefore records `reportable_result=no_stable_phenotype_solution`, `freeze_status=not_frozen_candidate_selection_failed_closed`, and `stability_refits_executed=0`.
- No frozen phenotype assignments or phenotype profiles were produced. Empty typed outputs are the correct fail-closed result, not missing work.
- The capability remains `analysis_only` with `scientific_validator_unavailable`; this run cannot support a phenotype discovery, stability, clinical-outcome, or publication claim.

## Efficient failure-to-closure sequence

- r3 exposed the missing typed window manifest and characterization owner before scientific execution.
- r4 executed candidate selection and exposed an incorrect attempt to route a prespecified scientific rejection into Coder repair; no Provider call occurred because the bounded reservation blocked transport.
- r5 closed scientific settlement but exposed missing declared candidate output registrations; again no Provider call occurred.
- r6 reused the same deterministic execution route and completed 3/3 with no Planner/Coder/Writer calls.

No full Provider canary was repeated between these owner-local fixes.

## Verification

- Focused signed-authority, runtime and representation suite: `49 passed`.
- Candidate registration and declared-product suite: `273 passed`.
- Broader trajectory suite: `183 passed, 2 skipped`, plus one pre-existing prompt-budget assertion (`42,217 > 42,000`) in a concurrently modified file; it was not changed or hidden by this work.
- Ruff and scoped diff checks passed.
- Exact image runtime identity: kernel identity SHA-256 `dcfc3ce161823854e4f625f14f4aaeed3e33882ee056ea46f07807a3aa7f7f9c`; 303 kernel files; runtime status ready.

## Remaining non-execution gates

- The legacy trajectory-bundle validator expects at least two wide fixed-window source inputs, while this signed long-tier plan uses the typed representation authority.
- Writer binding did not produce a substantive manuscript, and literature/novelty/figure/reportability gates remain open.
- These issues explain `artifact_valid=false` and the run-level scientific/reporting gate remaining false. They do not change the completed 3/3 execution and must not be described as paper readiness.

## Next gate

Dev9 E1-H3 is now execution-closed. Freeze the exact core contracts, run one full exact-head CI, then proceed to the post-Dev9 architecture batch (15-20 high-frequency ICU adapters, literature/novelty positioning, and problem-specific design alternatives). Do not enter Qualification12/Held-out27 until that freeze is recorded.
