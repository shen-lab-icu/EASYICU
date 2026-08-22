# H2 Dev9 1/1 fail-closed source-feasibility closure

Date: 2026-08-22 EDT

## Outcome

- Exact committed execution HEAD: `e455b25f36920aa1b0b3c8cea1fa51196597ecbf` (`main`, clean detached execution checkout).
- Exact runner image: `easyicu-research-agent:e455b25`, immutable digest `sha256:2e648ea42abdd5a02c044e53c3652bf0f95a2e1f34e75c6c412b588b3f542bbf`; runtime check returned `status=ready`, `network=none`.
- Run: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_e455b25_h2_execution_only_r7_20260822/h2_vasopressor_causal/aware/run_20260822T201422_1c46b0`.
- Required/completed steps: `1/1`; missing `[]`; failed `[]`; `execution_complete=true` and `step_scientific_requirements_complete=true`.
- The run remains `development_diagnostic` / `diagnostic_only` / `analysis_only`. It is not paper-authorized or manuscript-ready.

## Owner-local closure

1. Added the generic development execution-only projection for a signed `SourceFeasibilityRuntimeAuthority`.
2. Proved that only a digest-verified owner-compiled plan with zero materialized inputs may ignore an unrelated trajectory binding in development diagnostics. Data-consuming and formal paths remain fail closed.
3. When the sealed authority forbids a causal contrast, the benchmark launcher no longer declares a primary exposure/outcome contrast in `ResearchContext`.
4. The scientific-runtime owner now removes generic article-shaping additions and recompiles the exact one-step signed feasibility plan before final validation.

No conditional branch on the H2 task id, Sepsis, one database, or one manuscript result was added. The changes are authority/launcher/runtime-owner contracts.

## Verification

- Focused authority, launcher, trajectory-boundary and reporting-wiring suites: `43 passed` with Ruff clean.
- Fresh H2-only input materialization: `/Volumes/外置硬盘/easyicu_data/canonical9_miiv_h2_b90006c_20260822`; JSONL SHA-256 `f0f941363eb690c9e3973ed08882775c5bce159a5ac20bb58f7cf2041d709f69`, explicitly `paper_authority=false`.
- Final run plan contains one auxiliary step, `00_authority_compiled_source_feasibility`, with `inputs=[]` and method `signed_source_feasibility_fail_closed`.
- The typed output records `scientific_decision=blocked_by_source_authority`, `reason_code=H2_VERIFIED_NON_USE_UNAVAILABLE`, `verified_non_use_available=false`, `binary_control_arm_authorized=false`, `causal_contrast_authorized=false`, and `effect_estimate=null`.

## Provider and cost accounting

- Provider attempts: `0`; provider-reported calls/tokens/cost: `0 / 0 / $0.00`.
- Durable hard-stop ledger completed the task with no calls and no usage-unknown reservation.
- The local Writer stage attempted to reserve more tokens than the development ceiling and was blocked before any transport call. This does not change the completed 1/1 deterministic execution and cannot be presented as a manuscript result.

## Next gate

Proceed only to H3 Dev9 under its signed trajectory authority. Do not enter Qualification12/Held-out27 and do not run full CI until H3 closes and Dev9 is frozen.
