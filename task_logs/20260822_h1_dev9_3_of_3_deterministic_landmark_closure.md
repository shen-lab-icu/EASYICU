# H1 Dev9 3/3 deterministic landmark-survival closure

Date: 2026-08-22 EDT

## Outcome

- Exact committed HEAD: `87679130819036fc0744e721365d93f115843bd1` (`main`, clean detached execution checkout).
- Exact runner image: `easyicu-research-agent:8767913`, immutable image digest `sha256:794ec1fa97b2e871cf927cf7c07b45a2431e9f3af0549d86cb2034b3998270ec`; `tools/check_agent_runtime.py` returned `status=ready`, `network=none`.
- Run: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_a0991c1_h1_execution_only_r9_20260822/h1_ventilation_survival/aware/run_20260822T191503_97f329`.
- Required/completed steps: `3/3`; missing `[]`; failed `[]`; `execution_complete=true`.
- The run remains `development_diagnostic` / `diagnostic_only` / `analysis_only`. It is not a fresh paper-authorized run and is not manuscript-ready.

## Owner-local closure

1. Split the signed landmark survival method into a deterministic analysis owner and a separate source-bound deterministic figure renderer.
2. Made the renderer consume the exact typed KM, Cox and risk-set tables and copy those byte-identical sources under canonical local names declared by its `FigureContract`.
3. Fixed the generic `FigureSourceDataValidator`: a byte-identical upstream/source table is now accepted by exact SHA-256 lineage even when an all-numeric table has no textual join key. Non-identical files still undergo the existing row/value checks.

No conditional checks on `h1_ventilation_survival`, Sepsis, one database, or one benchmark id were added. The changes are method/authority/validator-owner contracts.

## Verification

- Focused authority, typed DAG, declared-product, figure-source, host-scaffold and plausibility suite: `519 passed`.
- Figure-source and signed-authority follow-up suites: `423 passed` after the exact-digest validator fix.
- Final checkpoint resume reused the same run and restarted only `02_authority_compiled_survival_figure`; Planner and Coder were not rerun.
- The three renderer source CSVs match their upstream source bytes exactly:
  - KM `87457464a1267041f3c559add9448b573cd28384f1c0affc38a7adcfb7da4227`
  - Cox `88a87084cf9ca1ad21f9fedbccb19670cbd171da9ebc20dee2ab80b989468fd0`
  - risk-set flow `b4488b253e5ce9a1acfe5328f8b029372da99d3b782150aa70076109973cd074`
- SVG/PDF/PNG and `landmark_survival_suite.figure_contract.json` were emitted; original-resolution visual review found no clipping or overflow.

## Development result and claim boundary

- Source ICU stays: 94,458; landmark population: 78,600; complete cases: 78,580; events: 8,921.
- Adjusted HR: 0.5276 (Wald 95% CI 0.4987-0.5581).
- Schoenfeld audit: global `p=7.889e-35`; exposure `p=0.05131`.
- The signed policy therefore records `violation_block_paper_authorization`; `paper_authorization_allowed=false`. The estimate is a descriptive prognostic association only and cannot be called a causal ventilation effect.

The source input is a development binding derived from the documented MIMIC-IV v3.1 export semantics. It is not a formal Held-out27 or paper input freeze.

## Provider and cost accounting

- Successful/charged Provider calls in the final H1 closure: 0; prompt/completion tokens: 0/0; reported cost: `$0.00`.
- The durable hard-stop ledger conservatively retains one `failed_usage_unknown` local transport attempt against `http://127.0.0.1:1/v1`: 160,820 reserved tokens / `$4.1682` upper bound. This was an intentionally unreachable local endpoint, not a successful external Provider call or reported charge.

## Next gate

Proceed only to H2 Dev9 under its sealed authority. Preserve H2's fail-closed non-use decision if no causal contrast is authorized; do not invent a control arm, do not enter Qualification12/Held-out27, and do not run full CI until Dev9 is frozen.
