# Dev9 deterministic reporting suffix and robustness-authority closure

Date: 2026-08-24  
Branch: `codex/dev9-quality-remediation`  
Code commits: `99f03ec`, `4a6bf92`  
Provider calls: 0

## Problems assigned to owners

1. `reporting/write_phase.py` returned on `stop_after_analysis` before invoking the provider-free `PublicationFigureSkill`. A valid deterministic M3 phenotype figure therefore remained a supporting figure instead of receiving its run-level article-display bundle.
2. `reporting/scientific_maturity.py` counted only `AnalysisPlan.robustness_specs` and the root generic `robustness_panel.json`. It ignored digest-bound family evidence already registered in EvidenceStore, including E1's host-owned association model grid and E2's signed spline-versus-linear functional-form sensitivity.

Neither defect required task-specific Planner/Coder behavior or a new statistical result.

## Repairs

- `99f03ec` runs only the deterministic publication-figure suffix before the optional Writer pause. Literature/manuscript drafting still remains paused, so the path makes no Provider call.
- `4a6bf92` lets the scientific-maturity owner accept only registered runner step summaries whose bytes match their EvidenceStore SHA-256 and whose typed receipt/row contract demonstrates an independent sensitivity axis.
  - `easyicu.association_model_grid_runtime_receipt/1` contributes timing, cohort, or model axes from exact reference-bound variants.
  - Family robustness rows contribute an axis only when `converged=true`, `independent_variant=true`, the axis is non-primary, and an evidence id is present.
  - Unregistered files, digest mismatches, non-independent documentation rows, and duplicate generic replays do not count.

## Focused verification

- Ruff: passed.
- Reporting/figure suffix focused suite: `77 passed` before commit `99f03ec`.
- Adjacent pipeline/completion/display/phenotyping suite: `23 passed, 1 deselected` before commit `99f03ec`.
- Scientific-maturity, reporting boundary, and association-model-grid suite after `4a6bf92`: `17 passed`.
- Real registered-evidence read-only probes:
  - E1: `cohort`, `model`, and `timing` axes recovered from `evidence/statistic_step_summary_a8a1605cb46f6176__step_summary.json`; the standard duplicate missingness replay no longer erases these completed axes.
  - E2: `functional_form` recovered from `evidence/statistic_step_summary_bab2f1b4c38782cc__step_summary.json`; with its locked missingness axis, it satisfies the two-axis development rule.
  - E3/M1/M2 remain at one genuine axis and therefore still require a new author-approved study version or must remain analysis-only.

## M3 provider-free real-evidence probe

The exact latest M3 registered step evidence was copied to an isolated `/tmp` run and passed through the current `PublicationFigureSkill`:

- generated: true
- Provider/Planner/Coder calls: 0
- contract SHA-256: `ca8b52b510bb85db247d99eb9fe96bcef327ff96de20f25cdd4ee0d1727d7434`
- PNG SHA-256: `aa8a6f1ae995c0595cb14777a7446f574d6396b5447b2a491e5a97be3784b0db`
- SVG SHA-256: `bd676423e2e78a79eb6ffcabcb5f61d1c6a88476619014cf4c5210faccfe3889`
- PDF SHA-256: `9b0630d14831ee86c71e6b9f30b44f8dc2b496c62a0f193901c6a8e088305823`
- TIFF SHA-256: `9ecb3489665b0efd4988b2e4eb974c5d6b5103f3ed0ce5d788e1408b293aec7b`

This proves the deterministic owner path but is not an exact-image full run and does not provide publication authority. The source run remains immutable.

## Remaining scientific decisions

The exact pending package is `benchmarks/figure2_canonical9/dev9_scientific_decision_packet_20260824.json` (SHA-256 `82d6df250b5a5bfdf3254205cc6497164976c63657a82a660200f0eadedea769`). It covers only changes that would alter time zero, adjustment, sensitivity definitions, prediction validation, or non-PH handling. No implementation should silently activate those choices before author approval.

Qualification12 and Held-out27 remain untouched.
