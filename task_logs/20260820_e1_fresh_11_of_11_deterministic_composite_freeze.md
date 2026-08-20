# E1 fresh 11/11 deterministic-composite freeze

Date: 2026-08-20

Task: `FIG2-DEV9-HELDOUT27` / E1 engineering acceptance

Verdict: fresh bounded E1 completed all 11 required execution and step-scientific-requirement contracts. This is a development/engineering closure only; the run remains `diagnostic_only`, `analysis_only`, and not paper-authorized.

## Fresh E1 authority

- Runtime source commit: `e689e91ade5ca22cbca64e531eec92a3d264ec6b`.
- Run: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_e1_e689e91_20260820_fresh/e1_sepsis3_prevalence_mortality/aware/run_20260820T160314_e82481`.
- `run_status.json` SHA-256: `dd90053cf2ebbd707ae74726b3a2af708a2421e77bae5f12bd226ddec6970f2c`.
- Plan SHA-256: `4d4d898db9c6409c8c67cf9170526c1931bb607b6f77c273191ed417e4cc665f`; it matches `progressive_planning_authority.json.analysis_plan_artifact_sha256`.
- Required/completed: `11/11`; missing `[]`; failed `[]`; scientifically incomplete `[]`; `execution_ok=true`.
- `primary_figure_suite` ran `deterministic_composite_descriptive_figure` and consumed four typed source tables. It produced PNG/SVG/PDF/TIFF plus source-data CSVs and a FigureContract. The execution suffix did not call Planner or Coder.
- Provider usage: 9 calls, all `role=writer`; Planner 0, Coder 0; 103,517 prompt + 16,671 completion = 120,188 tokens; provider-reported cost `$1.5353`; unknown usage 0. `cost_summary.json` SHA-256: `43b14615e9368b2a18163a16067cf5ae844eb8c041fc1f04e75def40c2c29607`.

## Authority boundary

- `status=diagnostic_only`, `forced_diagnostic_only=true`, maturity `analysis_only` (44).
- `artifact_valid=false`, `paper_authorized=false`, `evidence_complete=false`, `numeric_verified=false`, `analysis_validated=false`, and `manuscript_ready=false`.
- Therefore 11/11 proves the bounded E1 workflow executes and its deterministic composite adapter works. It does not make the checkpoint, its figures, or its analysis paper-ready. Held-out27 remains `0/27`.

## Generic owner fixes and final code freeze

- No E1/Sepsis-specific prompt, branch, variable, score, or expected answer was added.
- Generic fixes were limited to owner contracts: deterministic robustness/composite rendering, compiled effect authority, validation-only LOC handling, and exact resume path normalization for a same-SHA canonical plan/EvidenceStore alias.
- Final code freeze: `ce1223c82bd868f7968e464eaf61fdb2f4f9c8f6` (pushed `origin/main`). The fresh run predates this final freeze; intervening changes are generic authority/resume/governance fixes. Per the one-fresh-E1 policy, no second fresh E1 was run.
- Exact image: `easyicu-research-agent:ce1223c`; immutable image ID/repo digest `sha256:13657df0c4708473574b4e9660e16403155d47d5873939780575739c8120e5f2`; runtime check `status=ready`, `network=none`; execution-kernel identity `31da17efedb88a40926933b11589a95d3fa44a962ca2b3b6372b713469dd0321`.

## Verification

- Research-Agent exact-head CI: GitHub Actions run `32396320270`, all 3 jobs successful.
- Full exact-head CI: GitHub Actions run `32396320297`, all 7 jobs successful: Python 3.10/3.11/3.12 full suites, wheel/sdist install, and macOS/Windows/Ubuntu portability.
- Focused owner verification before freeze included exact capsule generation/audit/execution reuse and crash recovery; governance, typed-input/capsule, and architecture ratchets passed.

## Next gate

E1 is closed at the engineering level and the code/image contract is frozen. E2 may be scheduled next under Dev9, but Qualification12, Held-out27, formal runs, and paper claims remain gated. Do not rerun E1 merely to erase the documented commit gap.
