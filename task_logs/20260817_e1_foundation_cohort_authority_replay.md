# E1 Foundation cohort authority repair and zero-provider replay

Date: 2026-08-17  
Task: `FIG2-DEV9-HELDOUT27` / E1 development canary  
Branch: `integration/figure2-e1-h3-20260816`  
Implementation commit subject: `fix(planner): align foundation cohort authority`

## Finding

E1 job `88fce836c8c2` reused its validated outline and Foundation, then made two
schema-valid calls for the first step. Both candidates were rejected before
Execute with `unknown concept_id: stay_id`.

The strict Foundation schema treated selected `ResearchContext` variables as
sealed, materialized cohort columns. The deterministic `AnalysisPlan`
validator, however, consulted only the packaged concept dictionary and did not
scope validation to the current run's materialized columns. The Planner could
not repair this mismatch because a current-step response is not authorized to
rewrite the already sealed Foundation.

## Repair

- Moved the run-bound cohort concept roster into the progressive compiler
  owner so transport and host compilation consume one policy.
- Scoped both `ConceptPredicate` construction and final `AnalysisPlan`
  validation to the current `ResearchContext` columns. The scope restores the
  prior registry exactly and does not permanently register run-specific ids.
- Added a Foundation boundary check before the first step provider call or
  Foundation checkpoint publication.
- Preserved fail-closed behavior with the typed reason
  `progressive_foundation_cohort_invalid` at path `cohort`.

## Verification

- Focused planner/compiler and cohort-boundary matrix: `75 passed`.
- Ruff check on the three changed files: passed.
- `git diff --check`: passed.
- Exact zero-provider replay of
  `run_20260817T123234_d192cb/progressive_compile_failure_replay.json`:
  accepted the original revision-0 materialization, compiled one
  `cohort_denominator` step, and produced
  `artifact:analysis_cohort` plus `table:cohort_flow` with `provider_calls=0`.

## Remaining boundary

No post-fix paid canary was started in this session. The active execution
sandbox rejected both local TCP bind on `127.0.0.1:8877` and external DNS, so
the Web service and GitHub push could not run. These are environment failures,
not Research Agent findings. The branch is locally one commit ahead; E1 remains
unverified and must resume from job `88fce836c8c2` in a development session
that permits local listening and account-provider network access. Do not enter
E2 before E1 completes all 11 stages.
