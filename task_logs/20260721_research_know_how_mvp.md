# AGENT-KNOW-HOW-MVP — Research Know-How retrieval layer

Date: 2026-07-21

Branch: `codex/research-know-how-mvp`

Base: rebased onto `refactor/agent-control-plane@82427bb`

Scope: default-off planner context only; no Canonical9 refresh and no new execution tools

## Delivered

- Added strict `KnowHowCard`, `KnowHowCitation`, `KnowHowHit`, and
  `KnowHowRegistry` contracts under `research_agent/know_how/`.
- Added deterministic offline retrieval using question overlap, mapped analysis
  family, database, and available `ResearchContext` concepts. The hard limits
  are top-k 5, default top-k 3, 1,200 characters per card, and 8,000 characters
  for the full Planner projection.
- Added eight `curated_mvp` cards with at least two URL/DOI-backed sources each:
  AKI prediction, sepsis prognosis, lactate trajectories, vasopressor
  comparative effectiveness, ventilation liberation, mortality prediction,
  longitudinal phenotyping, and cross-database external validation.
- Added default-off `PipelineConfig` fields and pre-plan integration after the
  existing literature/blueprint stage. Enabled runs register
  `know_how_retrieval.json` and `know_how_prompt.md` in `EvidenceStore`.
- Added typed Planner adoption through `AnalysisPlan.know_how_refs`. Initial
  Planner output is restricted to this run's retrieved ids; duplicates and
  unknown ids fail structured parsing. Replanner preserves refs exactly, and
  resume revalidates the retrieved authority before reusing a plan.
- Kept empty refs out of serialized plans, so default-off plan bytes and
  provider call count remain unchanged. The opt-in smoke test compares enabled
  and disabled runs directly.
- Updated package data, public lazy exports, architecture description, and
  usage documentation. Built sdist/wheel and verified an isolated offline wheel
  install loads all eight cards.

## Verification

- Feature/config/planner suite on `14243b8`: `41 passed`; after the final
  rebase onto `82427bb`, the focused schema/retrieval/evidence/Planner/resume/
  opt-in smoke suite passed again: `16 passed`.
- Package-boundary and module-graph gates passed.
- ExperienceBank, PubMed/pre-plan literature, ResearchContext, Planner group:
  `84 passed`; its one initial Docker source-digest failure passed when rerun
  through the macOS subprocess sandbox.
- Resume failure reruns with Docker deliberately disabled: `17 passed`; one
  existing deterministic-policy test failed. A second existing resume test
  also fails. Both failures were reproduced unchanged on the parent baseline
  `c947105`, so they are not caused by this branch:
  - `test_resume_retires_unchanged_draft_after_deterministic_policy_supersession`
  - `test_resume_reaudits_material_deterministic_quarantine_repair[already-repaired-stale-finding]`
- The earlier adjacent Table 1 / execution / golden run on `14243b8` produced
  `103 passed` and four baseline-reproducible failures. Three source-contract
  failures were subsequently handled by upstream `82427bb`; this branch does
  not alter the execution phase. The remaining historical failure was a stale
  parent golden digest, not a Know-How behavior difference.
- `ruff`, `black`, `git diff --check`, and
  `research_agent_module_graph.py --diff` passed.
- Wheel inspection and isolated install: `offline_installed_cards 8`.

## Safety boundary

Cards remain advisory. Missing concepts are shown as unresolved. Retrieval does
not exclude patients, choose a time zero, choose an estimand, install software,
query a network service, or mutate the global case-neutral prompt. The feature
is disabled unless `enable_know_how=True`.
