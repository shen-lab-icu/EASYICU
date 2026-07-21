# AGENT-KNOW-HOW-GOVERNANCE-V2 — evidence-bound protocol retrieval

Date: 2026-07-21

Branch: `codex/research-know-how-mvp`

Base: rebased onto `refactor/agent-control-plane@82427bb`

Scope: default-off Planner knowledge layer; no Canonical9 refresh, no API run,
no new extraction, and no tool/software capability retrieval.

## Delivered: K1–K4 governance and offline acceptance

- Card schema v2 binds every design, stop, and confirmation item to one stable
  `claim_id`, exact text, evidence scope, and one or more `citation_ids`.
- Retrieval separates `topic_applicable` from `data_readiness`; a relevant card
  remains visible when required concepts are missing. It permits zero hits and
  never pads the result to a nominal top-k.
- Prompt projection is canonical structured JSON. Mandatory stop, confirmation,
  readiness, version/SHA, and citation coordinates are never truncated.
- Know-How is passed to `PlannerAgent` in its own labeled context section, not
  mixed into `ResearchContext.notes`.
- Plans persist only claim-level `know_how_decisions`. The earlier coarse
  `know_how_refs` list was removed before merge; adopted cards are derived from
  decisions with `disposition=adopted`.
- Source trust (`built_in_reviewed`, `project_reviewed`,
  `user_supplied_unreviewed`) is independent of scientific review status.
  User-controlled cards cannot self-assert trust or enter the default Planner.
- `clinical_reviewed` requires a version/content-digest-bound dual clinical and
  methods attestation; editing a card invalidates it. All bundled cards remain
  honestly `curated_mvp`.
- Added eight `curated_mvp` cards with at least two URL/DOI-backed sources each:
  AKI prediction, sepsis prognosis, lactate trajectories, vasopressor
  comparative effectiveness, ventilation liberation, mortality prediction,
  longitudinal phenotyping, and cross-database external validation.
- Canonical9 A offline matrix is 9/9: E2/M2/M3/H1/H2/H3 retrieve their intended
  card; E3 and M1 honestly retrieve none; bilingual aliases and missing-concept
  negatives are locked by tests.
- Full initial Planner request has an 80,000-byte pre-provider hard gate.
  `planner_prompt_metrics.json` records system/user/total bytes, approximate
  tokens, exact Know-How delta, selected-card count, and limit.
- Historical submission profiles remain byte-identical. The additive,
  non-default `npj_dm_know_how_dev/20260721` profile pins Know-How on; profile
  mismatch fails closed and the profile coordinate is written to the manifest.
- `PlannerKnowHowBinding` owns resume verification, prompt metrics, and evidence
  persistence outside `pipeline.py`; v2 adds only three net pipeline lines over
  the original MVP.

## Verification

- Know-How/profile focused suite: 64/64 after the claim-only plan migration.
- Expanded Know-How + plan/replan/resume authority suite: 121/122. The only
  failure is Docker source-image SHA mismatch against this uninstalled
  worktree, before the test reaches changed logic.
- Ruff, Black, py_compile, diff-check, and module graph pass.
- Architecture integration was moved out of `pipeline.py`; the additive feature
  baseline is re-emitted only after this extraction and recorded with the tool
  SHA.

## Safety boundary

Cards remain advisory. Missing concepts are shown as unresolved. Retrieval does
not exclude patients, choose a time zero, choose an estimand, install software,
query a network service, or mutate the global case-neutral prompt. The feature
is disabled unless `enable_know_how=True`.

## Not complete

- K5: the eight cards have not received clinical/methods claim-by-claim review;
  none is authorized for paper-facing science.
- K6: repeated blinded E2 Planner A/B (2–3 runs/arm or fixed recorded response
  component comparison first) has not run.
- K7: frozen B/C held-out generalization has not run. Tool/software capability
  retrieval is a separate later workstream and does not block this MVP.
- The fixed experiment data source is
  `/Volumes/外置硬盘/easyicu_data/full6_20260717`; do not re-extract six databases.
