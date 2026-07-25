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
- Added nine `curated_mvp` cards with at least two URL/DOI-backed sources each:
  AKI prediction, sepsis prognosis, lactate trajectories, vasopressor
  comparative effectiveness, ventilation liberation, mortality prediction,
  longitudinal phenotyping, cross-database external validation, and a narrow
  early-peak-lactate association card separated from the trajectory card.
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
- The E2 peak-lactate card was separated from the longitudinal
  trajectory/clearance card. The Canonical9 retrieval matrix and explicit
  peak-versus-trajectory negative control pass in a 46-test combined suite.
- `tools/run_research_know_how_planner_ab.py` now owns the repeated Planner-only
  comparison: paired OFF/ON order, opaque labels, a frozen blind rubric, exact
  prompt/call/token/wall evidence, and a fail-closed review-status gate.
- Prepare-only acceptance uses the immutable 94,458-stay E2 parent context
  (`context_sha256=a8199c621f5ce7f3ddb426a78514ecdbab5fc6ea130b89dcb3b7a35fb816262c`).
  After the exact enum/claim-output contract was added, OFF is 65,423 bytes
  and ON is 69,932 bytes (+4,509); ON selects only
  `early_peak_lactate_association`.
- The evidence review packet
  `docs/reviews/early_peak_lactate_association_20260721.json` binds the exact
  reviewable content SHA and remains `authorization=false` pending real dual
  signoff.
- A bounded live pre-A/B probe returned HTTP 200 but exposed an output-contract
  ambiguity before any scientific comparison: OFF used intuitive Table 1
  aliases (`binary`, `mann_whitney_u`, `chi_square`), and ON also emitted
  incomplete claim-decision coordinates. The probe was stopped rather than
  spending the production five-retry budget. The case-neutral Planner prompt
  now lists the exact Table 1 enums and exact claim-decision object; focused
  tests lock both. This probe is diagnostic and is not counted as K6 acceptance.
- The corrected bounded development A/B then completed all four Planner trials
  against the same fixed context. OFF produced 2/2 valid plans with one call
  each (43,186 total tokens; 203.1 s active wall). ON selected only
  `early_peak_lactate_association` and produced 2/2 valid plans, but one plan
  required a structured-output retry (3 calls, 73,996 total tokens; 324.0 s).
  The ON plans more consistently separated measured from unmeasured lactate,
  preserved the descriptive/noncausal estimand, recorded claim-level evidence,
  and avoided an early-mortality sensitivity that overlapped the 0–24 h
  exposure window. This is useful scientific framing, not a speed improvement.
- `docs/reviews/early_peak_lactate_planner_ab_20260721.json` binds the source
  manifest SHA and records the development comparison. Because the operator had
  access to the run manifest, this is explicitly a structural pre-review rather
  than an independent blind clinical review.

## Safety boundary

Cards remain advisory. Missing concepts are shown as unresolved. Retrieval does
not exclude patients, choose a time zero, choose an estimand, install software,
query a network service, or mutate the global case-neutral prompt. The feature
is disabled unless `enable_know_how=True`.

## Not complete

- K5: the E2 card has completed methods/evidence pre-review, but formal clinical
  and quantitative-methods attestation is still pending; the other eight cards
  remain unreviewed and no card is authorized for paper-facing science.
- K6: deterministic preparation, prompt budget, rubric, reviewed-card
  fail-close, and the four-trial development A/B are complete. Formal K6
  acceptance still requires independent blind scoring and may not use the
  development override for an unreviewed card.
- K7: frozen B/C held-out generalization has not run. Tool/software capability
  retrieval is a separate later workstream and does not block this MVP.
- The fixed experiment data source is
  `/Volumes/外置硬盘/easyicu_data/full6_20260717`; do not re-extract six databases.
