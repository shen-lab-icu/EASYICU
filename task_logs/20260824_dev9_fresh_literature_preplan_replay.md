# Dev9 fresh literature/preplan replay on additive profile

Date: 2026-08-24

## Exact coordinate

- HEAD: `0e6f91b` on `codex/dev9-quality-remediation`.
- Profile: `npj_dm_dev9_demo_dev/20260824`.
- Profile SHA-256:
  `03a984b85c87afcb03653369dd218e0d975a86b0dfd9cb88a653e28948caded0`.
- External receipt:
  `/Volumes/外置硬盘/easyicu_data/figure2_dev9_literature_preplan_replay_0e6f91b_20260824/replay_receipt.json`.
- Receipt SHA-256:
  `33ce97040789f34743a79d5a13f020e3f4e621b743f2b5667e0b368e342917f3`.
- Provider calls/tokens/cost: `0 / 0 / $0.00`.

## Outcome

The replay loaded each saved Dev9 `ResearchContext`, ran fresh live PubMed
retrieval through the additive profile, and attempted to review the exact saved
`AnalysisPlan` against the new literature bundle. It wrote only to a new
external audit directory and did not modify any historical run.

| Task | Retrieval | Current typed review | Exact remaining literature/plan blocker |
|---|---|---|---|
| M2 | 3 queries, 8 records, 5 design analogues | generated; `changes_required` | `DESIGN_ANALOGUE_NOT_BOUND_TO_PRIMARY_PLAN`; independent novelty review still required |
| M3 | 3 queries, 8 records, 2 design analogues | blocked before review | archived plan fails current schema: cohort predicate uses unknown concept id `susp_inf_n` |
| H2 | 3 queries, 8 records, 0 design analogues | generated; `changes_required` | `DESIGN_ANALOGUE_NOT_ESTABLISHED`; novelty not established; causal result remains unauthorized |
| H3 | 3 queries, 8 records, 1 design analogue | generated; `changes_required` | analogue not bound; method layers/design route and scientific-step method sources also unbound |

## Interpretation

- The retrieval/design-analogue implementation is working; the current
  blockers now belong to plan/schema/scientific-authority owners rather than
  search mechanics.
- M2 and H3 need a fresh Planner revision that explicitly binds the retained
  analogue and records what design element is used or rejected. Hand-editing
  those bindings would violate Planner ownership.
- M3 requires a fresh current-schema plan or an owner-approved generic legacy
  migration. Mapping `susp_inf_n` to `susp_inf` by hand is not safe: the saved
  column is a materialized count whereas the catalog concept is a source event.
- H2 correctly remains fail closed. Absence of an acceptable design analogue
  must not create a synthetic control arm or causal estimate.
- This replay is literature and plan-review evidence only. It is not execution,
  analysis validation, novelty attestation, a manuscript result, or publication
  authorization.

## Next efficient action

Run one bounded fresh Planner pass only for M2, M3, and H3 under the new
profile; require exact citation-to-step design bindings and a current typed
cohort contract. Keep H2 as a signed negative feasibility result. In parallel,
make the AI development choice for H1 non-PH handling and finish the M3
deterministic article/display suffix. Then replay only affected owners, freeze
one exact HEAD/image, and run one full exact-head CI before Qualification12.
