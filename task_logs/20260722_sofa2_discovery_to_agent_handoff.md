# SOFA-2 longitudinal discovery → Research Agent handoff

Date: 2026-07-22

## Outcome

`FIG5-DISC-017` is now connected to the standard Research Agent handoff as an
outcome-free `trajectory_clustering` task.  The handoff no longer invents a
predictor or mortality endpoint for concept-set analyses.  A parent task pack
contains six database-specific child handoffs for AUMC, eICU, HiRID,
MIMIC-IV, MIMIC-III, and SICdb.

The pack remains `hold / awaiting_human_confirmation`.  Idea Mining proved
longitudinal data readiness; it did not choose time zero, observation window,
time grid, trajectory representation, clustering method, stability threshold,
or cross-database matching metric.

## Durable evidence

- Discovery readiness:
  `research_output/experiments/FIG5-DISC-017/longitudinal_sofa2/longitudinal_discovery_manifest.json`
- Parent analysis task:
  `research_output/experiments/FIG5-DISC-017/analysis_task/longitudinal_analysis_task_pack.json`
- Standard discovery ledger:
  `research_output/experiments/FIG5-DISC-017/analysis_task/candidate_triage_report.json`
- Six child handoffs:
  `research_output/experiments/FIG5-DISC-017/analysis_task/databases/<database>/discovery_handoff.json`

## General engine changes

- Discovery handoff schema v3 supports both predictor/outcome and concept-set
  analysis shapes.
- Data acquisition can require an exact trajectory concept while operating
  without a target outcome.
- External evaluation JSONL accepts outcome-free longitudinal/clustering tasks.
- The launcher emits `longitudinal_trajectory_analysis`, preserves explicit
  analysis concepts, and no longer fills `agent_mined_idea` as a fake predictor.
- Existing Idea Mining ledgers now preserve `analysis_family` and
  `resolved_analysis_concepts` through the S6 boundary.

## Verification

- 138 Idea Mining/discovery/data-foundation/JSONL focused tests passed.
- 81 trajectory/discovery focused tests passed; two trajectory-mutation tests
  were excluded because they are already red at clean `aecdd7e` with the same
  missing `TRAJECTORY_PARQUET` fixture error (verified in a detached worktree).
- Ruff and py_compile passed.
- Architecture lower-is-better gate: zero regression.
- Research Agent module graph: zero regression / zero cyclic SCC.
- No API call, no scientific analysis, and no six-database re-extraction.

## Remaining scientific gate

Before any child becomes analysis-ready, freeze and review:

1. time zero;
2. observation window;
3. minimum measurement support;
4. time grid and aggregation;
5. trajectory representation;
6. class/feature selection method;
7. within-database stability threshold;
8. cross-database matching/transportability metric;
9. outcome-blind class discovery.

Only after this protocol and bounded prior-art review should the existing six
child handoffs be promoted through the human-confirmed analysis gate.
