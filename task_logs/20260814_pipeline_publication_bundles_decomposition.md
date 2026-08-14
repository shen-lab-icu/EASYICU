# 2026-08-14 — pipeline.py decomposition batch (P3 structure debt, batch 3)

## Scope

One owner, one batch: the publication-bundle renderer family in
`research_agent/pipeline.py` (functions spanning L6225–10943) moved to
`reporting/publication_bundles.py`. `ResearchAgentPipeline` (4,776 lines) and
every other function are untouched; bodies are byte-preserved line slices.

## Split rule (why 37 of 50 family members moved)

Tests monkeypatch names on the **pipeline module** and expect module-global
lookup (`pipeline`, `pipeline_module`, `pipeline_mod` aliases). A fixpoint
ejection ran before the move: any family member whose body references a
test-patched name (`_render_cohort_flow/overlap_...`,
`_resolve_upstream_analysis_family/method`,
`_migrate_legacy_resume_figure_render_edges`, ...) or any name defined
outside the family **stays in pipeline.py**, transitively. Result:

- moved (37, pure renderers + their private helpers): descriptive /
  prediction / cohort-overlap / cohort-flow / phenotype / absolute-risk
  renderers, table iterators, label helpers, `_AMBIGUOUS/_INCOMPATIBLE`
  vocabulary, `_iter_prior_output_tables` etc.
- stayed (13): the two test-patched renderers, the dispatch chain
  (`_renderer_for_upstream_*`,
  `_render_publication_bundle_from_prior_outputs_for_step`,
  `_render_authorized_sealed_publication_bundle`,
  `deterministic_figure_repair_id_for_upstream`, digest-seal helpers,
  `_resolve_upstream_analysis_method`, `_resolve_upstream_figure_data_family`).

pipeline.py: 11,273 → 8,543 loc. New owner 3,169 loc (relative imports
deepened one level; header + lazy imports rewritten mechanically, ruff-pruned).

## Verification

- Consumer suites (publication figures, figure rescues, executors,
  pilot fixes, clustering routing, resume-edge migration, host-services
  boundary): 242 passed across two runs.
- `test_resume.py`: 157 passed, 2 failed
  (`test_quarantine_policy_supersession_reclassifies_the_stored_error`,
  `test_resume_retires_unchanged_draft_after_deterministic_policy_supersession`)
  — verified pre-existing via path-scoped stash control (fail identically on
  the un-split pipeline.py).
- Pipeline end-to-end mock smoke (6 tests): passed.
- Module graph: `cyclic_scc_count` 0; `publication_bundles` has no edge back
  to pipeline. ruff clean.
- Arch ratchet: new owner appended to TARGET_FILES; baseline re-emitted with
  reason (no growth).

## Incident note (process, not code)

A failed `git stash push` followed by `git stash pop` accidentally popped an
unrelated foreign stash (`stash@{0}` from another branch), leaving conflict
state on `src/easyicu/api.py` / `src/easyicu/base.py`. Both were reset to
HEAD and the stray untracked file removed; all 7 pre-existing stash entries
remain intact. Lesson recorded: use path-scoped stash operations only.

## Follow-ups (not this batch)

- `ResearchAgentPipeline` (4,776-line class) is the remaining pipeline debt;
  it needs method-level extraction with characterization tests, not a
  mechanical split.
- The 2 pre-existing resume failures belong to the resume/quarantine lane.
