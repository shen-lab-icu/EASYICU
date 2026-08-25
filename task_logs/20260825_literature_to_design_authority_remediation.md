# Literature-to-design authority remediation

Date: 2026-08-25

Branch: `codex/literature-to-design-contract`

Parent freeze: `3c42bd63aefc4df2e14ce8ecd0093e9e6bf12e53`

## Problem

Dev9 evaluation reviewed high-quality comparator articles after execution, but
the Planner received mostly citation keys, generic method references, and a few
short excerpts. Full-text and supplement review were not compiled into a typed
pre-result design contract. Therefore an executable run could pass while its
population, timing, operational definitions, missingness/censoring, sensitivity
analyses, display suite, or conclusion boundary remained weaker than the
published comparison set.

## Repair

- Added `LiteratureDesignEvidenceCard`, a bounded full-text/supplement receipt.
  It stores source-backed paraphrases and digests, never the article body.
- Added seven required design dimensions: population; time zero/windows;
  variable operationalization; missingness/censoring; primary model and
  sensitivities; table/figure completeness; conclusion boundaries.
- Added candidate-level `adopt` / `adapt` / `diverge` / `not_applicable`
  decisions. The selected design must resolve all seven dimensions and bind an
  included direct comparator or design analogue.
- Strict preplan validation stops before Provider use when there is no eligible
  comparison source, no reviewed full text, incomplete supplement review, no
  recent comparison source, or incomplete seven-dimension coverage.
- Added the additive development profile
  `npj_dm_qualification12_design_dev/20260825`. Historical Dev9 profiles and
  serialized contracts remain unchanged.
- Added a generic zero-Provider saved-run auditor:
  `tools/audit_literature_design_authority.py`.

## Verification

- Focused owner/adjacent suite: `326 passed, 1 deselected`.
- Real pipeline pre-Provider test: `1 passed`; the forbidden Planner callback
  was never called.
- Blueprint-abort regression plus strict gate: `2 passed`.
- Ruff and `git diff --check`: pass.
- Full exact-head CI was intentionally not run during this scoped development
  iteration; project policy reserves it for the frozen formal checkpoint.

After the first push, Research-Agent CI run `32838056577` correctly rejected a
57-line `pipeline.py` growth. The follow-up owner refactor moved pre/post-plan
gate mechanics into the scientific-plan-review orchestration boundary:
`pipeline.py` is now 8,418 LOC versus the 8,419 baseline, the intentional new
planning owner is recorded in the module-graph baseline, cyclic SCC count
remains zero, and the architecture suite passes `50/50`.

## Dev9 zero-Provider shadow audit

Input root:
`/Volumes/外置硬盘/easyicu_data/figure2_dev9_84f31fd_exact_20260824`

Receipt:
`task_logs/20260825_dev9_literature_design_shadow_audit.json`

- Provider calls: `0`
- Runs inspected: `9`
- Pass under the new contract: `0/9`
- E1/E2 candidate counts: `3/3`; E3-M3/H1-H3: `0`
- Selected seven-dimension literature decisions: `0` for all nine
- Full-text design cards available to the archived Planner: `0` for all nine
- Four runs fail because no comparison source was established; five fail
  because an established comparison source lacked a reviewed design card.

This audit does not invalidate Dev9's frozen `76/76` engineering execution and
does not rerun or rewrite any result. It establishes that Dev9 would be stopped
before Planner under the new Qualification12 contract, exactly where the
previous workflow allowed literature-design omissions to surface only during
evaluation.

## Claim boundary

This change makes literature review executable as planning authority. It does
not claim the archived nine analyses are publication-ready, does not copy
published effects as expected answers, and does not authorize Held-out27.
