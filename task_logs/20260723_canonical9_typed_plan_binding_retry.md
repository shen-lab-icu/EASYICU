# Canonical9 typed plan binding retry

Date: 2026-07-23  
Task: `FIG2-CANONICAL9-REALRUN`  
Scope: case-neutral Planner/typed-cohort boundary repair

## Real-run finding

The fresh Luna E1 run under source image `3be1782` reached the LangGraph plan
node, then failed before step execution:

- batch:
  `/Volumes/外置硬盘/easyicu_data/canonical9_runs/batch_20260723_luna_miiv_3be1782`
- run:
  `e1_sepsis3_prevalence_mortality/aware/run_20260723T081753_88c914`
- exception:
  `MaterializedMetadataError: typed cohort definition could not be applied to its sealed universe`
- nested cause:
  the Planner used dictionary concept `sep3` with `aggregation=any`, while its
  own analysis-cohort step selected the sealed column `sep3_sofa2_max`.

The run was stopped before E2 proceeded. It is diagnostic only and has no
paper authority.

## Root cause

Two independent checks were missing at the same boundary:

1. The Planner cohort allowlist loaded only `concept-dict.json`; it omitted the
   packaged `sofa2-dict.json` overlay, so the prompt excluded valid concepts
   such as `sep3_sofa2`, `sofa2`, and the SOFA-2 components.
2. `PlannerAgent._parse` verified schema and global dictionary membership but
   did not verify that every primary/robustness cohort predicate could resolve
   against the current run's sealed typed column roster, exact materialized
   window, and aggregation before accepting the structured response.

This was not a LangGraph or memory failure. LangGraph propagated a plan-node
exception correctly; canonical cross-run memory remained disabled.

## Repair

- Merge both packaged dictionaries into the CTAS concept allowlist.
- Bind `sofa2-dict.json` into the deterministic Planner resource baseline.
- During Planner and Replanner structured parsing, validate every non-empty
  primary and robustness cohort definition against:
  - the immutable typed cohort column roster;
  - the analysis-cohort producer's declared inputs;
  - the ResearchContext source-concept, window, and aggregation descriptors.
- Raise a bounded, actionable `CohortSchemaError` when a legal global concept
  is not executable for this run. The existing structured retry then returns
  the exact typed source concepts and columns to the same Planner conversation.
- Preserve the existing fail-closed materializer; no fuzzy matching, silent
  aliasing, invented window, deterministic scientific choice, or E1 literal
  was added.

The archived failing plan now fails at the new parser boundary with:

```text
declared typed source concepts=['death', 'sep3_sofa2']
declared columns=['death', 'sep3_sofa2_max', 'stay_id']
```

An offline corrected projection using `sep3_sofa2` and the sealed 0--24-hour
coordinate passes the same validator.

## Verification

- focused parser + cohort schema: `56 passed`
- typed materialization/planning/Table 1 adjacency: `152 passed`
- broad provider/LLM/idea-mining/coder matrix:
  `347 passed, 1 skipped, 1 deselected, 1 existing static-entrypoint failure`
- architecture lower-is-better gate: passed; `agents/core.py` remains 3829 LOC
- module graph: passed; no new cycle
- resource/context baseline: deliberately regenerated and passed; maximum
  Planner-with-resources request is 37,801 bytes, below the 80,000-byte gate
- Ruff, Black, `py_compile`, and `git diff --check`: passed

Observed non-causal baseline issues were not mixed into this patch:

- three legacy scripts still construct `OpenAIClient` outside the factory;
- several old custom test clients are rejected by the current default-deny
  Provider registry;
- current Docker-image source mismatch blocks dirty-tree pipeline tests until
  the new source image is built;
- previously migrated private helper imports and plan-scope compatibility
  assertions remain separate work.

## Next action

Commit this isolated repair, build an immutable image from the clean commit,
generate a new execution identity/operator freeze, and launch a fresh
aware-only MIMIC-IV Canonical9 batch. Do not resume or reuse the failed
`3be1782` batch.
