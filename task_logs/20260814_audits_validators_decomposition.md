# 2026-08-14 — audits/validators.py decomposition batch (P3 structure debt, batch 2)

## Scope

One owner, one batch: the 13,815-line `research_agent/audits/validators.py`
(17 validator classes in one module) was split into sibling owner modules.
Concept-audit code (ConceptUsageAuditor, LLMConceptAuditor, and all
concept/AST helpers) **stays in validators.py** because
`test_validators.py:2762` monkeypatches `validators.authorized_complete` and
expects LLMConceptAuditor to resolve it via module-global lookup. Bodies
moved byte-for-byte (AST-computed cross-imports, `ruff --fix F401` pruning).

## New layout

| module | LOC | owns |
| --- | ---: | --- |
| `audits/validators.py` | 2,797 | concept-audit owner + re-export facade |
| `audits/figures.py` | 5,508 | FigureSourceDataValidator + FigureContractQualityValidator (mutually referencing; one module, no cycle) |
| `audits/cross_step.py` | 4,316 | cross-step lock/registered-output/fraction/contract/source-status + PrimaryModelContractValidator |
| `audits/statistical.py` | 682 | StatisticalValidator + StatisticalGuard |
| `audits/publication.py` | 210 | Replication/PublicationClaim auditors |
| `audits/clinical.py` | 164 | ClinicalConstraintValidator |
| `audits/cohort.py` | 132 | CohortAuditor |
| `audits/_v_support.py` | 209 | cohort_hygiene_findings + dedupe_findings |

Dependency direction: owner modules → `_v_support`/`schema`/contracts;
`validators.py` facade → all owners. No owner imports the facade back.

## Design notes

- `figures.py` keeps the two figure validators together: the splitter's AST
  pass found they reference each other's classes, so splitting them would
  create the package's first import cycle. Zero-SCC preserved
  (`cyclic_scc_count` 0, module count 506 → 513).
- The lazy-import cycle comments in `replication/paper.py`
  (`replication.paper ↔ audits.validators`) still hold: it now resolves
  against the slimmer validators.py; behavior unchanged.
- Two dead locals masked by the old validators.py F841 pin were deleted
  (`shared_columns` in figures.py).

## Verification

- `test_validators.py` + adjacent concept/replication/plausibility suites:
  316 passed; 3–4 `llm_concept_auditor_downgrade*` failures are pre-existing
  on the un-split baseline (git-stash control), owned by the concurrent
  E1-bias lane.
- validators-consumer batch (figures/contract/stratified/executors): 111
  passed; 1 failure (`ordered_stratified numeric_replay wiring`) verified
  pre-existing via stash control.
- End-to-end pipeline smoke (`test_pipeline.py -k mock`, 6 tests incl. full
  run with validators wired): passed.
- ruff clean on `audits/`; module graph zero SCC; arch ratchet: new owners
  appended to TARGET_FILES, baseline re-emitted with reason (only accepted
  growth: agents/replanner.py +14 loc from the prior reviewed fix).

## Follow-ups (not this batch)

- `figures.py` (5.5K) and `cross_step.py` (4.3K) are now the largest audit
  owners; a future batch may split them along class seams if characterization
  coverage exists — each needs its own owner contract.
- The pre-existing downgrade-test failures belong to the concept-audit lane.
