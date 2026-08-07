# 2026-08-07 external review closure from `58eaf72`

## Scope and baseline

This batch started from exact repository HEAD
`58eaf72a428c0f4093fef56a56209f45236bda4d` on
`fix/external-review-20260724-p0-p1`. It closes the two P1 scientific-contract
findings and the concrete P2 findings in the current review without running a
provider-backed Canonical9 experiment.

## Commits

| Commit | Owner and closure |
| --- | --- |
| `ada19d7` | KDIGO ascertainment: only a positive or a fully ascertained negative is definitive; partial and indeterminate phenotypes remain nullable. Technical load/schema/calculation failures now fail closed with stable component/reason codes. |
| `b9b9046` | Scientific adapters: enforce declared package version ranges, distinguish unavailable/incompatible/unresolved runtimes, separate unexpected competing-risk codes from unobserved declared risks, and bind the DoWhy identification object and assumptions. Also closes the explicit no-deps CI dependency list and Deptry distribution/import-name mappings. |
| `6cb7ed1` | Survival authority and contract strangler: a sealed host Cox owner consumes one digest-bound cohort, fits the exact Planner contract, runs the Schoenfeld diagnostic, and issues hash-bound result/diagnostic/analysis-frame evidence. Capability receipts now say `question_coordinates_resolved` and `claim_ceiling`; family/survival/model-token contracts moved out of `schema.py`, which remains a compatibility facade. |

## Scientific contract closure

### KDIGO

- Added the closed ascertainment states `positive`, `negative_complete`,
  `partial_no_observed_positive`, and `indeterminate`.
- Added component-level creatinine, urine-output, and RRT ascertainment plus an
  explicit observation-window coverage receipt.
- A component-negative row cannot become `AKI=False` without all three
  component negatives and an explicit complete observation-window receipt.
- Patient prevalence uses only positive plus complete-negative phenotypes;
  partial and indeterminate patients are reported separately.
- Source absence, patient-data absence, insufficient windows, missing weight,
  load errors, schema errors, and calculation errors no longer collapse into
  one ordinary-missingness path.
- `summarize_aki()` and `get_aki_incidence()` validate their public identity and
  schema contract before indexing.

### Host-owned survival execution

- The Planner now declares the exact input product, exposure encoding,
  covariates, missing-data policy, time unit/value horizon, and event value.
- Plan validation and runtime selection consume one shared ownership verdict;
  an unsupported survival contract cannot fall back to Coder execution.
- The generated script is fully sealed: its Coder-owned body is empty.
- The executor rechecks the input SHA before, during, and after setup; performs
  strict numeric and endpoint-domain checks; applies complete-case analysis and
  administrative censoring; fits Cox with `lifelines`; and runs the declared
  Schoenfeld PH diagnostic.
- The host receipt binds input evidence/SHA, canonical analysis-frame SHA,
  result evidence/SHA, PH evidence/SHA, rows/events, exact filter, formula,
  covariates, and installed package versions.
- The publication gate requires the host issuer/runner markers, reconciles the
  receipt with the Planner and endpoint, and independently hashes the
  materialized result and PH tables.
- The summary exports canonical HR interval/sample-size fields so the existing
  primary-effect and readiness pipeline consumes the host result without a
  second estimator path.

### Adapter and capability semantics

- `packaging.SpecifierSet` and `Version` now enforce every optional adapter's
  declared version range. Runtime receipt schema v2 records the expected range
  and installed version.
- scikit-survival reports an unexpected event code as a contract failure and an
  unobserved declared event type as `data_support_insufficient`.
- DoWhy receipt schema v2 records the normalized estimand hash/type,
  identification routes, declared-assumption hash, and assumption
  fingerprints; it still emits no effect estimate and does not promote causal
  capability.
- Pre-execution capability serialization no longer calls a coordinate check
  full grounding or a reportable ceiling current run status.
- `FamilyPrimaryResultRequirement`, `SurvivalAnalysisReceipt`, and the model
  token normalizer are now dependency-neutral owner contracts outside
  `schema.py`. This is an incremental strangler slice, not a claim that the
  remaining 3,295-line compatibility facade or 12,408-line `phase.py` has been
  fully decomposed.

## Verification on the resulting code

- Focused and adjacent review set: `201 passed, 1 skipped`.
- Earlier broad impacted set in the same batch: `333 passed, 2 skipped`.
- Portability workflow selection on macOS: `150 passed`; cross-layer safety
  selection: `4 passed, 92 deselected`.
- Full `ruff check src tests`: passed.
- Import-linter: 7 contracts kept, 0 broken.
- Deptry: no dependency issues after adding exact distribution/import-name
  mappings; no dependency rule was ignored.
- `git diff --check`: passed.
- Research-agent module graph baseline: passed.
- Wheel and sdist both built from the resulting tree, installed into clean
  Python 3.13 environments, passed `uv pip check`, imported the new contracts
  and executor, and contained the runner-image resources.
- Docker image `easyicu-research-agent:review-58eaf72` rebuilt from the
  resulting tree as image
  `sha256:9e413a48e41a4732cc2071ecbdec639b74ed7d80a0eb66dba37413b167cf3cc1`.
  Its runtime capability receipt reports network `none` and `lifelines`
  available.
- A real disconnected-container smoke fit completed Cox plus Schoenfeld PH on
  160 deterministic synthetic rows (30 events) and emitted the host receipt
  with `lifelines==0.30.3`.

## Release-gate boundary still open

The review findings above are closed in code, but the branch is not represented
as a fully green public current-SHA release:

1. The checked-in architecture ratchet was already stale at the starting
   `58eaf72`: it reports 21 lower-is-better regressions across historical large
   modules. The same gate remains red. This batch did not re-record that stale
   baseline to make the failure disappear.
2. The research-agent resource/context baseline was also already stale at the
   starting HEAD. The survival Planner contract intentionally changes H1 prompt
   content, so any future baseline move needs a separate, reviewed resource
   budget adjudication rather than an automatic refresh here.
3. A public Python 3.10/3.11 matrix and GitHub packaging/portability run cannot
   exist for `6cb7ed1` until the branch is pushed. No push was authorized in
   this task.
4. A full local research-agent suite was not used as a false green claim: the
   local environment lacks optional MCP collection dependencies, and the long
   non-MCP attempt exposed pre-existing external-corpus/naive-fixture/cache
   failures before it was stopped. Focused, adjacent, packaging, dependency,
   portability, and real Docker paths are the evidence for this patch.

No Canonical9 score changed. M3 remains unrun, and historical verify ledgers
must not be resumed as evidence for this code.
