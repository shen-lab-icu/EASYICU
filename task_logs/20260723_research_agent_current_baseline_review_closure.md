# Research-agent current-baseline review closure

Date: 2026-07-23 EDT  
Branch: `refactor/agent-control-plane`  
Review baseline: `1f932bd`  
Closure HEAD: `5a20b25`  
External review source: `task_logs/20260723_research_agent_full_code_review.md` at older baseline `518a231`

## Scope and authority boundary

This increment independently rechecked the highest-severity findings from the
Claude review against the current branch before changing code. It did not assume
that findings observed at `518a231` remained valid at `1f932bd`.

No Provider, Docker image, patient data, extraction, Canonical9 run, paper
authority, or external service was used. The work is limited to deterministic
source changes and offline regressions.

## Commits

| Commit | Closure |
|---|---|
| `89935d0` | Fail-close SOFT untraced manuscript numbers; preserve exposure-number identity; reject garbage concept selection instead of laundering it into a usable acquisition plan. |
| `cb1c220` | Replace survival-method and numeric-report lexical bypasses with structural checks; bind numeric headings; make Cox parsing scale-aware; restrict grouped Table 1 ownership to the exact analysis-cohort input. |
| `5458f18` | Replace overadjustment raw-substring matching with concept-token/alias/statistic identity matching; migrate one stale test import. |
| `f5888b5` | Schedule know-how cards inside one total prompt budget, retain mandatory stop/confirmation content, and expose the number of whole cards withheld. |
| `5afa1fb` | Compress the exposure-identity implementation without changing behavior so `plan_utils.py` returns to its frozen LOC baseline. |
| `11d9959` | Bind method compatibility to executable AST calls and variable flow; apply binary/count outcome-family rules only to declared outcomes. |
| `5a20b25` | Stage cohort adoption transactionally, restrict deterministic host receipts to explicit consumed inputs, align Planner/runtime raw-input rosters, preserve small trajectory signatures, expose the wrapped model identity to metering, and migrate stale custom mocks. |

## Confirmed current-baseline defects closed

1. **Untraced SOFT-mode manuscript numbers** now create a blocking numeric-audit
   finding and cannot leave `numeric_verified`/readiness green.
2. **Survival integrity activation** is based on result structure rather than a
   four-string method allowlist.
3. **Ordinary English near a metric** (for example “within the cohort”) no longer
   disables AUROC/Brier numeric comparison.
4. **Numeric Markdown headings** enter the same binding/evidence path as body
   claims; outline ordinals and Markdown syntax remain exempt.
5. **Grouped Table 1 standard-executor ownership** accepts no typed input or the
   exact `artifact:analysis_cohort`; subset/foreign/cohort-prefixed inputs remain
   agent-owned.
6. **Cox forest parsing** distinguishes coefficient/log-HR from hazard-ratio scale
   before transforming estimates and confidence intervals.
7. **Know-how retrieval with three or more cards** no longer fails only because
   individually valid cards exceed the aggregate prompt budget.
8. **Overadjustment detection** no longer rejects unrelated words such as
   `pancreatitis`, `increase`, or `mapped` because they contain short fragments
   like `crea` or `map`.
9. **Exposure identity** now preserves numeric distinctions rather than stripping
   every digit before fuzzy matching.
10. **Acquisition concept selection** fails explicitly when the model returns only
    garbage instead of silently producing an apparently materializable request.

These are ten defects, not the whole 26-finding report.

## Second high-priority closure

The current-baseline H5–H8 and H11–H13 reproduction is now closed:

- **H5/H6:** forbidden methods are matched only on executable AST calls and
  variables reaching those calls. Comments, docstrings, imports, unrelated
  variables, and binary covariates no longer create outcome-family findings.
  Count inference now requires an integer outcome with `SUM`. A newly exposed
  `map(...) → UMAP` substring false positive was also fixed.
- **H7:** each deterministic standard executor exposes a closed
  `consumed_input_keys` contract. Host receipts reject unknown keys and stamp
  only those explicit typed bindings. This is a standard-executor consumption
  contract, not general syscall tracing of arbitrary user code.
- **H8:** a prose-derived cohort is staged on a deep plan copy. The live plan is
  mutated only after materialization reports `applied` and authority rebind
  succeeds; failure leaves the executing plan unchanged and emits a typed error.
- **H11:** outbound trajectory removal is now limited to coordinates actually
  represented by the compact shared projection; small, non-compacted scientific
  signatures stay visible.
- **H12:** Planner structured retry uses the executable column-binding roster,
  so identity/time coordinates are rejected as raw step inputs before execution.
- **H13:** the reproducibility wrapper exposes the same resolved model identity
  recorded in its envelope, allowing the metering layer to attribute model and
  cost correctly.

Two know-how fixtures and three parameterized prose-cohort fixtures were
migrated from rejected custom subclasses to exact built-in composition mocks.
This closes the known mock authorization debt without weakening the production
provider boundary.

The remaining medium/low findings from the full review have not been declared
closed. M4 fraction/percentage envelope migration may now resume.

## Regression evidence

Focused and adjacency runs:

- first focused closure: `24 passed`
- related matrix: `79 passed`
- expanded focused matrix: `123 passed`
- survival integrity plus Cox scale: `19 passed`
- overadjustment suite: `29 passed`
- final bounded cross-area matrix: `231 passed, 271 deselected`

First-phase red categories were not hidden:

- Evidence/reporting adjacency produced `95 passed, 4 failed`; all four failures
  were the pre-existing Docker source-SHA mismatch
  (`expected fb19…`, `observed 0fae…`). Docker was not rebuilt in this task.
- Full know-how tests produced `16 passed, 2 failed`; both failures are the known
  legacy custom offline-mock authorization fixture migration debt:
  `test_replanner_cannot_remove_claim_decisions` and
  `test_opt_in_pipeline_smoke_adopts_card_without_extra_provider_calls`.
  They were excluded only from that first bounded matrix and were subsequently
  migrated and passed in the second closure.

Second closure matrix:

- 182 tests collected across method compatibility, trajectory contracts,
  cohort adoption, Table 1/missingness standard executors, prompt compaction,
  Planner parsers, cost attribution, and know-how.
- **180 unique functional tests passed.**
- The remaining 2 failures are Docker-bound cost-pipeline tests; both stop before
  pipeline execution because the installed image reports source SHA
  `0fae…` while the working tree expects `12f948…`. Docker was not rebuilt.
- The two Planner tests initially stale under the H12 roster change were updated
  to stop declaring `stay_id` as an executable input and then passed directly.
- Focused H5/H7/H8/H11/H12/H13 negative controls: **7 passed**.

Representative final command:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/research_agent/test_exposure_contract_audit.py \
  tests/research_agent/test_data_foundation.py \
  tests/research_agent/test_overadjustment_audit.py \
  tests/research_agent/test_numeric_provenance.py \
  tests/research_agent/test_numeric_binding_disambiguation.py \
  tests/research_agent/test_writer_hallucination_blocked_in_strict.py \
  tests/research_agent/test_pipeline_report_survival_integrity.py \
  tests/research_agent/test_figure_scale_and_ci_resolution.py \
  tests/research_agent/test_table_one_executor.py \
  tests/research_agent/test_research_know_how.py \
  tests/research_agent/test_evidence.py \
  tests/research_agent/test_publication_figures.py \
  tests/research_agent/test_pipeline.py \
  --deselect tests/research_agent/test_research_know_how.py::test_replanner_cannot_remove_claim_decisions \
  --deselect tests/research_agent/test_research_know_how.py::test_opt_in_pipeline_smoke_adopts_card_without_extra_provider_calls \
  -k 'not test_pipeline or manuscript_numeric_auditor or readiness_artifacts'
```

Static verification completed during the increment:

- Ruff: pass on changed source/tests
- `py_compile`: pass on changed production modules
- `git diff --check`: pass
- research-agent module graph: no new cycle
- architecture measurement: the same 12 pre-existing lower-is-better drifts remain;
  this task did not rewrite the baseline. `plan_utils.py` returned to its exact
  frozen LOC value, reducing a transient count of 13 back to 12.

At `5a20b25`, Ruff, Black, `py_compile`, `git diff --check`, and the module graph
pass; no new module cycle was introduced. The architecture gate still reports
the same 12 branch-level lower-is-better drifts, and the resource baseline
reports source-SHA drift after the intentional outbound projection change. No
baseline file was rewritten to manufacture a green result.

## Honest size accounting

`1f932bd..5a20b25` changes 33 files:

- production: `+683/-158`, net **+525**
- tests: `+907/-140`, net **+767**
- total: `+1590/-298`, net **+1292**

No new point-repair module was added and `pipeline.py` was not changed.
`execution/phase.py` gained the staged-commit wiring and exact receipt
arguments. This increment improves correctness but is not a production-code
reduction milestone; repair proliferation remains open.

## Next action

Resume M4 with the fraction/percentage canonical view in shadow/double-read
mode. Require E1/E2 archived replay to preserve finding, status, and artifact
SHA behavior before switching the Validator consumer. Continue consumer
migration in the fixed order Validator → Writer → readiness → scorer/Jury →
figure/source-data, deleting superseded raw parsers and point repairs per slice.
Canonical9 development or final authority must not be restarted from this
review increment.
