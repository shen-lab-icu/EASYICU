# GPT architecture review remediation

Date: 2026-07-26 EDT  
Task: `DATA-ARCH-REMEDIATION-20260726`  
Branch/HEAD at start: `fix/external-review-20260724-p0-p1@25039f4`

## Scope and concurrency boundary

The user supplied an external GPT review covering the data, Web, and
research-agent layers and asked for fixes and tests. The concrete correctness
findings were verified against the current tree before editing. A concurrent
Claude process was modifying and testing:

- `src/easyicu/research_agent/plan_utils.py`
- `src/easyicu/research_agent/reporting/readiness.py`
- `tests/research_agent/test_completion_axes.py`
- `tests/research_agent/test_step_completion_budgets.py`

This task did not edit those files or any frozen Figure 2 authority/baseline.

## Findings adjudicated and fixed

1. `align_to_icu_admission()` was a fake-success stub: it printed and returned
   unmodified data. It now fails explicitly with `NotImplementedError` and
   directs callers to canonical `load_concepts()` relative-time output.
2. `load_medications()` swallowed every per-concept exception and returned an
   unlabeled partial result. It now fails closed by default with a public
   `MedicationLoadError`. `allow_partial=True` is an explicit compatibility
   choice that emits `RuntimeWarning` and attaches a structured
   `easyicu_medication_load_report`.
3. Medication frame merging used substring-selected IDs/times and unconstrained
   outer merges. It now uses stable known key candidates and requires
   one-to-one keys, raising public `MedicationMergeError` before row
   multiplication.
4. deprecated `ConceptLoader._safe_load_table()` retried a full-table read after
   any projection exception. It now retries only an explicit missing-column
   projection error, and never retries in low-memory mode.
5. `import easyicu` installed process-wide pandas warning filters, printed
   import failures, and retained four permanently false feature branches.
   Filters and dead branches were removed; import failures are recorded without
   stdout output.

The review's LangGraph claim was outdated: the current pipeline is driven by
the compiled LangGraph and records a runtime receipt. Large pipeline/MCP/static
debt work was therefore recorded rather than mixed into this correctness
patch. Policy and sequencing are in `docs/deprecation_policy.md`.

## Real-data smoke

Read-only source: `/Volumes/外置硬盘/databases/mimiciv`.

A stay identifier was read from a prepared `inputevents` parquet batch without
printing the identifier. A two-concept cardiac medication request with
`allow_partial=True` returned:

```text
rows=0
loaded=[]
failed=[amiodarone, milrinone]
partial_warning=True
```

This is the intended contract: structural/data absence is visible and cannot
be mistaken for a complete successful bundle. No external-drive file was
modified.

## Tests

Focused and adjacent suites:

```text
58 passed
76 passed
99 passed
69 passed
Ruff: passed
git diff --check: passed
```

Core suite excluding research-agent tests and frozen Figure 2 benchmark tests:

```text
1584 passed, 54 skipped, 2 failed
```

The two failures were the known concurrent Agent snapshot drifts:

- `test_checked_in_architecture_baseline_has_no_regression`
- `test_checked_in_resource_context_baseline_has_no_drift`

Both failure reports named dirty research-agent files, not files changed by
this task. Re-running the same core suite with exactly those two snapshot
assertions deselected produced:

```text
1584 passed, 54 skipped, 2 deselected
```

An attempted broader run that included `tests/benchmarks/figure2_canonical9`
was stopped after the frozen scorer-tree digest correctly rejected the
concurrent dirty Agent tree. Frozen paper authority was not updated to make
that run green.

## Deferred work

- Do not split `pipeline.py` before Canonical9 is frozen.
- Keep current LangGraph authority; audit bypass entry points later.
- Evaluate official MCP SDK migration only with schema, audit, cancellation,
  timeout, and authority compatibility tests.
- Run Vulture, Deptry, and Import Linter in report-only mode first; these tools
  are not installed in the current environment.
- Remove `easyicu.easy`, old `ConceptLoader`, and the explicit alignment stub
  only in 2.0 after downstream usage scans and release notes.
