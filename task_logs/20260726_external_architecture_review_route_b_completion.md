# External architecture review — Route B completion

Date: 2026-07-26 EDT
Task: `ARCH-REVIEW-ROUTE-B-20260726`
Branch: `fix/external-review-20260724-p0-p1`
Starting base: `25039f4`
Tested implementation HEAD: `c466dca26435061de981d8f32523f188d8f76829`

## Decision

The user selected Route B: retain an explicit, framework-neutral state machine
and do not restore the former partial LangGraph integration.

This removes a shadow dispatcher and an unused dependency, not the workflow
contract. The production path still has:

- a `WorkflowEngine` interface and explicit `PipelineWorkflow`;
- typed workflow state and human-review decision models;
- pause/review/resume behavior within the same live process;
- run, artifact, digest, evidence, readiness, and execution-identity records.

The deliberately unsupported capability is durable resume after a service
restart. The retired LangGraph path did not provide that capability either; it
used no durable checkpointer and was not the authoritative production
dispatcher. The limitation is now explicit as `resume_scope="same_process"`.

After the paper freeze, a real Route A may be designed around a durable store
whose workflow state contains only `run_id`, artifact paths, and digests. Its
acceptance gate must include a service-restart recovery test. The former
half-integrated LangGraph path must not be restored as a substitute.

## Completed remediation

### Data and public API

- Fail-closed behavior replaced fake-success alignment, silent medication
  partials, unsafe medication merges, and arbitrary legacy full-table fallback.
- The monolithic public API was split into domain-owned modules while preserving
  the 1.x public surface through a lazy facade.
- Import-time process mutation was removed; compatibility setup is explicit.
- Canonical JSON serialization and digest generation now have one owner.

### Research-agent orchestration

- `PipelineConfig` is immutable configuration; live collaborators are carried
  by `PipelineServices`.
- The shadow LangGraph dispatcher and production imports were removed.
- `PipelineWorkflow` is the single explicit state machine behind the
  framework-neutral `WorkflowEngine` interface.
- Human-review data contracts remain first-class and same-process resume is
  tested.
- LangGraph was removed from project dependencies, the runner lock, and CI
  installation. Release tests reject its accidental reintroduction.
- Runtime capability context is reset around every test, closing a real
  order-dependent `ContextVar` leak without weakening production job scoping.

### MCP and Web-facing transport

- The custom MCP wire protocol was replaced with the official MCP Python SDK.
- Stdio and stateless Streamable HTTP `/mcp` transports retain authentication,
  scope, body-size, host/origin, concurrency, and timeout enforcement.
- The application contract is injected into the transport, eliminating the
  `mcp_server` ↔ `mcp_transport` import cycle.
- No route-specific frontend CSS or JavaScript was changed in this remediation.

### Maintainability and regression ownership

- Ruff, Deptry, Import Linter, and module-graph gates now enforce dependency
  boundaries.
- Regression tests were renamed and organized by functional owner.
- Extraction coverage audit tooling is Ruff-clean.
- The compact `easyicu.easy` 1.x compatibility shim remains; physical removal
  is a 2.0 change.
- Pydantic Settings and Tenacity were evaluated but not adopted: the current
  explicit configuration preserves import purity, while structured retry
  carries provenance semantics that a generic decorator would obscure.

## Commit batches

All batches below were local and unpushed at the time of this record. The
tracked upstream was `25039f4`; tested code was 17 commits ahead.

| Commit | Batch |
|---|---|
| `2e495ea` | `fix(data): fail closed legacy API paths` |
| `16457f0` | `refactor(runtime): make compatibility setup explicit` |
| `665acbd` | `build(lint): enforce dependency boundaries` |
| `2376eac` | `refactor(api): split public data domains` |
| `3f907f0` | `fix(agent): bind demotions and recovered methods` |
| `45b0336` | `refactor(agent): separate pipeline config and services` |
| `dff077c` | `refactor(agent): retire shadow LangGraph dispatcher` |
| `8e329af` | `refactor(agent): finalize explicit workflow engine` |
| `8146395` | `refactor(mcp): adopt official SDK transports` |
| `e360f22` | `refactor(api): make package facade lazy` |
| `eec5bf5` | `test: organize regressions by functional owner` |
| `406e783` | `refactor(authority): centralize canonical digests` |
| `208b016` | `fix(tools): clean extraction coverage audit` |
| `f48cca1` | `fix(api): preserve lazy export across legacy import` |
| `d993414` | `build(agent): finish LangGraph dependency cleanup` |
| `3db89d3` | `refactor(mcp): inject application contract into transport` |
| `c466dca` | `test(agent): isolate runtime capability context` |

## Verification

### Unfiltered full-suite diagnostic

The first complete run on `3db89d3` reported:

```text
9083 passed, 61 skipped, 113 failed
```

Every failure was classified:

- 105 were the same frozen Figure 2 scorer-tree digest mismatch
  (`73102c…` expected, `82406f…` current);
- 1 was the frozen research-agent resource/context baseline drift;
- 1 was the frozen architecture baseline drift;
- 6 were order-dependent Coder prompt tests caused by a leaked runtime
  capability provider.

The six functional failures were reproduced with a two-test sequence and
fixed by `c466dca`. The frozen paper scorer/resource/architecture baselines were
not refreshed to make the suite appear green.

### Complete functional suite

The final functional run excluded only the frozen Figure 2 evaluator directory
and the two frozen snapshot assertions:

```text
8931 passed, 61 skipped, 2 deselected, 1306 warnings
time: 2259.66s (37m39s)
START_SHA=c466dca26435061de981d8f32523f188d8f76829
END_SHA=c466dca26435061de981d8f32523f188d8f76829
SHA_STABLE=1
```

The command was:

```bash
python -m pytest -qq \
  --ignore=tests/benchmarks/figure2_canonical9/evaluator \
  --deselect=tests/test_research_agent_resource_baseline.py::test_checked_in_resource_context_baseline_has_no_drift \
  --deselect=tests/test_arch_measure.py::test_checked_in_architecture_baseline_has_no_regression
```

### Static, boundary, graph, and package gates

```text
python -m ruff check src/easyicu tests tools
  All checks passed

python -m deptry src/easyicu --no-ansi
  Success; no dependency issues

lint-imports
  4 contracts kept, 0 broken

python -m pytest -q tests/test_research_agent_module_graph.py
  11 passed

python tools/research_agent_module_graph.py \
  --diff tools/arch_baselines/research_agent_module_graph.json
  exit 0

python -m pytest -q tests/test_packaging_runner_image.py --run-packaging
  2 passed
```

The packaging gate built and installed a real wheel, read the runner image via
`importlib.resources`, and passed `pip check`.

### Real-data read-only smoke

Source:

```text
/Volumes/外置硬盘/databases/mimiciv
```

Using the final tested implementation, EasyICU sampled one MIMIC-IV stay
without printing its identifier and loaded raw heart-rate observations:

```text
sampled_patients=1
rows=9
concept=hr
id_column=stay_id
data_files_written=0
```

The smoke used `get_all_patient_ids(..., max_patients=1)` followed by
`load_concepts("hr", interval=None, concept_workers=1,
parallel_workers=1)`. A recursive modification-time scan confirmed that the
source directory was not written.

## Remaining explicit boundaries

- Route B supports same-process resume only.
- Durable restart recovery is a post-paper Route A workstream and needs a real
  durable store plus restart/multi-worker acceptance tests.
- Frozen Figure 2 scorer authority and the resource/architecture snapshots
  still require an explicit paper-authority review; they were intentionally not
  re-sealed during architecture cleanup.
- No push was performed by this task.
