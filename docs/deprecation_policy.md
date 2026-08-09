# EasyICU deprecation and architecture remediation register

Updated: 2026-07-26
Scope: follow-up to the user-supplied external GPT architecture review

This register separates correctness defects that must be fixed now from
structural work that must not destabilize the Canonical9 evidence baseline.
It is a migration contract, not a claim that every large file should be split
immediately.

## Closed in the current remediation

| Surface | Previous behavior | Current contract |
|---|---|---|
| `api.align_to_icu_admission()` | Printed a warning and returned the input unchanged, so `align_time=True` looked successful without aligning time. | Always raises `NotImplementedError`. Canonical relative-time output comes from `load_concepts()`; a fake success is no longer possible. Target removal: 2.0. |
| `api.load_medications()` | Caught every per-concept exception and silently returned whatever happened to load. Its heuristic merges could perform a many-to-many row multiplication. | Default is fail-closed with `MedicationLoadError`. Explicit `allow_partial=True` warns and attaches `DataFrame.attrs["easyicu_medication_load_report"]`. Merges require a shared stay/patient key and unique merge keys. |
| `load_concepts.ConceptLoader._safe_load_table()` | Any projection failure retried by reading the full table, including I/O, corruption, permission and memory errors. | Only an explicit missing-column projection error may use the compatibility retry. Low-memory mode never retries with a full-table read. |
| package import hygiene | `import easyicu` installed process-wide pandas warning filters, printed optional-import errors, eagerly loaded dozens of submodules, and retained four permanently false feature branches. | The root is now a small PEP 562 lazy facade backed by one declarative owner map. Plain import performs no runtime/cache/data/plot/agent initialization; 1.x names still resolve on access, and failures are recorded in `_IMPORT_ERRORS` without writing to stdout. |

## Deprecated compatibility inventory

| Compatibility surface | Replacement | 1.x policy | Removal target |
|---|---|---|---|
| `easyicu.easy` | Top-level `easyicu.load_*` / `easyicu.api` | Keep behavior and deprecation warning; do not add features. | 2.0 |
| `easyicu.load_concepts.ConceptLoader` | `easyicu.api.load_concepts()` and `ConceptResolver` | Keep only compatibility/safety fixes. It is not used by the canonical extraction path. | 2.0, after a usage scan in downstream repos |
| `easyicu.io.data_utils` and `easyicu.table.utils` ID conversion helpers | `easyicu.table.id_conversion` | Keep divergent historical behavior with call-time warnings; never silently redirect because the legacy direction is different. | 2.0 |
| `easyicu.io.id_mapping` non-working entry points | Canonical `load_concepts()` path | Continue failing explicitly instead of simulating success. | 2.0 |
| `align_to_icu_admission()` | Canonical relative-time output from `load_concepts()` | Fail explicitly in 1.x; do not restore the pass-through stub. | 2.0 |
| `ResearchAgentPipeline.run_with_graph()` / `research_agent.graph.build_pipeline_graph()` | `run()` / `orchestration.workflow.build_pipeline_workflow()` | The method warns and delegates; the retired builder refuses construction so it cannot recreate a shadow dispatcher. Human-review model imports remain compatible. | 2.0 |

Before a 2.0 removal, run a repository and downstream import scan, publish the
replacement table in release notes, and retain tests that prove the canonical
extraction path does not import the compatibility layer.

## Deferred structural work

### Research-agent orchestration

`research_agent/pipeline.py` remains a large public compatibility and
orchestration surface. Its size is real debt, but line-count-only extraction
would risk splitting scientific authority across two stores. After Canonical9
is frozen:

1. keep `PipelineConfig` as the only public configuration authority;
2. move behavior behind responsibility interfaces already represented by
   `authority/`, `gates/`, `execution/`, `repairs/`, and `reporting/`;
3. require architecture tests for import direction, module identity, authority
   ownership, and zero new cycles;
4. migrate one responsibility per change, with the full research-agent suite
   and evidence artifact comparison at every seam.

### LangGraph

Architecture decision: use one explicit EasyICU state machine and remove the
LangGraph dependency. The former graph checkpoint could not serialize the
plan-phase `EvidenceStore`, provider resolver, or run-scoped services, so it
stored those objects in a process-local dictionary. That gave EasyICU two
dispatchers without durable cross-process resume.

`orchestration.workflow.PipelineWorkflow` now owns
`plan → human_review → execute → write → finalise` directly. Runtime receipts
identify `explicit_state_machine`; human-review pauses still declare
`resume_scope="same_process"` until a complete artifact-rehydration contract
exists. EasyICU receipts, capsules, evidence and checkpoints remain the only
scientific/replay authority.

Callers depend on the structural `WorkflowEngine` interface rather than on a
framework-specific graph object. The post-paper durable-orchestration route is
deliberately not a restoration of the retired graph. Its acceptance contract is:

1. persisted state contains only `run_id`, artifact paths, immutable digests,
   phase/status values and schema-versioned review records;
2. `EvidenceStore`, provider resolution and run-scoped services are rebuilt by
   repositories/factories and their reconstructed authority is digest-checked;
3. a service-restart integration test pauses before review, destroys the first
   process, starts a second process and resumes to the same evidence outcome;
4. the runtime may advertise a durable resume scope only after that restart
   test passes.

### MCP

Architecture decision: the owner explicitly authorized the post-review
migration on 2026-07-26. `mcp_transport.py` now uses the stable official MCP
Python SDK line (`mcp>=1.28,<2`) for protocol negotiation, JSON-RPC validation,
stdio framing and stateless JSON Streamable HTTP. The upper bound prevents the
pending SDK v2 release from silently changing production behavior.

`mcp_server.py` continues to own the field-compatible tool schemas and
plain-Python `dispatch` seam. EasyICU continues to own scopes, provider opt-in,
path confinement, patient-data audit and EvidenceStore promotion. The removed
stdlib JSON-RPC/SSE bridge is not a compatibility authority.

Migration regressions use the official `ClientSession` and cover initialization,
tool listing/calls, schema rejection before dispatch, real subprocess stdio,
concurrency ceilings, cancellation, configured wait timeout, malformed and
oversized HTTP messages, Host/Origin protection, bearer authentication,
separate patient-data credentials and anonymous-loopback scope reduction.

### Static debt reporting

Ruff, Vulture, Deptry and Import Linter are declared development tools.
Dependency and import-direction checks run in CI, with responsibility contracts
covering the canonical data API and research-agent layers. Vulture remains
report-only because dynamic registrations and compatibility exports require
human classification.

The broad historical `F401`/`F841` exemptions remain visible debt. Remove them
one owned package at a time, beginning with new or already-clean modules, and
archive the classified Vulture report before deleting any dynamically
registered symbol.

### Shared primitives and deliberate trust boundaries

Exact canonical JSON and SHA-256 behavior is centralized in
`research_agent/canonical_json.py`. Authority modules supply any
schema-specific normalization first; protocol owners must request a trailing
newline explicitly. Golden byte/digest tests prevent a cleanup from silently
changing evidence identities.

Path and cache helpers are intentionally not collapsed merely because their
names look similar:

- `authority.filesystem.AnchoredDirectory` is a symlink-resistant,
  descriptor-anchored write boundary; MCP path resolution additionally owns
  process scopes and operator-configured disclosure roots.
- `api.cache` fingerprints source datasets for extraction reuse;
  `authority.pipeline_cache` validates content-addressed research-run
  authority. Sharing storage or invalidation rules would mix two trust
  domains.

Pydantic Settings and Tenacity remain evaluated, not adopted dependencies.
Environment reads stay at explicit runtime/CLI adapters so importing the
library does not create hidden configuration state. Model retries continue to
use the structured retry path because every attempt and parse failure is
scientific provenance; a generic retry wrapper may be introduced later only
for stateless transport calls that do not own attempt accounting.

### Package initialization

This pass removes global warning suppression, stdout printing, and dead flags.
The remaining import-time behavior—Windows stdio configuration and cache
manager initialization—needs a separate compatibility decision. Move it only
with explicit CLI/Web entry-point initialization and subprocess import tests;
do not combine it with data-correctness changes.

## Sequencing

1. Keep the correctness fixes and regression tests in the current data-fix
   lane.
2. Freeze and verify Canonical9.
3. Produce report-only static-debt baselines.
4. Decompose one research-agent responsibility at a time.
5. Migrate MCP transport only after the authority and evidence boundaries are
   stable.
6. Remove deprecated compatibility surfaces in 2.0.
