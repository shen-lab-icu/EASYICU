# Typed-input execution envelope — design and prototype handoff

Date: 2026-07-20
Branch: `window2/helper-bool-highs-20260720`
Base: `d7ca9af`

## Scope

This batch adds a standalone host-owned prototype in
`authority/typed_input_execution.py`. It is deliberately **not** connected to
the execution pipeline, online E2, publication authority, prompts, or repair
authorization.

The prototype answers one narrow question: after the host has loaded and
verified a typed input, what is the smallest boundary that can (a) expose only
content-addressed materializations to candidate code and (b) issue a durable
proof only when the exact materialization enters an explicit host-owned model,
table, or figure sink?

## Read-only inventory of current real loading paths

The inventory used the current E3 v14 development run:

`research_output/_development_postqc_n1000_20260719_v14/bench_e3_gpt56luna/E3_kdigo_gradient/aware/run_20260720T030326_ea84b7`

No online run file was modified.

| Downstream role | Logical input | Host-resolved binding | Candidate/host loader | Actual downstream sink | Present proof gap |
|---|---|---|---|---|---|
| Parquet → table | `artifact:analysis_cohort` in `02_table_one_by_kdigo_stage` | `evidence/table_step_artifact_26373c6e0f40ee28__analysis_cohort.parquet`, SHA starts `86ee43b9c3f1` | candidate `pd.read_parquet(artifact_path)` | constructed Table 1 written by `table.to_csv(...)` | opening the path is observable, but the current run has no host receipt joining the exact loaded bytes to the emitted table |
| Parquet/CSV → model | `artifact:analysis_cohort` in `06_secondary_icu_los_association` | same digest-bound cohort Parquet | candidate chooses `pd.read_parquet(path)` or `pd.read_csv(path)` from suffix | `QuantReg(y, X).fit(...)`, then estimate/source tables and plot | estimator-name or source-AST matching cannot prove that the authoritative frame, rather than a decoy frame, entered `y`/`X` |
| CSV → figure | `table:exposure_distribution`, `table:measurement_availability`, `table:missingness_audit`, and `table:structural_missingness_audit` in `03_exposure_and_data_quality_audit_figure` | four evidence CSV files with SHA prefixes `7062b20c47d9`, `209d447aa191`, `b6c94ec0f6e8`, `9d3573f7a9f3` | generated script delegates by parent digests to `_render_authorized_sealed_publication_bundle`; host renderer ultimately uses `figures/distribution_availability.py::_read_selected_columns` and `pd.read_csv(...)` | publication figure bundle | this is the best existing host-owned sink, but the input load and final renderer output still need one explicit receipt chain rather than path/prose claims |
| Historical direct figure path (contrast only) | heuristically selected CSV/Parquet tables in an older E3 candidate | candidate scans step output files rather than consuming a unique resolved binding | candidate `pd.read_csv`/`pd.read_parquet` | `ax.errorbar(...)` and `fig.savefig(...)` | demonstrates why path occurrence, one read, or one subscript cannot be treated as authoritative consumption |

The current candidate scripts therefore use real CSV and Parquet inputs in
table, model, and figure paths, but the trust boundary is inconsistent: some
paths are generated-code owned, while the sealed figure renderer is host owned.
The prototype intentionally models only the latter kind of provable boundary.

## Prototype contract

`TypedInputExecutionEnvelope` accepts only the `LoadedTypedInput` capabilities
returned by the preceding host-owned SDK. For each logical input it:

1. verifies step/code coordinates and the SDK payload/receipt binding;
2. serializes the immutable Arrow table to canonical, uncompressed Parquet;
3. publishes it under `objects/<materialized_sha256>.parquet` with anchored,
   no-follow, write-once filesystem primitives;
4. creates an execution receipt binding the source receipt/artifact, exact
   materialized bytes, payload digest, row identity, consumer step, and
   consumer-code SHA;
5. exposes a candidate manifest containing only content-addressed relative
   paths and verification coordinates, never the source evidence path or
   evidence ID; and
6. retains host-owned sink adapters and proof state outside that manifest.

`execute_host_sink(...)` reopens and re-verifies the materialized bytes and row
identity, invokes an inspectable host-owned adapter with the exact Arrow table,
and issues a proof bound to adapter implementation, output bytes,
materialization, row identity, step, and code.

`verify_required_sinks(...)` returns structured findings and a verified
mapping, not a boolean. Missing requirements, unknown inputs, duplicate
requirements, absent proofs, or multiple competing proofs all fail closed.

Merely printing a path, reading the file, selecting one column, or submitting a
caller-created DataFrame does not create a proof. The public verifier accepts no
candidate-provided proof collection and there is no `mark_consumed` API.

## Important fail-closed blocker discovered in the real run

The v14 resolved product contracts inspected above have physical columns and
transport metadata, but do not carry explicit `row_identity_column`,
`row_count`, or `row_identity_sha256` values. Consequently, the existing strict
typed-input SDK and this envelope would correctly reject these real bindings.

Central integration must first make row identity an explicit producer/Planner
contract for row-aligned products. For products whose semantics truly do not
require row alignment, the earlier completeness layer must carry an explicit
`row_identity_not_applicable` host decision. Neither the SDK nor this envelope
may infer identity from names such as `stay_id`, `record_id`, or the first
column.

## Deliberate prototype boundary

This module does not create an operating-system sandbox. The statement
“candidate code can see only envelope inputs” becomes true only when central
execution mounts the envelope root as the sole input mount and withholds the
host envelope object, original resolved manifest, evidence directory, and sink
adapter capabilities from the candidate process.

Until that central mount boundary exists, this prototype proves that its own
candidate manifest is source-path-free; it does **not** claim to prevent a
caller that independently grants broader filesystem access from seeing other
files.

Similarly, arbitrary candidate-owned estimators and plotting calls cannot
self-award a proof. If the statistical/model sink cannot be expressed through a
host-owned adapter receiving the exact materialized table, verification must
remain `unproven_downstream_sink` and the step must fail closed. No estimator
name allowlist or AST appearance check is a substitute.

## Future central wiring (not implemented)

1. After candidate code and its SHA are fixed, load every required typed input
   with `load_typed_input(...)`, construct one envelope, and mount only its
   content-addressed root plus source-path-free manifest into the sandbox.
2. Candidate code may request declared host sink operations, but cannot receive
   `_HostSinkAdapter`, mutate proof state, or submit a replacement DataFrame.
3. Before result/table/figure seal, require the exact sink kinds declared by the
   step's host contract and call `verify_required_sinks(...)`.
4. Only a finding-free verified mapping may be joined to result/evidence
   registration. A missing model/table/figure proof is a typed error, not a
   warning.
5. Reopen the materialization and verify its execution receipt again at seal to
   close the interval after sink execution.

The central implementation belongs at the candidate execution and seal
boundaries owned by the main window. This commit intentionally does not touch
`execution/phase.py`, `pipeline.py`, `repair_registry.py`, figure-renderer
authority, prompts/specs, or any run artifact.

## Tests and measured overhead

`tests/research_agent/test_typed_input_execution_envelope.py` contains 24
positive/adversarial tests. They cover CSV/Parquet normalization, content
addressing, receipt coordinates, row identity, all three sink families,
candidate proof injection, decoy DataFrames/paths, cross-input and
cross-step/code replay, changed bytes, symlink swap, missing/duplicate
requirements, failed/empty sinks, and the rule that print/read/subscript access
does not prove downstream consumption.

For a 1,000-row table, 12 local repetitions of envelope materialization plus a
model sink proof and verification (excluding the preceding SDK source load)
measured:

- median: **2.489 ms**
- p95: **3.890 ms**
- maximum: **4.808 ms**

This overhead is negligible relative to provider round trips. It is not an
online benchmark and does not predict model-fitting cost.
