# Typed-input consumption receipt — standalone authority batch

Date: 2026-07-20
Branch: `window2/helper-bool-highs-20260720`

## Scope completed

`authority/typed_input_receipt.py` is a case-neutral host authority leaf. It:

1. verifies the exact `resolved_inputs/<step>.json` bytes and schema;
2. joins one Planner-declared `input_key` to its unique binding and identity row;
3. opens the selected artifact through a no-follow regular-file descriptor;
4. verifies artifact SHA-256, physical schema, row count, ordered row-identity
   digest, uniqueness, and absence of missing identities;
5. returns an ephemeral `VerifiedTypedInputLoad` capability;
6. seals a strict frozen receipt only for the same, unchanged DataFrame object;
7. binds the receipt to the consumer step ID and consumer code SHA-256; and
8. re-opens all authority bytes when a durable receipt is verified.

The receipt schema is `easyicu.typed_input_consumption_receipt/1`. Unknown or
missing fields, a changed artifact, a changed manifest, identity exchange,
cross-step/code replay, duplicate identities, another DataFrame, and mutation
after load all fail closed.

The module does not infer consumption from AST names, a printed path, a column
reference, or a filename. It contains no case, database, exposure, outcome, or
benchmark vocabulary.

## Deliberate authority boundary

The standalone module proves that one exact DataFrame was presented to a
host-owned consumption boundary. It must not be imported by generated code as
a way for the generated script to award authority to itself. Central wiring
must call `seal_typed_input_consumption()` at the actual host-owned result,
model-design, or renderer sink, after the sink has used the supplied frame.

Consequently, the standalone batch intentionally does **not** claim that an
arbitrary model in generated Python consumed the frame. Proving that requires
a host-owned sink/SDK adapter; post-hoc AST appearance or a self-authored
`step_summary.input_bindings` entry remains insufficient.

## Minimal central hookup (not implemented here)

The following locations require changes in files reserved by the main window:

1. `execution/phase.py::_execute_one_step`, immediately after
   `_write_resolved_inputs_manifest(...)` and the existing
   `resolved_inputs_sha256 = sha256_of_file(...)`: preserve these two exact
   coordinates for every host consumption adapter.
2. `execution/runner.py` currently passes only
   `EASYICU_RESOLVED_INPUTS_JSON` into local/Docker execution. A generated-code
   path must not self-seal this new receipt. If a future typed SDK is mounted
   in the sandbox, the host must also supply the expected manifest digest and
   verify the returned receipt outside the writable step directory.
3. The host-owned renderer paths in `execution/phase.py` currently call
   `_write_host_input_binding_receipts(...)` after rendering. They must instead
   load inputs through `load_verified_typed_input_table(...)`, pass the exact
   `loaded.frame` to the renderer, and seal only after that renderer returns.
   The relevant call sites are the visual success path, final figure
   finalisation, and staged publication-figure repair paths.
4. `StepSummaryIntegrityValidator` should accept the strict receipt only after
   its caller supplies the resolved-manifest path/SHA, run root, consumer step,
   and consumer code SHA. Adding receipt prose without those host coordinates
   would recreate the existing fail-open weakness.

No central hookup was implemented because completing items 1 and 3 requires
the main-window-owned `execution/phase.py` and renderer authority flow.

## Existing contract dependency

Generic typed-table schema receipts currently contain physical columns but do
not always contain a Planner/producer-authorized `row_identity_column`,
`row_count`, and `row_identity_sha256`. This module correctly fails closed when
those coordinates are absent. The central integration must extend the upstream
typed product contract with an explicit row identity; the host must not guess
one from column names, dtypes, or position.

## Verification

- 15 standalone positive/adversarial tests.
- 367 focused typed-binding, lineage, step-summary, and declared-product tests.
- Ruff, Black, `py_compile`, and `git diff --check` pass.
- Module graph diff exits 0; current graph remains at zero cyclic SCCs.
- Architecture diff reports no lower-is-better regression in the measured
  execution/control-plane files.
