# Typed-input SDK central wiring handoff

Date: 2026-07-20
Branch: `window2/helper-bool-highs-20260720`
Base: `46689c4`

## Delivered boundary

`authority/typed_input_sdk.py` adds the host-owned
`load_typed_input(...) -> LoadedTypedInput` primitive.

The call accepts only the checkpoint-selected resolved-input manifest,
manifest SHA, run root, Planner input key, current consumer step ID, and
current candidate-code SHA. It does **not** accept an artifact override, a
caller DataFrame, or a caller statement that consumption occurred.

In one call it:

1. opens the resolved manifest and exact regular artifact through anchored,
   no-follow filesystem primitives;
2. verifies manifest SHA, binding identity (`input_key`, evidence ID, artifact
   SHA, producer), relative/absolute path agreement, opened artifact SHA, exact
   columns, and the explicit row-identity contract;
3. rejects unsafe paths, symlinks, unsupported transports, missing/duplicate
   row identity, or any attempt to infer an identity column;
4. seals a consumption receipt against the host-loaded frame;
5. reopens and reverifies manifest/artifact authority before returning, which
   closes the initial load-to-receipt mutation window; and
6. returns an immutable Arrow payload paired with that receipt in one
   `LoadedTypedInput` object.

CSV and Parquet (`.parquet`/`.pq`) are the only SDK transports. The existing
lower-level receipt loader remains unchanged for compatibility, but central
SDK consumers must use this narrower surface.

## Required central wiring boundary 1: candidate fixed, before execution

After the candidate script is final and its `consumer_code_sha256` is known,
but before any sandbox/runner execution:

1. use the checkpoint-selected resolved-input manifest and its recorded SHA;
2. call `load_typed_input` once for every Planner-declared, host-resolved
   file-backed typed input;
3. retain a host-owned `input_key -> LoadedTypedInput` mapping;
4. run `verify_step_typed_input_receipts` over the receipts from that mapping;
5. fail-close before provider/sandbox execution on any finding; and
6. inject only the SDK payloads into the execution adapter. Generated code may
   receive a host-provided table view, but it must not receive authority to
   choose a path or construct/submit receipts.

Do not accept a separately supplied `(DataFrame, receipt)` pair. The
`LoadedTypedInput` object is the capability joining payload and receipt.

## Required central wiring boundary 2: before result or figure seal

Immediately before sealing any result artifact or publication figure that
claims consumption of typed inputs:

1. use the same retained `input_key -> LoadedTypedInput` mapping from the
   pre-execution boundary;
2. call `verify_typed_input_consumption_receipt` for every receipt against the
   current manifest and artifact bytes;
3. rerun `verify_step_typed_input_receipts` for complete-set/current-code
   binding;
4. require the result/renderer adapter's input handles to be the same
   host-owned `LoadedTypedInput` entries, not caller-submitted tables or paths;
5. fail-close before seal/current registration if any artifact, manifest,
   step/code identity, row identity, or receipt-set coordinate changed.

This second verification is required because a file can change after the SDK
call returned. The SDK closes mutations during its own atomic call; the seal
boundary closes the later execution interval.

## Deliberate limits

- The SDK does not decide which Planner inputs are required.
- It does not guess row-identity columns. Missing explicit row identity is an
  authority failure for this row-aligned SDK.
- It does not prove statistical/model semantic use by inspecting estimator
  names. Central adapters must route the retained SDK payload into the actual
  model/result/renderer boundary.
- It does not write checkpoint/evidence/current authority.
- No execution, prompt/spec, repair registry, figure renderer, or online-run
  file is changed in this commit.

## Verification

- `tests/research_agent/test_typed_input_sdk.py`: 24 positive/adversarial cases.
- Existing typed-input receipt tests remain a separate lower-level contract.
- Architecture/module-graph/lint/format/diff checks are recorded in the commit
  handoff response.
