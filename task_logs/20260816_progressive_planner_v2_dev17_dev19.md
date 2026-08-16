# Progressive Planner v2: Dev17 decision gate through Dev24

Date: 2026-08-16
Task: `FIG2-DEV9-HELDOUT27`
Branch: `feat/figure2-dev9-heldout27-20260815`

## Authority boundary

All runs in this log are development diagnostics with
`paper_authority=false`. They use the E1 development binding receipt and may
not be promoted into Figure 2 results. Formal evaluation still requires a
fresh exact-head freeze and separate real-run authorization.

## Decision evidence

- Dev17, monolithic Planner at `0b8c9b2`, made five successful Provider calls
  but failed before analysis execution in five different contract states.
  Ledger totals were 182,490 tokens and an estimated USD 2.81576. This was the
  stop condition for further one-shot schema micro-patching.
- Commit `2211a13` introduced a case-neutral progressive skeleton, a host-owned
  compiler, immutable-prefix receipts, suffix-only repair, strict run-bound
  transport schemas, and an explicit `progressive_v2` execution coordinate.
  Before the real canary, the focused Planner/Prompt/configuration set passed
  524 tests with one environment skip, and the Figure 2 authority set passed
  146 tests.

## Real E1 diagnostics

### Dev18

Root:
`/Volumes/外置硬盘/easyicu_data/figure2_dev9_runs/batch_20260816_2211a13_e1_dev18`

The first logical Planner call returned no structured response because the
local Provider proxy reported HTTP 500 with upstream `stream INTERNAL_ERROR`.
The ledger recorded one transport attempt and zero response characters. This
run is Provider-failure evidence only and says nothing about Planner contract
acceptance.

### Dev19

Root:
`/Volumes/外置硬盘/easyicu_data/figure2_dev9_runs/batch_20260816_2211a13_e1_dev19`

The first skeleton needed one schema retry and then passed. Host compilation
subsequently advanced through suffix-only revisions instead of replacing the
compiled prefix. Six Planner calls completed, using 141,823 tokens and an
estimated USD 1.79579. Request payloads were 80,928 to 88,817 bytes, below the
111,844 to 119,799 bytes observed in Dev17 but still too large for a local
suffix repair.

Dev19 did not enter analysis execution. It exhausted four compile revisions
and stopped on stable compiler code
`progressive_product_has_multiple_owners`: steps `measurement_audit` and
`measurement_audit_detail` both declared
`table:missingness_measurement_audit`.

## Repair after Dev19

- The compiler now preflights independent step defects and returns one
  earliest suffix coordinate plus all currently observable typed findings.
  Final materialization reruns every check and remains fail closed.
- Suffix repair no longer repeats the full outbound ResearchContext, article
  contract, action catalog, and literature block. It receives the research
  question, selected family, immutable prefix receipts, current suffix, and
  typed compiler findings; run-bound schema rosters remain enforced by the
  transport.
- The shared Planner contract now publishes the complete five-field artifact
  consumption example. The legacy distribution example and progressive
  compiler both classify the result-bearing distribution as `secondary`; the
  compiler emits the canonical `descriptive` method and exact descriptive
  capability, and the deterministic distribution owner now claims that typed
  secondary result.
- The fixed monolithic Planner directive remains at its existing 51,600-byte
  ratchet rather than raising the context budget.
- Verification passes: 658 focused Planner, prompt, execution-owner,
  literature, identity, configuration, and archive tests passed with one
  environment skip; Ruff and `git diff --check` are clean.

### Dev20

Root:
`/Volumes/外置硬盘/easyicu_data/figure2_dev9_runs/batch_20260816_6c1f0e1_e1_dev20`

The exact `6c1f0e1` image made three successful strict-schema skeleton calls.
The transport receipts recorded 59,207 prompt tokens and 13,018 completion
tokens (72,225 total); the configured USD 10/30 per-million price table implies
USD 0.98261. Request payloads were 61,837 to 63,503 bytes. The run still did not
enter analysis execution.

The retries reduced five validation findings to one. Standard modules had
filled the schema-permitted `custom_method` field even though Pydantic correctly
reserved it for `custom_analysis`; the final response then omitted outputs for
the locked-grid `robustness_replay`, although those outputs are deterministic
runtime products. The final stable boundary was therefore schema-to-host
materialization, not statistics or Docker execution.

## Repair after Dev20

- The run-bound step transport is now a compact two-branch contract. Every
  standard module requires `custom_method=null`; only `custom_analysis` accepts
  a string, and that branch exposes only fields it can use. The resulting
  authority payload is 11,101 bytes, below the existing 12 KiB ratchet.
- The compiler now materializes the locked-grid replay's two unambiguous
  deterministic products, `table:robustness_matrix` and
  `table:robustness_summary`, when the skeleton uses `outputs=[]`. It also emits
  the existing typed `RobustnessReplaySpec`; the deterministic replay contract
  accepts the compiled step.
- The direct set passed 186 tests. The broader focused Planner/robustness set
  then passed 510 tests with one environment skip. Ruff and
  `git diff --check` are clean.

### Dev21 and Dev22

Both exact `61881c2` launches stopped on the first transport attempt with
HTTP 401 `Invalid API key`. They produced no structured response and are
transport/authentication diagnostics only, not evidence about the compiler.

### Dev23

Root:
`/Volumes/外置硬盘/easyicu_data/figure2_dev9_runs/batch_20260816_61881c2_e1_dev23`

The exact `61881c2` image completed nine Planner calls, using 129,594 tokens
and an estimated USD 1.79270. The final three suffix requests were 26,389,
26,968, and 27,506 bytes, confirming that compact suffix transport materially
reduced the prior 80--89 KiB request surface. The run still did not enter
Execute. Its terminal owner finding was
`progressive_duplicate_literature_source` at step
`scientific_sensitivity_table`: one sealed citation had been returned in more
than one design-binding record for the same step.

## Repair after Dev23

- The progressive compiler now deterministically coalesces repeated records
  for one citation key into the single `LiteratureDesignBinding` required by
  `AnalysisStep`. It preserves first-seen key and design-element order, keeps
  every distinct application/divergence statement, and does not invent new
  scientific content.
- Coalescing delegates final field limits to the existing
  `LiteratureDesignBinding` contract. If preserving all statements would
  exceed that contract, compilation fails with attributable code
  `progressive_literature_merge_overflow`; no text is truncated.
- The direct and adjacent progressive Planner, run-bound literature schema,
  literature authority, and package-direction set passed 48 tests. Ruff and
  `git diff --check` are clean.

### Dev24

Root:
`/Volumes/外置硬盘/easyicu_data/figure2_dev9_runs/batch_20260816_50a9b11_e1_dev24`

The exact `50a9b11` image completed four Planner calls, using 69,895 tokens
and an estimated USD 0.94927. Request payloads were 83,242, 84,037, 35,990,
and 33,774 bytes. The Dev23 repeated-source finding did not recur. Planning
instead stopped before Execute when the proposed Table 1 listed its `group_by`
column again as a row variable. `TableOneSpec` correctly rejected that
impossible display shape, but its raw Pydantic `ValidationError` escaped the
progressive compiler and therefore could not drive typed suffix repair.

## Repair after Dev24

- The compiler now treats a Table 1 `group_by` entry in the row roster as a
  redundant representation of the already-declared grouping coordinate and
  omits it deterministically. No analysis variable or group definition is
  removed: the grouping column remains bound in `TableOneSpec.group_by` and in
  the step inputs.
- A roster containing no distinct row variable still fails closed with
  `progressive_table_one_rows_missing`. Any remaining downstream
  `TableOneSpec` validation error is contained at the compiler boundary as
  `progressive_table_one_contract_invalid`, with step and field coordinates,
  rather than leaking an unattributable Pydantic exception.
- The progressive Planner and adjacent Table 1 contract/ordinal sets passed
  50 tests. Ruff and `git diff --check` are clean.

## Next gate

Build an exact-source image from the repair commit and run a fresh E1 Dev25.
Do not start E2 until E1 completes the full analysis, audit, figure, and report
workflow under development authority.
