# Progressive Planner v2: Dev17 decision gate through Dev19

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

## Next gate

Build an exact-source image from the repair commit and run a fresh E1 Dev20.
Do not start E2 until E1 completes the full analysis, audit, figure, and report
workflow under development authority.
