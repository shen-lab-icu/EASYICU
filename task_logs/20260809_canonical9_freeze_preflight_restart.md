# Canonical9 formal-freeze preflight restart

Date: 2026-08-09 EDT

## Outcome

- Kept the paper-facing batch closed and made zero Provider calls.
- Rejected the historical full-nine JSONL because its conceptual
  `operational_exposure` values no longer satisfy the current exact-column
  contract.
- Materialized all nine fresh development inputs from the verified native
  MIMIC-IV export and controlled identity bridge.  Every operational exposure
  now binds an exact sealed executable column; the final JSONL passed the full
  nine-task zero-Provider prompt preflight.
- Found and removed a second, stale Planner budget owner in Canonical9 prompt
  preflight.  Preflight now consumes the production Planner's recorded
  `limit_bytes` coordinate instead of maintaining the historical 80,000-byte
  constant.
- Closed the exact six CI regressions that remained after that correction.
  Catalog counts and SOFA2 group assertions now derive from the catalog owner
  instead of stale `288`/seven-item literals.  The resource baseline was
  re-adjudicated only after commit-level isolation showed that the reviewed
  SOFA2 evidence contract adds 147 bytes to M3 and the Planner compression
  removes 40 bytes from every task.

## Evidence

- Implementation commits: `b5b4ec649c23e923bed335a00f66dfbf31519bc6`
  and `39263dfa0354d6fc01d6c4856b26afb65cb00cd4`.
- Fresh full-nine materialization root:
  `/Volumes/外置硬盘/easyicu_data/canonical9_miiv_b7a6c79_20260809`.
- Full-nine JSONL SHA-256:
  `f89bee604de9da353f0cbed464b303b10b4ee98a434ee7cc29d5cd0fd898d1f0`.
- Passing full-nine zero-Provider prompt report:
  `/Volumes/外置硬盘/easyicu_data/canonical9_freeze_4cd2cbd_20260809/prompt_preflight_full9/canonical9_prompt_preflight.json`.
- Prompt report status: `passed`; task order 9/9; Provider calls 0.
- Verification: 43 Planner-budget/preflight tests, the exact six prior CI
  failures, and nine adjacent Patient/Cross-DB owner tests passed.
- Re-adjudicated offline resource maximum: 61,430/120,000 Planner bytes;
  Provider calls 0; patient-data reads 0.

## Remaining gates

- Wait for the exact-head CI matrix, then freeze a source-bound runner image
  and run the full-nine Docker resource preflight.
- Obtain real clinical-and-methods attestations for the exact E2/H2/H3 cards;
  the current `curated_mvp` cards cannot be promoted by the verifier.
- Verify the local Provider/model with a non-clinical smoke only after the
  zero-Provider gates are green.  Credentials must remain outside artifacts.
