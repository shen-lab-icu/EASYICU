# Canonical9 formal-freeze preflight restart

Date: 2026-08-09 EDT

## Outcome

- Kept the paper-facing batch closed and made zero Provider calls.
- Rejected the historical full-nine JSONL because its conceptual
  `operational_exposure` values no longer satisfy the current exact-column
  contract.
- Materialized a fresh E1 development canary from the verified native MIMIC-IV
  export and controlled identity bridge.  The new row binds the conceptual
  predictor `sepsis3` to the sealed executable column `sep3_sofa2_max`.
- Found and removed a second, stale Planner budget owner in Canonical9 prompt
  preflight.  Preflight now consumes the production Planner's recorded
  `limit_bytes` coordinate instead of maintaining the historical 80,000-byte
  constant.

## Evidence

- Implementation commit: `b5b4ec649c23e923bed335a00f66dfbf31519bc6`.
- E1 canary root:
  `/Volumes/外置硬盘/easyicu_data/canonical9_miiv_b7a6c79_20260809`.
- E1 JSONL SHA-256:
  `883ef35e9298490c97e160c54f0a475e10235ac45b5faf650565a22bdb5a4652`.
- Passing zero-Provider prompt report:
  `/Volumes/外置硬盘/easyicu_data/canonical9_freeze_b7a6c79_20260809/prompt_preflight_e1_v2/canonical9_prompt_preflight.json`.
- Focused verification: 43 Planner-budget and preflight tests passed.

## Remaining gates

- Resume the same fresh materialization root for the remaining eight tasks,
  then run full-nine prompt and Docker resource preflights.
- Wait for the exact-head CI matrix and freeze a source-bound runner image.
- Obtain real clinical-and-methods attestations for the exact E2/H2/H3 cards;
  the current `curated_mvp` cards cannot be promoted by the verifier.
- Verify the local Provider/model with a non-clinical smoke only after the
  zero-Provider gates are green.  Credentials must remain outside artifacts.

