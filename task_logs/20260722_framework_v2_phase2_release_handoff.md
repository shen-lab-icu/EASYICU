# Framework v2 Phase 2 release handoff

## Status

- Isolated worktree: `/Users/haibo/Documents/GitHub/.worktrees/easyicu-agent-phase2`
- Branch: `codex/agent-phase2-20260722`
- Base: `7d20f6f`
- Commits: `5e4f413`, `a9c7c83`, `1434aa9`
- Release commit: clean `1434aa98bd8408e5ca295191c1525c0951bfd3fe`
- Main branch integration: **DONE 2026-07-22** — independently reviewed and fast-forward
  merged; `refactor/agent-control-plane` HEAD is now the release commit `1434aa98…` itself.
  Mainline clean-release re-run passed 4/4 (117 tests). See
  `task_logs/20260722_framework_v2_phase2_mainline_merge.md` and
  `task_logs/20260722_framework_v2_phase2_mainline_release.json`.

## Batch 1: Coder resource wiring

The production Coder path now deterministically selects bounded Action, Software,
and Data resources from analysis family, step role, typed inputs, and runtime
constraints. Selection receipts are persisted and bound into prompt, authority
capsule, and resume coordinates. Zero matches are valid and selection uses no LLM.

## Batch 2: reviewed memory wiring

Only reviewed knowledge and promoted lessons belonging to the active profile may
enter Coder context. Quarantine, preferences, runtime notes, cross-profile objects,
and tampered receipts fail closed. Historical/default profiles retain their prior
semantic output because the feature is enabled only by additive profiles.

## Batch 3: capability closure

A missing capability causes a structured request and suspension before Planner or
Coder. There is no install, package guessing, or hot mutation. Approval is valid
only for a fresh run using a different registered profile whose expected runner
image digest equals the newly built image; resume and same-container continuation
are rejected.

## Verification

- Framework release: 4/4 passed.
- Focused production and authority suite: 117 passed, 21 warnings.
- Semantic golden: passed.
- Architecture lower-is-better gate: passed.
- Module graph / zero cyclic SCC: passed.
- Release report Git state: clean; status porcelain SHA is the empty SHA.
- Static release allowlist prohibits provider access and patient-data reads. This
  is not presented as OS-level runtime monitoring.

Release JSON is available in the isolated worktree at
`task_logs/20260722_framework_v2_phase2_release.json`.

## Honest remaining work

1. Independent review and main-branch integration.
2. Re-run the same clean release gate on the integrated main commit.
3. Clinical and methods review of ProtocolCards and product-level HITL UX.
4. Build and register a real new image/profile before exercising an approved
   capability; no hot installation is authorized.
5. Online Canonical9 experiments remain frozen until the user explicitly unfreezes
   them, and must reuse `/Volumes/外置硬盘/easyicu_data/full6_20260717`
   without re-extracting the six databases.
