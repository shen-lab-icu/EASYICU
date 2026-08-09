# Pi Workspace production-composition closure — 2026-08-09

## Scope

- Branch: `fix/pi-workspace-review-20260809`.
- Review baseline: `4fb66e3`.
- Implementation: `60fea79`.
- Task ID: `PI-WORKSPACE-COMPOSITION-CLOSURE`.
- Owner boundary: Web / Guided Pi Workspace.

This pass closes the latest review's one remaining release-blocking path and
two adjacent architecture findings. It does not expand into DNS pinning, live
Anthropic/Google canaries, multi-process compare-and-swap, or provider price
accounting.

## Decisions and implementation

1. **Declared workspace identity survives production composition.**
   `PiGatewayClient` now retains both declared and resolved session/workspace
   paths. `PiCopilotService` constructs `ProjectWorkspace` from the declared
   path, and `ToolExecutionContext` receives that already-sealed workspace
   object rather than resolving a bare path again for every tool call.
2. **Default-path derivation preserves ancestor identity.**
   The default workspace is derived from `declared_session_dir`, not the
   resolved session directory. This closes the normal `~/.easyicu` relocation
   path as well as explicit `cwd=` composition.
3. **Ancestor-symlink policy is explicit.**
   The workspace entry itself may not be a symlink. A stable ancestor symlink
   is supported for deliberate relocation, but its resolved identity is sealed
   once and any later retarget fails closed with
   `pi_workspace_base_root_changed` before creating files under the new target.
4. **Research artifact governance has one owner.**
   `agent_runs.project_artifact_governance()` now derives gate status,
   readiness, human sign-off, reportability, and claim ceiling. The Pi adapter
   only validates and projects that receipt; it no longer reimplements the
   state machine.

## Regression evidence

- The new real-composition regression first failed because a symlinked
  `PiGatewayClient(cwd=...)` was silently accepted by `PiCopilotService`.
- A second real-default-composition regression then exposed that resolving
  `session_dir` before deriving the workspace still erased an ancestor
  symlink. It also failed before the final fix.
- Final focused/affected gate: `102 passed` — all 97 Pi Copilot, gateway,
  workspace, route, static, provider, and installer tests plus 5 run-governance
  and sign-off owner tests.
- Ruff and `git diff --check` passed on every touched file.
- No frontend CSS/JS changed, so no new browser-layout or CSS-ownership surface
  was introduced in this pass.

## Release gate

- Previous exact-head scoped run `31296995067` is green for `4fb66e3`.
- Previous repository CI `31296995045` is still running; packaging and all
  portability jobs are green, while Python 3.10/3.11/3.12 tests remain active.
- Push `60fea79`, then require a new exact-head scoped gate and repository CI
  before merge. PR #7 remains the merge vehicle; this task does not merge it.

