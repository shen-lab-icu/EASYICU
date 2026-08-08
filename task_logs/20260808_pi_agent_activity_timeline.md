# PI-COPILOT-AGENT-TIMELINE — safe Agent activity and tool receipts

- Date: 2026-08-08
- Branch: `feat/pi-copilot-shell`
- Implementation commit: `7b8e77c`
- Owner: Web / Guided Pi Copilot

## Problem

The Pi shell used a real `AgentSession`, but the browser only rendered assistant text plus a minimal tool card. Pi lifecycle events such as agent start/settle, turn boundaries, tool progress, retries, and compaction were either discarded or not organized into a readable run trace. The result looked like ordinary streaming chat rather than an auditable agent workflow.

## Reference findings

- `pi-gui` treats Pi session data as the source of truth and derives a threaded timeline with collapsible tool calls: <https://github.com/minghinmatthewlam/pi-gui>
- Upstream Pi exposes SDK/JSON/RPC modes and structured messages/tool events; EasyICU continues to use the SDK rather than reimplementing an agent loop: <https://github.com/earendil-works/pi/tree/main/packages/coding-agent>
- EasyICU intentionally does not expose raw provider reasoning. The visible trace is restricted to lifecycle facts and bounded host receipts, not private chain-of-thought.

## Implementation

1. Added dependency-neutral `event-projection.mjs` as the single owner for browser-safe Pi event/session projection.
   - Projects agent/turn/assistant/tool/retry/compaction/settled lifecycle events.
   - Drops private reasoning updates.
   - Drops raw tool arguments and partial tool output.
   - Keeps bounded tool `status`, stable `code`, `summary`, and `owner` receipts.
2. Upgraded `screens-guided-pi.js` with a per-turn, collapsible Agent activity timeline.
   - Shows submitted, agent start, model turn, response phases, tool return, and settled state.
   - Shows run/tool duration, success/error, stable code, and owner boundary.
   - Handles failed/cancelled message jobs and missed terminal events without leaving a false running state.
3. Restored tool receipt cards from the Pi transcript after page refresh/session reopen. Live lifecycle detail remains a view-layer event trace; persisted tool receipts remain backed by Pi session messages.
4. Kept route ownership in `screens-guided-pi.js` and `guided-pi.css`; no Guided Pi selectors or workflow logic were added to catch-all CSS/JS.
5. Added the projection module to wheel/sdist manifests and advanced the private runtime directory revision so an older installed runtime cannot masquerade as the new contract.

## Verification

- Focused Pi/Web/route suite: `147 passed`.
- Projection/install/static focused suite after packaging changes: `18 passed`.
- Ruff, Node syntax checks, CSS brace/comment ownership guard, and `git diff --check`: passed.
- Isolated wheel build succeeded; both `main.mjs` and `event-projection.mjs` are present in the wheel.
- Real browser canary through the configured local provider:
  - two read-only `easyicu_inspect_context` calls completed;
  - stable code `study_context_projected` rendered;
  - owner `easyicu.webserver.study_contexts` rendered;
  - no configure/run/cancel grant selected;
  - no patient row or identifier sent.
- Session refresh/reopen restored the tool receipt card.
- Expanded desktop layout: no document, conversation log, activity-card, or tool-card horizontal overflow; browser console error list was empty.

## Visual evidence

- `output/ui_audit/pi_agent_timeline_20260808/01-before.png`
- `output/ui_audit/pi_agent_timeline_20260808/02-running.png`
- `output/ui_audit/pi_agent_timeline_20260808/04-final.png`

These files are QA artifacts, not source-of-truth project data.

## Remaining scope

- Next canary: one explicit Configure grant followed by authority-stale/rebind verification.
- Later UX phase may derive more historical turn timing from persisted Pi messages. Do not persist or display private reasoning to simulate a richer trace.
- Do not add generic shell, filesystem, coding, or scientific-authority tools; EasyICU owners remain authoritative.
