# Guided Pi timeline design QA

## Comparison target

- Source visual truth:
  - `/Users/haibo/Documents/GitHub/EASYICU/task_logs/browser_audit_20260808_pi_timeline/reference_codex_thread.png`
  - `/Users/haibo/Documents/GitHub/EASYICU/task_logs/browser_audit_20260808_pi_timeline/reference_codex_tool_timeline.png`
- Implementation screenshots:
  - `/Users/haibo/Documents/GitHub/EASYICU/task_logs/browser_audit_20260808_pi_timeline/easyicu_pi_timeline_complete.png`
  - `/Users/haibo/Documents/GitHub/EASYICU/task_logs/browser_audit_20260808_pi_timeline/easyicu_pi_tool_expanded.png`
- Same-input comparison evidence:
  - `/Users/haibo/Documents/GitHub/EASYICU/task_logs/browser_audit_20260808_pi_timeline/comparison.png`
- Route: `http://127.0.0.1:8765/?ui=20260808-project-rail1#guided`
- State: light theme, Chinese UI, selected local research project, real `gpt-5.6-luna` read-only turn, two completed EasyICU tools, completed reply; one additional capture has the first receipt expanded.

## Capture normalization

- Source pixels: complete Codex thread `704 × 918`; Codex tool focus `415 × 861`.
- Implementation pixels and CSS viewport: `1572 × 1354` at browser density 1 CSS pixel per captured pixel.
- Full-view comparison normalizes both complete-thread captures to a 700-pixel displayed content width and a common 720 × 970 cell.
- Focused comparison normalizes the tool reference to 760 pixels high and the implementation to 700 pixels wide inside common 720 × 800 cells.
- The sources are cropped Codex screenshots rather than a same-product full viewport, so the comparison is qualitative for hierarchy, typography, component shape, and rhythm; it does not claim pixel-identical outer-shell geometry.

## Primary interaction and runtime checks

- Selected an existing EasyICU research project and created an isolated Pi session.
- Sent a real read-only prompt without configure/run/cancel grants.
- Observed two bounded tools, completion, assistant response, and bottom composer.
- Expanded a tool receipt and verified tool name, stable code, duration, and owner remain available.
- Reloaded the route, reopened the project, and verified the duration/lifecycle marker, tool rows, and answer reconstruct from the persisted transcript.
- Restarted both the WebApp process and Pi sidecar, then reopened the project and verified the same persisted turn reconstructs again.
- Expanded the reconstructed lifecycle and verified only lifecycle facts and receipts are shown; private chain-of-thought remains hidden.
- Browser console errors after the final reload: none.
- Default desktop viewport: body `scrollWidth = clientWidth = 1572`; all activity/tool summaries `scrollWidth = clientWidth = 768`; horizontal overflow false.
- Composer bottom equals viewport bottom (`1354`), while the transcript owns vertical scrolling.

## Required fidelity surfaces

- Fonts and typography: 15px body/composer copy, 13px inline tool rows, 12px duration metadata, system UI stack, readable Chinese wrapping, no truncation in the measured state.
- Spacing and layout rhythm: 768px reading measure, right-aligned user bubble, unboxed activity/tool rows, 15–24px vertical rhythm, bordered container only for expanded receipt content.
- Colors and tokens: existing EasyICU surface/ink/hairline tokens retained; muted lifecycle copy plus text and colored pips for running/success/error states.
- Image and icon fidelity: no raster product art is involved; timeline icons use the existing EasyICU shared line-icon catalog, not new emoji, CSS drawings, or one-off SVG markup.
- Copy and content: user message, duration, semantic tool labels, status, assistant answer, stable receipt code, owner boundary, and private-reasoning disclaimer are all present.

## Comparison history

### Iteration 1

- Earlier P1 finding: the baseline displayed Agent activity and every tool receipt as separate bordered cards, unlike the Codex/pi-gui inline timeline; generic lifecycle rows dominated the conversation.
- Fix: adapted the MIT-licensed `pi-gui` timeline item structure—unboxed activity/tool headers, line icons, rotating disclosures, status pips, turn-duration divider, and a bordered body only when a receipt is expanded. User messages became right-aligned bubbles and assistant messages became plain reading-column prose.
- Post-fix evidence: `easyicu_pi_timeline_complete.png` and the upper half of `comparison.png`.

### Iteration 2

- Earlier P2 finding: the completed duration/lifecycle marker was initially page-memory-only and disappeared after reopening the Pi transcript.
- Fix: rebuilt one completed activity marker per persisted user turn from transcript timestamps, tool calls/results, and the final assistant message.
- Post-fix evidence: final `easyicu_pi_timeline_complete.png`; browser reload showed `耗时 7.0 秒`, both completed tool rows, and the same final response.

## Findings

- No actionable P0, P1, or P2 visual or interaction mismatch remains for the requested desktop Guided Pi conversation state.
- The right-hand EasyICU research workspace remains visible by design. It differs from Codex's single-purpose task pane but preserves the product requirement that study setup stays in the same Copilot screen.

## Follow-up polish

- P3: a future iteration could add a compact per-turn copy/retry action row after the Agent shell supports those commands; it is not present in the current product contract and does not block this timeline migration.

final result: passed
