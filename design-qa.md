# Guided Pi activity timeline design QA

## Visual truth and evidence

- Codex reference supplied by the user: `/var/folders/68/cz0swdq52vx1_rh5m4gql6v00000gn/T/codex-clipboard-ff0de24f-c147-4b48-a7e3-66602616bbb7.png`.
- Live EasyICU activity state: `task_logs/browser_audit_20260808_pi_freeze_round/04-live-tool-state.jpg`.
- Completed and expanded activity state: `task_logs/browser_audit_20260808_pi_freeze_round/06-expanded-final.jpg`.
- Final conversation state with safe inline Markdown: `task_logs/browser_audit_20260808_pi_freeze_round/07-final-markdown.jpg`.
- Same-input side-by-side comparison: `task_logs/browser_audit_20260808_pi_freeze_round/08-reference-comparison.jpg`.
- Route: `http://127.0.0.1:8765/?ui=20260808-project-rail1#guided`.
- Browser state: Chinese UI, existing local research project, real `gpt-5.6-luna` read-only turns, desktop viewport `1572 x 1354`.

The reference and implementation were placed in one comparison image before judging the result. The outer product shells intentionally differ; the comparison target is the conversation rhythm, activity semantics, disclosure behavior, typography, and reading hierarchy.

## Implemented interaction contract

- While a turn is active, one inline row names the current semantic action, for example `已读取研究配置`; a status pip carries running/success/error state.
- After completion, the turn becomes one compact summary such as `已检查工作区状态、已读取研究配置`; elapsed time is shown separately.
- Expanding the summary reveals individual tool receipts, stable codes, owner boundaries, and receipt details. It does not expose private chain-of-thought.
- Persisted transcripts reconstruct the same grouped activity timeline instead of falling back to independent tool cards.
- User messages remain right-aligned bubbles; assistant responses remain unboxed prose; the composer stays at the bottom of the conversation column.
- Assistant text supports escaped inline bold and code, preserving readable model responses without introducing HTML injection.

## Iteration record

### Iteration 1 — activity grouping

- Finding: the previous EasyICU page showed a generic Agent lifecycle card plus separate tool cards. It looked like a monitoring dashboard rather than a coding-agent conversation.
- Change: replaced the stack with one current-action row during execution and one collapsed turn summary after completion.
- Evidence: `04-live-tool-state.jpg` and `06-expanded-final.jpg`.

### Iteration 2 — semantic state and durable evidence

- Finding: generic labels such as `模型回合进行中` did not tell the user what the Agent was doing, while always-visible receipt cards overemphasized implementation detail.
- Change: mapped tools to human semantic actions, separated duration from the summary, and moved receipt-level evidence behind disclosure while keeping it persisted and attributable.
- Evidence: `06-expanded-final.jpg`.

### Iteration 3 — answer readability

- Finding: model replies containing lightweight Markdown were rendered literally, making the assistant prose feel less finished than the Codex reference.
- Change: added an escape-first inline Markdown renderer for bold and code.
- Evidence: `07-final-markdown.jpg`.

## Browser QA

- Real read-only conversation completed through the configured local model endpoint; three turns returned valid assistant replies.
- Activity state appeared during execution, completed summaries collapsed correctly, and receipt disclosure remained interactive.
- No horizontal clipping: body `1572/1572`, transcript `943/943`, composer `958/958` (`scrollWidth/clientWidth`).
- The bottom composer, transcript scrolling, project rail, and conversation text remain usable at the audited desktop viewport.
- CSS remains owned by `guided-pi.css`; no route-specific rules were appended to catch-all stylesheet files.

## Findings

- No actionable P0, P1, or P2 visual or interaction defect remains in the audited Guided Pi desktop conversation flow.
- EasyICU intentionally retains its research-project rail and study authority; parity with Codex applies to Agent activity presentation, not to copying Codex's product shell.

final result: passed
