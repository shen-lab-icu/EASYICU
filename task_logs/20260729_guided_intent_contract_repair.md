# WEB-COPILOT-COCKPIT-LITE · Guided research-intent contract repair

Date: 2026-07-29
Branch: `codex/web-copilot-cockpit-lite-20260729`
Worktree: `/Users/haibo/Documents/GitHub/.worktrees/easyicu-copilot-cockpit-lite`

## Problem reproduced

The interaction audit found four coupled failures:

1. A Home research question could enter the seeded demo parser and be rewritten into a different exposure, comparator, and study family.
2. Setup text containing broad words such as “patient” or “cohort” could be routed to Review Data and render an unrelated Kaplan–Meier panel.
3. The center, title, and right rail could project different stages after project creation or project restore.
4. The right-rail “Answer this question” action reused the full Data Extraction wizard, silently populating defaults and advancing a different state machine.

## Repair

- Added the pure, case-neutral `screens-guided-intent.js` owner.
  - Preserves `raw_question` verbatim.
  - Owns the typed field receipt, current field, explicit confirmation, and fail-closed text routing.
  - Has no API, DOM, localStorage, or execution dependency.
- Home questions now start a fresh Guided project boundary and bypass the seeded demo framing engine.
- Required setup is collected one field at a time inside the Copilot conversation.
- The right Study Brief reads the same intent receipt and no longer falls back to extraction, StudyContext, or demo defaults while that receipt owns setup.
- Project restore immediately re-renders the center, title, and right rail from the restored receipt.
- Confirming the brief prepares the Agent preflight but does not start it.
- Demo or mismatched sources cannot borrow the globally active export; the run button remains disabled until the confirmed source matches the registered active export.
- Editing a confirmed field invalidates the prepared Agent card and returns to the one-field receipt without opening Data Extraction.
- Restored run-agent setup no longer shows stale English suggestion chips.

## Automated evidence

Pure Node contracts:

```text
guided intent contract: 32 cases passed
guided study workspace: 36 cases passed
```

Focused Python tests:

```text
4 passed
```

Full Web UX/static route suite:

```text
81 passed, 1 failed
```

The one failure is the pre-existing isolated-worktree path assertion:

```text
test_native_extraction_feature_definition_manifest_records_callback_provenance
expected project_ref.hint == "EASYICU"
received "easyicu-copilot-cockpit-lite"
```

It is unrelated to Guided UI or this patch; all Guided and ownership tests passed.

## Browser evidence

Viewport: 1280 × 720 desktop.

- Exact question remained:
  `我想研究脓毒症患者早期使用血管活性药物是否影响28天死亡率`
- Free-text cohort answer remained in the intent setup.
- Data Extraction cards: `0`
- Unrelated Sepsis vs Non-sepsis KM panels: `0`
- Agent objective after restore: exact original question.
- Demo preflight run button: disabled.
- Right-rail cohort action owner: `data-gdsw-intent-field="cohort"`.
- Horizontal overflow: `scrollWidth === clientWidth === 1280`.

Artifacts:

- `output/ui-qa/20260729_guided_intent_repair/after-free-text-stays-in-brief.png`
- `output/ui-qa/20260729_guided_intent_repair/verified-restored-preflight.png`
- `output/ui-qa/20260729_guided_intent_repair/before-after-free-text-comparison.jpg`
- `design-qa.md`

## Safety / side effects

- No Agent run was started.
- Browser QA created one metadata-only local Guided folder:
  `~/easyicu/projects/guided-运行研究项目-9ee656`
- No backend execution owner was changed.
- No CSS file was added or modified.
- No merge or push was performed.
