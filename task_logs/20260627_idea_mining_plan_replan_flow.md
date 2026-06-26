# 2026-06-27 Idea Mining Plan/Replan Flow

## Scope

User feedback: Idea Mining must not jump from a mined idea directly to Agent handoff. It should first produce a human-reviewable study plan/replan draft that preserves the ICU clinical question, feasibility status, data-context confirmations, reference method motifs, and Agent execution boundary.

## Changes

- Added `/api/ideas/plan` in `src/easyicu/webserver/app.py`.
- Added metadata-only plan artifact generation in `src/easyicu/webserver/ideas/mining.py`.
  - Persists `idea_plan.json`.
  - Marks `agent_run_created=false`, `draft_unlocked=false`, `reportable=false`.
  - Requires later confirmation of local export/database source, cohort denominator, mapped modules/concepts, outcome/time window, analysis family, and prior-art decision.
  - Adds ICU-specific constraints and reference method patterns, e.g. critical-care treatment-strategy and target-trial-style planning guardrails for vasopressor/fluid ideas.
- Wired `window.EU_API.planIdea` in `src/easyicu/webserver/static/js/api.js`.
- Split Guided Copilot plan/replan renderer into owner file `src/easyicu/webserver/static/js/screens-guided-idea-plan.js`.
- Added owner CSS in `src/easyicu/webserver/static/css/guided-idea-plan.css`.
- Updated `src/easyicu/webserver/static/js/screens-guided.js` so Guided Idea Mining now flows:
  1. source clue
  2. idea ledger
  3. data context confirmation
  4. plan/replan before Agent handoff
- Updated `src/easyicu/webserver/static/js/screens-ideas.js` so Classic Idea Mining has the same explicit Plan/Replan step before handoff.
- Updated static route and backend tests:
  - `tests/test_webserver_idea_sources.py`
  - `tests/test_webserver_static_routes.py`

## Verification

- `./.venv/bin/python -m py_compile src/easyicu/webserver/ideas/mining.py src/easyicu/webserver/app.py`
- `/Users/haibo/.nvm/versions/node/v24.11.0/bin/node --check` on:
  - `src/easyicu/webserver/static/js/screens-guided.js`
  - `src/easyicu/webserver/static/js/screens-guided-idea-plan.js`
  - `src/easyicu/webserver/static/js/screens-ideas.js`
  - `src/easyicu/webserver/static/js/api.js`
- `./.venv/bin/python -m pytest tests/test_webserver_idea_sources.py tests/test_webserver_static_routes.py -q`
  - Result: 53 passed, 1 warning.
- Owner scan:
  - `gdi-plan-details` and `gdi-feature-row.one` are only in `guided-idea-plan.css`.
  - Guided plan renderer is in `screens-guided-idea-plan.js`; `screens-guided.js` only delegates and handles events.
  - No Guided plan widget selectors found in `redesign.css`, `tweaks.css`, broad `app.js`, or `tweaks.js`.

## Browser Smoke

Server restarted on `http://127.0.0.1:8765`.

Guided Copilot `测试1`:

- Article URL mode and manual mode render different full forms.
- NEJM URL seed without network opt-in produced a local idea ledger.
- The mined idea did not jump to Agent. It first showed `数据上下文尚未确认` and kept Plan/Replan locked.
- After confirming the active export context, the UI showed `生成研究计划`.
- Generated plan displayed:
  - analysis steps
  - reference method patterns
  - ICU constraints
  - remaining confirmations
  - replan notes box
  - freeze handoff button
  - disabled create-project button until handoff exists

Classic `#ideas`:

- Step navigation includes `计划 / replan`.
- The Plan/Replan step is locked until idea ledger and feasibility context exist.

Follow-up browser check after the source-mode matrix:

- Reopened Guided Copilot `测试1` on `http://127.0.0.1:8765/#guided`.
- The conversation restored the 4-step Idea Mining flow: source clue, candidate idea, data context, and pre-Agent plan/replan.
- The NEJM vasopressor/fluid seed showed data context confirmed, an active-export aggregate pre-experiment, `draft_plan_requires_user_review`, remaining confirmations, replan notes, and a disabled `创建 Agent 项目` button until handoff is frozen.
- No patient rows, full text, external provider call, or Agent execution was displayed as part of the planning step.

## Additional Source-Mode Smoke

Ran a temporary TestClient matrix with isolated `idea_runs` and `idea_history` paths, so no real project conversation or local export state was modified.

| Case | Source mode | Result | Plan/replan boundary |
|---|---|---|---|
| `manual_lactate_sepsis` | Manual clinical question | Mapped `sep3_sofa2`, `lact`, `death`; held because no export was registered in the isolated test context. | `plan_status=draft_plan_requires_user_review`; no Agent run; no draft unlock. |
| `article_nejm_vaso_fluids` | Article URL metadata for `10.1056/NEJMoa2516225` | Mapped vasopressor/fluid, shock, mortality and balance concepts. | Replan returned critical-care treatment strategy and target-trial-style patterns; handoff seed still required human confirmation. |
| `pdf_ards_peep` | Local PDF upload | Extracted metadata-only excerpt and mapped mechanical ventilation, mortality, and PEEP concepts. | Plan required local export, cohort denominator, module/concept, outcome/window, and prior-art decisions. |
| `folder_aki_fluid_balance` | Local literature folder | Scanned a local PDF folder without full-text persistence and mapped fluid, AKI, creatinine, urine, and mortality concepts. | Plan stayed metadata-only and did not create an Agent run. |
| `frontier_topic_no_network` | AI literature-discovery topic with network disabled | Returned `blocked_network_opt_in_required`; generated 3 queries but performed 0 network calls. | Correctly stopped before discovery/plan because user opt-in was missing. |

## Notes

This is intentionally still a pre-Agent planning artifact. It does not run planner/replanner/coder/analyzer/writer, does not claim novelty, and does not unlock manuscript draft content.
