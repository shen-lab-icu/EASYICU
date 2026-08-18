# Copilot activity visibility and Project Monitor boundary

- Date: 2026-08-17
- Worktree: `/Users/haibo/Documents/GitHub/EASYICU-figure2-integration`
- Branch / starting HEAD: `integration/figure2-e1-h3-20260816` / `66186c1`
- Final verification HEAD: `0d2434f` (concurrent progressive Planner commits were outside this task and did not overlap Web owners)
- Scope: Web frontend ownership and navigation only. No provider call, E1 run, scientific result, or publication authority was created.

## Decision

Guided Copilot is the conversational owner for requirements, study setup, the selected model connection, planning and confirmation, and the action that initiates governed Research Agent work. The legacy `#agent` route is renamed and constrained as **Project Monitor**: it may display or reconnect an existing run, open outputs and evidence, and support governed review, but it does not configure or initiate analysis.

The term `agent` is now explicitly disambiguated in the root `CLAUDE.md` and `AGENTS.md`:

- conversational UI, lifecycle, and tool visibility: `screens-guided-pi*.js`;
- scientific execution and evidence authority: `src/easyicu/research_agent/`;
- project/run/evidence visualization: legacy `#agent` Project Monitor.

## Implemented boundary

- Added `screens-guided-pi-activity.js` as the dedicated activity-display owner. It renders safe lifecycle, model phase, read/edit/tool, retry, compaction, duration, and artifact events in a collapsible timeline.
- The activity projection does not expose private chain-of-thought, raw tool arguments, credentials, patient rows, or host paths. It uses host-projected labels and artifact receipts.
- Removed provider/model setup, editable Planning Blocks, project creation, and run start/restart/promote controls from `screens-agent.js`; monitoring, reconnect/cancel for an already-started job, outputs, evidence, and review remain.
- Renamed the route and navigation to Project Monitor while preserving `#agent` as a compatibility route id.
- Moved analysis-start handoffs from Data Extraction, Patient Review, Cohort Statistics, and Cross-DB results to Guided Copilot. Existing-run review links may still enter Project Monitor.
- Updated Settings, Help, Idea Mining, legacy Guided copy, and Page Guide language so setup and execution no longer point users to Project Monitor.
- Reduced `screens-agent.js` from 2,485 to 1,974 lines and `agent.css` from 1,010 to 585 lines; extracted activity rendering reduced `screens-guided-pi.js` from 2,002 to 1,787 lines.

## Verification

Focused executable checks:

```text
143 passed, 5 warnings
```

Command scope:

```text
tests/test_pi_copilot_static.py
tests/test_static_frontend_ownership.py
tests/test_webserver_static_routes.py
tests/test_webserver_study_context_frontend.py
tests/test_webserver_patient_demo_data.py
tests/test_webserver_cohort_profile_ui.py
tests/test_webserver_crossdb_setup_frontend.py
```

All modified JavaScript owners also passed `node --check`.

Browser QA used the isolated development server `127.0.0.1:8898`; the existing `8877` process was not touched. At 1280×720:

- Cohort Statistics analysis CTA: `data-study-target="guided"`, label `Continue in Guided Copilot`, click result `#guided`.
- Project Monitor title: `Research Project Monitor`.
- Project Monitor forbidden controls: new project `0`, run initiation `0`, Planning Blocks `0`, provider inputs `0`.
- Project Monitor return links to Guided Copilot: `3` in the tested state.
- Page horizontal overflow: false on Cohort Statistics, Guided Copilot, and Project Monitor.
- Console warnings/errors: `0` / `0`.

Browser images:

- `output/playwright/cohort-to-copilot-boundary-loaded-1280x720.png`
- `output/playwright/project-monitor-boundary-1280x720.png`
- `output/playwright/guided-copilot-activity-1280x720.png`

The Guided activity demo used browser-local readiness mocks only to render the reviewer flow; it made no external provider call and is not E1 scientific evidence.

## Limits and next step

- This closes the frontend responsibility boundary and activity visibility task; it does not prove E1 completion or provider/model performance.
- The next scientific step remains one bounded E1 run from a normal Guided Copilot conversation, with the existing Research Agent evidence and publication gates unchanged.
- Concurrent progressive Planner edits in the same worktree are outside this task and were not altered or included in this verification claim.
