# Post-audit residual closure — 2026-08-12

- Modules: `DATA-FIX1`, `WEBAPP-FASTAPI-NATIVE-QA`, `PATIENT-CROSSDB-VISUAL-PARITY`, `FIG2-CANONICAL9-GATE`
- Scope: residual defects found after the D1–D3 / W1–W3 / A1–A3 remediation
- Workspace: shared dirty snapshot; no commit, push, formal run, full exact-head CI, or LOCK finalization

## Outcome

The remaining product defects in this review are closed in the current working-tree snapshot. Release integration is deliberately still fail-closed: several new production owner modules remain untracked in the shared workspace, and the concept lock remains genuinely unfinalized. Those two repository/release gates must be closed by one coordinated atomic commit and a real six-database extraction/QC cycle; neither was papered over in this task.

## Data residuals

- Fixed-horizon `mort_28d` summary and Kaplan–Meier vectors now apply the same event/time consistency filter. `event=True` after day 28 and `event=False` before day 28 are excluded and counted instead of producing contradictory event totals.
- Web data scanning now preserves typed database-detection failures. Ambiguous and unidentified prepared paths are returned as structured fail-closed findings rather than silently becoming `ready=true`.
- Database schema probing is case-insensitive for official table and extension names, including Linux/case-sensitive hosts.
- Content fingerprint exclusion now ignores an `exclude_dir` that is an ancestor or the same directory as the data root; only a proper descendant can be excluded. The content receipt index itself remains excluded.
- Cache-format tests now distinguish the opaque data cache artifact from its content-receipt index.

## Agent residual

- Literature binding policy now belongs to `research_agent/planning/literature_bindings.py`, a dependency-neutral planning contract.
- `planning/replan_gate.py` imports that owner directly. `agents/plan_payload.py` is a compatibility adapter/re-export, so planning no longer imports the agents layer.
- A package-direction regression walks planning imports and verifies that the adapter and owner expose identical callable contracts.
- Existing A1 and A3 fixes were reverified: sandbox child `NameError` remains coder-repairable, real backend startup failure remains fail-closed, and wide `*_first_time` stays an observation coordinate rather than clinical onset/initiation.

## Web residuals

- Pi workflow now includes a bounded, path-free `StudySetupReceipt` carrying `study_context_id`, exact revision, configured fields, and safe setup values. “研究配置” sends that authoritative receipt into the current Pi conversation and repeated projection is read-only.
- SPA navigation updates `document.title`, announces the route through a polite live region, and focuses the page heading or `main`. Route focus is preserved through asynchronous screen redraws.
- Cross-DB result tabs now implement unique tab/panel IDs, `aria-controls`, `aria-labelledby`, roving `tabindex`, and ArrowLeft/ArrowRight/Home/End navigation.
- Native responses add CSP, `nosniff`, no-referrer, and camera/microphone/geolocation denial headers. Google Fonts connections were removed; the UI uses local/system fallback fonts and CSP permits only same-origin fonts.
- The extraction screen's offline renal fallback count was updated from 29 to the current 35-concept owner catalog; stale agent asset-version assertions were updated to the actually wired owner versions.

## Verification

- Data/filter/security adjacent files: `125 passed`.
- Native static route suite: `74 passed`.
- Agent planning/dependency contracts: `50 passed`.
- Agent isolation/time-semantics residuals: `8 passed`.
- Pi workflow/round-trip focused selection: `69 passed`.
- Cross-DB shell/tab owner and accessibility contracts: `12 passed`.
- Cross-DB source/job/study-context continuity checks: `20 + 3 + 2 passed`.
- JavaScript syntax checks passed for `app.js`, `screens-guided-pi.js`, and `screens-viz-crossdb-results.js`.
- Ruff passed on the touched Python/test set; targeted `git diff --check` passed.
- Exact-snapshot browser QA: Patient route ended at `MAIN[tabindex=-1]`, title `患者审阅 — EasyICU`, live announcement `患者审阅`, no Google font links, and zero console errors/warnings. The temporary server and browser tab were closed.

## Honest remaining gates

1. `src/easyicu/data/concept-dict.LOCK.json` remains `finalized=false`; closing it requires a new six-database native-v2 extraction, QC, exact hash update, and deliberate seal.
2. New production owner modules such as `content_identity.py`, `databases/detection.py`, `planning/literature_bindings.py`, and other concurrent-lane owners remain untracked. A clean checkout would omit them. The shared workspace must be committed atomically after the active lanes agree on scope; this side task did not stage or commit another lane's work.
3. No full exact-head CI or formal Canonical9 run was started. Those remain freeze/merge/formal-experiment checkpoint actions.
