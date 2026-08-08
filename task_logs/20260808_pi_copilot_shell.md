# Pi Copilot Web-first vertical slice

Date: 2026-08-08

Branch: `feat/pi-copilot-shell`

Implementation commit: `da62b14`

Baseline: `a9610bf5dea50dfced80eba804959f8ba0e086f9`

## Outcome

EasyICU now has an opt-in Pi `AgentSession` shell inside Guided Copilot. Pi owns conversation history, streaming and its tool loop. EasyICU remains the owner of typed StudyContext, run submission, JobManager state, validation, evidence and publication gates.

The sidecar uses the official SDK package `@earendil-works/pi-coding-agent@0.84.1`, reviewed against upstream commit `9dd90a49711d088b86fdd9b4aea575913a8328a8`. The default shell provider is the local OpenAI-compatible endpoint `http://127.0.0.1:8317/v1` with model `gpt5.6 luna`; the credential is read only from `EASYICU_PI_API_KEY`.

## Implemented boundaries

- Added a long-lived Node sidecar with a strict versioned JSON-lines protocol and persistent Pi JSONL sessions.
- Disabled Pi built-in filesystem, edit, shell and discovery surfaces; registered exactly 15 EasyICU tools.
- Added FastAPI session/status/message/rebind/abort routes with strict request schemas.
- Added server-wide AI opt-in plus explicit per-session external-model opt-in.
- Added one-message `configure`, `run` and `cancel` grants held by FastAPI rather than model arguments.
- Added conversational StudyContext persistence through the existing typed owner; an authority revision change makes the chat stale until explicit rebind.
- Added PHI/path/credential-safe bounded projections and stable owner-attributable error codes.
- Kept full scientific provider runs, scientific crash-resume and replan fail-closed until their existing owners expose suitable public contracts.
- Added a route-owned Guided Pi UI and CSS; the legacy deterministic Guided flow remains a labelled local fallback.
- Packaged the Node manifest, lockfile, notices and entrypoint in wheel/sdist while excluding local `node_modules`.

The full decision record is `docs/pi_copilot_integration_architecture.md`.

## Verification

- Focused Web/Pi gate: `119 passed, 1 warning`.
- Ruff: all changed Python/tests passed.
- Node syntax/package check: passed.
- Import Linter: 7 contracts kept, 0 broken.
- deptry: no dependency issues.
- npm audit: 0 vulnerabilities across the installed production graph.
- Wheel and sdist: required Pi sidecar assets present; `node_modules` absent.
- User credential literal scan: absent from repository changes.
- Browser QA at 1440×900 and 1024×768: Pi ready/activation/legacy states render with zero horizontal overflow and zero console errors. No model prompt was sent.

The final repository-wide run collected 12,663 tests, but was stopped at 18% after repeated full-pipeline PDF generation made the run exceed the useful final gate. Its early Figure 2 failures all share the pre-existing root error `Figure 2 paper scorer tree digest mismatch`; this change modifies no `benchmarks/figure2_canonical9` file. The focused changed-domain gate remained fully green after the final bridge correction.

## Remaining work

- A fresh installed wheel needs `npm ci --ignore-scripts` inside the packaged `pi_copilot/node_app` directory before runtime status becomes ready; dependency installation is not performed implicitly by the Web server.
- Run one user-authorized local shell canary against the configured endpoint. Automated verification deliberately made no model/API calls.
- Decide phase 2 separately: a typed edit/execute/validate loop must reuse current Planner/Coder/evidence authority rather than give Pi direct code or filesystem mutation.
- Repair or intentionally reseal the Figure 2 paper-scorer tree in the benchmark workstream; it is outside this Web integration commit.

No remote push was performed.
