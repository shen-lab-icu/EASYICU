# Pi Copilot V1 freeze gate closure

- Date: 2026-08-08
- Module: web
- Task: `PI-COPILOT-V1-FREEZE-GATE`
- Branch: `feat/pi-copilot-shell`
- Starting commit: `a625509`
- Implementation commit: `6c3ec98`
- Review baseline: `df19b2bc1f86620a5c969fe648d41db2655b5349`
- Review input: `/Users/haibo/.codex/attachments/ec8cbfd8-91e8-42fc-ae53-e0a6e8398c10/pasted-text.txt`

## Outcome

The last confirmed Pi V1 P1 is closed: exact scientific paths longer than 4096 characters now fail closed instead of being silently truncated. The review's runtime-integrity performance concern was also confirmed by measurement and fixed without weakening the manifest: the complete private-runtime hash is verified once per gateway/sidecar lifetime, cached for ordinary status/session paths, and invalidated on close or unexpected sidecar exit.

No Pi scientific authority, filesystem authority, provider scope, or UI capability was expanded.

## Exact-path fail-closed contract

- `guided_sessions` owns a typed `GuidedProjectMigrationError` with stable code `guided_project_path_too_long`, precise field, and `max_length=4096` details.
- New Guided slot writes reject an overlong `active_export.path` or `extraction.export_dir` before persistence.
- Historical project migration maps the owner error to a 409 `PiCopilotError`; it does not create a StudyContext and does not write a ProjectAuthority binding.
- Internal path characters remain exact. Only leading/trailing whitespace is removed; paths at or below the limit are not normalized as prose.

## Runtime-integrity performance contract

The review was correct that the full private install scan was too expensive for a high-frequency path:

- Direct 11,044-file hash, cold: 2.30-2.75 s on the available macOS filesystem.
- Direct 11,044-file hash, warm: 0.63-0.73 s.
- Fresh current-source private install: 239 audited packages, 0 vulnerabilities.
- First gateway status on that private install: 794.0 ms, integrity true.
- Later status calls in the same gateway lifetime: 12.4 ms and 11.7 ms.
- Status after `close()` invalidated the receipt and performed a fresh full check: 732.2 ms, integrity true.

The cache lives only in the `PiGatewayClient` instance. Packaged development-runtime checks retain their existing cheap behavior. Runtime reinstall, gateway restart, sidecar close, or unexpected sidecar exit requires a new full private-runtime verification.

## Adjacent release-gate repair

The 203-test adjacency run exposed one pre-existing current-branch red: the Research artifact preview route added in `102ee52` was missing from the route-owner snapshot. The snapshot now records that route; no production routing behavior changed for this repair.

## Verification

- Focused contract/gateway/route owner gate: 68 passed.
- Complete Pi + route/security/static adjacency: 203 passed.
- Ruff: passed.
- `git diff --check`: passed.
- Real current-source private install and cold/warm integrity benchmark: passed.
- Real browser after server restart: existing project restored, Research conversation and clickable Table 1 artifact restored, readable preview showed denominator 140 and group rows 140/125/15.
- Browser viewport 1280×720: document width 1280/1280 and no visible non-scroll-owner horizontal overflow.

## Freeze boundary

Pi Copilot V1 framework hardening is locally ready to freeze. Remote current-SHA CI is not claimed because `6c3ec98` has not been pushed. The next product work should use real research-task feedback and the already documented explicit-follow-up/background-job UX; it should not reopen generic Pi shell authority.
