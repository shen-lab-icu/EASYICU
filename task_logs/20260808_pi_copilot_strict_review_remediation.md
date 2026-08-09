# Pi Copilot strict-review remediation

Date: 2026-08-08

Branch: `feat/pi-copilot-shell`

Baseline: `8f82a893bec29e834148df861488a167cc18c400`

Implementation commits:

- `314388b` — `fix(web): harden governed Pi Copilot boundary`
- `5f71666` — `fix(agent): use SciPy for subgroup Wald tail`

## Outcome

The four strict-review P1 findings are closed in code and focused regression
tests. Directly actionable runtime P2 findings are also closed: prompt-timeout
recovery, current-run binding semantics, private sidecar CWD, bounded JSONL
retention, shell token hard stop, hidden raw reasoning, 32-hex correlation
digests, Node-version validation, explicit fresh-wheel installation, and the
O24 SciPy chi-square survival function.

Pi remains a UX/session/tool-loop shell. It still has no generic filesystem,
shell, network, Planner, Coder, EvidenceStore, paper authorization, full
provider-run, scientific replan, or crash-resume authority.

## Review finding closure

1. **One-use grants:** FastAPI now owns `HostTurnGrant`; each action is consumed
   atomically once. A second same-action tool call returns
   `pi_action_grant_consumed`.
2. **Per-tool authority freshness:** every host tool call revalidates study
   revision, active job and current run. A successful setup/run/cancel mutation
   invalidates the rest of that turn until explicit rebind.
3. **Credential and filesystem isolation:** Node receives a strict basic
   runtime + `EASYICU_PI_*` environment allowlist, uses a private empty CWD, and
   requires Node `>=22.19.0`.
4. **Value-safe projections:** job labels/free-form reasons and full generic
   readiness objects are no longer projected. Owner-specific semantic
   allowlists plus a recursive value scanner cover the complete tool result,
   including summaries and authority values.
5. **Runtime hardening:** prompt timeout performs best-effort abort/state
   refresh; retention disposes and deletes only evicted `.jsonl` files proven
   inside the private session root; raw reasoning is forced off and not
   streamed; a session token budget blocks the next call before its reserved
   output would exceed the ceiling.
6. **Installability:** `easyicu copilot install` copies the packaged exact
   runtime into a private versioned directory and runs
   `npm ci --ignore-scripts` with an installer environment allowlist. Server
   startup never installs dependencies.
7. **O24 kernel:** subgroup interaction Wald tails now use
   `scipy.stats.chi2.sf`; the hand-written incomplete-gamma approximation was
   removed.

## Verification

- Web/Pi + adjacent route/static gate: **128 passed**.
- O24 subgroup-focused gate in canonical Python 3.11.15 environment:
  **6 passed / 8 deselected**.
- Ruff on changed Python/test owners: passed.
- Node syntax checks for sidecar and Guided Pi owner: passed.
- Import Linter: **7 kept / 0 broken**.
- deptry: no dependency issues.
- npm production audit: **0 vulnerabilities** across 238 dependencies.
- wheel + sdist: built successfully with `uv build`; installer, lockfile,
  sidecar entrypoint and notices present; `node_modules` absent.
- Browser QA at 1440×900 and 1024×768: zero horizontal overflow, zero console
  error/warning, no raw-thinking control or transcript marker.

The repository-wide suite was not rerun for this scoped review closure. The
canonical agent baseline immediately before this change was already
9855 passed / 13 skipped / 0 failed; this turn used changed-owner and boundary
gates instead of repeatedly paying the full pipeline/PDF cost.

## Real endpoint canary boundary

No model prompt was sent. The local endpoint is listening on port 8317, but the
WebApp/Codex process did not have `EASYICU_PI_API_KEY` injected. Browser QA
therefore correctly showed `api_key_configured` unavailable. The credential
was not copied from chat into a command, file, log, commit, or process listing.

The remaining product gate is one user-authorized canary after the user sets
`EASYICU_PI_API_KEY` in the WebApp launch environment and restarts it:

```text
inspect context -> inspect capability -> one configure -> stale/rebind
-> one preflight -> inspect run -> abort/reopen consistency
```

The canary must record tool order, verify a second same-action mutation is
blocked, and scan browser/model-visible results for paths, secrets and row
identifiers. Phase 2, direct code editing, full scientific provider runs,
artifact editors and session trees remain outside this task.
