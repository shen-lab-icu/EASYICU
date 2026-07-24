# Control-plane review — REQUEST_CHANGES blocker fixes

## Scope

An independent review of branch head `37c77d2` (control plane, execution
isolation, recovery authority, config entry, concurrency, CI) returned
`REQUEST_CHANGES` with 1 P0, 3 P1, 2 P2, and 1 architecture-debt note. Each
finding was verified against the actual source before fixing (the review was
static-only — it could not resolve GitHub DNS to run pytest). All six were
real; no reviewer hallucinations this round.

This log covers the **four merge-blockers** (P0 + three P1). The two P2s
(`as_kwargs` deep-copy, CI path-filter/pyarrow/ruff) and the ~11k-line
`execution/phase.py` debt are tracked follow-ups, not merge-blockers.

Safety boundary unchanged: no Provider, Luna, Docker, patient data, extraction,
or Canonical9 run; no authority issuance, merge, or evidence-gate loosening.
Every fix is pure code + unit tests. All four TIGHTEN security.

## Fixes

### P0 — `7d98197` CodeRunner control-file symlink traversal

`CodeRunner` wrote `analysis.py` and `run.log` with raw `Path.write_text`. The
macOS sandbox grants generated code write access to the whole step dir, and the
step dir is reused across repair attempts, so a prior attempt could leave either
control file as a symlink pointing at a host file outside the sandbox; the
host's next write would follow the link and clobber the victim — a
write-through-symlink escape. All three CodeRunner control-file writes now route
through the existing atomic, symlink-safe `_replace_regular_file_atomically`
(already used for the sibling `.run_artifact_authority_snapshot.json`):
`os.replace` onto a symlinked path overwrites the link with a fresh,
exclusively-created, single-hardlink regular file instead of writing through it.
No new utility introduced; DockerRunner already had its own guard.
Regression: `test_code_runner_control_writes_never_follow_planted_symlinks`
(both control files pre-planted as symlinks → victim bytes survive, both become
real single-link regular files). Proven non-vacuous: old `write_text` clobbers
the victim; `os.replace` leaves it intact.

### P1-a — `147bb6f` non-bool `allow_unsafe_host_fallback` coercion

`self.allow_unsafe_host_fallback = bool(value)` made `bool("false")` → `True`,
so a quoted YAML/TOML/env/JSON value routed through
`PipelineConfig.runner_kwargs` would silently enable unsafe host execution. The
flag now accepts only `True`/`False`/`None`; any other type raises `TypeError`.
The `None` path (env `EASYICU_ALLOW_UNSAFE_HOST_FALLBACK` via `_env_flag`) is
unchanged. Regression: `test_code_runner_rejects_non_bool_unsafe_host_fallback`
parametrized over `"false"/"0"/"no"/0/1`.

### P1-b — `2b37234` run-input capsule-digest TOCTOU

`_verified_run_input_capsule_digest` hashed the working copy on read #1, re-read
it on read #2 for the sealed byte comparison, and returned the read-#1 digest —
a swap between the two reads could return a digest that never matched the
compared bytes, and an `OSError` from the reread escaped the
`RunInputIdentityError` boundary. Both copies are now read exactly once into
memory; the digest, the record check, and the sealed byte-equality check all
derive from those buffers, and read failures convert to `RunInputIdentityError`.
Mirrors the M8-A envelope-sidecar loader hardening (`37c77d2`). Regression:
`test_run_input_capsule_read_error_is_typed`.

### P1-c — `1e6ce7b` timeout reaps the whole process group

`subprocess.run(timeout=...)` sends the timeout kill only to the direct child,
so a background process double-forked by generated code survived and could keep
mutating step outputs after evidence collection (and, on the opt-in unsafe
host-fallback path, run with host access). All six CodeRunner generated-code
execution sites now route through `_run_capturing_with_descendant_reaping`,
which on POSIX launches the child in a new session (`start_new_session=True`)
and `killpg(group, SIGKILL)`s on timeout before re-raising `TimeoutExpired` with
the captured partial output (so the existing bytes-safe timeout handler is
unchanged). Non-POSIX keeps plain `subprocess.run`. Trusted interpreter
capability probes and DockerRunner are untouched. Test migration: the CodeRunner
tests that monkeypatched `subprocess.run` now patch the helper (mechanical:
signature `(cmd, *, cwd, env, timeout)`, target repointed). Two new regressions:
`test_run_capturing_reaps_whole_process_group_on_timeout` (start_new_session +
killpg(pgid, SIGKILL) fire, partial output preserved) and
`test_run_capturing_returns_completed_process_for_real_command` (real Popen
happy path).

## Verification

```bash
.venv/bin/python -m pytest -q \
  tests/research_agent/test_runner.py \
  tests/research_agent/test_runner_timeout_bytes.py \
  tests/research_agent/test_execution_phase_contract.py
```
Runner + timeout: 46 passed. Phase-contract (incl. capsule): 121 passed in the
combined envelope adjacency. Ruff, Black (`--target-version py310`), and
`git diff --check` clean on every changed file.

Production delta across the four commits: `runner.py` +symlink-guard routing,
strict flag check, and the reaping helper (net small); `phase.py` capsule read
`+17/-2`.

## Pre-existing state (NOT introduced by these fixes — bisect-confirmed)

- `tests/research_agent/test_resume.py` has ~20 failures whose root cause is the
  planner phase getting the generic `MOCK RESPONSE — no live LLM configured`
  instead of scripted JSON. These reproduce identically at `2b37234` (before
  P1-c) via `git stash` bisect, and touch the planner/graph/mock path, which
  none of these four fixes go near. Flagged for separate attention; not a
  blocker the review raised.
- The arch gate reports 12 regressions vs baseline — identical at `acef706` and
  HEAD (pre-existing branch drift, zero new), matching prior worktree review.

## Not done (explicit)

P2-a (`PipelineConfig.as_kwargs` deep-copies live runtime objects via `asdict`),
P2-b (`research_agent_ci.yml` path filter misses `pyproject.toml`/public
API/concept, pins `pyarrow>=14` vs project `>=23.0.0`, no ruff step, only
py3.10/3.11), and the `execution/phase.py` size debt remain as tracked
follow-ups. No PR opened from this environment (`gh` is not installed here);
origin already carries P0/P1-a/P1-b, HEAD is ahead by P1-c only.
