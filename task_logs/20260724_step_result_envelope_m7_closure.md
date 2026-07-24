# StepResultEnvelope M7 closure — attempt-bound identity, required query, descriptor path, ok-fail-close

## Scope

- Task: `AGENT-STEP-RESULT-ENVELOPE-CONVERGENCE`
- Baseline: `b31b1ee` (the M7 producer/loader lifecycle documentation commit)
- Closure commit: `b8460af`
- Branch: `refactor/agent-control-plane`
- Push state: the four M7 commits `e10e74a` / `922eb3e` / `c9671ff` / `b31b1ee`
  were pushed to `origin/refactor/agent-control-plane` **by the reviewer/user,
  not by this session**; `origin` now points at `b31b1ee`.  The closure commit
  `b8460af` is **not** pushed (working tree clean, ahead of origin by 1).
- Safety boundary unchanged: no Provider, Luna, Docker, extraction,
  patient-data mutation, Canonical9 run, authority issuance, merge, or push.
  No repair-cap increase, no evidence-gate loosening, no second verdict
  consumer switched, no baseline rewrite, `percentage_identity.py` retained.

This is a single closure increment on top of `b31b1ee` — no history rewrite —
that hardens the M7 sidecar lifecycle along six axes the review named. It does
**not** connect a second consumer: registered-output, Writer, readiness,
scorer, Jury, and figure/source-data remain untouched, the snapshot stays
`shadow=true` / `paper_authorized=false`, and paper-facing status is still
**0/9**.

## The six fixes

1. **Attempt-bound evidence identity (schema `/2`).** `_sidecar_evidence_id`
   now hashes `step_id + attempt_id + checkpoint_id + script_evidence_id +
   content_sha256`. A second successful attempt of the same step/script/content
   is therefore a **new** record, not a dedupe. The step-scoped alias re-points
   to the latest attempt (`EvidenceStore.publish_step_success_aliases`
   overwrites the current pointer — verified directly), so a query naming the
   earlier attempt is stale and only the current attempt is recoverable.
   `SIDECAR_SCHEMA_VERSION` is bumped to `easyicu.step_result_envelope_sidecar/2`.

2. **Required, non-empty query coordinates.**
   `StepResultEnvelopeSidecarQuery.attempt_id` / `checkpoint_id` are now required
   (no default) and validated non-empty in `__post_init__`; the loader always
   compares them (the old `is not None` bypass is removed). A caller can no
   longer omit them to sidestep the current-attempt binding.

3. **Descriptor-anchored artifact resolution.** The loader resolves the on-disk
   sidecar through the shared `verified_run_evidence_path(evidence_store.root,
   record)` guard, which rejects — in one check — a parent-directory symlink, a
   `..` / absolute / escaped `relative_path`, a final symlink, a non-regular
   file, and a digest mismatch. The prior bare `Path.is_symlink()` +
   hand-rolled digest comparison is gone; all path/tamper failures collapse to
   the single reason `artifact_path_unverified`.

4. **`status==ok` fails closed if the sidecar cannot be sealed.** In
   `_execute_one_step`, a missing snapshot, a fail-closed prepare, or a
   registration error on a successful step now converts the step to a typed
   `result_envelope_sidecar` `contract_failed` finding (`severity="error"`)
   instead of committing silently. The step never reaches `StepEvidenceCommit`,
   so no alias promotes. Non-`ok` steps still publish no sidecar.

5. **Real negative regressions added** (all against the real `EvidenceStore` /
   `StepEvidenceCommit`, mock LLM + in-process runner; no Provider/Docker):
   two identical-content successive attempts, a rolled-back third attempt, an
   omitted/empty attempt or checkpoint, a parent-directory symlink, a
   `relative_path` traversal, and the live `ok` path failing closed on both a
   `None`-return and a registration error.

6. **Docs** — this log + `项目进度/agent/CURRENT.md` + `项目进度/README.md`
   updated to the actual HEAD/origin/push state; no claim that a second
   consumer is connected.

## Verification

All runs use the repo `.venv/bin/python` (Python 3.13.5).

Focused core matrix (was 183 at M7):

```bash
.venv/bin/python -m pytest -q \
  tests/research_agent/test_step_result_envelope.py \
  tests/research_agent/test_step_result_envelope_sidecar.py \
  tests/research_agent/test_execution_phase_contract.py \
  tests/research_agent/test_resume_revalidation.py
```

Result: `190 passed` (+7 new sidecar tests).

Full adjacency + sidecar + golden + evidence:

```bash
.venv/bin/python -m pytest -qq \
  tests/research_agent/test_step_result_envelope.py \
  tests/research_agent/test_step_result_envelope_sidecar.py \
  tests/research_agent/test_execution_phase_contract.py \
  tests/research_agent/test_resume_revalidation.py \
  tests/research_agent/test_anti_pipeline_robustness.py \
  tests/research_agent/test_step_summary_integrity.py \
  tests/research_agent/test_trajectory_stability_pipeline_success.py \
  tests/research_agent/test_trajectory_stability_pipeline_terminal.py \
  tests/research_agent/test_validators.py \
  tests/research_agent/test_bound_percentage_identity_repair.py \
  tests/research_agent/test_char_golden_run_bundle.py \
  tests/research_agent/test_evidence.py \
  tests/research_agent/test_evidence_registration.py \
  tests/research_agent/test_evidence_strict.py \
  tests/research_agent/test_char_evidence_authority.py
```

Result: `571 passed, 83 warnings`.

Static gates: Ruff `All checks passed`, Black clean (after formatting the test
file), `git diff --check` clean, module graph
`test_production_research_agent_import_graph_is_acyclic` passed — the new
`execution.envelope_sidecar -> authority.runtime_artifacts` import edge does
**not** create a cycle (`execution` already depends on `authority`).

## Guarantee → test mapping (closure)

- Fix 1 — attempt-bound identity, re-point, stale/recoverable:
  `test_second_successful_attempt_supersedes_first_as_current`
- Fix 1 — schema `/2` round-trips prepare→publish→load:
  the refreshed positive `test_committed_sidecar_is_recoverable_and_outside_raw_outputs`
  (+ every existing loader test, now on `/2`)
- Fix 2 — required, non-empty query coordinates:
  `test_query_requires_nonempty_attempt_and_checkpoint`
- Fix 3 — descriptor-anchored path: `test_loader_rejects_symlinked_sidecar`,
  `test_loader_rejects_parent_directory_symlink`,
  `test_loader_rejects_relative_path_traversal`,
  `test_loader_rejects_tampered_bytes`,
  `test_live_sidecar_fails_closed_on_tampered_bytes`
  (all now assert `artifact_path_unverified`)
- Fix 4 — `status==ok` fail-close (both None-return and registration error):
  `test_live_ok_step_fails_closed_when_sidecar_cannot_publish[return_none-...]`
  and `[raise-...]`
- Fix 5 — rolled-back third attempt keeps the second current:
  `test_rolled_back_third_attempt_keeps_second_as_current`

## Honest size / status

Closure production is `+140/-39` net across two files
(`execution/envelope_sidecar.py` +81/-27, `execution/phase.py` +59/-12; net
`+101`). The 39 deletions are lines this closure rewrote that earlier M7
commits had added, so the endpoint diff `acef706..HEAD` collapses to `+618/-0`
production (net `+618`); the closure's own churn is the `+140/-39` above.
Cumulative M7 tests/fixtures over the same range are `+1150/-5`.

Architecture: the branch's regressed-metric **count is 12 at both `acef706`
and HEAD** (freshly re-confirmed this session via a throwaway worktree at
`acef706`). This closure enlarges the magnitude of the already-red
`phase.py` / `_execute_one_step` LOC metrics but introduces **no** new
regressed metric; no baseline file was modified. The golden bundle fixture is
**byte-unchanged** — the sidecar evidence ids are volatile-normalized and the
record count is stable at four (one per `ok` step), so the schema `/2` + new
identity did not alter the characterized surface.

Still only the producer + loader lifecycle. Registered-output, Writer,
readiness, scorer, Jury, nine-question, and paper authority are **not** claimed
complete and were not switched. Next work: switch **one** registered-output
consumer onto this loader in a separate reviewed patch after observing the
boundary — not multiple consumers at once.
