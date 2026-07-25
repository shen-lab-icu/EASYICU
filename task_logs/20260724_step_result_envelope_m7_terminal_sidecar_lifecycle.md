# StepResultEnvelope M7 — terminal rebinding + transactional sidecar + unique loader

## Scope

- Task: `AGENT-STEP-RESULT-ENVELOPE-CONVERGENCE`
- Baseline: `acef706` (implementation baseline `334fbb7`)
- Commits: `e10e74a` (rebinding primitive) → `922eb3e` (sidecar writer + loader)
  → `c9671ff` (live wiring on the step success path)
- Branch: `refactor/agent-control-plane` (working tree clean, ahead of origin
  by 3, not pushed)
- Safety boundary: no Provider, Luna, Docker, extraction, patient-data
  mutation, Canonical9 run, authority issuance, merge, or push. No repair-cap
  increase, no evidence-gate loosening, no new single-question
  variables/prompts/repairs, no baseline rewrite. Large typed input is not
  re-hashed.

This increment establishes a **unique envelope-authority lifecycle** that a
future completed-step consumer can adopt, and live-wires only the *producer*
(and its loader) into the real success path. It does **not** switch a second
verdict consumer: Writer, readiness, scorer, Jury, figure, source-data, and
registered-output authority are untouched, and `percentage_identity.py` is
kept.

## What the three commits do

### 1. Terminal-status rebinding primitive (`e10e74a`)

`result_envelope.rebind_step_result_status(envelope, *, status)` is a pure
digest owner. The sealed compiler reads the step output once; after the Critic,
deterministic findings, and the seal recheck decide the final step status, the
snapshot's envelope is rebound to that terminal status. Rebinding only
recomputes the content digest — it never re-reads cohort/CSV/JSON/patient data.
It fails closed on a non-string/empty status, an original envelope whose digest
does not already verify, or a rebound envelope that fails self-verification
(status tampering).

### 2. Sealed sidecar writer + loader (`922eb3e`)

`execution/envelope_sidecar.py` (new, self-contained module):

- `prepare_step_result_envelope_sidecar(...)` — pure; rebinds the snapshot
  envelope to the terminal status and builds a canonical payload + metadata
  binding **step_id / attempt_id / checkpoint_id / schema version / content
  SHA / source-summary SHA / script evidence id / terminal status**, with
  `paper_authorized=false` stamped explicitly. Returns `None` (fail-closed,
  publishes nothing) when the snapshot is absent, the terminal status is not
  `ok`, a required coordinate is missing, the step identity disagrees, or the
  rebind fails. Performs **no filesystem reads**.
- `publish_step_result_envelope_sidecar(...)` — registers the payload via
  `register_text(kind="log", producer="step_result_envelope_sidecar",
  publish_aliases=False)`, so the alias sits in `pending_success_aliases` and
  is promoted only by the existing `StepEvidenceCommit` transaction. The
  sidecar therefore lives **outside** the raw step outputs and rides the
  **existing** success transaction — no second transaction is introduced.
- `load_current_step_result_envelope_sidecar(...)` — the single recovery path
  for both fresh and resume. It resolves **only** via
  `evidence_store.aliases().get(alias)` — never `EvidenceStore.get`'s fuzzy
  prefix fallback — then validates record kind, producer, `produced_by_step`,
  metadata schema/step/script/terminal-status/paper-authority/attempt/
  checkpoint, rejects a symlinked path, verifies the on-disk bytes' SHA against
  the record, re-parses and self-verifies the envelope (digest/step/status/
  schema/content/source), and re-derives the exact evidence id. Any mismatch
  returns a typed `StepResultEnvelopeSidecarUnavailable(reason)`; it never
  fabricates authority.

### 3. Live wiring on the success path (`c9671ff`)

In `_execute_one_step`, after the terminal status is confirmed `ok` **and**
quarantine cleanup succeeds, **before** the existing `StepEvidenceCommit`, a
single call to `publish_terminal_step_result_envelope_sidecar(...)` (a helper
in the sidecar module, extracted to keep the god-function footprint minimal)
prepares and registers the sidecar with `publish_aliases=False`. Because it
rides the existing transaction: a rolled-back commit leaves an *unpublished*
evidence record that the loader — resolving only through `aliases()` — can
never recover as current authority.

## Verification

All runs use the repo `.venv/bin/python` (Python 3.13.5).

Focused core matrix:

```bash
.venv/bin/python -m pytest -q \
  tests/research_agent/test_step_result_envelope.py \
  tests/research_agent/test_step_result_envelope_sidecar.py \
  tests/research_agent/test_execution_phase_contract.py \
  tests/research_agent/test_resume_revalidation.py
```

Result: `183 passed`.

Comprehensive adjacency + sidecar + golden:

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
  tests/research_agent/test_char_golden_run_bundle.py
```

Result: `501 passed, 83 warnings`.

Evidence store / registration / strict / authority:

```bash
.venv/bin/python -m pytest -qq \
  tests/research_agent/test_evidence.py \
  tests/research_agent/test_evidence_registration.py \
  tests/research_agent/test_evidence_strict.py \
  tests/research_agent/test_char_evidence_authority.py
```

Result: `63 passed`.

Static gates: Ruff `All checks passed`, Black `3 files would be left unchanged`
(the Python 3.13/3.15 parse warning is a pre-existing environment quirk, not a
failure), `py_compile` OK, `git diff --check acef706..HEAD` clean, module graph
`test_production_research_agent_import_graph_is_acyclic` passed (11 passed).

## Guarantee → test mapping

Terminal rebinding (`tests/research_agent/test_step_result_envelope.py`):
- rebinds status and recomputes digest —
  `test_rebind_step_result_status_rebinds_status_and_recomputes_digest`
- fail-close on empty/blank status (3 params) —
  `test_rebind_step_result_status_rejects_empty_status`
- fail-close on invalid original digest —
  `test_rebind_step_result_status_rejects_invalid_original_digest`
- fail-close on hand-tampered status —
  `test_rebind_step_result_status_rejects_hand_tampered_status`

Sidecar producer + transactional publish + unique loader
(`tests/research_agent/test_step_result_envelope_sidecar.py`):
- full metadata binding on `ok` — `test_prepare_binds_terminal_status_and_full_metadata`
- prepare fail-close without publishing (7 params) — `test_prepare_fails_closed_without_publishing`
- step-identity disagreement — `test_prepare_rejects_step_identity_disagreement`
- prepare does no filesystem reads — `test_prepare_performs_no_filesystem_reads`
- committed sidecar recoverable, outside raw outputs — `test_committed_sidecar_is_recoverable_and_outside_raw_outputs`
- fresh and resume use the same loader — `test_fresh_and_resume_use_the_same_loader`
- helper chains prepare+publish — `test_publish_terminal_helper_chains_prepare_and_publish`
- helper publishes nothing when fail-closed — `test_publish_terminal_helper_publishes_nothing_when_fail_closed`
- uncommitted alias not current authority — `test_uncommitted_sidecar_is_not_current_authority`
- rolled-back commit leaves no current authority — `test_rolled_back_commit_leaves_no_current_authority`
- legacy store not auto-promoted — `test_legacy_store_without_sidecar_is_not_auto_promoted`
- loader negatives — stale attempt / wrong checkpoint / wrong script / wrong
  step / non-successful status / foreign record kind / wrong producer / wrong
  sidecar schema / paper-authority metadata / tampered bytes / symlink /
  internally inconsistent envelope —
  `test_loader_rejects_{stale_attempt,wrong_checkpoint,wrong_script_binding,
  non_successful_terminal_status,wrong_step,foreign_record_under_sidecar_alias,
  wrong_producer,wrong_sidecar_schema,paper_authority_metadata,tampered_bytes,
  symlinked_sidecar,internally_inconsistent_envelope}`
- live `_execute_one_step` success path publishes recoverable sidecar —
  `test_live_success_path_publishes_recoverable_sidecar`
- live path fail-close on stale attempt / missing step —
  `test_live_sidecar_fails_closed_on_stale_attempt_and_missing_step`
- live path fail-close on tampered bytes —
  `test_live_sidecar_fails_closed_on_tampered_bytes`

## Honest size/status

Production is `+517/-0` net across three files:
`execution/envelope_sidecar.py` +453 (new), `execution/result_envelope.py`
+35, `execution/phase.py` +29 (~28 of which land inside the already-over-budget
`_execute_one_step` god function via the minimized live-wire block, +1 import).
Tests/fixtures are `+890/-5`, including 26 new sidecar tests, 4 new rebinding
test functions (6 test ids), and 3 live integration tests that reuse the
existing trajectory-stability success harness.

Architecture: the branch's regressed-metric **count is 12 at both `acef706` and
HEAD** — freshly re-confirmed this session via a throwaway worktree at
`acef706`. My three commits increase the *magnitude* of already-red phase.py /
`_execute_one_step` LOC metrics but push **no** previously-green metric over its
threshold, i.e. **zero new regressions**. No baseline file was modified. The
resource baseline's pre-existing offline-envelope drift is likewise unchanged.

The golden bundle fixture moved additively only: current_evidence 34→38,
current_aliases 36→40, current_self_aliases 34→38 — exactly four sidecar
records/aliases, one per `ok` step; every other bundle key is byte-identical
(proven in the prior session by extracting the raw records/aliases, confirming
removal of the four reproduces the pre-change SHAs, then regenerating the
fixture from the proven bundle).

The snapshot remains `shadow=true` / `paper_authorized=false`. This is the
producer + loader lifecycle only. Registered-output, Writer, readiness, scorer,
Jury, nine-question, and paper authority are **not** claimed complete and were
not switched. Paper-facing status stays **0/9**. Next work should switch one
registered-output consumer onto this loader in a separate reviewed patch after
observing the boundary — not broaden to multiple consumers at once.
