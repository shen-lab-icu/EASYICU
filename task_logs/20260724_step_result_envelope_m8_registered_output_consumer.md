# StepResultEnvelope M8 — first real registered-output consumer migration

## Scope

- Task: `AGENT-STEP-RESULT-ENVELOPE-CONVERGENCE` (Phase A of the收尾 plan)
- Baseline: `7f8b7ad` (M7 closure)
- Commits this turn: `37c77d2` (M8-A loader hardening) → `82631c4` (M8-B1
  consumer, pure/unwired)
- Branch: `refactor/agent-control-plane`; not pushed by this session.
- Safety boundary unchanged: no Provider, Luna, Docker, extraction,
  patient-data mutation, Canonical9 run, authority issuance, merge, or push.
  No repair-cap increase, no evidence-gate loosening.

M8 migrates the FIRST real downstream consumer onto the M7 sidecar loader: the
cross-step registered-output validator. Delivered this turn in two verified
slices; the live final-gate wiring (B2) is deliberately held as its own slice
because it touches both the main and the resume final-gate paths.

## M8-A — loader TOCTOU / read-error hardening (`37c77d2`)

`load_current_step_result_envelope_sidecar` verified the file via
`verified_run_evidence_path` (which digests it with `.open`) but then re-read it
with `read_bytes` to parse. That second read is now treated as untrusted: an
`OSError` returns `artifact_read_failed`, and the re-read bytes are re-hashed
against `record.sha256` so a payload swapped in the TOCTOU window returns
`artifact_digest_mismatch` instead of being parsed as authority. Tests:
`test_loader_returns_typed_unavailable_on_read_oserror`,
`test_loader_rejects_bytes_changed_after_verification`.

## M8-B1 — envelope-authoritative consumer, pure and unwired (`82631c4`)

`RegisteredOutputEnvelopeConsumer` replaces the DEAD M3 registered-output
shadow scaffold (only the offline replay tool referenced it; the live pipeline
never wired it). Upstream table presence is read ONLY from the canonical
`StepResultEnvelope` recovered through the sidecar loader — never a raw
`evidence_ids` / `output_files` glob, and never an envelope-or-legacy choice.

Two lanes, decided per upstream step by the step's own record (resume-safe):

- **modern run** — the record's `evidence_ids` declare a sidecar
  (`step_record_declares_sidecar`). The canonical envelope must load and
  self-verify; a missing / stale / coordinate-drifted / tampered / unreadable
  sidecar is a typed fail-close (`registered_output_sidecar_unrecoverable`).
- **legacy archived run** — no sidecar was ever declared. The legacy raw parse
  runs in an explicit diagnostic lane; findings are `diagnostic_only` with
  `paper_authority=False`, never paper authority.

The resume-safety signal is deliberately the per-record `evidence_ids` list
(preserved across resume), not a fragile run-level store scan — so a modern
run's broken sidecar fails closed while a resumed run whose upstream records
simply never had sidecars stays in the diagnostic lane.

Deleted: `compare_registered_output_shadow`,
`RegisteredOutputShadowComparison`, `registered_output_shadow_blocking_finding`
(envelope_shadow.py nets **-78** lines). Added the pure
`canonical_registered_output_table_artifacts` helper. No parallel parser is
introduced — the consumer inherits the single legacy `_availability_blocks` /
`_table_artifacts` parse for the diagnostic lane.

The offline replay tool (`tools/replay_step_result_envelopes.py`) migrates to
the consumer's **detection-equivalence** check (it must flag the same upstream
steps as the legacy validator; the diagnostic lane adds provenance detail so a
byte comparison no longer applies) and falls back to an empty read-only
evidence view for anchor-less archived runs.

### Tests (all real, non-zero, against a real EvidenceStore + StepEvidenceCommit)

- `test_canonical_helper_reports_upstream_table_presence`
- `test_registered_output_consumer_matches_legacy_on_real_hit` — a real
  registered-output claim: canonical verdict is a NON-ZERO finding agreeing
  with the legacy verdict (not zero-finding equivalence)
- `test_registered_output_consumer_fails_closed_on_missing_sidecar`
- `test_registered_output_consumer_fails_closed_on_stale_attempt`
- `test_registered_output_consumer_fails_closed_on_incomplete_coordinates`
- `test_registered_output_consumer_fails_closed_on_tampered_sidecar`
- `test_registered_output_consumer_ignores_failed_upstream_step`
- `test_registered_output_consumer_uses_legacy_diagnostic_lane_for_archived_run`

## Verification

```bash
.venv/bin/python -m pytest -q \
  tests/research_agent/test_step_result_envelope.py \
  tests/research_agent/test_step_result_envelope_sidecar.py
```
→ `99 passed`.

Adjacency (envelope + sidecar + validators + phase-contract + resume + golden +
anti-pipeline + step-summary): `502 passed`. Ruff, Black, `git diff --check`,
and the module graph (`test_production_research_agent_import_graph_is_acyclic`)
all pass — the new `audits.envelope_consumers → execution.envelope_sidecar`
import edge is acyclic (audits already depends on execution).

## Honest size / status

M8-A production `+19/-6`; M8-B1 production `+213/-142` (net `+71`;
envelope_shadow.py is `-78`). The consumer is larger than the deleted dual-read
wrapper because it is now real two-lane authority with fail-close, not a shadow
comparator; no parallel parser was added.

**NOT done, explicitly:** the consumer is NOT yet wired into the live final
gate (B2). The migration is therefore not consumer-complete: Writer, readiness,
scorer, Jury, figure/source-data are untouched; the snapshot stays
`shadow=true` / `paper_authorized=false`; paper-facing status is **0/9**. Phase
B (offline E1/E2 replay + development Docker image) and Phase C (real Luna E1
development canary) have NOT started — they follow B2 landing and are the real
patient-data / Provider work, gated on the supervisor reviewing Phase A.

Next: **M8-B2** — wire `RegisteredOutputEnvelopeConsumer` into
`_evaluate_final_deterministic_gates` (both call sites), construct the read-only
evidence view safely for the resume path, update the characterization test
(`test_sealed_envelope_wires_only_the_final_fraction_consumer`), and re-verify
golden + adjacency.
