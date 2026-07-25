# StepResultEnvelope M6-B — sealed final fraction consumer

## Scope

- Task: `AGENT-STEP-RESULT-ENVELOPE-CONVERGENCE`
- Baseline: `c02546d`
- Implementation: `334fbb7`
- Branch: `refactor/agent-control-plane`
- Safety boundary: no Provider, Docker, extraction, patient-data mutation,
  Canonical9 run, authority issuance, merge, or push.

This increment switches exactly one live decision surface: the bounded
fraction/percentage Validator at the shared final deterministic gate. Fresh
final execution and resume revalidation use the same gate; the early
pre-registration repair gate continues to use the legacy Validator.

The final gate still executes the legacy audit once, then requires its complete
finding payload to be byte-equivalent to the sealed envelope's canonical scalar
view. Exact agreement retains the legacy findings. Missing/failed compiler
output, invalid envelope digest, source/status drift, normalization errors,
invalid scalar reconstruction, or finding drift produces one blocking
`step_summary_fraction_scale` finding. There is no fallback that can silently
restore a legacy pass.

The snapshot remains `shadow=true` and `paper_authorized=false`; this consumer
switch does not grant paper authority or change Writer/readiness/scorer/Jury.
The pre-execution `percentage_identity.py` AST guard remains because it owns a
different failure stage.

## Verification

Focused compiler/final/resume matrix:

```bash
.venv/bin/python -m pytest -q \
  tests/research_agent/test_step_result_envelope.py \
  tests/research_agent/test_execution_phase_contract.py \
  tests/research_agent/test_resume_revalidation.py
```

Result: `145 passed`.

Expanded adjacent matrix:

```bash
.venv/bin/python -m pytest -qq \
  tests/research_agent/test_step_result_envelope.py \
  tests/research_agent/test_execution_phase_contract.py \
  tests/research_agent/test_resume_revalidation.py \
  tests/research_agent/test_anti_pipeline_robustness.py \
  tests/research_agent/test_step_summary_integrity.py \
  tests/research_agent/test_trajectory_stability_pipeline_success.py \
  tests/research_agent/test_trajectory_stability_pipeline_terminal.py \
  tests/research_agent/test_validators.py \
  tests/research_agent/test_bound_percentage_identity_repair.py
```

Result: `460 passed, 83 warnings`.

Ruff, Black check, `py_compile`, `git diff --check`, and the module graph all
passed. The branch's 12 previously recorded architecture/resource baseline
drifts remain open; no baseline file was changed.

## Archived E1/E2 replay

The repository replay tool read the exact archived development runs and wrote
only disposable shadow output under the external disk:

`/Volumes/外置硬盘/easyicu_data/diagnostics/envelope_m6b.vFlK5Z`

- E1 `run_20260723T211020_5733af`: 8 envelopes, 0 normalization errors,
  0 fraction mismatches.
- E2 `run_20260723T235937_f4d63c`: 9 envelopes, 0 normalization errors,
  0 fraction mismatches.

All 17 steps had `fraction_shadow_exact=true`. The corpus contains zero legacy
fraction findings, so this proves exact no-finding equivalence on the archived
runs, not a nonzero production hit. The adversarial unit corpus separately
locks 21 positive/negative shapes and source/status/digest/normalization drift.

## Honest size/status

Implementation commit: seven files, `+170/-24`; production is `+48/-12` and tests
are `+122/-12`. This is the first live envelope consumer, but it is still a
dual-read migration gate, not the final deletion step. It removes no
scientific rule and adds no task-specific vocabulary. Next work should observe
this boundary before switching a second consumer; do not broaden directly to
Writer, readiness, scorer, Jury, or registered-output authority in one patch.
