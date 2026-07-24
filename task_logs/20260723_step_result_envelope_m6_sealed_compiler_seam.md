# StepResultEnvelope M6-A — final/resume sealed compiler seam

## Scope

- Task: `AGENT-STEP-RESULT-ENVELOPE-CONVERGENCE`
- Baseline: `2947ec6`
- Implementation: `f59e75f69f0b4d5e07b08ce7d8de522526e904f8`
- Branch: `refactor/agent-control-plane`
- Safety boundary: no Provider, Docker, patient-data extraction, Canonical9 run,
  authority issuance, merge, or push.

This increment adds exactly one canonical `StepResultEnvelope` compiler call to
the shared final deterministic gate evaluator. Fresh execution and resume
revalidation already use that evaluator, so both paths compile once at the
post-repair, pre-publication boundary. Early/draft validation stays on the
legacy path.

The compiled snapshot remains `shadow=true` and `paper_authorized=false`. It is
returned alongside final gate findings but is excluded from `all_findings()`;
no Validator, Writer, readiness, scorer, Jury, or publication decision consumes
it in this increment.

## Guarantees

1. Compiler exceptions and digest failures return typed failures without
   falling back to an unsealed result.
2. Host and container input paths are projected only from host-resolved typed
   bindings into opaque `evidence:<id>@sha256:<digest>` references.
3. Binding path structure, run-root containment, exact relative/absolute path
   agreement, regular-file status, and symlink rejection are checked at the
   seam.
4. Large inputs are not rehashed by the normalizer. Byte-level verification
   remains owned by the existing final integrity validators, avoiding duplicate
   cohort I/O.
5. The raw `step_summary.json` bytes and output artifact bytes are not mutated.

## Tests

Focused final/resume/compiler matrix:

```bash
.venv/bin/python -m pytest -q \
  tests/research_agent/test_step_result_envelope.py \
  tests/research_agent/test_execution_phase_contract.py \
  tests/research_agent/test_resume_revalidation.py
```

Result: `144 passed`.

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

Result before the final no-rehash regression was added: `458 passed`; the added
regression then passed independently. Ruff, Black check, `py_compile`,
`git diff --check`, and the zero-cycle module graph all passed.

The architecture and resource baseline commands remain red for the branch's
already-recorded lower-is-better/source drift. The baseline was not rewritten:
`arch_measure` reports the same 12 open regressions; the current increment adds
13 lines and two import edges to `execution/phase.py`, while the new 115-line
compiler lives outside that historical target list.

## Archived diagnostic replay

Read-only replay used the exact archived development runs:

- E1: `run_20260723T211020_5733af` — 8/8 snapshots ready.
- E2: `run_20260723T235937_f4d63c` — 9/9 snapshots ready.

An initial replay without resolved-input projection correctly found
`absolute_unbound_path` on the E1/E2 steps that named upstream cohorts. Replaying
with the evidence ledger's exact resolved binding IDs, paths, and digests
produced 17/17 ready snapshots and zero normalization errors. No archived files
were modified and these diagnostic artifacts remain non-paper-authoritative.

## Honest size/status

This is an additive compiler seam: four files, `+310/-1` in the implementation
commit, including 194 test lines and a 115-line new production module. It does
not itself close repair proliferation or delete a legacy parser. The next
increment must switch only the final/resume fraction consumer to the sealed
snapshot with exact dual-read comparison and fail-closed compiler/error
handling; early validation must remain legacy until it has a separate draft
contract.
