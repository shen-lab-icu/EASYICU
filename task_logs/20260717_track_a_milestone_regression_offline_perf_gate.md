# Track A milestone regression and offline performance gate

Date: 2026-07-17 EDT

Branch: `refactor/agent-control-plane`

Tested HEAD: `f8946b1ba97f`

Review base: `9b2fe21`

## Decision

The fixed-HEAD regression and offline performance/authority gate is green. The
same archived E3 run may now proceed to the online acceptance step, resuming
only `02_exposure_derivation_and_qc`. This log does **not** claim that the real
provider-call, provider-token, active-wall, or E3-local-prepare performance
targets have already passed; those require the one authorized same-run resume.

No evidence, provenance, concept-audit, statistical, or fail-closed gate was
relaxed to clear this milestone.

## Changes since the last accepted review

| Commit | Scope | Resolution |
|---|---|---|
| `f0a3797` | visual repair governance tests | Updated stale mocks to the strict typed patch transport; production gate unchanged. |
| `dfdf5f7` | compatibility repair tests | Replaced full-script patch-channel replies with `easyicu.code_patch/1`; full rewrite remains separately authorized and charged. |
| `68e6985` | post-repair concept-gate tests | Preserved distinct logical repair attempts and removed a false-green patch/rewrite mock path. |
| `f8946b1` | submission dictionary authority | Added immutable `npj_dm/20260716` for the intentional SICdb mortality authority change; retained `npj_dm/20260708` and its old SHA for replay. |

The first three discrepancies were test-contract drift. The fourth was a real
default-profile lock omission introduced when `ea9fc98` intentionally added
`HospitalDischargeType` to the SICdb in-hospital-death source. The canonical
dictionary was not reverted, the historical profile was not overwritten, and
the paper benchmark CLI's deliberately frozen `npj_dm/20260611` default was not
changed as part of this engine milestone.

## Full serial regression

The canonical environment was the main worktree `.venv` with `PYTHONPATH=src`,
`PYTHONHASHSEED=0`, `MPLBACKEND=Agg`, and all BLAS thread counts set to one.
The suite was split into 24 fresh pytest processes and run serially; xdist and
real-LLM flags were not used. The manifest contained every current
`tests/research_agent/test_*.py` file exactly once:

- files: `253`;
- sorted file-list SHA-256:
  `c1db562e969d2bd2e4031ce40d613c497f59e500c6b29259696959d07afccdd0`;
- result: **4,628 passed, 10 skipped, 0 failed**;
- warnings: `104` (non-failing, chiefly expected anti-pipeline warning probes);
- summed pytest time: `1,819.86 s` (`30 min 19.86 s`).

| Shard | Result | Pytest time (s) |
|---|---:|---:|
| S00 | 25 passed | 3.30 |
| S01 | 267 passed, 19 warnings | 349.80 |
| S02 | 310 passed | 2.18 |
| S03 | 202 passed | 3.52 |
| S04 | 79 passed | 174.50 |
| S05 | 157 passed | 1.97 |
| S06 | 229 passed | 1.46 |
| S07 | 73 passed | 0.88 |
| S08 | 37 passed | 10.77 |
| S09 | 44 passed, 1 warning | 5.44 |
| S10 | 85 passed | 2.06 |
| S11 | 44 passed | 4.64 |
| S12 | 58 passed | 1.77 |
| S13 | 63 passed | 1.67 |
| S14 | 70 passed | 1.55 |
| S15 | 46 passed, 83 warnings | 1.73 |
| S16 | 369 passed, 1 warning | 413.16 |
| S17 | 336 passed, 7 skipped | 63.92 |
| S18 | 350 passed | 289.08 |
| S19 | 326 passed | 88.65 |
| S20 | 405 passed | 120.00 |
| S21 | 379 passed, 1 skipped | 130.08 |
| S22 | 333 passed, 2 skipped | 130.79 |
| S23 | 341 passed | 16.94 |

This includes all five characterization files, the golden bundle, the meta
benchmark/capability-drift probes, full `test_pipeline.py`, full
`test_resume.py`, provider receipts, capsule replay, evidence authority,
figure-source trace, routing, and anti-pipeline coverage.

## Focused offline performance/authority gate

The following focused suite passed **163/163** in `4.92 s`:

- `tests/test_agent_perf_baseline.py`;
- `test_coder_prompt_budget.py`;
- `test_step_authority_capsule_integration.py`;
- `test_step_executor.py`;
- `test_resume_revalidation.py`;
- `test_provider_budget.py`;
- `test_repair_coordination.py`.

The read-only fail-closed harness recomputed the archived E3 baseline without
modifying it:

- `15` real provider calls = `12` step-scoped + `3` planner;
- `7` repair calls and `1` blocked request that is not counted as a call;
- `366,592` deduplicated tokens;
- Step 02: `6` calls, including `4` repair calls;
- Step 02 active wall: `373.5 s`;
- Step 02 sandbox compute: `1.719 s`;
- `4` resumes.

The current harness adds only schema-aware empty logical-repair fields when
reading the old schema-v2 receipt; the baseline numbers and every source digest
remain unchanged.

## P0-5 gate status

| Gate | Offline status | Evidence / boundary |
|---|---|---|
| Clean step calls `<=3` | structurally green | Initial generation, final audit and analyzer share one durable budget; provider-budget tests are green. |
| Pre-audit repair `<=4`; patch+rewrite `<=5` | structurally green | Patch and explicit rewrite are separately charged in one logical transaction. A post-audit semantic repair necessarily includes the failed audit and mandatory final re-audit; it must not be forced under the pre-audit call count by weakening audit. |
| Same authority digest: zero repeated generation/execution/audit | green | Capsule selection outranks legacy code; replay bypasses `StepExecutor`, never constructs an LLM auditor, and reruns deterministic gates. |
| Scoped transport size | green as bytes | Archived E3 Step 02 capture is initial `39,343/42,000`, patch `28,397/30,000`, rewrite `63,896/65,000` bytes. Bytes are not reported as model tokens. |
| Resume local preparation `<10 s` | pending online E3 | Synthetic capsule/revalidation suite is fast, but the real E3 preparation path must be timed. |
| Initial `<=12k` and repair `<=8k` provider tokens | pending online E3 | The local endpoint's usage receipt is the authority; byte limits are only a fail-before-provider transport gate. |
| Calls/tokens/active wall `>=50%` lower on the wasteful Step 02 path | pending online E3 | Acceptance targets are measured against 6 calls and 373.5 s active wall; no synthetic result may substitute. |

## Authorized next step and invariants

Resume the same archived E3 run using the configured loopback endpoint
`http://127.0.0.1:8317/v1` and model `gpt-5.6-luna`, with these invariants:

1. run a completion probe first and stop on non-200;
2. resume only `02_exposure_derivation_and_qc`;
3. Step 01's selected capsule, evidence digest, provider receipt and audit
   history must remain unchanged;
4. do not reset or delete the old provider receipt and do not grant a fourth
   logical repair attempt;
5. record real provider categories, usage tokens, local preparation time,
   active wall, sandbox time and final gate status;
6. if the step fails closed, preserve that result and diagnose it; do not turn
   a data/scientific decision into a deterministic pass.

