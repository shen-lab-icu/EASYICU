# Track A online E3 acceptance

Date: 2026-07-17 EDT

Branch / engine commit: `refactor/agent-control-plane@d75136d`

Run: `research_output/_diagnostic_e3_8317_fresh_ceb00f2_20260716T072600Z/E3_kdigo_gradient/aware/run_20260716T072721_7fd5c5`

Endpoint / model: loopback `8317` / `gpt-5.6-luna`

Task: `AGENT-TRACK-A-PERF-REFACTOR`

## Decision

The single authorized same-run E3 acceptance succeeded. Track A's control-plane
code and the waste-path online performance acceptance are complete at
`d75136d`; the shared engine is frozen for the next experiment steps.

The run stopped deliberately after `02_exposure_derivation_and_qc`, so the
run-level status remains `diagnostic_only`. That is expected and is not a
scientific failure. Figure 2 remains 6/9 until the complete E3 development run
and the other unfinished questions pass their reportability gates.

## Authority-pinned execution

The current engine was imported from an ephemeral mirror with only the packaged
concept dictionary restored to the archived authority. Preflight asserted:

- concept dictionary SHA:
  `bc377779ce0f6b7983b2f8f527a37c1c394cc38e4a64055c9d9268b5f4d451ea`;
- SOFA2 dictionary SHA:
  `b26e36b6ef5ea947027c8f7cd514fc5174545aa658187d6bdb8ec43f2a80b6aa`;
- engine SHA:
  `514a3b4765b109c6ea808cc32f22bd9aea849d9d1f1ff1865a3e460970915a5d`;
- validator SHA:
  `c7bb47b07f6078f4cf34413391df531d8bc6a7b6a1bc846810750cfe8578bc82`;
- `verify_replay_dict_match(...) == []`;
- all imported EasyICU modules were contained in the mirror.

No `EASYICU_DICT_PATH` overlay or submission profile was used. SDK retries were
zero, BLAS threads were one, replanning was disabled, the logical LLM repair
limit remained three, and resume/stop were both fixed to Step 02.

The `/models` and four-token completion probes returned 200; the completion was
exactly `OK`. Probe calls are not part of the research-run provider ledger.

## Step 01 immutable-authority result

Current validator drift legitimately appended a new
`revalidated_without_execution=true` checkpoint for `00_probe` and
`01_cohort_flow`. Step 01 was not executed and made no provider call.

Against the complete pre-run backup:

- the full `steps/01_cohort_flow` tree is byte-identical;
- the Step 01 provider receipt remains SHA
  `d2d4760d32f67e5248252d9d02fcb1cd93d291865c75dff11ce692499547c4e7`;
- all seven selected evidence rows and their evidence files are byte-identical;
- `run_input_capsule.json` is byte-identical;
- the old audit log is an exact byte prefix of the new append-only log;
- no Step 01 authority capsule was created;
- dictionary/SOFA2 SHA coordinates are unchanged. The mutable root fingerprint
  file refreshed only its `computed_at` timestamp.

The Step 01 current-record digest changed because of the legitimate
revalidation checkpoint. It is not an immutable authority and is not reported
as unchanged.

## Step 02 scientific and gate result

Latest Step 02 record:

- `status=ok`;
- `returncode=0`;
- `result_evidence_sealed=true`;
- `outputs_safe_to_collect=true`;
- deterministic, contract, final-concept and LLM-concept approvals all bind
  executed code SHA
  `0842e1855383e766a74587862de3b3aebb8a9dfb1b6accb17712726f76591b7d`;
- LLM concept audit completed with zero errors;
- stat, clinical, contract, guard, usage and figure-source findings are all
  empty;
- Critic status is `pass`, with no concerns or unsupported claims;
- exactly one deterministic repair was applied:
  `lossy_numeric_coercion_guard_v1`;
- logical LLM repairs remain exactly three (`concept`, `concept`, `concept`);
  no fourth repair was granted;
- the provider ledger added exactly one `concept_audit`, and no repair call.

All eight registered Step 02 evidence files are present and match their sealed
SHA. They include the executed code, four typed tables, step summary, Critic
report and interpretation. The four declared tables are:

- `aki_stage_max_ordered.parquet`;
- `aki_stage_distribution.csv`;
- `aki_stage_source_availability.csv`;
- `aki_stage_numeric_coercion_audit.csv`.

The locked cohort contains 74,829 rows. Ordered-stage counts are
37,433 / 14,061 / 19,593 / 3,621 with 121 preserved missing values and zero
invalid values; all measurement-provenance comparisons report zero
discordance. These are agent-produced development artefacts, not hand-entered
paper results.

## Online performance acceptance

Process start was `05:53:32.439415Z`; Step 02 started at
`05:53:34.015421Z`, completed at `05:54:00.783060Z`, and the run stopped at the
requested checkpoint.

| Metric | Old failed Step 02 path | Accepted resume | Reduction / result |
|---|---:|---:|---:|
| Real provider calls | 6 | 1 | 83.3% lower |
| LLM repair calls | 4 | 0 | 100% lower |
| Provider tokens | 142,724 | 30,204 | 78.8% lower |
| Active wall | 373.5 s | 26.768 s | 92.8% lower |
| Sandbox execution | 1.719 s | 1.181 s | 31.3% lower |
| Local preparation | not previously isolated | 1.576 s | `<10 s` target passed |

The accepted call used 29,159 prompt and 1,045 completion tokens, below the
predeclared resume ceilings of 62,219 prompt and 71,362 total tokens. The fresh
initial-generation / LLM-repair per-call token targets were not exercised:
this resume eliminated both categories, so their effective usage on this
wasteful path was zero. No claim is made here about a future fresh step's
per-call prompt size.

Planner calls remain three, resume receipts increased from four to five, and
the run ledger reconciles exactly: 16 total calls = 13 step-scoped + 3 planner.

## Next action

Continue the E3 development run from the next unfinished step using this frozen
engine and the same authority-pinned execution method. Do not modify shared
engine code in response to ordinary warnings or an unfavorable scientific
result. Only a reproducible, case-neutral correctness or data-integrity defect
may reopen the engine, with a new protocol/version and invalidation analysis.

Track B structural cleanup remains postponed until the active Canonical9
development runs are closed; it must not delay the experiments again.
