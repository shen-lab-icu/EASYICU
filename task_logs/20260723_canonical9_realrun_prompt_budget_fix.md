# Canonical9 real-run repair prompt budget fix

Date: 2026-07-23 EDT
Task: `AGENT-CANONICAL9-REALRUN-PROMPT-BUDGET`

## Trigger

The first mount-corrected MIMIC-IV E1 aware run reached the fifth planned
analysis step after completing the cohort definition, grouped Table 1,
prevalence/mortality summary, and publication figure. The generated
`04_missingness_and_measurement_audit` script then called
`measurement_provenance_receipt` with two positional arguments. Mechanical
preflight correctly emitted typed reason
`host_helper_call_signature_invalid`, but the minimal-patch request was rejected
before transport because it exceeded the 30,000-byte Coder patch budget.

Failed development-run evidence is preserved at:

`/Volumes/外置硬盘/easyicu_data/canonical9_runs/batch_20260723_luna_miiv_7894816_mountfix1/e1_sepsis3_prevalence_mortality/aware/run_20260723T073738_e030a0`

That interrupted batch is not paper authority and will not be resumed or
promoted.

## Fix

- Keep the existing 30,000-byte minimal-patch transport ceiling.
- Compute the exact byte capacity left after system messages, host authority,
  typed repair ticket, method contract, diagnostic envelope, and compact
  scientific context.
- Shrink only the redundant source-code neighbourhood to the remaining
  capacity.
- Require every applicable host-typed helper name and executable source-line
  anchor to remain in the excerpt.
- If those typed coordinates cannot fit, keep the request oversized so the
  existing preflight fails closed before a Provider call.
- Move the fitting and prompt rendering logic into `repairs/patch.py`; the gated
  `agents/core.py` surface decreases from 3,829 to 3,825 lines.

## Verification

Focused and adjacent offline matrix:

```text
529 passed, 5 deselected in 7.50s
```

The five exclusions are pre-existing branch fixtures: one migrated private
helper import and four custom fake clients rejected by the current Provider
factory when external transport is disabled. This change does not modify
Provider authorization.

The new negative/integration regression is:

```text
test_real_sized_helper_repair_compacts_code_not_typed_authority
```

It proves that a real-sized host attachment plus wide candidate script remains
within 30,000 bytes while preserving the exact helper call, typed reason,
source line, and complete host authority.

Additional checks:

- Ruff, Black, `py_compile`, and `git diff --check`: passed.
- Architecture lower-is-better gate: passed; `agents/core.py` 3,829 → 3,825.
- Module graph / zero-cycle gate: passed.
- Resource/context baseline: regenerated deterministically and passed. The
  measured Planner envelope remains below its 80,000-byte ceiling; the small
  recorded Planner-context delta predates this repair and comes from the
  already-committed outbound-safe projection.
- No real Provider, Docker execution, or patient-data read was used by the fix
  verification.

## Next authority transition

Build and validate a new immutable Docker image from the clean fix commit,
generate a fresh expected execution identity and operator freeze declaration,
then start a new MIMIC-IV aware-only batch. Do not reuse the interrupted batch.
