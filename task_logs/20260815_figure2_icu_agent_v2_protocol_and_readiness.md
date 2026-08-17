# Figure 2 ICU Research Agent v2: protocol and development readiness

Date: 2026-08-15
Task: replace the development-contaminated Canonical9-as-primary design with a
strict Dev9 + Qualification12 + Held-out27 experiment contract.  This log is
development evidence only; it grants no Provider, patient-data, formal-run, or
publication authority.

## P0 baseline adjudication

- Active implementation worktree:
  `/Users/haibo/Documents/GitHub/.worktrees/easyicu-fig2-dev9-heldout27-20260815`
- Branch: `feat/figure2-dev9-heldout27-20260815`
- Starting HEAD: `738f95cd88a7fc181e2cb3ea3403357bef26b68a`
- The dirty historical `feat/e1-h3-20260815` worktree was inspected but left
  untouched. Its extra strict-numeric alias layer was not transplanted because
  the added test still exposed the numeric Series directly to the existing
  sink; its SOFA changes also conflicted with the committed typed-input/Arrow
  dependency direction.
- The separate Git task-scope guard worktree remains a governance change and
  was not mixed into this scientific experiment branch.

## Frozen evaluator-side bundle

Owner: `benchmarks/figure2_icu_agent_v2/`.

- Dev9: the nine E1-H3 development questions; architecture changes are allowed,
  paper authority is forbidden.
- Qualification12: `MG01`-`MG12`; reusable non-paper generalization and
  fail-closed probes, never part of the primary denominator.
- Held-out27: 27 distinct item-scoped questions, no numeric answer or expected
  direction, exactly 9 basic / 9 intermediate / 9 advanced.
- Database coverage: MIMIC-IV 5, MIMIC-III 4, eICU 6, AUMC 5, HiRID 3, SICdb 4.
- Study-family coverage: descriptive 5, association 6, prediction 4,
  time-to-event 4, causal emulation 4, phenotyping 4.
- Every item requires the exact same 11-stage ICU research action space and is
  scored on nine frozen evidence/safety dimensions.
- Primary formal policy is aware-only, one run per task, no reuse/resume,
  cross-run memory, development sampling, post-hoc retry, or result-driven
  modification. Every failure remains in the denominator.

Content identities:

- protocol: `1323a6fba6d7b48b820233ce3dba9941e8305b0ecd57039cb80fce71a211529a`
- action space: `dc7e4cf2778225c5ffb07699fe1ee2ef941a72faa445e2897e0c27320e47dd9a`
- Qualification12 taskbank: `d7dadaf9a0ea9383c283fcc76027ec4d0453b0ba73308ac0d99a8a2d1543d9fd`
- Held-out27 taskbank: `55b711822008583a319d0dfd5f7f62bb6c8b585adacec59f8f7bf45f787ccf59`

## Generic architecture repair

The production concept-catalog owner lacked a general class of derived ICU
outcomes. `DERIVED_CONCEPT_HINTS` now supplies ICU readmission, 28/90/365-day
mortality, ICU-free days at day 28, ventilator-free days at day 28, general
culture positivity, and blood-culture positivity. Binary and non-binary
determinability remain explicitly different; the change does not mention a
benchmark item or alter the global Planner/Coder prompt.

## Development reachability receipt

Command:

```bash
PYTHONPATH=src:. .venv/bin/python \
  tools/audit_figure2_icu_agent_v2_readiness.py \
  --development-full6-root /Volumes/外置硬盘/easyicu_data/full6_20260717 \
  --output task_logs/20260815_figure2_icu_agent_v2/development_readiness_v1.json
```

The audit read `run_manifest.json` plus Parquet footers only; it did not read
patient rows. Observed modules were 19 each for MIMIC-IV, MIMIC-III, eICU, and
AUMC, and 16 each for HiRID and SICdb.

- registered primary scientific contract: 27/27
- required concepts catalogued: 27/27
- required columns observed in the target development database: 27/27
- development-ready: 27/27
- formal-ready: 0/27
- `paper_authority=false`

Receipt:
`task_logs/20260815_figure2_icu_agent_v2/development_readiness_v1.json`

Receipt SHA-256:
`51550a45e642cbd03eec1b39d046d7ea091fb7e06be02e06ee4d61cc796cb554`

Development manifest SHA-256:
`3cd70b85daabf6470ea19ed315cf07b88d6fcc0737cdfa4980c4639d109e9ed2`

Formal readiness is deliberately zero because fresh native-v2 typed inputs,
clinical review, methods review, and the exact execution-environment freeze do
not yet exist. The 2026-07-17 full6 tree remains an immutable development
vintage and cannot be promoted into paper evidence.

## Verification

```text
36 passed in 2.02s
Ruff: All checks passed
git diff --check: passed
```

The focused set covers the production concept owner, bundle identity and
coverage, duplicate-key/digest/stage/path negative cases, development
reachability, and task-attributed missing-column failure.

No full exact-head CI was started for this patch. A separate Claude full-suite
process was already running in another worktree and did not block development.

## Next gate

Run Dev9 in development mode in fixed E1-H3 order. A failure may drive only a
case-neutral owner-contract repair with focused negative tests. After Dev9 is
complete, run Qualification12 and independent E2/H2/H3 clinical/methods review;
only then prepare native-v2 inputs and the atomic formal freeze.
