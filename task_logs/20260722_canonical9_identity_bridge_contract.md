# Canonical9 identity-bridge contract (offline handoff only)

Date: 2026-07-22 EDT  
Task: `FIG2-CANONICAL9-GATE/P5-IDENTITY-BRIDGE-CONTRACT`  
Commits: `59d53b3 feat(figure2): define identity bridge handoff contract`; `dccc4d9 fix(figure2): harden identity bridge descriptor reads`

## Scope and outcome

The historical `full6_20260717` export has stay-level identifiers but no
patient-level identity relation.  This change adds a strict, small JSON
descriptor for a possible *future* controlled identity bridge.  It binds a
future protected mapping to the historical export manifest/content digests,
per-source source-snapshot and mapping artifact digests, aggregate cardinality
facts, and source-semantic attestations.  It never reads a mapping artifact,
full6 parquet, patient row, Provider, Docker image, or Canonical9 task.

The contract is deliberately outside `realrun_authority.py` and the launcher.
Even an authorized data-lane descriptor reports `real_run_authorized=false`;
it can only be presented for later native typed-materialization review.  P4
production input authority, E2/H2/H3 human/scientific gates, and final operator
freeze remain mandatory.

## Negative guarantees

- Exact six-source order and audited identifier names are required.
- HiRID and SICdb require explicit owner semantic attestation.
- Duplicate stay mappings, empty mapping artifacts, unknown extra fields,
  duplicate JSON keys, non-canonical JSON, symlink descriptors, and malformed
  contract digests fail closed.
- A structural retrofit or an identity-bridge descriptor cannot directly become
  paper authority or execute a run.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='src:.' /opt/anaconda3/bin/python -m pytest -q \
  -p no:cacheprovider -p no:randomly \
  tests/benchmarks/figure2_canonical9/test_identity_bridge_contract.py \
  tests/benchmarks/figure2_canonical9/test_development_repair_framework.py \
  tests/benchmarks/figure2_canonical9/test_realrun_authority.py
```

Result: `96 passed`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='src:.' /opt/anaconda3/bin/python -m pytest -q \
  -p no:cacheprovider -p no:randomly tests/benchmarks/figure2_canonical9
```

Result: `434 passed` (the prior 425-test Canonical9 baseline plus nine new
bridge-contract tests).

`ruff`, `black --check`, `py_compile`, `git diff --check`, architecture
measurement, and the zero-cycle module graph also passed.  No external model,
Docker, mapping data, or patient data was touched.
