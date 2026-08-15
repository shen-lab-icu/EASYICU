# Canonical9 full0717 controlled identity-bridge build

## Scope

Owner instruction on 2026-07-22 selected the latest external-drive
`full6_20260717` export as the only clinical payload and delegated the choice
of a safe identity route.  The selected route reads the six source
stay-to-patient relations only; it does not re-extract clinical concepts,
modify full0717, invoke a Provider, invoke Docker, or run Canonical9.

## Artifact

- implementation commit: `ccd3e6b`
- private output root: `/Volumes/外置硬盘/easyicu_data/full6_20260717_identity_bridge_20260722`
- descriptor: `identity_bridge_contract.json`
- descriptor SHA-256: `4092104a40d2b22d80a93cf00323be0f3c048bfc3d9ea6bb002124361ecce794`
- permissions: directory `0700`; descriptor and mapping Parquet members `0600`
- contract status: `data_lane_authorized=true`,
  `eligible_for_native_materialization_review=true`,
  `real_run_authorized=false`

## Coverage checks

| Source | mapped stays | unmapped | duplicate stays | max stays/patient |
| --- | ---: | ---: | ---: | ---: |
| MIMIC-IV | 94,458 | 0 | 0 | 41 |
| MIMIC-III | 61,532 | 0 | 0 | 41 |
| eICU | 200,859 | 0 | 0 | 26 |
| AmsterdamUMCdb | 23,106 | 0 | 0 | 8 |
| HiRID | 33,905 | 0 | 0 | 1 |
| SICdb | 27,386 | 0 | 0 | 12 |

HiRID's source documentation defines its Patient ID per ICU admission, so the
one-to-one cardinality is a source limitation rather than evidence of
cross-admission patient linkage.  SICdb uses `CaseID -> PatientID`; the bridge
contract was corrected from its former conservative placeholder.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='src:.' /opt/anaconda3/bin/python -m pytest -q \
  -p no:cacheprovider -p no:randomly \
  tests/benchmarks/figure2_canonical9/test_identity_bridge_contract.py \
  tests/benchmarks/figure2_canonical9/test_identity_bridge_builder.py
```

Result: `13 passed`.

Black, Ruff, `py_compile`, and `git diff --check` passed for the builder and
its tests.  The bridge contract loader independently verified the real private
descriptor after build.

## Remaining boundary

The next allowed operation is native typed-materialization review using this
private bridge.  It must produce fresh typed cohort/trajectory authorities and
cannot be skipped by adding a patient column to a historical parquet.  E2's
formal ProtocolCard review, H2's medication-coverage contract, H3's reviewed
redesign decision, and the P4 final operator freeze remain separate gates.
