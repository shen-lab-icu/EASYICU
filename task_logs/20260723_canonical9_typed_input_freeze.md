# Canonical9 MIMIC-IV typed input freeze

Date: 2026-07-23 EDT

## Outcome

- Materialized the exact ordered Canonical9 from the verified MIMIC-IV
  full0717-v2 native export without cross-database inputs.
- Reused one verified export snapshot across all nine materializations; outputs
  were written directly to the external drive.
- Froze all nine benchmark-owner scientific identities in
  `benchmarks/figure2_canonical9/canonical_run_input_bindings_v2.json`.
- Added `tools/freeze_canonical9_run_inputs.py` so the selector is regenerated
  from the typed JSONL and verified authorities rather than hand-edited.

## Frozen inputs

- Root:
  `/Volumes/外置硬盘/easyicu_data/canonical9_miiv_native_20260723_final_28f4f06`
- JSONL:
  `/Volumes/外置硬盘/easyicu_data/canonical9_miiv_native_20260723_final_28f4f06/canonical9_miiv.jsonl`
- Cohorts: 9/9, each 94,458 ICU stays.
- Trajectories: H2 2,687,299 rows; H3 19,067,154 rows.
- M2 patient grouping: 94,458 unique stay identities from 65,366 patient
  groups; grouping is the prefix before `:s`.
- E1/H1/H2 positive-only events were projected with explicit structural-zero
  receipts. Two out-of-bound lactate source values were excluded only under the
  recorded `exclude_with_receipt` policy.

## Authority

- Run-input selector state: 9/9 `ready`.
- Selector SHA-256:
  `e888e483231708cd52208290df1f487904fe4eac4873616192509b8523271c0d`.
- H2/H3 use run-input capsule v3 and bind typed trajectory authority; the
  remaining seven use capsule v2.
- This freeze authorizes exact typed inputs only. It does not by itself
  authorize a Provider call or paper-facing result.

## Verification

```text
110 passed
```

The matrix covered the strict selector, scoring-input authority, benchmark
sealer, and Figure 2 scorer. The generator's `--check` mode reproduced the
selector byte-for-byte.

Luna transport was checked without patient data: the loopback endpoint
`127.0.0.1:8317` was listening and advertised `gpt-5.6-luna`.

The existing `easyicu-research-agent:1.0.0` image was rejected for the formal
run because its installed research-agent source digest
`0faeb3a6…6626` differs from the current expected digest
`c4d00e47…ddbc`. A matching immutable image must be built and frozen before
the first real prompt.

The first matching overlay image exposed one final execution-identity wiring
gap: Docker preflight already exported the verified immutable `image_id`, but
`execution_identity_for_pipeline()` recorded only an optional profile-side
expected digest. Paper profiles without a capability-activation pin therefore
lost the real image coordinate. The identity now binds the validated runtime
image and rejects any mismatch with an independently expected digest. Focused
execution-identity/realrun/pipeline checks passed (`102 passed`); architecture
and module-graph gates showed no regression.
