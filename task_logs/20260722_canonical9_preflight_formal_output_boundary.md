# Canonical9 offline preflight: formal-output and subprocess-boundary closure

Date: 2026-07-22

## Scope

This is a preflight-only increment for the synthetic E1--E3 partial-flow smoke.
It does not invoke a real Provider, Docker, patient data, or frozen Canonical9
bindings, and it does not alter `src/easyicu/research_agent/**`.

## Review disposition

The useful parts of the independent P1 review were integrated manually onto the
current mainline. Its commit could not be merged directly because it was based
on `1434aa9` and reinstated a custom `MockLLMClient` subclass, which the current
reviewed Provider-authority boundary correctly rejects. The integration retains
the reviewed built-in `PatternScriptedMockLLMClient` delivery path.

## Guarantees now exercised

- Every live formal expected output has a one-to-one `ProductMapping`. A
  `produced` declaration requires both a plan step and an evidence-ID prefix.
  The resolved map is persisted as `preflight_product_map.json`, with declared
  and observed fulfillment read from the real manifest.
- The offline smoke honestly produces only deterministic Table 1
  (`table_step_artifact_...`). Planned-only nodes and sealed figures stay
  non-paper outputs; missing mapping or a removed artifact fails closed in
  regression coverage.
- The subprocess runner receives explicit
  `network_policy="none"` and `allow_unsafe_host_fallback=False`. The latter
  pin defeats `EASYICU_ALLOW_UNSAFE_HOST_FALLBACK=1`; every recorded subprocess
  step must report `requested_network_policy="none"` and
  `isolation_degraded=false`.
- An unavailable isolation backend writes its runtime receipt, avoids pipeline
  construction, and makes the CLI return a structured blocked result with exit
  code 2.

## Verification

```bash
MPLCONFIGDIR=/private/tmp/easyicu_preflight_p3_final PYTHONPATH="src:." \
  /opt/anaconda3/bin/python -m pytest -q \
  tests/benchmarks/figure2_canonical9/preflight/ -p no:randomly
# 81 passed

ruff check benchmarks/figure2_canonical9/preflight \
  tests/benchmarks/figure2_canonical9/preflight
black --check benchmarks/figure2_canonical9/preflight \
  tests/benchmarks/figure2_canonical9/preflight
python -m py_compile benchmarks/figure2_canonical9/preflight/*.py
git diff --check
# all passed
```

## Remaining boundary

This is now an honest partial-flow smoke, not a paper-readiness claim. A real
Canonical9 result still requires the approved full6 input authority, frozen
execution identity, an explicit real-Provider authorization, and a fresh full
`--arms aware` run.
