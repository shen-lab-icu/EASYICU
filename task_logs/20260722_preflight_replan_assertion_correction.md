# Canonical9 preflight substantive-replan assertion correction

Date: 2026-07-22

## Finding

An independent replay found that
`test_substantive_replan_cap_stops_real_replanner_loop` selected the typed
finding through `finding["code"] == "replan_budget"`. The production manifest
stores this discriminator in `finding["validator"]`; the actual control path
already emitted `replan_budget_exhausted=True` correctly.

## Correction

The test now selects `finding["validator"] == "replan_budget"`. No pipeline
control logic, limits, acceptance policy, Provider boundary, or real-run
configuration changed.

## Verification

```bash
MPLCONFIGDIR=/private/tmp/easyicu_preflight_replan_fix PYTHONPATH="src:." \
  /opt/anaconda3/bin/python -m pytest -q \
  tests/benchmarks/figure2_canonical9/preflight/test_e1e3_preflight.py::test_substantive_replan_cap_stops_real_replanner_loop \
  -p no:randomly
# 1 passed
```

The complete offline preflight batch is rerun before commit. This remains
synthetic, zero-Provider, and diagnostic-only.
