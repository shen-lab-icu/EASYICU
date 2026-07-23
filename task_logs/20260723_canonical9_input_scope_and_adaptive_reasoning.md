# Canonical9 E1 input-scope correction and adaptive reasoning profile

Date: 2026-07-23  
Task: `FIG2-CANONICAL9-GATE`  
Implementation commit: `14795cb8ccc4ea04128e38e0e89a290176e22b2e`

## Outcome

The failed Luna E1 run exposed a general contract error: a step's
`planner_declared_inputs` is its permitted consumer scope, not a declaration
that the physical `COHORT_PARQUET` must contain exactly those columns. The
physical locked cohort can legitimately be a strict superset. Generation and
repair prompts now state that distinction, and a narrow deterministic repair
changes only the exact archived closed-world assertion into a required-column
presence check. It never drops columns, changes values, or relaxes the
fail-closed behavior for missing declared inputs.

An explicit experimental reasoning profile, `adaptive_v1`, was added without
changing the default. It assigns:

- Planner and initial Coder: `medium`
- Analyzer, Writer, and Literature: `low`
- Coder repair after a validated failure: `high`

`provider_default` remains the EasyICU CLI default. The proxy-wide default was
not changed. The standalone Tier-2 jury is off by default and is not part of
this profile or the current Canonical9 run.

## Reproducibility and authority

The selected profile is bound into the Canonical execution configuration,
Provider authorization manifest, execution identity, and result payload.
Provider dispatch identity includes the canonical request `extra_body`, so an
in-place effort mutation is rejected before transport. Each recorded Provider
call now includes the actual reasoning effort and elapsed milliseconds, and
the envelope summary includes the set of efforts and total recorded call time.
Results from `provider_default` cannot be reused as `adaptive_v1` results.

The local Luna endpoint accepted minimal non-clinical requests with
`reasoning.effort=low`, `medium`, and `high`. This proves transport capability
only; it does not establish a performance advantage.

## Verification

- Focused authority/provider/input-scope matrix: `254 passed, 4 deselected`.
- The four deselected integration cases were blocked by the expected
  current-source versus old-Docker-source digest mismatch. They must be rerun
  after building an immutable image from the final clean documentation commit;
  the source-identity gate was not weakened.
- Legacy Provider authorization schema references under `src/`, `tests/`,
  `tools/`, and `benchmarks/`: `0`.
- Ruff and `git diff --check`: passed.
- Static module graph: no new cycle.
- Architecture baseline records the necessary Coder prompt/repair seam and
  deterministic repair registration; the execution god-function remains
  unchanged in line count and own-scope names.
- Resource/context measurements are numerically unchanged; only the reviewed
  `agents/core.py` source digest changed.

## Next action

Build a fresh immutable Docker image from the final clean commit and rerun the
four digest-blocked integration cases. Then launch a fresh E1 engineering run
with `--reasoning-effort-profile adaptive_v1`. Do not promote the profile to
default unless E1 remains scientifically/contract valid and shows a material
end-to-end latency improvement without more Planner retries or repairs. Paper
authority remains `0/9` until accepted Canonical9 artifacts exist.
