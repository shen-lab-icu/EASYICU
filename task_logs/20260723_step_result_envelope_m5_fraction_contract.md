# StepResultEnvelope M5-B fraction contract convergence

> Date: 2026-07-23 23:32 EDT  
> Task: `AGENT-STEP-RESULT-ENVELOPE-CONVERGENCE / M5-B1+B2`  
> Baseline: `8645848`  
> Commits: `c1ea67a`, `285dec3`

## Outcome

The fraction/percentage lane now has one case-neutral field vocabulary and
does not execute the legacy bounded-metric Validator twice during an envelope
dual-read.

- `contracts/fraction_scale.py` owns metric-key normalization and scale
  descriptor recognition.
- The legacy Validator, envelope scalar preservation, and pre-execution
  percentage repair use that shared vocabulary.
- `StepSummaryFractionEnvelopeDualReader` evaluates the legacy Validator once
  and passes the exact findings into the canonical comparison.
- `percentage_identity.py` remains because it is a pre-execution AST guard;
  the envelope consumer is a post-execution result audit. Removing either
  would change failure timing.

This slice does not wire the envelope into live execution, change a finding,
raise a repair cap, relax an evidence gate, or add case-specific vocabulary.
Across both commits production changed by `+48/-53` (net deletion: 5 lines).

## Verification

The adjacent matrix passed:

```text
310 passed in 15.27s
```

Static checks passed:

- Ruff;
- Black check;
- `py_compile`;
- `git diff --check`;
- module graph diff, with no new cycle.

Archived, metadata-only replays also remained exact:

| Run | Envelopes | Normalization errors | Validator mismatches | Fraction mismatches |
|---|---:|---:|---:|---:|
| E1 | 8 | 0 | 0 | 0 |
| E2 | 9 | 0 | 0 | 0 |

External diagnostic outputs:

- `/Volumes/外置硬盘/easyicu_data/canonical9_shadow_envelopes/m5b2_single_legacy_audit/e1_run_20260723T211020_5733af`
- `/Volumes/外置硬盘/easyicu_data/canonical9_shadow_envelopes/m5b2_single_legacy_audit/e2_run_20260723T235937_f4d63c`

No Provider, Docker, extraction, raw patient table, or Canonical9 execution was
started.

## Next bounded increment

Audit the two live invocation times of
`_step_deterministic_contract_findings` before any consumer switch. The early
pre-registration gate and final deterministic gate must receive envelopes
bound to the same summary/status/output authority without duplicate file
normalization. If that cannot be achieved as a small fail-closed replacement
with net deletion, keep the live path unchanged and design the compiler
injection seam first.
