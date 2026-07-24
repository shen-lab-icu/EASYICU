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

The live timing audit found three distinct contexts rather than one reusable
result:

1. the early pre-registration gate reads a draft inside the bounded repair
   loop; a later deterministic/LLM repair can clear and replace its output
   directory;
2. the fresh-run final gate reads the post-figure-repair, sealed output view;
3. resume revalidation materializes a temporary verified evidence view before
   running the same final gate.

Therefore a draft envelope cannot be cached and reused as final authority. The
next safe slice is a single sealed compiler seam after all pre-seal figure
repair and before result evidence registration, plus an equivalent compiler
from the materialized verified resume view. Final/resume may then dual-read the
sealed envelope; the early repair gate remains on the legacy view until a
separate draft-stage contract exists. Any implementation must retain identical
finding payloads and fail closed on compiler/digest/status drift.
