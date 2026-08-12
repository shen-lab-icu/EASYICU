# Agent publication-authority review remediation

- Date: 2026-08-12
- Branch: `fix/pi-workspace-review-20260809`
- Review baseline: `f6603713f7b16ea6a22d061c7877058479aaefb5`
- Scope: Agent-layer manuscript semantic authority, numeric miscitation repair,
  and CLI-provider transport boundaries
- Publication/benchmark status: unchanged; Canonical9 remains 4/9 and paper
  authority remains frozen

## Adjudication

The external review's two P0 findings and one P1 finding were reproducible and
correct:

1. A valid evidence ID proved provenance but could be attached to an unrelated
   qualitative scientific assertion.
2. Numeric citation repair selected an ordered owner when several different
   statistical facts had the same value.
3. A directly constructed CLI provider could reach the subprocess without the
   factory authorization check, and CLI/standalone AgenticCoder subprocesses
   inherited the parent environment.

The `execution/phase.py` size/coupling observation remains separate P2
architecture debt and was not mixed into this safety patch.

Before implementation, 13 focused adversarial tests produced 12 failures and
1 pass, reproducing the reported paths.

## Implemented boundaries

### 1. Host-derived `ScientificClaim` authority

- Added a small typed owner contract under
  `research_agent/authority/scientific_claims.py`.
- Only the reviewed deterministic adjusted-association result schema can mint
  a claim. The host derives exposure, outcome, direction, estimand,
  population, analysis role, and adjustment set from the digest-registered
  summary. Direction is derived from the confidence interval relative to the
  estimand's null.
- Runner-supplied `scientific_claims` are rejected. LLM-generation modes cannot
  mint claims even if they imitate the deterministic summary shape.
- The stored claim registry is re-derived from the immutable evidence bytes on
  every read; metadata drift, digest drift, stale coordinates, and duplicate
  references fail closed.
- The Writer receives exact `{claim:<step>.<claim>}` tokens. A token must be a
  complete standalone sentence; the host renders the sentence and adds its
  evidence citation. Free-form association, comparison, change, similarity,
  or interpretation prose cannot acquire semantic authority by borrowing a
  valid evidence ID.
- The current compiler is intentionally narrow. Unsupported qualitative claim
  shapes expose no token and remain blocked rather than being guessed.

### 2. Numeric miscitation repair

- Auto-repair now runs only when exactly one registered numeric claim owns the
  value and that owner is citable.
- Any second owner prevents mutation because the current `NumericClaim`
  contract does not encode enough estimand/exposure/outcome/population identity
  to prove equivalence.
- Adversarial regressions cover AUROC versus mortality rate, OR versus HR,
  cohort denominator versus deaths, proportion versus percentage, and primary
  versus sensitivity analyses. Ambiguity remains unchanged and the strict
  numeric binder blocks it.

### 3. CLI transport authorization and subprocess environment

- `CLIAgentLLMClient.complete()` now repeats the provider-authorization check
  immediately before checking or launching the CLI, matching the existing
  OpenAI-compatible transport boundary.
- Added one reviewed environment builder shared by CLI providers and the
  standalone AgenticCoder path. It passes only basic runtime/config variables,
  backend-specific authentication, and explicitly named input coordinates.
- AWS, GitHub, database, proxy, and other unrelated parent secrets are not
  inherited. Codex and Claude authentication are separated.
- Standalone AgenticCoder delegation also requires the explicit external-LLM
  opt-in; otherwise it uses the receipt-aware fallback.

## Verification

Canonical interpreter: `.venv/bin/python` (Python 3.11).

- Evidence/Writer adjacent set: `86 passed`.
- Provider/numeric adjacent set: `180 passed`.
- Adjusted-association/execution-authority adjacent set: `138 passed, 1 skipped`.
- Additional focused scientific-claim tests: `17 passed`.
- CLI/AgenticCoder tests after the final env cases: `34 passed`.
- Static architecture policy: `2 passed`.
- Research-agent module graph diff: clean.
- Ruff on every touched Python file: clean.
- `git diff --check`: clean.
- Final non-overlapping command selection: `414 passed, 1 expected skip`.

Per the development checkpoint policy, no local full-suite or full exact-head
matrix was started for this scoped fix. The pushed exact head must supply the
remote CI evidence before it is treated as a stable/formal checkpoint.

## Explicit non-claims

- No Provider call, patient-data read, Docker analysis, benchmark run, rubric
  change, or paper-authority promotion occurred.
- Canonical9 remains 4/9.
- This patch does not claim that all possible scientific statement types have
  a renderer. Unsupported types fail closed.
