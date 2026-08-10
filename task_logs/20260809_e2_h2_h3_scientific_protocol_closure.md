# E2 / H2 / H3 scientific protocol closure

Date: 2026-08-09
Module: `benchmark实验`
Task: `FIG2-CANONICAL9-SCIENCE-PROTOCOL-V2`
Implementation commit: `d530ba9`

## Scope and authority boundary

The attached GPT review was treated as AI-assisted pre-review, not clinical or methods attestation. This change prepares exact, digest-bound review inputs. It does not authorize Canonical9 provider runs and does not change Canonical9 question wording, the shared Planner/system prompt, or the frozen paper rubric.

The previous `f188acb` image, formal input, execution identity, and launch/finalize helpers do not bind the new protocol content. They remain historical preflight evidence only and must not be used for the formal 27-run batch.

## Implemented scientific contracts

### E2 — early peak lactate and in-hospital mortality

- ICU admission is time zero; exposure is maximum valid typed lactate in ICU hours 0–24, in mmol/L.
- The primary estimand is an adjusted descriptive association among eligible lactate-measured stays, not a causal effect.
- The full eligible cohort retains measured/unmeasured fractions, standardized differences, death/discharge before 24 h, observation opportunity, and measurement timing/count audits.
- Linear and prespecified restricted-cubic-spline forms are compared without outcome-driven form selection.
- Current 2026 adult Surviving Sepsis Campaign guidance is bound to the protocol.

Protocol content SHA-256: `2fa81bad5ab217c0fef0af6180e9899218b9cd4bf983817f91aad4f1be6306cb`

### H2 — vasopressor target-trial feasibility

- A positive typed `inputevents` record supports recorded administration in hours 0–24.
- An absent record means no recorded administration, not verified non-use.
- The current source therefore cannot authorize a binary control arm or causal contrast and must return `H2_VERIFIED_NON_USE_UNAVAILABLE`.
- The intended future target trial predeclares eligibility, strategies, ICU-admission time zero, 24 h grace period, clone-censor-weight handling, 28-day competing-risk outcome, per-protocol RD/RR, baseline timing, stabilized IPTW, positivity/balance thresholds, weight truncation, and sensitivity analyses.
- TARGET 2025, SSC 2026, and the MIMIC-IV `inputevents` source documentation are bound to the protocol.

Protocol content SHA-256: `3abd1083b604c5d84f2923f0a219845d5fe95f9d5728741c946d3a78a9955305`

### H3 — longitudinal phenotype redesign

- The previous k=6 protocol and observed mean ARI 0.5357 remain a terminal fail-closed result.
- The new protocol is a separately versioned, outcome-blind redesign: ICU hours 0–72, 12 h windows, SOFA-2 total/components plus lactate, observed-data likelihood, no zero/LOCF imputation.
- Candidate k is frozen to 2–6 and selected once by minimum BIC; minimum cluster fraction is 5%.
- Stability is exactly 100 80% subsamples, base seed 1729, all 100 successful, mean ARI at least 0.70.
- A failed selected solution yields “no stable phenotype solution”; there is no post-result alternate k, seed, threshold, feature set, exclusion, or imputation rescue.
- Any transportability claim additionally requires external-database reproducibility under the same frozen protocol.

Protocol content SHA-256: `4636c5cb57e6e01d4fc6348f726e17a6ef7402df74763471e31efac0c2958529`

## Engineering boundary

`benchmarks/figure2_canonical9/case_scientific_protocol.py` is the benchmark-local owner. It exposes strict frozen models, task-specific loading, normalized content hashing, and stable owner-attributable errors. Materialization, the JSONL adapter, and scientific authority consume this typed contract. The authority schema now binds both the KnowHow card and the case protocol, preventing a signed card from being paired with a swapped protocol.

## Verification

- Focused contract/regression set: 135 passed, 0 failed.
- Canonical9 + KnowHow + resource baseline set: 548 passed, 50 skipped, 0 failed in 72.52 s.
- Ruff passed for all changed Python files.
- `git diff --check` passed.
- Resource-context baseline was regenerated with an explicit reason because the three reviewed card projections and digests intentionally changed.

## Remaining blocker

All three protocols remain `human_attestation_pending`. A real clinical reviewer and a real methods reviewer must review the exact card and protocol digests. Any reviewer-requested content change creates a new SHA and new digests. Only after dual sign-off may the project rebuild fresh materialization, image, science authority, execution identity, ledger, and output root and begin the aware-only E1 canary followed by E2–E9 ×3.

## Primary references

- Surviving Sepsis Campaign 2026 adult guideline: <https://pubmed.ncbi.nlm.nih.gov/41869847/>; DOI `10.1097/CCM.0000000000007075`.
- TARGET Statement 2025: <https://www.bmj.com/content/390/bmj-2025-087179>; DOI `10.1136/bmj-2025-087179`.
- MIMIC-IV ICU `inputevents` documentation: <https://mimic.mit.edu/docs/IV/modules/icu/inputevents.html>.
- Multicenter validation of sepsis phenotypes, 2026: <https://pubmed.ncbi.nlm.nih.gov/42223936/>; DOI `10.1001/jamanetworkopen.2026.16134`.
