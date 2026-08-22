# E2 zero-Provider preflight: exact human-review blocker

Date: 2026-08-21

Task: `FIG2-DEV9-HELDOUT27` / `e2_lactate_mortality`

Verdict: E2 must not start a Provider/Planner run yet. The exact E2 scientific protocol is still `human_attestation_pending`, and its KnowHow card is still `curated_mvp`. The enforced authority requires real, digest-bound clinical and methods approval before any E2 execution.

## Verified exact inputs

- Protocol: `benchmarks/figure2_canonical9/protocols/e2_lactate_mortality_20260809.json`
  - file SHA-256: `08706e8d095131292e89fd27d4393f928396b2302da00ac39f4f7b74983b6961`
  - normalized content SHA-256: `a3213192320226ddfd7c767885686551fca63e1df0ecfb863279233e52286a86`
  - runtime projection SHA-256: `1a267cd6e03452c2bbf5cf71d7281837f7d9529ba91f15901bbf9f6b39fcdd23`
  - review state: `human_attestation_pending`
- KnowHow card: `src/easyicu/data/research_know_how/early_peak_lactate_association.json`
  - version: `1.1.0`
  - file SHA-256: `6dd71c90dfb32a55d63d2a219bd198d669f88f9606695992013fcc416c56b9ec`
  - reviewable-content SHA-256: `13be93660e3f46eb95c52c030124379750b0dd03c30e36f22bf0beafd5898f93`
  - review state: `curated_mvp`

## Machine checks

- Scientific-protocol authority, real-run authority, and KnowHow evaluation tests: `129 passed`.
- Worktree was clean and synchronized at docs tip `022af34`; no competing Claude, pytest, or benchmark process was active.
- Exact E1 code freeze/image/CI remain unchanged at `ce1223c` / `sha256:13657df0...e5f2`.
- The current 2026-08-15 Dev9 input bundle contains E1 only. An older 2026-08-09 E2 development materialization exists, but it is not a fresh E2 input frozen against the post-E1 execution coordinate and cannot be silently promoted.

## Required external decision

Two real reviewers must review the exact protocol/card content above:

1. an ICU/critical-care clinical reviewer;
2. a clinical-epidemiology/biostatistics methods reviewer.

If either requests a content change, create a new version and recompute all digests. If both approve unchanged content, record a `clinical_reviewed` card attestation binding both approvals, card version/content digest, protocol content digest, and runtime projection digest. The agent must not fabricate these identities or approvals.

## Shortest continuation after attestation

1. Verify the signed E2 card and scientific-protocol authority locally with zero Provider calls.
2. Materialize one fresh E2-only development input against the current dictionaries and build an E2 development profile/exact image only if the existing frozen profile cannot bind it.
3. Run one fresh bounded E2; repair only generic owner contracts, then report required-step completion, Provider cost, and authority limits. Do not enter E3, Qualification12, or Held-out27 until E2 closes.
