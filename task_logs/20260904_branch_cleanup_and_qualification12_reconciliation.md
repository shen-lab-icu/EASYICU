# Branch cleanup and Qualification12 reconciliation

Date: 2026-09-04 EDT

## Scope

The repository was reduced to the two durable local lanes that remain active:

- `main`, for the product and governed data foundation;
- `codex/experiment-v2-1-review-20260901`, fixed at the reviewed Figure 2
  candidate `b89ba94`.

The experiment candidate was not rebased, merged, or otherwise rewritten.

## Completed branch actions

- Fast-forwarded the two-commit IMV targeted-refresh branch into `main` after
  focused review, then deleted `origin/fix/imv-targeted-refresh-20260904`.
- Deleted `origin/codex/literature-to-design-contract` after proving that its
  tip was already an ancestor of `origin/main`.
- Archived the superseded AKI source-native branch at
  `archive/aki-source-native-profiles-20260826-06bd4c4` and deleted
  `origin/feat/aki-source-native-profiles`. Its implementation already had a
  patch-equivalent successor on `main`; its remaining documentation described
  an obsolete primary-authority relationship and was deliberately not merged.
- Confirmed that `main` contains no `septic_shock_sepsis3_2016` definition,
  implementation, or registry reference. Archived the withdrawn prototype at
  `archive/septic-shock-withdrawn-20260831-7e88628` and deleted the local
  branch `codex/septic-shock-sepsis3-20260823`.
- Archived the original Qualification12 preflight lane at
  `archive/qualification12-preflight-20260827-ce60832`, selectively reconciled
  the valid data-foundation changes, and deleted both the original local branch
  and the temporary reconciliation branch.

## Qualification12 commit adjudication

Eleven commits retained valid, general data-foundation behavior and were
adapted onto current `main`: eICU ICU-unit type, exact requested-ID boundaries,
streamed mapping isolation, categorical batch invariance, dependency request
serialization, measured one-shot/fallback planning, RSS evidence, and the
eICU/MIMIC-IV/MIMIC-III resource profiles.

Eight commits were not merged:

- seven old `meta_generalization` readiness/materialization/profile commits
  belonged to the superseded pre-v2.1 Qualification12 route and would conflict
  with the current immutable v2.1 formal experiment contract;
- the Sepsis-3 septic-shock commit encoded the definition that was later
  withdrawn as clinically invalid.

The reconciliation also preserved current Web progress localization, updated
the demographics fallback count, and moved added tests into their owner
directories instead of growing the existing oversized Web test modules.

## Validation

- IMV branch before merge: `68 passed, 1 skipped`; Ruff and diff-check passed.
- Qualification12 reconciliation: `480 passed`; follow-up tool and test-owner
  checks `28 passed`.
- Changed Python files passed Ruff; changed JavaScript files passed
  `node --check`; `concept-dict.json` parsed; range `git diff --check` passed.
- After cleanup, both remaining worktrees were clean and synchronized with
  their upstream branches.

These checks establish the scoped engineering merge and branch cleanup. They
do not constitute full exact-head CI, real six-database extraction, a formal
Qualification12 run, or paper authority.
