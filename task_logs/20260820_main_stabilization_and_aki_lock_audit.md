# Main stabilization and seven-day AKI lock audit — 2026-08-20

## Scope and baseline

- Starting tree: clean, synchronized `main@c93b251cb4f49e5c96fadbe80f45302b4d58e30f`.
- No concurrent Claude, pytest, extraction, benchmark, or Web server process was
  editing or executing the repository.
- This pass repairs the eight failures reproduced from GitHub CI run
  `32256223775` and audits the finalized seven-day AKI foundation lock. It does
  not run a new clinical extraction and does not create paper results.

## Repairs

1. The SIC extreme-interval regression now supplies the required
   `cases.ICUOffset` origin instead of bypassing the raw SIC time contract.
2. The Extraction screen's renal fallback count is synchronized with the live
   backend catalog (`40`).
3. Immutable 20260817 E1 profiles remain replayable; additive 20260819
   canary/reviewed-demo profiles bind the finalized dictionary SHA
   `e3fd2fcb...38b00` and unchanged SOFA-2 SHA `71d67c47...cbbc3`.
4. The case-neutral Planner directive is nine bytes smaller and remains below
   its 51,600-byte fixed-cost ratchet.
5. Cohort prose translation no longer publishes materialized universe columns
   into a process-global registry. Predicate construction and execution receipt
   validation use bounded scopes tied to the actual materialized columns.
6. Generated catalog counts and the provider-free Canonical9 resource baseline
   are synchronized with the finalized dictionary and prompt bytes.

## Verification

- Original failure owners plus adjacent cohort/profile/catalog/resource tests:
  `93 passed, 8 warnings`.
- New run-local registry regressions and the two formerly order-dependent
  lifecycle guards: `4 passed`.
- Portability subset including SIC time quarantine: `5 passed, 245 deselected`.
- Full-six release sealer unit/negative contracts: `22 passed`.
- Ruff: `All checks passed`; `git diff --check`: clean.
- The first exact-commit matrix (`32330563773`) completed all portability and
  packaging jobs and executed the Python 3.11 functional suite as `14394
  passed, 197 skipped`; its only remaining gate was the checked-in architecture
  ratchet because `execution/phase.py` had grown by 13 lines. The materialized
  schema scope was moved to its cohort owner instead of refreshing the ratchet;
  the focused architecture and lifecycle rerun then passed (`15 passed`).
- A second full cross-version exact-commit GitHub Actions matrix is the final
  release gate and must be recorded by the workspace progress dashboard before
  this checkpoint is called stable.

## Foundation lock and artifact boundary

`_validate_foundation_lock()` passes with:

- lock SHA: `0c0df237e0b9b90870590ff7e1737d4e4bec56a8d757f271b19b0078de38b211`
- locked run: `full6_native_v2_kdigo_7d_baseline_9913f31c_20260819`
- concept dictionary SHA: `e3fd2fcb9d4a65fdaa58c5bc1edece0b1d8e7c685c13310bedef86fdd7138b00`
- SOFA-2 dictionary SHA: `71d67c479dfef8d0aad1f6fb02d1ca9dbc4243ea4f10b84e33ba8c9ced0cbbc3`
- clinical contracts SHA: `8bd3bc11073b98883b610d2fe3f7855013a5b2203c4325f9dfaf3d794f113679`
- clinical validator SHA: `a41bddae25be29f6f40150241ae29f985288243a3c09f3ded5951c4dcda63cb7`
- data-source registry SHA: `ec828bfccada3e2900b768308cd1aecf680c81c6c89f4aa4a3be2d35190f9a6b`

The repository retains the patient-level audit
`docs/aki-patient-level-audit-20260819.md`, including six positive examples,
six complete-negative examples, whole-export recomputation, and the 90-trajectory
audit claim bound in the finalized lock. However, exact-name searches under
`/Users/haibo/Documents`, `/Users/haibo/Desktop`, `/Users/haibo/Downloads`,
`/Volumes/外置硬盘/easyicu_data`, and `/Volumes/外置硬盘/EasyICU_归档` found no
retained directory for the named 9913 candidate. Therefore:

- the foundation semantics and repository hashes are finalized and usable;
- the historical candidate's claims remain documented evidence;
- a publication release package cannot be re-sealed or independently re-read
  from local bytes until that candidate is restored or a fresh six-database
  native-v2 extraction is produced.

Do not equate the passing foundation lock with a currently retrievable sealed
six-database release package.
