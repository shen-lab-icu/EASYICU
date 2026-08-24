# Unified product exact-head desktop release verification

Date: 2026-08-24

## Outcome

The unified Web/Desktop/Copilot candidate was exercised as a product and then packaged from an exact, clean source commit without touching `main`.

- Worktree: `/Users/haibo/Documents/GitHub/EASYICU-unified-product-20260823`
- Branch: `codex/easyicu-unified-product-20260823`
- Exact package source: `26960a71a0da5b8e315223ef455195225f93f175`
- Upstream base: `origin/main@8115f93`
- Push/merge: not performed
- Existing worktrees/releases: preserved; no stash, prune, delete, or overwrite was performed

The packaged desktop application is still a thin Tauri shell around the same FastAPI native WebApp. The Web and App therefore share the product implementation; the App only changes when it is rebuilt from a newer Web source commit.

## Product UAT

An isolated FastAPI service ran at `127.0.0.1:8897` with a temporary `EASYICU_HOME`. The in-app browser covered every primary route, compatibility route, and the requested conversational data workflow.

- Primary routes: `entry`, `guided`, `ideas`, `extraction`, `patient`, `cohort`, `crossdb`, `agent`, `tutorial`, `dictionary`, `settings`, `states`
- Compatibility routes: `help`, `assistant`, `audit`, `sofareclass`, `icd`
- Invalid route: safely rewrote to `#entry`
- All 18 requested route checks resolved to the expected page title; desktop/laptop horizontal overflow was 0 and the console had 0 errors/warnings.
- Home: English/Chinese switch, research-question draft, and `Start Guided Copilot` were exercised.
- Guided Copilot: the draft was preserved, outcome selection worked, and the conversation advanced to data-source configuration (step 1/8).
- Data Extraction: recommended seeded-demo extraction completed with six module ledgers plus manifest, correctly stated that no files were written, and opened Patient Review.
- Patient Review: rendered 48 synthetic entities, 19 modules, tables, features, and next-step controls.
- Cohort Statistics: exercised Group contrast, Survival curves, Coverage audit, Cohort profile, and SOFA reclassification.
- Cross-database: six offline fallback sources completed with 19 modules and 307 comparable feature profiles; Overview, Coverage, Distributions, and Quality & provenance all rendered.
- Project Monitor: Overview, Runs, Outputs, Draft, and Evidence rendered the correct read-only/fail-closed states.
- Idea Mining: a local idea ledger was created; feasibility correctly remained blocked without an active export. External PubMed/network access was not enabled.
- Data Dictionary: `lactate` returned four matching concepts.

No blocking product defect was found, so no source-code fix was added after UAT.

## Focused regression

- Python desktop/Copilot/static contracts: `148 passed`, 5 warnings.
- Canonical JavaScript contracts: `27/27` passed.
- Rust/Tauri tests after the frozen backend resource was generated: `3 passed`, 0 failed.

The first pre-build Rust invocation stopped at `resource path resources/backend doesn't exist`. This is the expected build-order prerequisite in a fresh worktree: `desktop/scripts/build_macos.py` generates that resource before Tauri compilation. The post-generation Rust run passed.

## Exact-head build and artifacts

Build command:

```text
cd /Users/haibo/Documents/GitHub/EASYICU-unified-product-20260823/desktop
python3 scripts/build_macos.py
```

Stable local release copies:

- `output/releases/26960a7/EasyICU.app` (approximately 980 MB)
- `output/releases/26960a7/EasyICU_1.0.0_aarch64.dmg` (446,681,040 bytes; approximately 426 MB)
- DMG SHA-256: `761c0b09d8f5b0f0e2b507161795996b8ce8005699e4be4dddbd24878c256ea9`

The older `output/releases/905d0b8/` release was not overwritten.

## Package verification

- `codesign --verify --deep --strict`: passed for the stable App copy.
- Architecture: thin `arm64` Mach-O.
- Signature: ad-hoc; no Team ID. This is an internal Apple Silicon package, not a notarized public release.
- `hdiutil verify`: valid checksum.
- The DMG was mounted read-only; the App inside passed strict deep signature verification.
- Bundle contents include `screens-guided-pi-data-preview.js`, `guided-pi-data-preview.css`, `screens-agent-run-history.js`, `screens-extraction.js`, `screens-icd.js`, and the Guided Copilot owner modules. This confirms that the package is not the older `905d0b8` UI.

## Native application smoke

Computer Use launched the stable App copy, not the build-tree bundle.

1. The Tauri startup screen reached the full EasyICU Home at dynamic loopback port `57302`.
2. `Start Guided Copilot` opened the production `#guided` conversation route.
3. `Data workspace` opened the production `#extraction` page with the native sidebar and recommended extraction action.
4. `Cmd+Q` performed a clean quit.
5. After quit, no listener remained on `57302`, and no App/backend process from this release path remained.

No provider was invoked, no credential was entered, and no real patient data was used during this release smoke.

## Claim boundary and next gate

This evidence supports a unified, locally packaged Apple Silicon product candidate and validates the demonstrated UI/data flows on desktop/laptop viewports. It does not establish notarization, Intel compatibility, complete-database clinical validity, provider-backed scientific correctness, or formal manuscript readiness. The remaining product decision is user acceptance followed by an explicit merge/push decision; `main` remains untouched until then.
