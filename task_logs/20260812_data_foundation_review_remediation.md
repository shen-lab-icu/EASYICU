# Data foundation static review adjudication and remediation

- Date: 2026-08-12
- Baseline: `fix/pi-workspace-review-20260809@b4015bf`
- Implementation commit: `854ff4c`
- Module / task: `DATA-FIX1 / DATA-FOUNDATION-REVIEW`
- Scope: release foundation gate, native-export provenance, legacy projection cache, source-config fail-closed behavior, registry compatibility metadata, concept overlay semantics

## Adjudication

The review correctly identified one open release-state blocker and four reproducible code defects. Its database-metadata finding was directionally correct but broader than the current typed profile contract: `data-sources.json` owns display and ICU-identifier metadata, not every raw event/value/time column. The production-binding proposal and strict callback-parameter schema are valid architecture work, not defects that should be mixed into this correctness patch.

| Finding | Decision | Result |
|---|---|---|
| Critical 1: concept foundation not finalized | **Correct; remains OPEN as release state** | The shipped lock is intentionally `finalized=false`; its recorded concept/SOFA2 hashes also differ from the current files. The AUMC RRT code fix is present, but only a new six-database extraction, QC pass and deliberate lock finalization can close this item. No dictionary hash or finalized flag was rewritten to simulate completion. |
| Critical 2: sealer does not enforce the lock/RRT correction | **Correct; code CLOSED** | The minimum semantic ancestor is now the AUMC interval fix `187c6123ea59b4d904a2594d755de4186dc249b5`. The sealer requires a finalized regular lock, exact concept/SOFA2 hashes, and matching extraction-manifest hashes for concept dictionary, SOFA2 dictionary, clinical contracts and data sources. Release/native contract revisions were advanced. |
| Major 1: legacy projection-cache order dependence | **Correct; CLOSED** | A table cache entry is reused only when its columns cover the requested projection; a later wider request reloads instead of silently omitting callback inputs. Preload uses the same coverage rule. |
| Major 2: duplicate database metadata | **Partly correct; actionable drift CLOSED** | Supported public keys and display labels now come from the typed registry. AUMC is correctly marked as a database-wide millisecond offset clock. Raw event/value/unit compatibility columns remain in their existing owner because the current profile does not declare them; moving them without a typed replacement would be speculative. |
| Major 3: `load_src_cfg()` unknown-source fail-open | **Correct; CLOSED** | Unknown names now propagate the typed registry/profile `KeyError`. Custom sources require an explicitly supplied `DataSourceRegistry`; no empty source config is synthesized. |
| Major 4: conflicting overlay semantics | **Correct; CLOSED** | Resource extras and `from_multiple_json()` now share `ConceptDictionary.update()` patch semantics: sources merge by database and omitted metadata survives. Existing SOFA2 overlays now retain published bounds directly. |
| Major 5: general production binding for all phenotypes | **Valid P2 architecture work; not part of this hotfix** | SOFA2's production-binding gate should be generalized in a separate characterized change with typed contracts for each phenotype. No unbounded cross-phenotype refactor was attempted here. |

The additional suggestions to move all callback parameters under an explicit `params` object and split very large clinical modules also remain separate migrations. Both require dictionary/caller characterization and are not necessary to close the demonstrated release, cache, configuration and overlay defects.

## Owner contracts

- Native extraction owns four content hashes in each `_manifest.json`: `concept_dictionary_sha256`, `sofa2_dictionary_sha256`, `clinical_contracts_sha256`, and `data_sources_sha256`.
- The full-six release sealer owns the finalized-lock check, semantic minimum-commit ancestry, cross-database hash equality and immutable release receipt.
- `ConceptLoader` compatibility cache owns projection coverage and may not reuse a narrower frame for a wider concept request.
- `DataSourceRegistry` owns explicit custom-source registration; `load_src_cfg()` resolves only registered keys/aliases.
- `ConceptDictionary.update()` owns the single patch merge contract used by all multi-resource loaders.

## Verification

- New negative regressions on the old implementation: `11 failed / 11 selected`.
- Same regressions after the patch: `11 passed`.
- Full directly related set: `125 passed` covering release sealer, native export, legacy loader, patient-filter/config, resources/overlay and database profiles.
- Adjacent data/contract set: `253 passed / 11 real-data skipped` covering clinical contracts, catalog consistency, metadata sidecars, derived/logical concepts, data correctness, API ownership/cohort, Web column metadata, publication QC, extraction grouping and re-extraction tooling.
- Final release/native subset after contract revision bump: `48 passed`.
- Ruff and `git diff --check`: passed.
- The real repository lock was explicitly probed and correctly raised `ReleaseValidationError: concept foundation lock is not finalized`; therefore no existing full-six product can be falsely resealed as current.

Per the development-test policy, no full exact-head matrix was started locally. No full-six extraction was launched while other high-memory work was active. Critical 1 remains blocked on a future clean-commit six-database re-extraction, QC and intentional lock finalization.
