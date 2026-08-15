# SOFA-2 production conformance v3

- Task: `SOFA2-PRODUCTION-CONFORMANCE-V3`
- Branch: `fix/pi-workspace-review-20260809`
- Reviewed baseline: `8dec86e1b85727e8a42afa89183dac16a9238cb2`
- Implementation commit: `4b82772` (`fix(clinical): close SOFA-2 production conformance gaps`)
- Scope: data foundation + Agent Clinical Trust Gate

## Outcome

The three remaining review blockers are closed in the production path:

1. A valid P/F or S/F observation is scoreable when persistence is unknown. Only an explicit `oxygenation_sustained_1h=False` excludes a documented transient episode shorter than one hour. The shipped dictionary → `ConceptResolver.load_concepts()` → component callback → scorer path now has a resolver-level golden regression; P/F 180 produces respiratory score 2.
2. The aggregate callback applies groupwise last-observation-carried-forward before the 24-hour worst-value window. A previously observed component therefore remains available after hour 24 when no new measurement exists.
3. Structural component absence remains `NaN` for completeness accounting and is imputed only by the score sum. `sofa2_n_components` now reports 6, 5, or 0 for truly observed domains instead of counting synthetic zero columns.
4. Every SOFA-2 component and the aggregate contract now names the real resolver, production callback, and exact dictionary-owned runtime inputs. Contract validation rejects dictionary/contract/fixture input drift.

The older `blood_cell_ratio` callback also fails closed when WBC cannot be loaded or time-aligned. It returns an unavailable percentage plus a stable assessment reason instead of passing through an absolute numerator.

## CI reconciliation

The reviewed baseline's Main CI had four pre-existing failures. This patch closes their stale generated-state/test isolation causes without weakening production gates:

- regenerated catalog count: 10 clinical contracts;
- re-adjudicated the provider-free, patient-data-free resource baseline after dictionary digest changes; prompt-byte summaries did not move;
- isolated two superseded-finding lifecycle tests from the independent scientific-capability claim-ceiling gate.

## Verification

- Focused clinical/callback gate: `184 passed`.
- Expanded SOFA, resolver, concept catalog, completeness, missingness, resource and historical-CI gate: `247 passed, 13 skipped` (real-database partition tests skipped by marker).
- Historical Main-CI failure subset plus clinical contracts: `39 passed`.
- Ruff: passed for all changed Python files.
- JSON parsing: passed for the contract registry, SOFA-2 dictionary, and changed golden fixtures.
- Generated clinical matrix/catalog equality: passed.
- `git diff --check`: passed.

No provider call and no patient-data read was used for this work.

## Remaining declared limitations

- Six-database results remain `mapping_only`; no database is promoted to algorithm-golden or independently clinically validated.
- Renal mappings do not yet provide reliable permanent RRT termination and non-renal-only indication signals in every database.
- `other_vaso` does not yet prove continuous intravenous infusion for at least one hour in every database, and treatment-ceiling/unavailable-vasopressor state is not mapped.
- Respiratory treatment-ceiling/unavailable-support and explicit transient-episode ascertainment are not mapped across all databases. Unknown transience no longer suppresses a valid ratio, while an explicitly transient observation remains excluded.

These are typed ascertainment limitations, not silently synthesized evidence.
