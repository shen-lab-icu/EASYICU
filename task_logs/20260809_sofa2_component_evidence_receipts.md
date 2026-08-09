# SOFA-2 component evidence receipts and ascertainment closure

- Date: 2026-08-09
- Branch: `fix/pi-workspace-review-20260809`
- Implementation: `7f7ba49a221943f98c86e355a6a735ef2c876561`
- Task: `SOFA2-COMPLETENESS-RECEIPTS-V5`
- Review baseline: `818e1137753df6e7ac328a251b809e10b1231b8c`

## Outcome

The remaining external-review P1 and three P2 findings are closed in code:

1. Each of the six SOFA-2 component callbacks now emits component-owned
   `*_observed` and `*_available` receipts derived from raw domain inputs.
   The production aggregate sums those receipts and no longer infers evidence
   completeness from a non-null component score. A missing physiology domain
   may still contribute the score's documented missing-as-normal zero while
   contributing zero to both completeness counts.
2. The clause-specific public concept is now
   `sofa2_cns_delirium_tx_ascertainment`. The old
   `sofa2_cns_ascertainment` remains a deprecated compatibility alias.
   Non-zero GCS rows return `not_score_relevant` for this clause rather than
   claiming whole-CNS completeness.
3. Deprecated `ClinicalConceptContract.golden_vector` again prefers the
   runtime vector, preserving the pre-split compatibility meaning.
4. Empty aggregate cohorts expose the same aggregate schema as non-empty
   cohorts; `keep_components=True` also materializes all component and receipt
   columns.
5. Direct and dictionary/resolver regressions cover the strict CNS score,
   proxy sensitivity, precise ascertainment concept, legacy alias, all six
   component receipts, and the aggregate receipt path.

The public description was narrowed to “database operationalization of SOFA-2
with conservative handling of unconfirmed delirium-treatment proxies”. The
catalog, generated conformance matrix, native catalog bootstrap, and route-owned
fallback counts were regenerated or synchronized.

## Verification

- Focused clinical/API/catalog/static route gate: `135 passed, 70 deselected`.
- Clinical conformance marker: `56 passed, 12770 deselected`.
- Focused SOFA-2 and clinical-contract files: `58 passed` before generated-doc
  synchronization; the final 135-test gate includes the synchronized result.
- Ruff on changed Python files: passed.
- `git diff --check`: passed.
- SOFA-2 and clinical-contract JSON parse: passed.
- Static web catalog generation check: passed.
- Progress lint is recorded in the follow-up handoff commit.

## Authority boundary

This patch does not run a Provider, access patient data, change Canonical9
paper authority, or upgrade any database from `mapping_only`. Treatment
ceiling, delirium indication/negative coverage, intermittent/non-renal RRT,
and `other_vaso >=1h` still require database-specific evidence and independent
clinical review.
