# Dev9 stratified PubMed retrieval and lineage shadow probe

Date: 2026-08-24

## Outcome

- Implementation commit: `b959469` on isolated branch `codex/dev9-quality-remediation`.
- The Research Agent literature owner now issues two or three complementary,
  prespecified PubMed strata, round-robin selects candidate PMIDs, and performs
  one bounded ESummary/EFetch hydration pass.
- Every retained citation records the exact query or queries that returned its
  PMID. Query membership still grants no comparator authority; the existing
  source-backed population/exposure/outcome/design screen remains fail closed.
- PubMed citation keys now include the source-issued PMID. This closes a real
  author/title/year collision that had cross-wired one E2 screening decision
  with a different displayed paper.
- Generic metadata normalization removes operational suffixes such as
  `window(s)` from clinical search terms, maps the owner concept `aki_stage` to
  its KDIGO literature identity, and derives bounded topic/study-intent clauses
  for contexts without a single primary exposure.
- No E1, Sepsis, Dev9 task id, expected result, database-specific, or manuscript
  answer branch was added. No Planner, Coder, Writer, or other Provider call was
  made.

## Exact Dev9 shadow probe

The probe read each saved `research_context.json` from the accepted Dev9 run,
removed only the historical runtime-only `materialized_inputs` field for current
strict-schema loading, and ran `LiteratureAgent(enable_pubmed=True)` without
writing into any old run directory.

| Case | PubMed strata | Returned | Direct-comparator candidates | Interpretation |
|---|---:|---:|---:|---|
| E1 | 3 | 8 | 0 | Retrieval works; no retained paper established the exact Sepsis-3 plus SOFA-2 exposure role. |
| E2 | 2 | 8 | 2 | Two source-backed lactate/in-hospital-mortality candidates survived; horizon-mismatched, pediatric, and lactated-Ringer records failed closed. |
| E3 | 2 | 8 | 1 | KDIGO identity repair changed the prior empty result into bounded AKI literature with one direct candidate. |
| M1 | 2 | 8 | 0 | Related bilirubin studies returned, but no record passed every declared axis. |
| M2 | 3 | 8 | 0 | Prediction literature returned; no single exposure exists, so P/E/O direct-comparator authority was not fabricated. |
| M3 | 3 | 8 | 0 | Sepsis classification/subphenotype papers ranked first; absence of a primary exposure keeps the direct-comparator role closed. |
| H1 | 3 | 8 | 0 | Mechanical-ventilation retrieval is now live; returned studies did not establish ventilation timing/status as the declared exposure with the exact endpoint. |
| H2 | 3 | 8 | 0 | Vasopressor/propensity literature returned; the Dev9 context has no authorized exposure/control axes, consistent with its positivity fail-closed result. |
| H3 | 3 | 8 | 0 | Sepsis trajectory/subphenotype papers ranked first; no direct P/E/O authority was inferred for the latent-class design. |

These counts are retrieval/screening diagnostics, not a gold answer, literature
review completion, novelty attestation, article-grade result, or publication
authorization. Direct candidates still require human comparison of population,
time zero, estimand, adjustment, and endpoint compatibility.

## Verification

- `ruff format --check`: passed after formatting.
- `ruff check`: passed.
- Focused and adjacent literature/scientific-review suite:
  `121 passed, 1 deselected`.
- `git diff --check`: passed.
- Live PubMed probe: all 9/9 contexts returned records after the generic repairs.
- Provider calls/tokens/cost: `0 / 0 / $0.00`.
- Matching image: not built; this change affects host-side preplan retrieval and
  does not justify an image/full-CI checkpoint before the remaining Dev9 gates
  close.

## Remaining gates

1. The current `direct_comparator` contract is intentionally P/E/O-shaped. M2,
   M3, H2, and H3 need a separately typed design-analogue/novelty-review path;
   they must not be made to pass by inventing a primary exposure.
2. Bind a new additive qualification/formal execution coordinate to live
   bibliographic retrieval; historical frozen profiles must remain immutable.
3. Obtain the outstanding author decisions, especially H1 non-proportional
   hazards strategy, then finish M3 deterministic article/display suffix.
4. Only after those gates close: zero-Provider replay of affected owners, one
   exact HEAD/image freeze, and one full exact-head CI before Qualification12.

