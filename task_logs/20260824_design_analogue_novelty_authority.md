# Dev9 design-analogue and novelty-authority closure

Date: 2026-08-24

## Outcome

- Implementation commit: `50f2e32` on isolated branch
  `codex/dev9-quality-remediation`.
- A sealed research context without a `primary_exposure` may now retain a
  typed `design_analogue` when a PubMed record has an eligible observational
  publication type plus source-backed ICU population, clinical-topic, and
  analysis-intent evidence.
- A design analogue is never labelled a direct exposure/outcome comparator.
  Contexts with a declared primary exposure continue to require the existing
  direct-comparator contract.
- Planner review and scientific-maturity gates now require the appropriate
  comparison-source type and require its exact citation key to govern a primary
  analysis step.
- Novelty remains unsigned until an independent reviewer completes all six
  prespecified comparison dimensions and binds the exact context, plan, and
  literature digests.
- No benchmark task id, expected numeric result, manuscript answer, or
  E1/Sepsis-specific execution branch was added.

## Zero-Provider live PubMed shadow probe

The probe loaded the current saved Dev9 research contexts from the mounted
external drive, removed only the historical runtime-only `materialized_inputs`
field for strict current-schema loading, and ran
`LiteratureAgent(enable_pubmed=True, pubmed_retmax=8)` without writing to the
old run directories.

| Case | PubMed strata | Returned | Included design analogues | Interpretation |
|---|---:|---:|---:|---|
| M2 | 3 | 8 | 5 | Five ICU mortality prediction-model designs passed; single-predictor association papers were rejected. |
| M3 | 3 | 8 | 2 | Two source-backed sepsis endotype/subphenotype designs passed. |
| H2 | 3 | 8 | 0 | Returned vasopressor literature did not establish both topic and causal/propensity design in the title; the question stays fail closed. |
| H3 | 3 | 8 | 1 | PMID 35786445, the organ-dysfunction trajectory subphenotyping study, passed as a design analogue. |

The five M2 candidates were PMIDs `40833967`, `38369749`, `40073651`,
`37001474`, and `36312291`. The two M3 candidates were PMIDs `28864056` and
`42187539`. These are candidate comparison sources, not gold answers,
independent novelty attestations, article-grade results, or publication
authorization.

## Verification

- Focused plus adjacent literature/planning/maturity/pipeline/reporting tests:
  `194 passed, 292 deselected`.
- Ruff: passed.
- `git diff --check`: passed.
- Live probe: M2 `5`, M3 `2`, H2 `0`, H3 `1` design analogues.
- Provider calls/tokens/cost: `0 / 0 / $0.00`.
- Matching image/full CI: not run; they remain a single freeze checkpoint after
  the remaining Dev9 blockers close.

## Remaining gates

1. H2 still has no acceptable comparison source and remains scientifically
   blocked rather than being made to pass.
2. An author decision is still required for the H1 non-proportional-hazards
   strategy; M3 still needs its deterministic article/display suffix closed.
3. Enable live bibliographic retrieval only through a new additive execution
   profile; do not mutate historical frozen profiles.
4. Then run the affected-owner zero-Provider replay, build one exact-HEAD image,
   and run one full exact-head CI before Qualification12. Held-out27 remains
   out of scope until those gates close.
