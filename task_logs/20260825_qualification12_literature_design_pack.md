# Qualification12 literature design pack

Date: 2026-08-25  
Branch: `codex/literature-to-design-contract`  
Starting HEAD: `aa9f4a67bf47cd315ca824baf9eb806558a3dafa`  
Provider calls / tokens / cost: `0 / 0 / $0`

## Outcome

Prepared a machine-readable, reviewed literature seed for all 12 Qualification
items. Each item has two recent or methodologically necessary open-access
sources and aggregate coverage of the seven planning dimensions:

1. study population;
2. time zero and analysis windows;
3. variable operationalization;
4. missingness and censoring;
5. primary model and sensitivity analyses;
6. table and figure completeness;
7. conclusion boundaries.

The tracked pack contains 24 card instances referencing 23 unique articles and
84 bounded source-backed facts. Twenty unique articles had published
supplements and those files were reviewed; three sources had no published
supplement. Published effect estimates are explicitly excluded from benchmark
expected answers.

## Artifacts

- Tracked seed pack:
  `benchmarks/meta_generalization/qualification12_literature_design_pack_20260825.json`
  (`sha256:6c91e987a33244bed54f3b83253032b6141c46183cfe63326bbef5ebb60c4fd6`)
- External source/full-text/supplement pack:
  `/Volumes/外置硬盘/easyicu_data/qualification12_literature_design_aa9f4a6_20260825/source_pack/`
- Zero-Provider audit receipt:
  `/Volumes/外置硬盘/easyicu_data/qualification12_literature_design_aa9f4a6_20260825/qualification12_literature_design_audit.json`
  (`sha256:e591b6406ea1285d37feceee367a0238eb7732a76e67abef08357f55633038d4`)
- Rebuilder: `tools/build_qualification12_literature_design_pack.py`
- Auditor: `tools/audit_qualification12_literature_design_pack.py`

The pack rebuild is byte-identical from the external manifests
(`rebuild_match=true`).

## Source pairs

| Item | Reviewed sources | Design use |
|---|---|---|
| MG01 | PMC9547456; PMC11867898 | ICU transfusion cohort design; current transfusion evidence/reporting |
| MG02 | PMC10186077; PMC9198202 | early sedation definitions, adjustment and ventilation outcomes |
| MG03 | PMC11579024; PMC7906666 | driving-pressure derivation; time-varying ventilation burden |
| MG04 | PMC9362765; PMC8116825 | new-onset AF ascertainment; beta-blocker confounding analogue |
| MG05 | PMC12528449; PMC11102905 | 48-hour prediction validation; SICdb data/time structure |
| MG06 | PMC9250715; PMC11059505 | trajectory clustering, stability and validation |
| MG07 | PMC12220764; PMC7810439 | measurement frequency; informative observation methods |
| MG08 | PMC9810617; PMC9848213 | MIMIC-IV linkage boundary; post-ICU outcome ascertainment |
| MG09 | PMC6132188; PMC12084561 | eICU medication coverage; cross-database mapping limits |
| MG10 | PMC8486643; PMC12425674 | sepsis antibiotic clocks; time-zero heterogeneity |
| MG11 | PMC7906666; PMC10685677 | real graded ventilation intensity and duration burden |
| MG12 | PMC10760471; PMC10008759 | competing-risk estimands; ICU AKI competing-event example |

Sources are design/reporting comparators, not numerical gold standards. For the
five deliberate fail-closed items, an article showing that a method is possible
with richer data does not authorize EasyICU to fabricate missing variables,
follow-up, exposure levels or an unsupported runner.

## Generic contract repair

`LiteratureAgent` previously re-screened every bound seed using title/context
heuristics, even when the seed already represented the exact research question
and carried a reviewed full-text card. This could discard a legitimate
`design_analogue` before planning. The generic repair preserves the reviewed
decision only when all of the following hold:

- the seed question exactly matches the sealed `ResearchContext` question;
- the citation has an included direct-comparator/design-analogue decision;
- a reviewed card exists for the same citation;
- the card role equals the screening role.

Ordinary Idea Mining seeds, question-mismatched seeds and records without a
reviewed card continue through context-bound re-screening. No MG-specific,
database-specific or sepsis-specific branch was added.

## Verification

- External digest + typed authority audit: `12/12 pass`, missing/extra tasks
  `[]/[]`, manifest errors `[]`, Provider `0`.
- Literature owner and pack tests:
  `33 passed, 1 deselected`.
- Benchmark spec, runner and architecture checks:
  `115 passed`, module graph unchanged, `7 warnings`.
- Ruff: all changed Python files passed.
- `git diff --check`: passed.
- Deterministic pack rebuild: byte-identical.

## Authority boundary and next gate

This completes the literature-design input and its zero-Provider preflight. It
does **not** start Qualification12, does not produce an analysis, and carries no
manuscript or paper authority. The next permitted experiment is one bounded
Planner canary using the strict Qualification12 profile. Acceptance requires
2-4 question-specific candidate designs and explicit seven-dimension
adopt/adapt/diverge decisions while preserving the expected fail-closed outcome
for MG08-MG12. Held-out27 remains closed.
