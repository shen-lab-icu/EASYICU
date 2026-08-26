# H1 literature-coverage and framework audit

Date: 2026-08-25 EDT

Task: `FIG2-DEV9-HELDOUT27` / framework-level literature authority repair

Disposition: **the nine-reference reader is under-supported by argument role;
do not repair it by manually appending references**

## Reproduced authority chain

The frozen run's `preplan_literature_bundle.json` contains 10 records, while the
strict v8 reader cites nine. The missing item is the contextual `ricu` software
record. More importantly, the bundle declares:

- `search_conducted=false`;
- no enabled or returning retrieval source;
- no dated query;
- no screening decision;
- no direct comparator or design-evidence card.

The nine cited records cover one database source, two clinical definitions and
six general methods/reporting sources. They do not include a ventilation study
that can position the clinical question. Writer attempted to cite
`time_varying_ventilation_intensity`, a key absent from the immutable bundle;
the fail-closed reader correctly removed the unsupported sentence. The final
count therefore reflects missing upstream authority, not a bibliography-format
problem.

The run's scientific-maturity owner already blocks publication with:

- `TOP_JOURNAL_LITERATURE_SEARCH_NOT_ESTABLISHED`;
- `NOVELTY_POSITIONING_NOT_ESTABLISHED`;
- `FINAL_PLAN_LITERATURE_BINDING_INCOMPLETE`;
- `SCIENTIFIC_STEP_METHOD_SOURCE_NOT_BOUND`;
- `APPLICABLE_METHOD_LAYERS_NOT_BOUND`.

The local manuscript citation audit nevertheless says the surviving exact keys
are "role-appropriate". Its pass means only that the remaining citation syntax
is bound and section-level minima are present; it must not be read as evidence
of adequate prior-art coverage or novelty.

## Dated search performed for this audit

Search date: 2026-08-25 EDT.

Primary verification source: PubMed. Grambsch and Therneau was verified against
the Oxford Academic journal record because the article has no PubMed record.

Search strata:

1. mechanical-ventilation intensity or duration and ICU mortality;
2. early invasive ventilation, first 24 hours, ICU mortality and landmark/time
   alignment;
3. MIMIC-IV ventilation and 28-day mortality;
4. proportional-hazards diagnostics and non-proportional hazards;
5. restricted mean survival time as an absolute survival estimand.

This was a targeted authority audit, not a systematic review. It does not
create PRISMA authority and must not be copied into a run as though the
pipeline performed the search.

## Citation-existence and metadata audit

The 12 records in the revised provider-free H1 pack were rechecked on
2026-08-25 using PubMed/NCBI ESummary and EFetch, Crossref DOI metadata, and
official publisher pages. All 12 resolve to real sources, but the shared
registry contained metadata defects that a title-only search would not catch:

- `anderson_landmark_1983` used an expanded title that is not the source title;
  the verified title is `Analysis of survival by tumor response.`
- `ricu_2023` was represented as generic `Software` with only a GitHub URL. It
  is a GigaScience journal article (2023, DOI
  `10.1093/gigascience/giad041`, PMID `37318234`).
- `vincent_sofa_1996`, `singer_sepsis3_2016`, and
  `johnson_mimiciv_2023` were real but lacked verified DOI and/or PMID fields in
  the offline registry.

The citation schema now carries source-issued bibliographic notices. The
verified STROBE record exposes its 2008 Annals of Internal Medicine erratum;
the MIMIC-IV record exposes both Scientific Data author corrections
(`10.1038/s41597-023-01945-2` and `10.1038/s41597-023-02136-9`). Writer digest
and BibTeX export preserve those notices. No checked record carried a PubMed
retraction or expression-of-concern relation at the audit date; this is a
dated metadata result, not a permanent guarantee.

The methodology pack is now schema v5 because the exact Anderson title and
bibliographic-notice contract change its frozen content. The public demo
literature rows were corrected from the same sources so the demonstration no
longer teaches stale metadata. Combined literature/method/plan/maturity/Copilot
static regression: 213 passed, 2 deselected; Node syntax, Ruff, and
`git diff --check` pass.

## Verified comparison and method matrix

| Source | Verified identifier | Role for this study | What it supports | Why it is not an exact direct comparator |
|---|---|---|---|---|
| Urner et al., *Lancet Respir Med* 2020 | PMID 32735841; DOI 10.1016/S2213-2600(20)30325-8 | High-impact design analogue | Time-varying ventilation intensity, longitudinal exposure, mortality, informative censoring | Includes only ventilated acute-respiratory-failure patients; exposure is intensity over time; outcome is ICU mortality |
| Yarnell et al., *Crit Care* 2023 | PMID 36814287; DOI 10.1186/s13054-023-04307-x | Near clinical/design comparator | Invasive-ventilation initiation, 28-day mortality, MIMIC-IV plus AmsterdamUMCdb, explicit intervention timing | Hypoxemic-respiratory-failure target-trial emulation; compares initiation thresholds, not early binary status |
| Chen et al., *J Intensive Care* 2023 | PMID 38031184; DOI 10.1186/s40560-023-00709-9 | Same-database related comparator | MIMIC-IV, ventilation duration/intensity and 28-day ICU mortality | Includes only ventilated patients and uses dynamic mechanical power rather than early status |
| Schuijt et al., *Crit Care* 2021 | PMID 34362415; DOI 10.1186/s13054-021-03710-6 | Clinical analogue | Ventilation intensity and 28-day mortality | COVID-19 acute respiratory failure; no unventilated comparator |
| Bellani et al., *JAMA* 2016 | PMID 26903337; DOI 10.1001/jama.2016.0291 | Clinical epidemiology context | Multicentre ICU ventilation practice, ARDS burden and mortality | Descriptive ARDS cohort, not the target exposure contrast |
| Amato et al., *N Engl J Med* 2015 | PMID 25693014; DOI 10.1056/NEJMsa1410639 | Mechanistic/clinical context | Ventilator intensity measure and survival | ARDS trial-data mediation analysis; not early ventilation status |
| Grambsch and Therneau, *Biometrika* 1994 | DOI 10.1093/biomet/81.3.515 | Method authority | Weighted-residual proportional-hazards diagnostics and time-varying coefficients | Method source, not clinical prior art |
| Royston and Parmar, *Stat Med* 2011 | PMID 21611958; DOI 10.1002/sim.4274 | Method authority | Prespecified RMST when proportional hazards are doubtful | Method source developed around randomized trials; observational interpretation still requires confounding limits |

No retrieved PubMed record matched all of these H1 coordinates at once:
general adult ICU population, invasive ventilation status classified within the
first 24 hours, an unventilated comparator, a 24-hour landmark, and 28-day
mortality. That absence is a search result requiring reproducible pipeline
confirmation, not evidence of novelty.

## Top-journal comparison

The main contrast with the published high-impact papers is not prose polish:

- Urner uses a clinically coherent ventilated population, longitudinal
  exposure, an analysis that addresses informative censoring, and multicentre
  prospective registry data. H1 uses a binary first-24-hour status in one
  database and cannot reconstruct complete ventilation history.
- Yarnell makes the decision point, eligibility, treatment strategies,
  time-varying confounding route, absolute 28-day risk and external-cohort
  comparison explicit. H1 has a landmark association but no causal initiation
  strategy and no external validation.
- Bellani provides broad multicentre clinical context and transparent patient
  flow. H1 is currently source-specific and the title still overstates
  "duration/status" relative to the executed binary status exposure.
- The H1 analysis appropriately rejects a constant-hazard headline, but its
  plan lacked explicit method authority for the PH diagnostic and RMST
  estimand. The manuscript cited neither source because neither existed in the
  shared method pack.

## Framework repair in this branch

The general methodology owner now includes two case-neutral cards:

1. `survival_assumption` — Grambsch and Therneau governs proportional-hazards
   diagnostics and the rule that a rejected assumption cannot support a
   constant-HR headline.
2. `survival_estimand` — Royston and Parmar governs prespecification and
   reporting of RMST, including the restriction horizon, group values,
   contrast, uncertainty and adjusted/unadjusted status.

The plan reviewer now requires those layers only when a survival plan actually
declares Cox/PH-diagnostic or RMST outputs. This is output-sensitive and
case-neutral; it does not force ventilation-specific literature into a global
prompt. New runs receive both verified sources through the frozen method pack,
and a survival plan that produces the corresponding outputs without binding
their cards fails locally.

The literature owner now also exposes one shared manuscript-citable
projection. The immutable bundle remains a candidate universe for search
audit, but:

- a curated record with no retrieval-screening decision remains citable;
- a retrieved record is citable only with an unambiguous explicit `include`;
- an `exclude` or conflicting disposition fails closed;
- Planner allowed keys, hypothesis generation, Writer digest, manuscript
  audit/repair, scientific-plan binding, LaTeX and both bibliography exporters
  consume that same projection.

This closes a distinct authority leak: before the repair, a retrieved record
could carry `disposition=exclude` yet still appear as `curated_context` in the
Writer digest, pass the exact-key manuscript audit and be exported to BibTeX.
Screened-out records are no longer silently deleted; they remain in
`LiteratureBundle.citations` with their query and decision so the search flow
is inspectable without granting manuscript authority.

A provider-free replay of the frozen H1 context against the revised pack yields
12 curated records and requires exactly `reporting_standard`,
`time_alignment`, `survival_assumption`, and `survival_estimand`. The frozen
historical plan remains unbound; this replay verifies the new pre-plan contract
without rewriting old evidence.

A bounded live PubMed canary after the query-stratification and authority
projection repair issued four queries, retrieved eight unique records and
retained 20 candidate/curated records in total. Urner 2020
(`urner_time_2020_32735841`) was retained as a candidate rather than being
erased by global ranking. All eight retrieved records failed the exact H1
screen, so PRISMA remained `eligible=0`, `included=0`; the citable projection
contained the 12 curated records only, and Urner correctly remained
`disposition=exclude`, `citable=false`. This demonstrates retrieval recall and
citation authority as separate contracts. It does not establish that the
screen is complete or that H1 is novel.

Focused regression: 170 passed, 2 deselected. Adjacent Pipeline/Writer/
hypothesis authority regression: 60 passed, 1 skipped, 296 deselected. Ruff and
`git diff --check` pass.

## Reference-count conclusion

There is no defensible universal target number. For this concise manuscript,
nine is still too few because the clinical-comparison and survival-estimand
roles are empty, not because nine violates a journal quota. A reasonable next
reader would likely land near 15--20 unique references after the pipeline
retains and screens the strongest clinical analogues plus the two missing
survival-method sources. The repaired framework currently authorizes 12, not
20: adding the audited clinical analogues still requires source-backed
role review rather than changing their screen to `include` for the sake of a
larger count. The final number must be the consequence of closed argument
roles, deduplication and exact citation use.

The current v8 reader remains the corrected analysis-only artifact. It should
not be manually edited or promoted. A new reader requires a dated,
receipt-bound retrieval seed, exact record-level screening, literature-bound
plan regeneration and the existing scientific-maturity gates.
