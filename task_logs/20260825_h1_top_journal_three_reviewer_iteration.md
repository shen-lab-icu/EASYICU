# H1 evidence-bound manuscript repair and top-journal comparison

> Superseded for numeric-provenance validation by `e081003` and
> `20260825_numeric_provenance_framework_audit.md`. The v7 PDF remains only a
> historical content-review artifact; use the framework-v8 reader for current
> analysis-only review.

Date: 2026-08-25 EDT

Task: `FIG2-DEV9-HELDOUT27` / H1 manuscript-only development iteration

Disposition: **major revision; analysis-only; not authorized for submission**

## Evidence reviewed

- Source run: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_h1_writer_rebind_71b9717_r2_20260825/h1_ventilation_survival/aware/run_20260825T204850_622da7`
- Final reader: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_h1_reader_final_v7_20260825/`
- Final PDF SHA-256: `c9da5c1d5890d5fe8336e338d1bf5b9f6c86bf4c642697acdd31de453af29026`
- Reader provenance: 20 numeric claims; zero Provider calls during deterministic reader repair; claim ceiling `analysis_only`; publication authorization `false`.
- Final fresh Writer run: 10 calls; 265,508 prompt tokens; 8,004 completion tokens; 273,512 total; provider-reported estimated cost USD 2.89520; no heuristic usage accounting.
- Primary published anchor: Urner et al., *Lancet Respiratory Medicine* 2020, PMID 32735841, DOI 10.1016/S2213-2600(20)30325-8, https://pubmed.ncbi.nlm.nih.gov/32735841/.
- Supporting time-varying design comparator: Schuijt et al., 2021, PMID 34869421, https://pubmed.ncbi.nlm.nih.gov/34869421/.
- Reporting standards: STROBE checklist, https://www.strobe-statement.org/fileadmin/Strobe/uploads/checklists/STROBE_checklist_v4_combined.pdf; RECORD statement, https://pmc.ncbi.nlm.nih.gov/articles/4595218/.

## Repair completed in this iteration

The execution owner now emits a typed `easyicu.survival_reporting/1` result that withholds an invalid constant hazard-ratio headline after proportional-hazards rejection and exposes the non-PH alternatives. Writer evidence prioritizes this contract. Two fresh Writer runs nevertheless omitted the numerical RMST contrast, so the reader now fails closed to an owner-issued deterministic projection rather than relying on repeated stochastic retries.

The final reader reports both analysis scales without conflating them:

- unadjusted 27-day post-landmark RMST: 24.841 versus 25.146 days; signed difference −0.305 days (95% CI −0.412 to −0.197; p=2.80×10^-8);
- adjusted interval-specific hazard ratios: 0–7 days 0.538 (95% CI 0.497–0.583), 7–14 days 0.559 (0.500–0.625), and 14–27 days 0.482 (0.430–0.541);
- constant Cox hazard ratio remains unauthorized as the headline estimate.

The reader also repairs the misplaced structured-abstract conclusion, resolves bibliography numbering and scientific notation, removes bare scaffold markers, and excludes the primary figure because the source run's publication-figure visual QA failed. The PDF is eight A4 pages; pages 1, 4 and 8 were visually inspected, and the LaTeX log contains no undefined-reference or overfull-box finding.

## Top-journal comparison matrix

| Dimension | Current H1 manuscript | Published anchor/package expectation | Assessment | Required action |
|---|---|---|---|---|
| Population and setting | Single MIMIC-IV source; 94,458 source stays, 78,600 landmark eligible, 78,580 complete cases | Lancet anchor used a prospective registry across nine ICUs and clearly separated descriptive, complete-case and imputed populations | Partial | State dates/sites and transportability limits; add independent validation before a broad clinical claim |
| Exposure and time zero | Binary first-observed invasive ventilation status within 24 h; 24-h landmark | Anchor treated ventilation intensity as a longitudinal exposure over the ventilation course | Major gap | Narrow the title/question to first-24-h status or execute a genuinely longitudinal duration/intensity analysis |
| Outcome and estimand | 28-day mortality; unadjusted RMST plus adjusted interval HRs | Anchor explicitly aligned longitudinal exposure, outcome, censoring and joint-model estimand | Partial | Keep RMST and adjusted HRs separated; justify which is primary and why their adjustment levels differ |
| Confounding and temporal role | Age, sex, Charlson and SOFA-2 used, but roster and baseline temporal roles are not author-confirmed | Published comparators justify adjustment variables and model longitudinal physiology/censoring | Blocker | Author-confirm exact roster and pre-exposure roles; reassess whether early SOFA can be post-exposure or mediating |
| Missingness and censoring | Complete-case analysis; ventilation timing is extensively missing; no imputation sensitivity | Lancet anchor added a full-cohort multiple-imputation joint-model analysis and addressed informative censoring | Blocker | Prespecify at least one missing-data/measurement sensitivity and one distinct exposure/window or cohort sensitivity |
| Robustness | No converged variant beyond the registered primary suite | Published packages report multiple sensitivity definitions/models and substantial supplements | Blocker | Execute at least two independent, source-supported robustness axes |
| Literature and novelty | Methods citations pass, but no direct comparator is bound to the executed plan; novelty remains unestablished | Top-journal papers position the exact population, exposure, estimand and contribution against direct prior art | Blocker | Run a dated search, bind retained direct comparators, and complete the six-dimension novelty matrix |
| Figures | Source data/contracts exist, but publication SVG failed overlap QA and is excluded from the reader | Anchor uses clear study flow and longitudinal association displays | Blocker | Redraw from registered source data and rerun visual/export QA; do not restore the current failed figure |
| Reporting | Reader is complete and traceable; source audit addresses 17/22 STROBE items | STROBE/RECORD expect setting dates, selection flow, missingness, confounder rationale, sensitivity analyses, generalisability, funding and data provenance | Major gap | Close the five open STROBE items and RECORD-specific coding/selection detail |

## Reviewer 1 — Clinical and editorial relevance

**Recommendation: major revision.** The repaired manuscript is substantially safer than the previous draft: it no longer promotes the rejected constant Cox effect, and it now gives an absolute-time contrast alongside time-varying adjusted associations. The question/title, however, still says “duration/status” while the executed exposure is a binary first-24-hour status representation. That mismatch is editorially material because the Lancet comparator studies ventilation intensity longitudinally across the course of ventilation. The manuscript must either narrow its claim and title to early observed status or execute the longitudinal duration/intensity study it currently implies. Generalisability from one routinely collected database also remains limited. External validation is a prerequisite for a broad clinical conclusion, not a sentence-level polish item.

Major concerns:

1. Exposure wording overstates what was measured.
2. The cohort represents ICU stays, not necessarily independent patients; patient identity authority is absent.
3. The lower hazards among early ventilated stays are readily confounded by selection, severity and survivor/landmark conditioning and must not be interpreted as benefit.
4. The final title, author list, affiliations, ethics, funding and contributor roles still require human authority.

Minor concerns:

1. Use “Kaplan–Meier plug-in RMST” consistently and keep “unadjusted” adjacent to every RMST mention.
2. Keep exposed and comparator group labels visible in the final figure and table.

## Reviewer 2 — Statistical and epidemiologic methods

**Recommendation: major revision.** The non-proportional-hazards response is appropriate: a constant HR is withheld, RMST is reported, and three prespecified time intervals are shown. The main unresolved issue is estimand coherence. The RMST contrast is unadjusted, whereas interval HRs are adjusted for age, sex, Charlson burden and SOFA-2. These answer different statistical questions and cannot jointly function as one “primary effect” without a hierarchy and rationale. The adjustment roster is Planner-selectable rather than author-confirmed, and the temporal status of early SOFA relative to ventilation classification is not established. Complete-case analysis is inadequate as the only missing-data route given the documented timing-variable missingness.

Major concerns:

1. Confirm the exact adjustment set and provide a causal/temporal rationale even though the estimand remains associational.
2. Prespecify and run at least two independent robustness axes: one missingness/measurement route and one exposure-window/cohort-definition route.
3. Explain informative censoring, landmark conditioning and exclusions with a denominated flow diagram.
4. Consider whether an adjusted absolute-time estimand or a carefully justified marginal standardisation would better align the primary absolute result with the adjusted association analysis; do not compute it post hoc without a new protocol.

Minor concerns:

1. Report exact event counts and follow-up summaries next to every analysis denominator.
2. Avoid treating very small p-values as the main evidence; magnitude, uncertainty and robustness are more important.

## Reviewer 3 — Reproducibility, reporting and figures

**Recommendation: major revision.** The evidence lineage is unusually strong for a development manuscript: every reported number in the reader is provenance-bound, the reader build is provider-free, and the failed figure is excluded instead of silently accepted. Literature citation coverage now passes for Introduction, Methods and Discussion. Nevertheless, the run itself did not perform a current direct-comparator search, the executed scientific step lacks an exact method/comparator citation binding, novelty is not established, and five STROBE items remain open. RECORD-specific reporting of cohort selection, code lists/algorithms and data-cleaning provenance needs to be reader-facing rather than confined to internal artifacts. The figure cannot be restored until overlap QA passes.

Major concerns:

1. Bind the Lancet direct comparator and relevant design-method sources into a new final plan, not only into this external review memo.
2. Close the dated search and six-dimension novelty-positioning receipt.
3. Complete the article display suite and redraw the source-data-backed figure.
4. Close the open STROBE/RECORD items and include a reproducible supplement inventory.

Minor concerns:

1. Replace the research-question title with a concise declarative title after the exposure decision is confirmed.
2. Keep the draft watermark until authorship and scientific review are closed.

## Synthesis and next gate

All three reviewers independently reach **major revision**. The writing/reader repair is complete enough for scientific discussion: the structured abstract is repaired, the non-PH results are complete, citations render, the PDF has no detected layout defect, and the failed figure is excluded. This does **not** make the study top-journal ready. The remaining blockers are scientific and authority-bound rather than copy-editing defects.

The next authorized iteration should begin only after two author decisions are recorded in a new StudyContext revision:

1. Is the intended exposure **first-24-hour ventilation status** or a genuinely longitudinal **duration/intensity** exposure?
2. Which exact adjustment roster and two independent sensitivity axes are approved, with their pre-exposure temporal rationale?

After those decisions: run the dated literature/comparator search; bind method and comparator keys to the plan; execute missingness plus exposure/window sensitivities; redraw the figure from registered source data; then repeat this three-reviewer comparison. External validation remains required before a broad submission claim.
