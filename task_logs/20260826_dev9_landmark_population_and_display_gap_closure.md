# Dev9 landmark population and display gap closure

Date: 2026-08-26  
Authority: AI development review; all scientific outputs remain `analysis_only`  
Numerical answer-key policy: published articles were used as design and display comparators, never as expected effect sizes.

## What was repaired

1. The landmark spline runtime now emits a signed nested population flow, marginally standardised absolute-risk curve, and (where prespecified) a variable-opportunity sensitivity. The old relative-effect curve, contrasts, and linear sensitivity are replayed from the same frozen cohort and must remain numerically identical.
2. Table 1 and the primary model population can no longer silently appear to be the same cohort. A generic population-alignment owner requires either equal denominators or an explicit scope label.
3. A descriptive measurement-state table can no longer satisfy a requested adjusted absolute-risk product merely because both contain the words “absolute risk”.
4. Landmark result figures now separate adjusted absolute risk, the continuous relative association, and prespecified contrasts. Routine missingness/measurement plots remain supplementary unless the measurement process is itself the research question.
5. Article-suite QA no longer equates figure-file count with scientific completeness. One main figure is acceptable for a narrow question only when it is a multi-panel scientific-result figure bound to at least two declared source-data products.
6. Log-ratio axes now use shared plain-number clinical ticks and suppress overlapping scientific-notation minor labels.
7. Failed-closed analyses must still emit diagnostic figures and tables, but those displays remain diagnostics and cannot be relabelled as effects or selected phenotype solutions.

## Nine-case scientific and display audit

| Case | Current result/display judgment | Closed in this change | Remaining boundary (not repairable by drawing another panel) |
|---|---|---|---|
| E1 Sepsis-3 | Result-bearing article display | Existing denominator, absolute risk, adjusted association, sensitivity, and component audit retained | Repeated-stay dependence cannot be resolved without patient identity; chart review and external reproduction remain absent |
| E2 lactate | Result-bearing narrow association article | Landmark flow, complete-case scope, standardised absolute risk, variable-opportunity sensitivity, and readable three-panel primary figure | Residual measurement-by-indication and external reproduction remain |
| E3 KDIGO | Result-bearing gradient article | Source-cohort Table 1 scope is explicit | Baseline kidney definition validation, prospectively authorised temporal estimand, and external reproduction remain |
| M1 bilirubin | Result-bearing measurement-selection article | Landmark flow, complete-case scope, and standardised absolute risk added; missingness remains a main scientific result because it is central to the question | Measurement-by-indication and external reproduction remain |
| M2 prediction | Internally validated prediction article | Existing discrimination, calibration, Brier, decision-curve, leakage and repeated-split displays retained | No temporal/external validation or recalibration; not deployable |
| M3 static subtype | Negative/unstable result-bearing article | Existing structure, profiles, stability and algorithm-disagreement displays retained without forced naming | Missing-data method source layer and external phenotype reproduction remain |
| H1 ventilation/survival | Result-bearing survival article with method limitation | Existing risk set, PH diagnostic and PH-free RMST display retained | Informative missingness/censoring sensitivity and external reproduction remain |
| H2 vasopressor causal | Failed-closed diagnostic article | One main identifiability diagnostic figure and one main diagnostic table are required instead of zero displays | A verified comparator and positivity are absent; no causal effect may be estimated |
| H3 trajectory clustering | Failed-closed diagnostic article | Candidate-selection diagnostic plus feature-availability supplement retained instead of zero displays | No stable internally selected class solution; alternative algorithms and external reproduction remain |

## Comparator interpretation

The full-text/supplement shadow review remains the comparator source. It covers the declared seven dimensions: population, time zero/window, variable definition, missingness/censoring, model/sensitivity, figure/table completeness, and conclusion boundary. The comparator’s number of figures is not copied mechanically: a broad lactate paper analysing admission, maximum, and time-weighted lactate may need several primary figures, while E2 asks one narrower post-landmark maximum-lactate estimand and can use one evidence-dense three-panel primary figure plus supplementary measurement diagnostics.

## Acceptance boundary

The final package is accepted for Dev9 development inspection only when all nine inventories have no unresolved placement/purpose or planning gaps, all exported figures have source data and contracts, E2/M1 legacy relative results remain numerically identical in provider-free replay, and focused contract/rendering tests pass. This does not authorize a manuscript claim, Qualification12 result, Held-out27 result, clinical conclusion, or paper submission.
