#!/usr/bin/env python3
"""Build the reviewed Qualification12 literature-design seed pack.

The tracked pack stores bounded paraphrases and cryptographic receipts, not
article bodies or published effect estimates. Source files remain in the
external evidence pack supplied with ``--source-pack-root``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REVIEWED_AT = "2026-08-25T12:00:00-04:00"
DIMENSIONS = (
    "study_population",
    "time_zero_and_windows",
    "variable_operationalization",
    "missingness_and_censoring",
    "primary_model_and_sensitivities",
    "table_and_figure_completeness",
    "conclusion_boundaries",
)


ITEMS: dict[str, dict[str, Any]] = {
    "MG01": {
        "query": "intensive care red blood cell transfusion mortality early cohort",
        "sources": [
            ("PMC9547456", "design_analogue", "ICU transfusion cohort and confounding-control design."),
            ("PMC11867898", "design_analogue", "Current critical-care transfusion evidence and reporting framework."),
        ],
        "evidence": {
            "study_population": ("PMC9547456", "The cohort enrolled adults across multiple ICUs but analyzed ICU survivors, so its population restriction must not be copied into an admission-based eICU target trial.", "Methods: study population; Discussion: selection bias"),
            "time_zero_and_windows": ("PMC9547456", "Transfusion was observed during the ICU stay and follow-up began at ICU discharge; this differs from a first-24-hour exposure anchored at ICU admission and exposes survivor-selection risk.", "Methods: exposure and follow-up; Supplementary analyses"),
            "variable_operationalization": ("PMC11867898", "The guideline separates transfusion strategies by explicit hemoglobin thresholds and clinically defined subgroups; an eICU study must instead prespecify a documented RBC administration rule and cannot equate an order with administration without validation.", "Guideline PICO questions and recommendations"),
            "missingness_and_censoring": ("PMC9547456", "The cohort used multiple imputation and a missingness-aware sensitivity approach, while also acknowledging selection and loss-to-follow-up mechanisms; eICU exposure absence requires a separate structural-absence audit.", "Statistical analysis; Supplementary methods"),
            "primary_model_and_sensitivities": ("PMC9547456", "The analysis combined survival models, propensity weighting and alternate estimands; this supports prespecified confounding adjustment and overlap diagnostics, not automatic reuse of a one-year survivor model.", "Statistical analysis; propensity and sensitivity sections"),
            "table_and_figure_completeness": ("PMC11867898", "The guideline presents evidence profiles by PICO, intervention threshold, outcome and certainty; the study should likewise show cohort flow, baseline balance, exposure prevalence, adjusted effects and absolute mortality with uncertainty.", "Evidence tables and recommendation summaries"),
            "conclusion_boundaries": ("PMC9547456", "The authors explicitly retain residual-confounding and survivor-selection limitations; the Qualification12 result may support an adjusted association only, not a causal claim that transfusion changes mortality.", "Discussion: limitations"),
        },
    },
    "MG02": {
        "query": "mechanically ventilated early deep sedation prolonged mechanical ventilation cohort",
        "sources": [
            ("PMC10186077", "direct_comparator", "Prospective early-sedation cohort with ventilation outcomes."),
            ("PMC9198202", "direct_comparator", "Independent early deep-sedation cohort and reporting analogue."),
        ],
        "evidence": {
            "study_population": ("PMC10186077", "A prospective multicenter cohort enrolled adults expected to require invasive ventilation and sedation beyond 48 hours; that eligibility condition can select for prolonged ventilation and must be adapted for HiRID.", "Methods: patients and study setting"),
            "time_zero_and_windows": ("PMC10186077", "Sedation depth was summarized during the first 48 hours after ventilation-related ICU care, whereas the target question requires an exact ventilation-start time zero and a first-24-hour exposure window.", "Methods: early sedation period"),
            "variable_operationalization": ("PMC9198202", "Deep sedation was defined from recorded RASS or SAS values using prespecified cutoffs; HiRID tier 0-3 therefore needs an auditable mapping from irregular scores and a rule for patients with no assessment.", "Methods: exposure definition"),
            "missingness_and_censoring": ("PMC10186077", "Sedation assessments and follow-up eligibility were protocol-defined, but routine HiRID scores are irregular; measurement frequency and sparse assessment must be reported rather than treated as exchangeable missing values.", "Methods: data collection; study flow"),
            "primary_model_and_sensitivities": ("PMC10186077", "The cohort used adjusted time-to-event models and propensity-score matching; an ordinal dose-response analysis should add a trend test, flexible tier contrasts and sensitivity to the tier-construction rule.", "Statistical analysis; matched analyses"),
            "table_and_figure_completeness": ("PMC9198202", "The report includes participant flow, baseline comparisons, adjusted outcome analyses and supplementary covariate detail; the target analysis should additionally plot tier-specific absolute prolonged-ventilation risk with uncertainty.", "Figures, main tables and supplement"),
            "conclusion_boundaries": ("PMC9198202", "The observational dual-center design cannot eliminate confounding by illness severity or treatment indication; a monotonic tier association would not establish that deeper sedation causes prolonged ventilation.", "Discussion: limitations"),
        },
    },
    "MG03": {
        "query": "driving pressure extubation mortality time varying mechanical ventilation cohort",
        "sources": [
            ("PMC11579024", "direct_comparator", "First-day driving-pressure cohort with explicit derivation and missing-data analysis."),
            ("PMC7906666", "design_analogue", "Prospective time-varying ventilation-intensity analysis."),
        ],
        "evidence": {
            "study_population": ("PMC11579024", "The MIMIC-IV cohort restricted to acute hypoxemic respiratory failure and required plateau pressure plus PEEP in the first ICU day; this documents the severe complete-case selection that driving-pressure eligibility can create.", "Methods: participants; Figure 1"),
            "time_zero_and_windows": ("PMC7906666", "Ventilation intensity was handled longitudinally during mechanical ventilation, illustrating that ventilator exposure belongs on a ventilation-aligned clock rather than an ICU-admission clock.", "Methods: longitudinal exposure assessment"),
            "variable_operationalization": ("PMC11579024", "Initial driving pressure was calculated as plateau pressure minus PEEP from first-day values and then dichotomized; the target must certify compatible measurements and prespecify how repeated pairs are selected.", "Methods: data collection and exposure definition"),
            "missingness_and_censoring": ("PMC11579024", "Patients lacking either plateau pressure or PEEP were excluded and remaining covariate gaps were multiply imputed; the target should quantify both derivation failure and downstream covariate missingness.", "Methods: exclusions and statistical analysis"),
            "primary_model_and_sensitivities": ("PMC7906666", "The registry study modeled ventilation intensity as time-varying and examined cumulative burden with adjusted survival methods; extubation and death require an estimand that respects their competing terminal states rather than a cause-naive composite shortcut.", "Methods: statistical models and sensitivity analyses"),
            "table_and_figure_completeness": ("PMC7906666", "The report pairs exposure distributions with adjusted intensity-response and cumulative-burden displays; the target should show risk sets, event counts and the consequences of treating death jointly or separately from extubation.", "Main figures and appendix"),
            "conclusion_boundaries": ("PMC11579024", "The single-center retrospective study notes residual confounding, measurement availability and population restrictions; a first-day driving-pressure association cannot prove a ventilator-setting intervention effect.", "Discussion: limitations"),
        },
    },
    "MG04": {
        "query": "intensive care new onset atrial fibrillation beta blocker observational cohort",
        "sources": [
            ("PMC9362765", "design_analogue", "ICU new-onset atrial-fibrillation ascertainment and outcome design."),
            ("PMC8116825", "design_analogue", "Premorbid beta-blocker exposure and confounding-control analogue."),
        ],
        "evidence": {
            "study_population": ("PMC9362765", "The ICU epidemiology study distinguishes new-onset atrial fibrillation from pre-existing disease using linked clinical history; the MIMIC-IV cohort needs an explicit prior-AF lookback and exclusion receipt.", "Methods: cohort and AF history"),
            "time_zero_and_windows": ("PMC9362765", "New-onset AF is observed during critical illness after ICU entry; exposure classification must therefore finish before outcome surveillance begins to avoid immortal-time and reverse-causation bias.", "Methods: AF timing and follow-up"),
            "variable_operationalization": ("PMC9362765", "AF ascertainment combines clinically recorded rhythm information with diagnostic history; absence of a code alone is not equivalent to a validated no-AF state.", "Methods: outcome ascertainment"),
            "missingness_and_censoring": ("PMC9362765", "Routine rhythm documentation varies with monitoring and clinical concern; the target should report ascertainment sources and monitoring opportunity instead of assuming undocumented AF is uniformly absent.", "Methods and limitations: routine-data ascertainment"),
            "primary_model_and_sensitivities": ("PMC8116825", "The beta-blocker study uses multivariable adjustment and propensity-based analyses for a strongly confounded treatment exposure; the target additionally needs overlap, treatment-indication and early-outcome sensitivity checks.", "Statistical analysis and supplementary propensity analyses"),
            "table_and_figure_completeness": ("PMC9362765", "The epidemiology report separates cohort construction, AF incidence and downstream outcomes; the target should add exposure balance, propensity overlap and cumulative AF incidence with death/discharge handling made explicit.", "Study flow, tables and outcome figures"),
            "conclusion_boundaries": ("PMC8116825", "The exposure analogue is premorbid beta-blocker use in sepsis and mortality, not early ICU treatment for AF prevention; it supplies design cautions but no direct efficacy evidence for the target question.", "Discussion: interpretation and limitations"),
        },
    },
    "MG05": {
        "query": "ICU dynamic 48 hour mortality prediction vital signs calibration external validation",
        "sources": [
            ("PMC12528449", "direct_comparator", "Dynamic 48-hour ICU mortality prediction with validation and interpretability."),
            ("PMC11102905", "design_analogue", "SICdb content, time structure and cross-database comparison."),
        ],
        "evidence": {
            "study_population": ("PMC12528449", "The dynamic prediction study defines eligible ICU stays and separates development from external validation cohorts; SICdb analysis must use patient-level splits and state whether repeated stays cross partitions.", "Methods: cohorts and validation design"),
            "time_zero_and_windows": ("PMC12528449", "Predictions are updated on a rolling ICU timeline for a future 48-hour horizon; the target instead fixes first-six-hour predictors at admission and must exclude information recorded after that window.", "Methods: prediction times and horizon"),
            "variable_operationalization": ("PMC11102905", "The SICdb descriptor documents its high-resolution vital-sign structure and differences from MIMIC-IV; each summary feature needs unit harmonization, sampling rules and a stable stay/patient key.", "Dataset structure and comparative analysis"),
            "missingness_and_censoring": ("PMC12528449", "The model describes preprocessing and incomplete data handling across time; first-six-hour exclusion can create informative selection, so window completeness and excluded-patient outcomes must be audited.", "Methods and supplement: preprocessing"),
            "primary_model_and_sensitivities": ("PMC12528449", "The study evaluates discrimination, calibration and external performance with interpretable machine learning; the target should prespecify patient-level resampling, calibration intercept/slope, Brier score and a simple baseline model.", "Methods: model development and validation"),
            "table_and_figure_completeness": ("PMC12528449", "Performance is accompanied by cohort detail, temporal behavior, calibration and feature interpretation; AUROC alone is insufficient for acceptance.", "Main performance figures and supplementary validation"),
            "conclusion_boundaries": ("PMC11102905", "SICdb is a single-center data resource with database-specific structure and coverage; internal discrimination cannot establish transportability or clinical utility without external validation and decision-impact testing.", "Discussion: database comparison and limitations"),
        },
    },
    "MG06": {
        "query": "sepsis physiologic trajectory subphenotype reproducibility external validation",
        "sources": [
            ("PMC9250715", "direct_comparator", "Multi-cohort organ-dysfunction trajectory phenotyping."),
            ("PMC11059505", "direct_comparator", "Time-series sepsis subphenotypes with validation."),
        ],
        "evidence": {
            "study_population": ("PMC9250715", "The study derives sepsis trajectory phenotypes in one cohort and evaluates them in multiple independent cohorts; the MIMIC-IV target must define septic-shock onset and derivation eligibility before clustering.", "Methods: study cohorts and inclusion"),
            "time_zero_and_windows": ("PMC9250715", "Organ-dysfunction trajectories are aligned to a clinically defined early sepsis period and followed over repeated intervals; ICU admission is not an interchangeable substitute for shock onset.", "Methods: trajectory construction"),
            "variable_operationalization": ("PMC11059505", "The time-series study standardizes repeated physiologic variables before clustering and then characterizes groups clinically; feature units, binning, scaling and directionality must be frozen before outcome comparison.", "Methods: variables and time-series preprocessing"),
            "missingness_and_censoring": ("PMC11059505", "Trajectory preprocessing includes imputation and minimum data requirements; sparse stays should be flagged and cluster stability rechecked under alternate imputation rather than silently completed.", "Methods and supplement: missing-data processing"),
            "primary_model_and_sensitivities": ("PMC9250715", "The multi-cohort study combines trajectory clustering with cross-cohort validation and alternate characterization; acceptance requires cluster-number justification, resampling stability and a held-out reproducibility metric.", "Methods: clustering, validation and sensitivity analyses"),
            "table_and_figure_completeness": ("PMC9250715", "The report shows cohort flow, trajectory profiles, phenotype composition and external-cohort behavior; outcome-by-cluster tables are descriptive and should be separated from the clustering objective.", "Main figures, tables and supplement"),
            "conclusion_boundaries": ("PMC11059505", "Data-driven clusters depend on variables, sampling and algorithm choices; labels must remain descriptive unless they are stable across perturbations and an independent cohort, and outcome differences are not causal effects.", "Discussion: limitations"),
        },
    },
    "MG07": {
        "query": "ICU electronic health record measurement frequency informative missingness laboratory",
        "sources": [
            ("PMC12220764", "direct_comparator", "ICU measurement-frequency and missing-data process analysis."),
            ("PMC7810439", "design_analogue", "Methodological review of informative observation in routine health data."),
        ],
        "evidence": {
            "study_population": ("PMC12220764", "The empirical study compares measurement behavior across ICU patients and clinical strata; eICU analyses must retain unit type, hospital and stay identifiers so institutional practice is not mistaken for physiology.", "Methods: cohort and strata"),
            "time_zero_and_windows": ("PMC12220764", "Measurement frequencies are evaluated over fixed ICU-time intervals; the target should prespecify a common observation window and handle short stays without rewarding longer opportunity time.", "Methods: observation windows"),
            "variable_operationalization": ("PMC12220764", "The study treats measurement counts and missing-data rates as outcomes rather than nuisance fields; lactate frequency should be defined as both any measurement and opportunity-adjusted count.", "Methods: measurement-frequency definitions"),
            "missingness_and_censoring": ("PMC7810439", "The review shows that whether and when clinicians record a variable can carry prognostic information and violate missing-at-random assumptions; imputation must not erase the process being studied.", "Review: informative presence and observation taxonomy"),
            "primary_model_and_sensitivities": ("PMC12220764", "The empirical analysis compares frequencies while accounting for patient and care-context factors; eICU should use clustered uncertainty and test robustness to stay length, severity and hospital mix.", "Statistical analysis and supplementary models"),
            "table_and_figure_completeness": ("PMC12220764", "The report visualizes missingness and measurement-rate variation across variables and groups; the target should show denominators, zero-inflation, hospital/unit distributions and adjusted contrasts.", "Main figures and supplementary tables"),
            "conclusion_boundaries": ("PMC7810439", "Observation-process associations can reveal care patterns and bias but do not identify why a clinician ordered a test; unit-level differences cannot be interpreted as quality or causal effects without additional design.", "Review: discussion and limitations"),
        },
    },
    "MG08": {
        "query": "ICU discharge readmission death external validation MIMIC-IV follow-up",
        "sources": [
            ("PMC9810617", "design_analogue", "MIMIC-IV data scope and linkage boundary."),
            ("PMC9848213", "design_analogue", "Post-ICU readmission-or-death ascertainment and external validation."),
        ],
        "evidence": {
            "study_population": ("PMC9810617", "MIMIC-IV links hospital and ICU events within one health system and deidentifies longitudinal records; it is not a population-wide post-discharge registry.", "Data descriptor: scope and modules"),
            "time_zero_and_windows": ("PMC9810617", "Hospital discharge timestamps exist, but observation after discharge depends on later encounters returning to the same source; a 30-day complete follow-up window is not certified by ICU tables alone.", "Data descriptor: hospital admissions and date handling"),
            "variable_operationalization": ("PMC9848213", "The validation study defines a composite of ICU readmission or death after ICU discharge using site-specific electronic records; both event components and their ascertainment window require explicit linked sources.", "Methods: outcome definition"),
            "missingness_and_censoring": ("PMC9848213", "External validation relies on a known at-risk interval and recorded post-discharge events; absent linkage cannot be repaired by censoring everyone at discharge because that leaves no requested follow-up.", "Methods: follow-up and validation cohort"),
            "primary_model_and_sensitivities": ("PMC9848213", "The study evaluates external performance and retraining for a post-ICU outcome; such modeling is meaningful only after event and censoring certification, so the current Qualification12 item must fail before estimation.", "Methods: external validation and retraining"),
            "table_and_figure_completeness": ("PMC9848213", "The report includes cohort flow, outcome prevalence, discrimination and calibration across sites; no survival curve should be drawn when post-discharge risk sets are unobservable.", "Results tables and validation figures"),
            "conclusion_boundaries": ("PMC9810617", "The database descriptor supports in-system retrospective research but does not claim complete out-of-system readmission capture; the correct result is a data-linkage gap, not a low readmission estimate.", "Data descriptor: limitations"),
        },
    },
    "MG09": {
        "query": "eICU medication mapping drug names crosswalk multi center database heterogeneity",
        "sources": [
            ("PMC6132188", "design_analogue", "eICU medication tables, hospital coverage and data heterogeneity."),
            ("PMC12084561", "design_analogue", "Cross-database drug-name harmonization and covariate-shift methods."),
        ],
        "evidence": {
            "study_population": ("PMC6132188", "eICU aggregates stays from many hospitals with different interfaces and table coverage; exposure resolvability must be checked at hospital and stay levels before defining the cohort.", "Data descriptor: participating units and table coverage"),
            "time_zero_and_windows": ("PMC6132188", "Medication and infusion records use event offsets relative to ICU care, but coverage and charting conventions differ; a first-day window requires a certified event-time field for the named agent.", "Data descriptor: medication and infusion tables"),
            "variable_operationalization": ("PMC6132188", "Medication orders and infusion records are distinct and an order does not guarantee administration; an absent crosswalk entry cannot be replaced with a guessed synonym or therapeutic class.", "Data descriptor: medication and infusionDrug"),
            "missingness_and_censoring": ("PMC6132188", "Table availability varies across hospitals, creating structural absence that is different from a documented unexposed state; site coverage must be reported before any comparison.", "Data descriptor: data completeness and limitations"),
            "primary_model_and_sensitivities": ("PMC12084561", "The harmonization study maps heterogeneous EHR features for prediction and addresses covariate shift; mapping algorithms may propose candidates but cannot certify a causal exposure without source-table validation.", "Methods: feature mapping and reweighting"),
            "table_and_figure_completeness": ("PMC12084561", "Cross-site performance and feature-availability summaries make heterogeneity visible; this item instead needs a mapping receipt listing searched tables, terms and unresolved status before any clinical figure.", "Methods and results: mapping and site comparisons"),
            "conclusion_boundaries": ("PMC12084561", "Successful predictive harmonization does not prove that a specific administered drug is present in eICU; unresolved exposure identity must remain a fail-closed capability result.", "Discussion: limitations and transportability"),
        },
    },
    "MG10": {
        "query": "sepsis antibiotic timing symptom onset emergency department meta analysis guideline",
        "sources": [
            ("PMC8486643", "design_analogue", "Current sepsis guideline definitions for antimicrobial timing."),
            ("PMC12425674", "design_analogue", "Recent antibiotic-timing meta-analysis and time-zero heterogeneity."),
        ],
        "evidence": {
            "study_population": ("PMC8486643", "The guideline separates septic shock or high-likelihood sepsis from possible sepsis without shock; timing recommendations are conditional on these clinically assessed populations.", "Antimicrobial timing recommendations"),
            "time_zero_and_windows": ("PMC8486643", "Antibiotic clocks are anchored to recognition and urgent clinical assessment, not an unrecorded biologic symptom-onset time; substituting ICU admission would change the estimand.", "Guideline rationale for timing"),
            "variable_operationalization": ("PMC12425674", "The meta-analysis compares prespecified treatment-delay thresholds but documents inconsistent definitions of sepsis, shock and time zero across studies; the target exposure cannot be built without an observable onset anchor.", "Methods: exposure definitions; subgroup analyses"),
            "missingness_and_censoring": ("PMC12425674", "Included observational studies vary in timestamp availability and adjustment, producing substantial design heterogeneity; an entirely absent symptom-onset field is a structural data gap, not imputable missingness.", "Risk of bias and heterogeneity sections"),
            "primary_model_and_sensitivities": ("PMC12425674", "Pooled timing associations use stratified meta-analysis and sensitivity analyses across definitions; these methods cannot rescue a local cohort whose requested time origin is unobserved.", "Statistical analysis and sensitivity analyses"),
            "table_and_figure_completeness": ("PMC12425674", "Forest plots, subgroup results and study-definition tables expose between-study variation; the Qualification12 output should instead present a concise time-origin failure receipt and no effect plot.", "Study characteristics, forest plots and supplement"),
            "conclusion_boundaries": ("PMC8486643", "Guideline recommendations reflect clinical urgency and evidence certainty, not proof that an ICU database can estimate effects from symptom onset; fail-closed preserves the requested scientific question.", "Recommendations and evidence-quality statements"),
        },
    },
    "MG11": {
        "query": "mechanical ventilation dose response intensity duration mechanical power mortality",
        "sources": [
            ("PMC7906666", "design_analogue", "Time-varying graded ventilation-intensity and cumulative burden."),
            ("PMC10685677", "design_analogue", "Joint mechanical-power intensity and duration surface."),
        ],
        "evidence": {
            "study_population": ("PMC7906666", "The prospective registry studies invasively ventilated adults with repeated ventilator measurements; a binary ever-ventilated flag lacks the exposure detail used to study intensity-response.", "Methods: cohort and ventilation measurements"),
            "time_zero_and_windows": ("PMC10685677", "Mechanical-power burden is indexed jointly by intensity and duration during ventilation; dose begins only after ventilation starts and cannot be aligned solely to ICU admission.", "Methods: dynamic power-time construction"),
            "variable_operationalization": ("PMC10685677", "The exposure combines graded mechanical power with accumulated ventilation time, producing an interpretable two-dimensional burden; yes/no ventilation contains neither quantity nor duration.", "Methods: exposure calculation"),
            "missingness_and_censoring": ("PMC7906666", "Repeated ventilator data and covariates require explicit handling of missing measurements and informative treatment duration; binary exposure cannot identify unrecorded dose values through imputation.", "Methods and appendix: longitudinal missing data"),
            "primary_model_and_sensitivities": ("PMC7906666", "The study uses adjusted time-varying and cumulative-exposure models with sensitivity analyses; a dose-response contract requires at least three ordered levels or a continuous dose before such models are eligible.", "Statistical analysis and appendix"),
            "table_and_figure_completeness": ("PMC10685677", "The power-time burden is displayed as a response surface with supporting distributions and adjusted analyses; drawing a gradient from two binary groups would be a fabricated scientific figure.", "Main dynamic burden figures"),
            "conclusion_boundaries": ("PMC7906666", "Observed ventilation intensity is confounded by disease severity and treatment changes; even a real graded association would not prove causality, while the current binary input cannot answer the dose-response question at all.", "Discussion: limitations"),
        },
    },
    "MG12": {
        "query": "acute kidney injury renal replacement therapy cumulative incidence death competing risk ICU",
        "sources": [
            ("PMC10760471", "design_analogue", "Competing-risk estimands and model-selection tutorial."),
            ("PMC10008759", "design_analogue", "Severe-AKI cohort using death as a competing event."),
        ],
        "evidence": {
            "study_population": ("PMC10008759", "The SALTO cohort follows critically ill patients with severe AKI who survive to a defined landmark; its eligibility differs from an all-ICU RRT-initiation risk set and makes the landmark population explicit.", "Methods: cohort and follow-up"),
            "time_zero_and_windows": ("PMC10008759", "Long-term kidney outcomes are anchored to a prespecified post-acute landmark with subsequent follow-up; an ICU RRT analysis likewise needs certified event times from ICU admission for both RRT and death.", "Methods: outcomes and timeline"),
            "variable_operationalization": ("PMC10760471", "Competing-risk analysis requires mutually exclusive event types and valid event times; a generic death flag or an uncertified RRT indicator is insufficient to construct the cumulative-incidence function.", "Tutorial: competing events and data structure"),
            "missingness_and_censoring": ("PMC10008759", "The cohort distinguishes observed kidney outcomes, death and follow-up status; unknown event identity cannot be treated as ordinary right censoring without changing the cumulative-incidence estimand.", "Methods and supplement: outcome ascertainment"),
            "primary_model_and_sensitivities": ("PMC10760471", "The tutorial distinguishes cause-specific hazards from Fine-Gray subdistribution models and cumulative incidence; a Cox hazard is not a substitute for a CIF, and model choice depends on the scientific estimand.", "Tutorial: cause-specific and subdistribution approaches"),
            "table_and_figure_completeness": ("PMC10008759", "The report presents cumulative-incidence displays with death handled as a competing event and provides detailed outcome definitions; the target should produce no CIF until both event processes and a deterministic runner are certified.", "Outcome figures and supplementary definitions"),
            "conclusion_boundaries": ("PMC10760471", "Competing-risk methods answer different etiologic and prognostic questions and require explicit interpretation; the correct current output is an unsupported-estimand receipt, not a cause-naive survival claim.", "Tutorial: interpretation and conclusion"),
        },
    },
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_pack(source_pack_root: Path, task_bank_path: Path) -> dict[str, Any]:
    source_manifest = _load_json(source_pack_root / "source_manifest.json")
    review_manifest = _load_json(source_pack_root / "review_manifest.json")
    metadata = {row["pmcid"]: row for row in source_manifest["sources"]}
    reviews = {row["pmcid"]: row for row in review_manifest["sources"]}
    tasks = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in task_bank_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    if set(tasks) != set(ITEMS):
        raise ValueError("Qualification12 task IDs do not match the reviewed pack")

    built_items = []
    for task_id, spec in ITEMS.items():
        task = tasks[task_id]
        citations = []
        decisions = []
        cards = []
        evidence_by_source: dict[str, list[dict[str, str]]] = {
            pmcid: [] for pmcid, _, _ in spec["sources"]
        }
        for dimension in DIMENSIONS:
            pmcid, summary, locator = spec["evidence"][dimension]
            evidence_by_source[pmcid].append(
                {
                    "dimension": dimension,
                    "source_backed_summary": summary,
                    "locator": locator,
                }
            )
        for pmcid, role, rationale in spec["sources"]:
            source = metadata[pmcid]
            review = reviews[pmcid]
            key = f"{pmcid.lower()}_{task_id.lower()}"
            citations.append(
                {
                    "key": key,
                    "title": source["title"],
                    "year": source["year"],
                    "venue": source["journal"],
                    "relevance": rationale,
                    "doi": source["doi"],
                    "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
                    "pmid": source["pmid"],
                    "publication_types": [],
                }
            )
            decisions.append(
                {
                    "citation_key": key,
                    "source": "europe_pmc_review",
                    "disposition": "include",
                    "evidence_role": role,
                    "rationale": rationale,
                    "query": f'EXT_ID:{source["pmid"]} AND SRC:MED',
                    "population_match": role == "direct_comparator",
                    "exposure_match": role == "direct_comparator",
                    "outcome_match": role == "direct_comparator",
                    "design_excerpt_available": True,
                    "publication_type_eligible": True,
                }
            )
            cards.append(
                {
                    "schema_version": "easyicu.literature_design_evidence/1",
                    "citation_key": key,
                    "evidence_role": role,
                    "access_mode": "open_access_fulltext",
                    "full_text_locator": source["full_text_locator"],
                    "full_text_sha256": review["full_text_sha256"],
                    "supplement_status": review["supplement_status"],
                    "supplement_sha256": review.get("supplement_sha256"),
                    "reviewed_at": REVIEWED_AT,
                    "evidence": evidence_by_source[pmcid],
                }
            )
        record_queries = {
            citation["key"]: [
                f'EXT_ID:{citation["pmid"]} AND SRC:MED'
            ]
            for citation in citations
        }
        bundle = {
            "research_question": task["question"],
            "citations": citations,
            "prisma": {
                "identified": 2,
                "duplicates_removed": 0,
                "screened": 2,
                "eligible": 2,
                "included": 2,
            },
            "search_provenance": {
                "schema_version": "easyicu.literature_search_provenance/1",
                "curated_seed_count": 0,
                "sources_enabled": ["europe_pmc_review"],
                "sources_returning": ["europe_pmc_review"],
                "search_queries": {
                    "europe_pmc_review": [
                        *[query for queries in record_queries.values() for query in queries],
                    ]
                },
                "record_queries": record_queries,
                "search_conducted": True,
                "searched_at": REVIEWED_AT,
                "note": (
                    "The two retained identifiers were verified through Europe PMC; "
                    "open full text plus any published supplement was reviewed. The "
                    "manual candidate-screening query is stored at item level and its "
                    "unbounded result count is not represented as a PRISMA flow. "
                    "Published effect estimates are not expected answers."
                ),
            },
            "authority_trace": None,
            "screening_decisions": decisions,
            "design_evidence_cards": cards,
        }
        built_items.append(
            {
                "task_id": task_id,
                "title": task["title"],
                "expected_behavior": task["expected_behavior"],
                "expected_gap_reason": task["expected_gap_reason"],
                "manual_screening_query": spec["query"],
                "bound_preplan_literature": bundle,
            }
        )

    return {
        "schema_version": "easyicu.qualification12_literature_design_pack/1",
        "profile_ref": "npj_dm_qualification12_design_dev/20260825",
        "reviewed_at": REVIEWED_AT,
        "selection_policy": {
            "sources_per_item": 2,
            "required_dimensions": list(DIMENSIONS),
            "full_text_required": True,
            "supplement_disposition_required": True,
            "recent_source_window_years": 5,
            "published_effects_are_expected_answers": False,
            "note": (
                "Articles are external design and reporting benchmarks. Their numerical "
                "results must never be copied into benchmark expectations or used to "
                "override database-specific estimates."
            ),
        },
        "source_manifest_sha256": __import__("hashlib").sha256(
            (source_pack_root / "source_manifest.json").read_bytes()
        ).hexdigest(),
        "review_manifest_sha256": __import__("hashlib").sha256(
            (source_pack_root / "review_manifest.json").read_bytes()
        ).hexdigest(),
        "items": built_items,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pack-root", required=True, type=Path)
    parser.add_argument(
        "--task-bank",
        type=Path,
        default=Path("benchmarks/meta_generalization/meta_benchmark.jsonl"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_pack(args.source_pack_root, args.task_bank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
