# EasyICU research-agent capability matrix

_Generated from `easyicu.research_agent.capability_registry`. Do not edit by hand — edit the registry and regenerate._

**Primary analysis** = how the reported estimand is computed. **Figure** = how the publication figure is rendered. The two are independent: a family can have a deterministic figure while its primary analysis is LLM-coded.

| Study-design family | Primary analysis | Primary estimand | Runner | Figure | Fail-closed when contract unmet |
| --- | --- | --- | --- | --- | --- |
| Survival / time-to-event | deterministic ✅ | Cox proportional-hazards hazard ratio (+ Kaplan-Meier curve data) | `survival_primary_cox` | deterministic ✅ (`time_to_event`) | Runner blocks (status=blocked) when the follow-up column is absent or uncensored; the survival-estimand integrity gate rejects any headline that is not the deterministic Cox HR. |
| Causal inference / target-trial emulation | deterministic ✅ | Stabilised-IPTW marginal odds ratio (+ covariate balance, propensity, target-trial protocol) | `causal_primary_iptw` | deterministic ✅ (`causal_emulation`) | Runner blocks with 'Missing required causal columns' or 'Exposure groups too small'; positivity enforced by propensity trimming. The design schematic reads the runner's target_trial_protocol.csv. |
| Association — graded ordinal exposure (dose-response) | deterministic ✅ | Adjusted odds ratio per +1 stage (trend) + per-stage forest + monotonicity | `ordinal_dose_response` | deterministic ✅ (`base_association_skill`) | Runner blocks with 'Could not resolve a graded ordinal exposure (>=3 levels)'; a binary/continuous exposure is never coerced into a grade. Routes only on an explicit dose-response signal. |
| Association — general (non-graded) | LLM-coded ⚠️ | LLM-coded adjusted association (logistic/linear); bound via NumericClaim + primary-effect extractor | — | deterministic ✅ (`base_association_skill`) | LLM code failure -> code_repair -> if still failing the step fails, the execution gate floors the status to diagnostic_only, and the specific error is surfaced (never a silent pass). |
| Prediction / risk modelling | LLM-coded ⚠️ | LLM-coded discrimination + calibration (AUROC, calibration); value-provenance verified | — | deterministic ✅ (`prediction`) | LLM code failure -> repair -> fail-closed. manuscript_numeric_auditor catches rounded/hallucinated metrics (caught AUROC 0.766->0.7 in a pilot). |
| Phenotyping / clustering | LLM-coded ⚠️ | LLM-coded cluster solution + stability; outcome-by-cluster kept descriptive (not causal) | — | deterministic ✅ (`phenotyping`) | figure_strategy anti-pattern blocks 'clusters are causal entities'; an LLM failure fails closed to diagnostic_only. |
| Descriptive / measurement audit | LLM-coded ⚠️ | LLM-coded descriptive summaries / Table One / measurement-process audits | — | deterministic ✅ (`base_association_skill`) | Evidence STRICT mode blocks unbound sentences; the plausibility gate flags implausible descriptives before they reach the manuscript. |

## Auxiliary deterministic runners (support, not family-primary)

| Runner | Purpose | Fail-closed |
| --- | --- | --- |
| `cohort_definition_overlap` | Overlap / concordance of alternative cohort definitions. | Blocks with a reason when no alternative definition is registered. |
| `cohort_definition_sensitivity` | Re-fit the primary estimand under alternative cohort definitions. | Degrades to a CLEAN skip (status=skipped, not_applicable) when no alternative_cohort_attrition.csv exists upstream — it does NOT block (this removed the H2 'produce the missing file' replan loop). |
| `missingness_measurement_audit` | Per-concept measured-vs-missing counts + structural-vs-measurement split for a missingness / measurement-process audit step (never imputes). | Blocks with a reason when no <concept>_measured columns resolve. Owns the audit so the LLM coder no longer times out on it (~27.6 min then fail); the figure step renders via the data_quality->missingness renderer. |
| `trajectory_clustering` | Deterministic phenotyping partition: features over OBSERVED trajectory windows (never zero-imputed), silhouette-selected k, seed-stability, and a DESCRIPTIVE outcome-by-cluster contrast (adjusted_effect=None). | Blocks with a specific reason when no trajectory columns resolve — never fabricates a partition. Supports the phenotyping figure (cluster_characteristics + clustering_metrics) without binding a scalar primary estimand, so phenotyping stays LLM-coded-primary for the report layer's no-deterministic-primary waiver. |

## Known unsupported estimands (explicit boundaries)

Deliberately out of scope — these must **fail closed**, not be approximated by a nearby estimand:

- **Competing-risks cumulative incidence (Fine-Gray / CIF)** — No deterministic runner. A cause-naive Cox HR is NOT a CIF, so a competing-risks question (e.g. RRT with death as a competing risk) must fail closed to diagnostic_only — not be answered with a Cox HR. Exercised by meta-benchmark probe MG12.

## Fail-closed / gap-report ladder

What happens when no valid runner or data contract exists — the pipeline fails **closed** with a surfaced reason, never open:

- **1. Deterministic runner match** — A family's preflight predicate fires only for its PRIMARY result step (not a figure/sensitivity step). If it fires and the data contract is met, the deterministic estimand is used and owns its step contract.
- **2. Runner contract unmet** — The deterministic runner writes status=blocked + a SPECIFIC blocking_reason (e.g. missing exposure/outcome column, degenerate groups, non-ordinal exposure). It never guesses a surrogate — case-specific values come from research_context.json only. Auxiliary steps degrade to status=skipped + not_applicable when their input is legitimately absent.
- **3. No deterministic runner for the family** — The LLM coder generates the analysis; code_repair applies deterministic post-failure repairs (KeyError strip, missing-helper restore, ...).
- **4. Output / validity gates (fail-closed)** — execution_complete (any failed step -> False); evidence_complete (STRICT: unbound citations blocked); numeric_verified (value-level provenance: hallucinated numbers blocked); analysis_validated (plausibility + survival-estimand + figure-credit + headline==primary-estimand gates); replan_budget (runaway loop -> advisory if converged-clean with a bound deterministic primary, else demote).
- **5. Verdict** — The status ladder (publication_ready > manuscript_ready > analysis_only > diagnostic_only) and the scorecard tristate (gate_reportable / analysis_only / diagnostic_only) floor to diagnostic_only whenever a gate fails, with the specific reason surfaced. INVARIANT: a capability gap is always reported, never silently filled with a fabricated result.

