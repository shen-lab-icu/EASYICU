# EasyICU research-agent capability matrix

_Generated from `easyicu.research_agent.capability_registry`. Do not edit by hand — edit the registry and regenerate._

**Primary analysis** = how the reported estimand is computed. **Figure** = how the publication figure is rendered. The two are independent: a family can have a deterministic figure while its primary analysis is LLM-coded.

| Study-design family | Primary analysis | Primary estimand | Runner | Figure | Fail-closed when contract unmet |
| --- | --- | --- | --- | --- | --- |
| Survival / time-to-event | LLM-coded ⚠️ | Agent-coded time-to-event estimand under the declared survival method; value/provenance checked | — | deterministic ✅ (`time_to_event`) | The agent step fails when certified follow-up/event inputs are absent, and survival plausibility/provenance gates reject invalid event counts, effect scales, or unsupported estimands. |
| Causal inference / target-trial emulation | LLM-coded ⚠️ | Agent-coded causal contrast under a declared target-trial/identification strategy; assumptions and balance checked | — | deterministic ✅ (`causal_emulation`) | The agent step fails when its declared exposure, outcome, time zero, or adjustment inputs cannot be resolved; balance, positivity, and causal-language gates reject unsupported claims. |
| Association — graded ordinal exposure (dose-response) | LLM-coded ⚠️ | Agent-coded ordered-exposure association under the declared trend method; per-stage products value-checked | — | deterministic ✅ (`base_association_skill`) | The ordered-product contract rejects fewer than three declared levels, invalid level ordering, cohort drift, or missing trend statistics; a binary/continuous exposure is never coerced into an ordinal gradient. |
| Association — general (non-graded) | LLM-coded ⚠️ | LLM-coded adjusted association (logistic/linear); bound via NumericClaim + primary-effect extractor | — | deterministic ✅ (`base_association_skill`) | LLM code failure -> mechanical code_repair only (no deterministic association refit or estimator substitution) -> if still failing the step fails, the execution gate floors the status to diagnostic_only, and the specific error is surfaced (never a silent pass). |
| Prediction / risk modelling | LLM-coded ⚠️ | LLM-coded discrimination + calibration (AUROC, calibration); value-provenance verified | — | deterministic ✅ (`prediction`) | LLM code failure -> repair -> fail-closed. manuscript_numeric_auditor catches rounded/hallucinated metrics (caught AUROC 0.766->0.7 in a pilot). |
| Phenotyping / clustering | LLM-coded ⚠️ | Agent-planned cluster solution; outcome-by-cluster kept descriptive (not causal) | — | deterministic ✅ (`phenotyping`) | figure_strategy anti-pattern blocks 'clusters are causal entities'; an LLM failure fails closed to diagnostic_only. |
| Descriptive / measurement audit | LLM-coded ⚠️ | LLM-coded descriptive summaries / Table One / measurement-process audits | — | deterministic ✅ (`base_association_skill`) | Evidence STRICT mode blocks unbound sentences; the plausibility gate flags implausible descriptives before they reach the manuscript. |

## Auxiliary deterministic runners (support, not family-primary)

| Runner | Purpose | Fail-closed |
| --- | --- | --- |
| `absolute_risk_context` | Render descriptive exposure prevalence and absolute-risk context from an explicit product contract. | Declines figure/primary-effect contracts and blocks when the declared descriptive columns are unavailable. |
| `robustness_sensitivity` | Replay an agent-locked primary model across prespecified robustness variants. | Requires a locked model/specification contract and never selects the primary exposure, outcome, cohort, or estimator. |
| `missingness_measurement_audit` | Per-concept measured-vs-missing counts + structural-vs-measurement split for a missingness / measurement-process audit step (never imputes). | Blocks with a reason when no <concept>_measured columns resolve. The figure step renders the registered audit product via the data_quality->missingness renderer. |
| `trajectory_cluster_stability` | Compute a complete planner-owned, digest-bound trajectory-cluster stability specification without selecting the representation, model, cluster count, resampling design, seed policy, or decision threshold. | Requires one dedicated stability owner, exact typed upstream products, and the closed supported refit contract. Unsupported or failed refits remain diagnostic and never fall back to coder repair, another method, another seed policy, or another cluster count. |

## Known unsupported estimands (explicit boundaries)

Deliberately out of scope — these must **fail closed**, not be approximated by a nearby estimand:

- **Competing-risks cumulative incidence (Fine-Gray / CIF)** — No deterministic runner. A cause-naive Cox HR is NOT a CIF, so a competing-risks question (for example, an event with death as a competing risk) must fail closed to diagnostic_only — not be answered with a Cox HR.

## Fail-closed / gap-report ladder

What happens when no valid runner or data contract exists — the pipeline fails **closed** with a surfaced reason, never open:

- **1. Agent method and product contract** — The planner/coder owns the scientific method, cohort, exposure and outcome. Deterministic code is limited to validated calculation primitives or an explicit auxiliary product contract; it does not preflight-replace a primary estimand.
- **2. Runner contract unmet** — An auxiliary runner writes status=blocked + a specific blocking_reason when its declared standardized inputs are missing or invalid. It never guesses scientific variables or a surrogate method. Optional auxiliary steps degrade to status=skipped + not_applicable when their input is legitimately absent.
- **3. Agent execution** — The agent generates the planned primary analysis; code repair and statistical validators may repair implementation faults but never replace the declared scientific method.
- **4. Output / validity gates (fail-closed)** — execution_complete (any failed step -> False); evidence_complete (STRICT: unbound citations blocked); numeric_verified (value-level provenance: hallucinated numbers blocked); analysis_validated (plausibility + survival-estimand + figure-credit + headline==primary-estimand gates); replan_budget (runaway loop -> advisory only after a clean, bound primary result, else demote).
- **5. Verdict** — The status ladder (publication_ready > manuscript_ready > analysis_only > diagnostic_only) and the scorecard tristate (gate_reportable / analysis_only / diagnostic_only) floor to diagnostic_only whenever a gate fails, with the specific reason surfaced. INVARIANT: a capability gap is always reported, never silently filled with a fabricated result.
