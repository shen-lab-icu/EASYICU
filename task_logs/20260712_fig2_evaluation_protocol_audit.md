# Figure 2 nine-task evaluation protocol audit

> Task ID: `FIG2-EVAL-PROTOCOL-AUDIT`
> Date: 2026-07-12
> Scope: local rubric implementation, external scientific-agent benchmarks, and npj Digital Medicine fit.
> User priority: finish the experiments before manuscript assembly; use E2/E3/H3 to expose general framework problems, then freeze and run the paper-facing evaluation.

## Executive conclusion

The five intended dimensions are conceptually sound: scientific plan, code execution, result validity, evidence binding, and audit/conclusion safety. The current implementation is not yet publication-grade as a formal benchmark because the canonical nine have been used as development cases, most tasks lack frozen independent result references, the paper-facing plot substitutes internal “result sanity” for result validity, the task-specific safety key is not connected, and Tier 2/Tier 3 external evaluation has not been run.

The correct paper positioning is therefore:

> a prespecified nine-task ICU research capability suite

not:

> a comprehensive benchmark proving general ICU research ability.

Nine tasks are not inherently too few. BLADE was built around 12 deep open-ended questions, but paired them with hundreds of expert decisions, repeated runs, evaluator validation and uncertainty. EasyICU can retain the nine deep tasks if it adds independent references, frozen configuration, repetitions, blind expert review and an untouched held-out suite.

## Local implementation findings

1. `src/easyicu/research_agent/icu_agent_bench.py` explicitly labels the machinery an internal evaluation protocol rather than a published benchmark. It records no external adjudication/public frozen manifest, one partially frozen synthetic gold task, and unexecuted Tier 2/Tier 3 evaluation.
2. `evaluation_scorecard.py` defines the intended five dimensions, but `plot_canonical9_scorecard.py` excludes `result_validity` because only 1/9 tasks is scored and replaces it with `result_sanity`, derived from the same run's numeric/analysis validator status.
3. The same plot excludes `audit_conclusion_safety` because all imported rows are 1.0 while lacking a per-task hazard key, then replaces it with reporting completeness. The displayed five columns therefore differ from the declared five constructs.
4. The canonical JSONL carries question/outcome/predictor/kind but not the full evaluator-side hazard/forbidden-claim specification that the scorecard expects. The safety dimension is consequently not task-specific in the current paper-facing artifact.
5. The task prompts contain protocol-rich instructions (ordered exposure handling, complete-case bias, immortal time, confounding/positivity, split leakage and calibration). That is valid for testing protocol execution, but it does not test whether the agent independently discovers those hazards.
6. The tasks provide broad method-family coverage but are concentrated in MIMIC-IV, first-24-hour data and mortality-related endpoints. Figure 2 therefore tests methodological breadth on a common ICU substrate; cross-database generality must come from Figure 3 and held-out variation.
7. The current plot reads one scorecard per task and computes a mean over whichever dimensions happen to be scored. This is not a reliability estimate and makes rows with different missing dimensions incomparable.
8. The existing scorecard source points to June 13 artifacts and a machine-specific project path. It is stale relative to the current 6/9 state and cannot be promoted to the final manuscript.

## Answer-leakage boundary

The agent must never receive evaluator gold values or case-specific solution code.

Allowed evaluator-side preparation:

- freeze the research question, input SHA and task type;
- independently compute hidden oracle checks for cohort size, exposure/outcome definition, estimand, numeric tolerance and required structural outputs;
- define atomic required-hazard and forbidden-claim checks;
- define multiple scientifically acceptable method families where more than one approach is valid;
- have ICU and statistical reviewers validate the evaluator package before the final run.

Not allowed:

- placing oracle numbers in the agent prompt;
- adding task IDs/variables/answers to shared prompts or runners;
- choosing the exposure, cohort, model or cluster count inside deterministic runners;
- changing the task or score after seeing the final result;
- treating oracle calculations as manuscript findings. Manuscript findings still come from the research-agent pipeline; the oracle is a hidden sanity check only.

## Revised experiment sequence

### Phase D — development stress completion

1. Continue Fresh E3 from Step 06, then H3 and E2.
2. Use `aware` only, one heavy run at a time, completion probe first, SDK retries 0 and step-level `stop-after`.
3. A failed step may trigger a shared-engine change only when the defect is a case-neutral invariant. Case-specific requirements remain in the item/rubric.
4. Each defect gets one general repair plus focused regression and targeted rerun. A repair requiring benchmark keywords, a primary deterministic scientific runner, or relaxed meta-generalization probes is rejected.
5. All of these runs remain development evidence, even if they become reportable, because the cases have influenced architecture development.

### Phase F — protocol and framework freeze

Freeze one versioned manifest containing:

- all nine questions and input/data SHA values;
- one code commit and dependency lock;
- one model/provider/version/date and prompt hash;
- temperature, seed, time/token budget, timeout and retry policy;
- per-task expected artifacts, acceptable method families, hidden numeric/structural oracle, hazards and forbidden claims;
- scorer version and task-applicability map;
- rule for external/API failure and a predeclared retry/pass@k policy.

After this point, final outcomes do not justify further tuning. A genuine framework bug creates a new protocol version and invalidates affected canonical results.

### Phase C — final canonical evaluation

- Run all nine fresh under the same frozen configuration. Development checkpoints are not final artifacts.
- Recommended strong design: three independent fresh runs per task (`9 x 3`) with all attempts counted, matching the repeated-attempt pattern used in scientific-agent benchmarks.
- If compute/time forces a smaller design, run one pass@1 canonical for all nine plus predeclared repeated runs for high-variance sentinel tasks, and describe the result only as case-suite evidence rather than a reliability estimate.
- Report per-task results and failure types; do not average heterogeneous dimensions or omit NA dimensions from a composite mean.
- Report wall time, model calls/tokens or the closest available resource measure, replan/repair counts and external failures.

### Phase H — held-out and human validation

- Add 3–6 tasks/variants never used to repair the framework, including a positive task, an expected fail-closed task, a different database, a different time origin/outcome and a different missingness mechanism.
- Keep their evaluator oracle hidden and prohibit engine tuning after opening results; failures are reported.
- Have ICU/clinical-research and biostatistics/informatics experts validate the task/oracle packages before execution.
- Use blinded independent reviewers for final plan appropriateness and conclusion safety. The target is four reviewers for medical-research evaluation when feasible, with weighted kappa/Krippendorff alpha and adjudication rules reported.
- LLM jury may be a supplementary audit only, calibrated against the human labels; it is not the primary scientific-validity judge.

## Figure 2 scoring design

Keep the original five concepts, but operationalize them with task-specific atomic criteria:

1. **Scientific plan**: blinded expert assessment of estimand/cohort/method/sensitivity rationale, not merely the presence of a table and figure.
2. **Execution**: deterministic completion of all required primary outputs; external failures are separately classified.
3. **Result validity**: hidden independent oracle plus task-specific structural and numerical checks; no substitution with “internal validator passed.”
4. **Evidence binding**: current producer, SHA, source-data traceability and verified manuscript numbers.
5. **Safety/claim calibration**: task-specific hazard key and forbidden claims, with blind clinical/statistical review.

Reporting completeness belongs in Extended Data/Supplement. The main plot should show the five dimensions and overall tri-state per task, without a mean-score bar. NA remains NA.

## Comparator decision

If the manuscript claims that the EasyICU architecture improves performance relative to a generic agent, a paired comparator or ablation is required. A resource-bounded option is to preselect one easy, one medium and one hard task and compare the same backbone/budget under:

- generic code agent;
- full EasyICU;
- EasyICU with evidence/audit gates removed or disabled.

This would be a new ablation authorization; it must not be run as the historical `naive` arm without the user's explicit approval. If no comparator is run, the manuscript claim must be limited to an auditable capability demonstration rather than causal attribution of improvement.

## npj Digital Medicine assessment

EasyICU is plausibly in scope as an innovative clinical-informatics and AI research platform, not merely an off-the-shelf LLM study. Acceptance cannot be guaranteed. The current nine single development runs, self-scored without comprehensive gold references, would carry high reviewer risk and could be considered a small preliminary validation.

The risk becomes materially lower if the submission includes:

- a frozen same-configuration canonical batch;
- independent task-specific oracle and human blind review;
- repetitions and uncertainty;
- an untouched held-out suite;
- a small paired baseline/ablation or appropriately bounded claims;
- complete public protocol, rubric, prompts/configuration, traces, failures, source data and code;
- cross-database validation in Figure 3 and pipeline-produced discovery evidence in Figure 5.

Primary-source notes are preserved in `sources/research_fig2_evaluation_npj_20260712.md`.
